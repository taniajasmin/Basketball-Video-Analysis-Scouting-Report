import cv2
import numpy as np
import os
import base64
from pytubefix import YouTube
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
MOTION_THRESHOLD_DEFAULT = 50
MAX_FRAMES_PER_SEGMENT_DEFAULT = 20
SEGMENT_DURATION_DEFAULT = 300
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
JPEG_QUALITY = 80
MAX_FRAMES = 200
RETRIES = 5
DELAY = 2

# Valid colors for jersey validation
VALID_COLORS = [
    'red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple',
    'gray', 'grey', 'pink', 'brown', 'navy'
]
SIMILAR_COLORS = {
    'blue': ['navy'],
    'green': []
}

# Color ranges for detection (gold, silver, cyan removed)
COLOR_RANGES = {
    'red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
    'blue': [([110, 120, 70], [130, 255, 255])],
    'green': [([40, 120, 70], [70, 255, 255])],
    'yellow': [([20, 120, 70], [40, 255, 255])],
    'white': [([0, 0, 200], [180, 30, 255])],
    'black': [([0, 0, 0], [180, 255, 30])],
    'orange': [([10, 120, 70], [20, 255, 255])],
    'purple': [([130, 120, 70], [160, 255, 255])],
    'gray': [([0, 0, 50], [180, 30, 200])],
    'grey': [([0, 0, 50], [180, 30, 200])],
    'pink': [([160, 120, 70], [170, 255, 255])],
    'brown': [([10, 60, 20], [30, 255, 100])],
    'navy': [([100, 120, 20], [110, 255, 100])]
}

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY not found in .env file.")
    raise ValueError("OPENAI_API_KEY not found in .env file.")
client = OpenAI(api_key=api_key)

def suggest_color(invalid_color):
    """Suggest valid colors based on invalid input."""
    invalid_color = invalid_color.lower()
    for valid_color, similar in SIMILAR_COLORS.items():
        if invalid_color in similar or invalid_color == valid_color:
            return valid_color
    if invalid_color in ['gold', 'golden']:
        return 'yellow'
    if invalid_color in ['silver', 'metallic']:
        return 'gray'
    if invalid_color in ['cyan', 'turquoise']:
        return 'blue'
    return None

def is_color_present(frame, color_name):
    """Check if a specified color is prominent in the frame."""
    if color_name not in COLOR_RANGES:
        return False
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = COLOR_RANGES[color_name]
    mask = None
    for lower, upper in ranges:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        temp_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)
    
    color_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    return color_pixels / total_pixels > 0.02

def download_youtube_video(url):
    try:
        logging.info(f"Downloading video: {url}")
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first() or \
                 yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            logging.error("No suitable video stream found.")
            return None
        video_path = stream.download(output_path='.', filename=f"game_video_{yt.video_id}.mp4")
        logging.info(f"Downloaded to {video_path}")
        return video_path
    except Exception as e:
        logging.error(f"Error downloading video: {str(e)}")
        return None

def calibrate_motion_parameters(video_path):
    logging.info("Calibrating motion parameters...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return MOTION_THRESHOLD_DEFAULT, MAX_FRAMES_PER_SEGMENT_DEFAULT, SEGMENT_DURATION_DEFAULT
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_duration = total_frames / fps if fps > 0 else 0
    if total_duration <= 0:
        logging.warning("Invalid duration or FPS.")
        return MOTION_THRESHOLD_DEFAULT, MAX_FRAMES_PER_SEGMENT_DEFAULT, SEGMENT_DURATION_DEFAULT
    
    motion_values = []
    prev_frame = None
    sample_frames = min(500, total_frames)
    frame_count = 0

    while cap.isOpened() and frame_count < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_prev, gray_current)
            motion = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1])
            motion_values.append(motion)
        prev_frame = frame.copy()
        frame_count += 1

    cap.release()
    
    if not motion_values:
        logging.warning("No motion data collected.")
        return MOTION_THRESHOLD_DEFAULT, MAX_FRAMES_PER_SEGMENT_DEFAULT, SEGMENT_DURATION_DEFAULT

    motion_threshold = np.percentile(motion_values, 75)
    motion_density = len(motion_values) / total_duration
    max_frames_per_segment = max(10, min(50, int(20 * (motion_density / 10))))
    segment_duration = max(60, min(600, int(300 / (motion_density / 10))))
    logging.info(f"Calibrated: threshold={motion_threshold:.2f}, max_frames={max_frames_per_segment}, duration={segment_duration}")
    return motion_threshold, max_frames_per_segment, segment_duration

def extract_frames_with_motion(video_path, jersey_color="Unknown"):
    logging.info("Extracting frames with motion detection...")
    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return []
    
    motion_threshold, max_frames_per_segment, segment_duration = calibrate_motion_parameters(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_duration = total_frames / fps if fps > 0 else 0
    frames = []
    prev_frame = None
    segment_start = 0
    frame_count = 0
    color_check_frames = 100
    color_detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < color_check_frames and jersey_color != "Unknown":
            if is_color_present(frame, jersey_color):
                color_detected_count += 1

        if prev_frame is not None:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_prev, gray_current)
            motion = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1])
            
            max_allowed_frames = max_frames_per_segment * (int(total_duration // segment_duration) + 1)
            if motion > motion_threshold and len(frames) < max_allowed_frames:
                resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                frames.append(buffer.tobytes())

        prev_frame = frame.copy()
        frame_count += 1
        current_time = frame_count / fps
        if current_time - segment_start >= segment_duration:
            segment_start = current_time

    cap.release()
    logging.info(f"Extracted {len(frames)} frames from {total_frames}.")
    
    if jersey_color != "Unknown" and color_detected_count < color_check_frames * 0.2:
        logging.warning(f"Jersey color '{jersey_color}' rarely detected.")
        print(f"Warning: Color '{jersey_color}' not prominent in video.")
    
    if len(frames) > MAX_FRAMES:
        sample_rate = len(frames) // MAX_FRAMES
        frames = frames[::sample_rate]
        logging.info(f"Sampled to {len(frames)} frames.")

    if 'game_video' in video_path:
        try:
            os.remove(video_path)
            logging.info(f"Deleted: {video_path}")
        except Exception as e:
            logging.warning(f"Failed to delete {video_path}: {str(e)}")

    return frames

def analyze_frame_with_gpt(frames, max_frames=MAX_FRAMES, retries=RETRIES, delay=DELAY):
    logging.info(f"Analyzing {min(len(frames), max_frames)} frames...")
    descriptions = []
    total_batches = (min(len(frames), max_frames) + 19) // 20
    
    for i in range(0, min(len(frames), max_frames), 20):
        logging.info(f"Processing batch {i//20 + 1}/{total_batches}...")
        batch = frames[i:i + 20]
        
        for frame_idx, frame in enumerate(batch):
            base64_image = base64.b64encode(frame).decode('utf-8')
            prompt = (
                "Describe the basketball gameplay action in this frame, focusing on:\n"
                "- Scoring (points, field goal attempts, made shots)\n"
                "- Assists (passes leading to scores)\n"
                "- Rebounds (offensive or defensive)\n"
                "- Steals or blocks (defensive plays)\n"
                "- Mistakes (missed shots, turnovers)\n"
                "Provide a concise description without game context."
            )
            
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ],
                        max_tokens=300
                    )
                    descriptions.append(response.choices[0].message.content)
                    time.sleep(delay)
                    break
                except Exception as e:
                    logging.error(f"Error analyzing frame: {e}")
                    if "rate limit" in str(e).lower() or "connection" in str(e).lower():
                        if attempt < retries - 1:
                            sleep_time = delay * (2 ** attempt)
                            logging.info(f"Retrying ({attempt+1}/{retries}) after {sleep_time}s...")
                            time.sleep(sleep_time)
                        else:
                            logging.warning("Max retries reached. Skipping frame.")
                            descriptions.append(None)
                    else:
                        logging.warning(f"Unexpected error: {e}. Skipping.")
                        descriptions.append(None)
                        break
    
    logging.info(f"Analyzed {len(descriptions)} frames.")
    return descriptions

def aggregate_stats(descriptions, is_team=False):
    logging.info("Aggregating stats...")
    stats = {
        'points': 0,
        'field_goal_attempts': 0,
        'field_goal_made': 0,
        'assists': 0,
        'rebounds': 0,
        'steals_blocks': 0,
        'weaknesses': []
    }
    processed_frames = set()
    valid_descriptions = [desc for desc in descriptions if desc is not None]
    
    if not valid_descriptions:
        logging.warning("No valid descriptions.")
        return stats

    for desc in valid_descriptions:
        desc_id = hash(desc)
        if desc_id in processed_frames:
            continue
        processed_frames.add(desc_id)

        desc_lower = desc.lower()
        if "score" in desc_lower or ("made" in desc_lower and "shot" in desc_lower):
            stats['points'] += 2 if is_team else 1
            stats['field_goal_made'] += 1
            stats['field_goal_attempts'] += 1
        elif "shot" in desc_lower:
            stats['field_goal_attempts'] += 1
        if "assist" in desc_lower or ("pass" in desc_lower and "score" in desc_lower):
            stats['assists'] += 1
        if "rebound" in desc_lower:
            stats['rebounds'] += 1
        if "steal" in desc_lower or "block" in desc_lower:
            stats['steals_blocks'] += 1
        if any(w in desc_lower for w in ["missed", "turnover", "error"]):
            stats['weaknesses'].append(desc)
    
    stats['fg_percentage'] = (stats['field_goal_made'] / stats['field_goal_attempts'] * 100) if stats['field_goal_attempts'] > 0 else 0
    
    if not is_team and len(valid_descriptions) > 0:
        stats['points'] = max(1, int(stats['points'] / (len(valid_descriptions) / 20)))
    
    logging.info(f"Stats: Points={stats['points']}, FG%={stats['fg_percentage']:.1f}%, Rebounds={stats['rebounds']}")
    return stats

def generate_player_report(stats, player_name, jersey_number, jersey_color, height, gender, opponent_team):
    logging.info(f"Generating report for {player_name}...")
    current_time = datetime.now()
    date_str = current_time.strftime("%m/%d/%Y %I:%M %p +06")

    prompt = (
        f"Generate a basketball scouting report for {player_name} (Jersey Number: {jersey_number}, Jersey Color: {jersey_color}, Height: {height}, Gender: {gender}) against {opponent_team} on {date_str}:\n"
        f"- Points Per Game (PPG): {stats['points']}\n"
        f"- Field Goal Percentage (FG%): {stats['fg_percentage']:.1f}%\n"
        f"- Rebounds Per Game (RPG): {stats['rebounds']}\n"
        f"- Assists Per Game (APG): {stats['assists']}\n"
        f"- Steals & Blocks: {stats['steals_blocks']}\n"
        f"- Weaknesses: {', '.join(stats['weaknesses'][:3]) if stats['weaknesses'] else 'None'}\n"
        f"Format with sections:\n"
        f"- Overview: Summarize {player_name}'s performance, noting {height} and {jersey_color} jersey.\n"
        f"- Strength: Highlight skills, considering physical attributes.\n"
        f"- Weaknesses: List weaknesses and suggest improvements for {player_name}.\n"
        f"- Projection: Assess {player_name}'s potential, factoring {height}.\n"
        f"Be harsh and specific."
    )
    
    logging.debug(f"Player prompt: {prompt}")
    
    for attempt in range(RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            if "rate limit" in str(e).lower():
                sleep_time = DELAY * (2 ** attempt)
                logging.info(f"Retrying ({attempt+1}/{RETRIES}) after {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

def generate_team_report(stats, jersey_color, gender, opponent_team):
    logging.info("Generating team report...")
    current_time = datetime.now()
    date_str = current_time.strftime("%m/%d/%Y %I:%M %p +06")

    prompt = (
        f"Generate a basketball team scouting report for a team (Jersey Color: {jersey_color}, Gender: {gender}) against {opponent_team} on {date_str}:\n"
        f"- Total Points: {stats['points']}\n"
        f"- Field Goal Percentage (FG%): {stats['fg_percentage']:.1f}%\n"
        f"- Total Rebounds: {stats['rebounds']}\n"
        f"- Total Assists: {stats['assists']}\n"
        f"- Steals & Blocks: {stats['steals_blocks']}\n"
        f"- Weaknesses: {', '.join(stats['weaknesses'][:3]) if stats['weaknesses'] else 'None'}\n"
        f"Format with sections:\n"
        f"- Overview: Summarize team performance, noting {jersey_color} jerseys.\n"
        f"- Offensive Efficiency: Analyze scoring and assists.\n"
        f"- Defensive Strength: Analyze rebounds, steals, blocks.\n"
        f"- Weaknesses: List weaknesses and suggest improvements.\n"
        f"- Projection: Assess team potential.\n"
        f"Be harsh and specific."
    )
    
    logging.debug(f"Team prompt: {prompt}")
    
    for attempt in range(RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            if "rate limit" in str(e).lower():
                sleep_time = DELAY * (2 ** attempt)
                logging.info(f"Retrying ({attempt+1}/{RETRIES}) after {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

def main():
    logging.info("Starting scouting report generator...")
    print("Choose report type:\n1. Single Player Report\n2. Team Scouting Report")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in ['1', '2']:
        logging.error("Invalid report type choice.")
        print("Invalid choice. Enter 1 or 2.")
        return

    is_team_report = (choice == '2')

    if is_team_report:
        logging.info("Collecting team information...")
        print("\nTeam Information")
        while True:
            jersey_color = input("Jersey Color (e.g., blue, red): ").strip().lower()
            if not jersey_color:
                logging.error("Jersey color is required.")
                print("Jersey color is required.")
            elif jersey_color not in VALID_COLORS:
                logging.warning(f"Invalid color '{jersey_color}'.")
                suggestion = suggest_color(jersey_color)
                suggestion_text = f" Did you mean {suggestion}?" if suggestion else ""
                print(f"Invalid color. Choose from: {', '.join(VALID_COLORS)}{suggestion_text}")
            else:
                if jersey_color in SIMILAR_COLORS:
                    similar = SIMILAR_COLORS[jersey_color]
                    print(f"Did you mean {jersey_color} or one of {', '.join(similar)}?")
                    confirm = input(f"Enter 'yes' to confirm {jersey_color}, or a different color: ").strip().lower()
                    if confirm != 'yes' and confirm in VALID_COLORS:
                        jersey_color = confirm
                    elif confirm != 'yes':
                        continue
                break
        gender = input("Gender (Male/Female): ").strip() or "Unknown"
        opponent_team = input("Opponent team name: ").strip() or "Unknown"
    else:
        logging.info("Collecting player information...")
        print("\nPlayer Information")
        player_name = input("Player name: ").strip()
        if not player_name:
            logging.error("Player name is required.")
            print("Player name is required.")
            return
        jersey_number = input("Jersey number: ").strip() or "Unknown"
        while True:
            jersey_color = input("Jersey Color (e.g., blue, red): ").strip().lower()
            if not jersey_color:
                logging.error("Jersey color is required.")
                print("Jersey color is required.")
            elif jersey_color not in VALID_COLORS:
                logging.warning(f"Invalid color '{jersey_color}'.")
                suggestion = suggest_color(jersey_color)
                suggestion_text = f" Did you mean {suggestion}?" if suggestion else ""
                print(f"Invalid color. Choose from: {', '.join(VALID_COLORS)}{suggestion_text}")
            else:
                if jersey_color in SIMILAR_COLORS:
                    similar = SIMILAR_COLORS[jersey_color]
                    print(f"Did you mean {jersey_color} or one of {', '.join(similar)}?")
                    confirm = input(f"Enter 'yes' to confirm {jersey_color}, or a different color: ").strip().lower()
                    if confirm != 'yes' and confirm in VALID_COLORS:
                        jersey_color = confirm
                    elif confirm != 'yes':
                        continue
                break
        height = input("Height (e.g., 6'9): ").strip() or "Unknown"
        gender = input("Gender (Male/Female): ").strip() or "Unknown"
        opponent_team = input("Opponent team name: ").strip() or "Unknown"

    print("\nProvide game video:\n1. Local video file\n2. YouTube link")
    video_choice = input("Enter 1 or 2: ").strip()

    video_path = None
    max_path_attempts = 3
    if video_choice == '1':
        for attempt in range(max_path_attempts):
            video_path = input("Video file path (e.g., 'C:/path/to/video.mp4'): ").strip()
            if os.path.exists(video_path):
                break
            logging.error(f"File not found at {video_path}. Attempt {attempt+1}/{max_path_attempts}.")
            print("File not found. Provide a valid path.")
            if attempt == max_path_attempts - 1:
                logging.error("Max attempts reached for video file path.")
                print("Max attempts reached. Try again with a valid path.")
                return
    elif video_choice == '2':
        youtube_url = input("YouTube link: ").strip()
        if not youtube_url:
            logging.error("YouTube URL is required.")
            print("YouTube URL is required.")
            return
        video_path = download_youtube_video(youtube_url)
        if video_path is None:
            logging.warning("YouTube download failed. Falling back to local file.")
            for attempt in range(max_path_attempts):
                video_path = input("Video file path (e.g., 'C:/path/to/video.mp4'): ").strip()
                if os.path.exists(video_path):
                    break
                logging.error(f"File not found at {video_path}. Attempt {attempt+1}/{max_path_attempts}.")
                print("File not found. Provide a valid path.")
                if attempt == max_path_attempts - 1:
                    logging.error("Max attempts reached for video file path.")
                    print("Max attempts reached. Try again with a valid path.")
                    return
    else:
        logging.error("Invalid video source choice.")
        print("Invalid choice. Enter 1 or 2.")
        return

    logging.info("Extracting frames...")
    frames = extract_frames_with_motion(video_path, jersey_color=jersey_color)
    if not frames:
        logging.error("No frames extracted.")
        print("No frames extracted. Cannot generate report.")
        return

    logging.info("Analyzing frames...")
    gameplay_descriptions = analyze_frame_with_gpt(frames)
    if all(desc is None for desc in gameplay_descriptions):
        logging.error("Failed to analyze frames.")
        print("Failed to analyze frames. Cannot generate report.")
        return

    logging.info("Aggregating stats...")
    stats = aggregate_stats(gameplay_descriptions, is_team=is_team_report)

    if is_team_report:
        logging.info("Generating team report...")
        report = generate_team_report(stats, jersey_color, gender, opponent_team)
    else:
        logging.info(f"Generating player report for {player_name}...")
        report = generate_player_report(stats, player_name, jersey_number, jersey_color, height, gender, opponent_team)

    print("\nScouting Report:")
    print(report)
    output_file = "scouting_report.txt"
    try:
        with open(output_file, "w") as f:
            f.write(report)
        logging.info(f"Report saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save report: {str(e)}")
        print(f"Failed to save report to {output_file}.")

if __name__ == "__main__":
    main()
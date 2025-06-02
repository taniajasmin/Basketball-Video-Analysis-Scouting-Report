import cv2
import numpy as np
import base64
from openai import OpenAI
import time
import os
from pytubefix import YouTube
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it in the .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to download YouTube video using pytubefix
def download_youtube_video(url):
    try:
        yt = YouTube(url, use_po_token=True)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            print("No suitable video stream found.")
            return None
        video_path = stream.download(output_path='.', filename='game_video.mp4')
        return video_path
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        print("Ensure the video URL is valid, accessible, and not restricted (e.g., private or age-restricted).")
        return None

# Function to calibrate motion detection parameters
def calibrate_motion_parameters(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 1
    motion_values = []
    prev_frame = None
    sample_frames = min(500, total_frames)  # Sample up to 500 frames for calibration
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
        print("No motion data collected. Using default parameters.")
        return 50, 20, 300

    # Calculate motion threshold as the 75th percentile to capture significant motion
    motion_threshold = np.percentile(motion_values, 75)
    # Adjust max frames per segment based on motion density
    motion_density = len(motion_values) / total_duration if total_duration > 0 else 1
    max_frames_per_segment = max(10, min(50, int(20 * (motion_density / 10))))
    # Adjust segment duration: shorter for fast-paced, longer for slow-paced
    segment_duration = max(60, min(600, int(300 / (motion_density / 10))))
    print(f"Calibrated parameters: motion_threshold={motion_threshold:.2f}, max_frames_per_segment={max_frames_per_segment}, segment_duration={segment_duration}")
    return motion_threshold, max_frames_per_segment, segment_duration

# Function to extract frames with motion detection
def extract_frames_with_motion(video_path):
    # Calibrate parameters dynamically
    motion_threshold, max_frames_per_segment, segment_duration = calibrate_motion_parameters(video_path)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 1
    frames = []
    prev_frame = None
    segment_start = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_prev, gray_current)
            motion = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1])
            if motion > motion_threshold and len(frames) < max_frames_per_segment * (total_duration // segment_duration + 1):
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())

        prev_frame = frame.copy()
        frame_count += 1
        current_time = frame_count / fps if fps > 0 else 0
        if current_time - segment_start >= segment_duration:
            segment_start = current_time

    cap.release()
    print(f"Extracted {len(frames)} frames based on motion detection.")
    return frames

# Function to analyze frames with OpenAI GPT-4o API in batches
def analyze_frame_with_gpt(frames, max_frames=200, retries=5, delay=2):
    descriptions = []
    for i in range(0, min(len(frames), max_frames), 20):
        batch = frames[i:i + 20]
        batch_descs = []
        for frame in batch:
            base64_image = base64.b64encode(frame).decode('utf-8')
            prompt = (
                "Describe the basketball gameplay action in this frame, focusing on:\n"
                "- Scoring (points, field goal attempts, and made shots)\n"
                "- Assists (passes leading to scores)\n"
                "- Rebounds (offensive or defensive)\n"
                "- Steals or blocks (defensive plays)\n"
                "- Mistakes (missed shots, turnovers)\n"
                "Provide a concise description without game context like scores or team names."
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
                    batch_descs.append(response.choices[0].message.content)
                    time.sleep(delay)
                    break
                except Exception as e:
                    print(f"Error analyzing frame: {e}")
                    if "rate limit" in str(e).lower() or "connection" in str(e).lower():
                        if attempt < retries - 1:
                            print(f"Retrying ({attempt+1}/{retries}) after {delay} seconds...")
                            time.sleep(delay)
                            delay *= 1.5
                        else:
                            print(f"Max retries reached for this frame. Skipping.")
                            batch_descs.append(None)
                    else:
                        print(f"Unexpected error: {e}. Skipping frame.")
                        batch_descs.append(None)
                        break
        descriptions.extend(batch_descs)
    return descriptions

# Function to aggregate stats from frame descriptions
def aggregate_stats(descriptions, is_team=False):
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
        print("No valid frame descriptions to analyze. Returning default stats.")
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
    return stats

# Function to generate single player scouting report
def generate_player_report(stats, player_name, jersey_number, jersey_color, height, gender, opponent_team):
    current_time = datetime.now()
    date_str = current_time.strftime("%m/%d/%Y %I:%M %p +06")  

    prompt = (
        f"Generate a basketball scouting report for {player_name} (Jersey Number: {jersey_number}, Jersey Color: {jersey_color}, Height: {height}, Gender: {gender}) against {opponent_team} on {date_str}:\n"
        f"- Points Per Game (PPG): {stats['points']}\n"
        f"- Field Goal Percentage (FG%): {stats['fg_percentage']:.1f}%\n"
        f"- Rebounds Per Game (RPG): {stats['rebounds']}\n"
        f"- Assists Per Game (APG): {stats['assists']}\n"
        f"- Steals & Blocks: {stats['steals_blocks']}\n"
        f"- Weaknesses: {', '.join(stats['weaknesses']) if stats['weaknesses'] else 'None'}\n"
        f"Format the report with these sections:\n"
        f"- Overview: Summarize the player's performance.\n"
        f"- Strength: Highlight key skills based on stats.\n"
        f"- Weaknesses: List weaknesses and suggest specific improvements.\n"
        f"- Projection: Assess future potential.\n"
        f"Be harsh and specific."
    )
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating report: {e}")
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit for report generation, retrying ({attempt+1}/3) after {2**attempt} seconds...")
                time.sleep(2**attempt)
            else:
                return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

# Function to generate team scouting report
def generate_team_report(stats, jersey_color, gender, opponent_team):
    current_time = datetime.now()
    date_str = current_time.strftime("%m/%d/%Y %I:%M %p +06")  

    prompt = (
        f"Generate a basketball team scouting report for a team (Jersey Color: {jersey_color}, Gender: {gender}) against {opponent_team} on {date_str}:\n"
        f"- Total Points: {stats['points']}\n"
        f"- Field Goal Percentage (FG%): {stats['fg_percentage']:.1f}%\n"
        f"- Total Rebounds: {stats['rebounds']}\n"
        f"- Total Assists: {stats['assists']}\n"
        f"- Steals & Blocks: {stats['steals_blocks']}\n"
        f"- Weaknesses: {', '.join(stats['weaknesses']) if stats['weaknesses'] else 'None'}\n"
        f"Format the report with these sections:\n"
        f"- Overview: Summarize the team's performance.\n"
        f"- Offensive Efficiency: Analyze scoring and assists.\n"
        f"- Defensive Strength: Analyze rebounds, steals, and blocks.\n"
        f"- Weaknesses: List weaknesses and suggest specific improvements.\n"
        f"- Projection: Assess future potential.\n"
        f"Be harsh and specific."
    )
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating report: {e}")
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit for report generation, retrying ({attempt+1}/3) after {2**attempt} seconds...")
                time.sleep(2**attempt)
            else:
                return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

# User input and report generation
def main():
    print("What type of scouting report would you like to generate?")
    print("1. Single Player Report")
    print("2. Team Scouting Report")
    choice = input("Enter 1 or 2: ")

    if choice not in ['1', '2']:
        print("Invalid choice. Please run the script again and enter 1 or 2.")
        return

    is_team_report = (choice == '2')

    if is_team_report:
        print("\nTeam Information")
        opponent_team = input("Opponent team name: ") or "Unknown"
        jersey_color = input("Jersey Color: ") 
        gender = input("Gender (Male/Female): ") 
    else:
        print("\nPlayer Information")
        player_name = input("Player name: ") 
        jersey_number = input("Jersey number: ") 
        jersey_color = input("Jersey Color: ") 
        height = input("Height (e.g., 6'9): ") or "Unknown"
        gender = input("Gender (Male/Female): ") 
        opponent_team = input("Opponent team name: ") or "Unknown"

    print("\nHow would you like to provide the basketball game video?")
    print("1. Upload a local video file")
    print("2. Provide a YouTube link")
    video_choice = input("Enter 1 or 2: ")

    video_path = None
    if video_choice == '1':
        video_path = input("Enter the path to your video file (e.g., 'C:/path/to/video.mp4'): ")
        if not os.path.exists(video_path):
            print("File not found. Please provide a valid path.")
            return
    elif video_choice == '2':
        youtube_url = input("Enter the YouTube link: ")
        video_path = download_youtube_video(youtube_url)
        if video_path is None:
            video_path = input("Download failed. Enter the path to your video file (e.g., 'C:/path/to/video.mp4'): ")
            if not os.path.exists(video_path):
                print("File not found. Please provide a valid path.")
                return
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")
        return

    print("Extracting frames from the video with motion detection...")
    frames = extract_frames_with_motion(video_path)
    if not frames:
        print("No frames extracted. Cannot generate a meaningful report.")
        return

    print("Analyzing frames with GPT-4o API...")
    gameplay_descriptions = analyze_frame_with_gpt(frames)
    if all(desc is None for desc in gameplay_descriptions):
        print("Failed to analyze any frames. Cannot generate a meaningful report.")
        return

    print("Aggregating stats...")
    stats = aggregate_stats(gameplay_descriptions, is_team=is_team_report)

    if is_team_report:
        print("Generating team scouting report...")
        report = generate_team_report(stats, jersey_color, gender, opponent_team)
    else:
        print("Generating scouting report for", player_name, "...")
        report = generate_player_report(stats, player_name, jersey_number, jersey_color, height, gender, opponent_team)

    print("Scouting Report:")
    print(report)

if __name__ == "__main__":
    main()

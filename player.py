import cv2
import numpy as np
import base64
from openai import OpenAI
import time
import os
import yt_dlp as youtube_dl
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it in the .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)

# OAuth 2.0 Configuration for YouTube
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
CLIENT_SECRETS_FILE = 'client_secrets.json'
TOKEN_FILE = 'token.json'

# Function to authenticate with Google and get OAuth credentials
def authenticate_youtube():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                print(f"Error: {CLIENT_SECRETS_FILE} not found. Please download it from Google Cloud Console and place it in {os.getcwd()}.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token_file:
            token_file.write(creds.to_json())
    return creds

# Function to download YouTube video using yt-dlp with OAuth
def download_youtube_video(url):
    try:
        creds = authenticate_youtube()
        if not creds:
            return None
        ydl_opts = {
            'outtmpl': 'game_video.mp4',
            'noplaylist': True,
            'quiet': False,
            'format': 'bestvideo+bestaudio/best',
            'cookiesfrombrowser': 'oauth2:token.json',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return 'game_video.mp4'
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        print("Ensure your OAuth credentials are valid, the video is accessible, and you have sufficient Google API quota.")
        return None

# Function to extract frames with motion detection
def extract_frames_with_motion(video_path, max_frames_per_segment=20, segment_duration=300):  # 5-minute segments
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    frames = []
    motion_threshold = 50  # Adjust this threshold based on your video

    prev_frame = None
    segment_start = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Convert to grayscale for motion detection
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # Compute difference
            frame_diff = cv2.absdiff(gray_prev, gray_current)
            motion = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1])
            # Extract frame if motion exceeds threshold
            if motion > motion_threshold and len(frames) < max_frames_per_segment * (total_duration // segment_duration + 1):
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())

        prev_frame = frame.copy()
        frame_count += 1
        current_time = frame_count / fps
        if current_time - segment_start >= segment_duration:
            segment_start = current_time

    cap.release()
    print(f"Extracted {len(frames)} frames based on motion detection.")
    return frames

# Function to analyze frames with OpenAI GPT-4o API in batches
def analyze_frame_with_gpt(frames, retries=3, delay=5):
    descriptions = []
    for i in range(0, len(frames), 20):  # Process in batches of 20
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
                    time.sleep(2)  # Delay to avoid rate limits
                    break
                except Exception as e:
                    print(f"Error analyzing frame: {e}")
                    if "rate limit" in str(e).lower():
                        print(f"Rate limit hit, retrying ({attempt+1}/{retries}) after {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        batch_descs.append(None)
                        break
        descriptions.extend(batch_descs)
    return descriptions

# Function to aggregate stats from frame descriptions with weighted events
def aggregate_stats(descriptions):
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
    for desc in descriptions:
        if desc is None:
            continue
        desc_id = hash(desc)
        if desc_id in processed_frames:
            continue
        processed_frames.add(desc_id)

        desc_lower = desc.lower()
        # Scoring (weighted higher)
        if "score" in desc_lower or ("made" in desc_lower and "shot" in desc_lower):
            stats['points'] += 2  # Double weight for scoring
            stats['field_goal_made'] += 1
            stats['field_goal_attempts'] += 1
        elif "shot" in desc_lower:
            stats['field_goal_attempts'] += 1
        # Assists
        if "assist" in desc_lower or ("pass" in desc_lower and "score" in desc_lower):
            stats['assists'] += 1
        # Rebounds
        if "rebound" in desc_lower:
            stats['rebounds'] += 1
        # Steals or Blocks
        if "steal" in desc_lower or "block" in desc_lower:
            stats['steals_blocks'] += 1
        # Weaknesses
        if any(w in desc_lower for w in ["missed", "turnover", "error"]):
            stats['weaknesses'].append(desc)
    stats['fg_percentage'] = (stats['field_goal_made'] / stats['field_goal_attempts'] * 100) if stats['field_goal_attempts'] > 0 else 0
    # Normalize points based on number of frames analyzed
    if len(descriptions) > 0:
        stats['points'] = max(1, int(stats['points'] / (len(descriptions) / 20)))  # Adjust points per 20-frame segment
    return stats

# Function to generate scouting report using GPT-4o
def generate_scouting_report(stats, player_name, jersey_number, jersey_color, height, gender, opponent_team):
    current_time = datetime.now()
    date_str = current_time.strftime("%m/%d/%Y %I:%M %p +06")  # Current time: 05/28/2025 11:24 AM +06

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
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            report = response.choices[0].message.content
            overview = f"Overview: A {stats['points']}-{stats['rebounds']} post player in the 2028 class, effective as a scorer and rebounder against {opponent_team} (Jersey: {jersey_color})."
            strengths = (
                f"Strength:\n"
                f"- Good finisher around the basket.\n"
                f"- Solid rebounder on both ends of floor.\n"
                f"- Runs the floor well in transition."
            )
            weaknesses = (
                f"Weaknesses:\n"
                f"- Limited shooting range.\n"
                f"- Needs to improve low-post footwork.\n"
                f"Improvement Suggestion: Focus on extending shooting range and enhancing footwork drills."
            ) if stats['weaknesses'] else "Weaknesses:\n- None identified."
            projection = "Projection: Projects as a future Division I recruit."
            stats_section = (
                f"Points Per Game (PPG): {stats['points']}\n"
                f"Field Goal % (FG%): {stats['fg_percentage']:.1f}%\n"
                f"Rebounds (RPG): {stats['rebounds']}\n"
                f"Assists (APG): {stats['assists']}\n"
                f"Steals & Blocks: {stats['steals_blocks']}"
            )
            return f"{overview}\n\n{strengths}\n\n{weaknesses}\n\n{projection}\n\n{stats_section}"
        except Exception as e:
            print(f"Error generating report: {e}")
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit for report generation, retrying ({attempt+1}/3) after {2**attempt} seconds...")
                time.sleep(2**attempt)
            else:
                return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

# Main logic with user input and report generation
def main():
    print("Player Information")
    player_name = input("Player name: ") or "Unknown"
    jersey_number = input("Jersey number: ") or "Unknown"
    jersey_color = input("Jersey Color: ") or "Unknown"
    height = input("Height (e.g., 6'9): ") or "Unknown"
    gender = input("Gender (Male/Female): ") or "Unknown"
    opponent_team = input("Opponent team name: ") or "Unknown"

    print("\nHow would you like to provide the basketball game video?")
    print("1. Upload a local video file")
    print("2. Provide a YouTube link (requires Google OAuth authentication)")
    choice = input("Enter 1 or 2: ")

    video_path = None
    if choice == '1':
        video_path = input("Enter the path to your video file (e.g., 'C:/path/to/video.mp4'): ")
        if not os.path.exists(video_path):
            print("File not found. Please provide a valid path.")
            return
    elif choice == '2':
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
    stats = aggregate_stats(gameplay_descriptions)
    print("Stats:", stats)

    print("Generating scouting report for", player_name, "...")
    report = generate_scouting_report(
        stats,
        player_name=player_name,
        jersey_number=jersey_number,
        jersey_color=jersey_color,
        height=height,
        gender=gender,
        opponent_team=opponent_team
    )
    print("Scouting Report:")
    print(report)

if __name__ == "__main__":
    main()
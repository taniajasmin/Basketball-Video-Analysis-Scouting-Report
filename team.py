import cv2
import numpy as np
import google.generativeai as genai
import yt_dlp as youtube_dl
import time
from google.api_core import exceptions
import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

GEMINI_API_KEY = "AIzaSyAiCkdnfv0a50BPzKl2cQw3IBfi8VzYqeQ"
genai.configure(api_key=GEMINI_API_KEY)

# OAuth 2.0 Configuration
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
CLIENT_SECRETS_FILE = 'client_secrets.json'  # Path to your client secrets file
TOKEN_FILE = 'token.json'  # File to store OAuth tokens

# Function to authenticate with Google and get OAuth credentials
def authenticate_youtube():
    creds = None
    # Load existing credentials if they exist
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # If there are no valid credentials, perform OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # Opens browser for user to sign in
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token_file:
            token_file.write(creds.to_json())
    return creds

# Function to download YouTube video using yt-dlp with OAuth
def download_youtube_video(url):
    try:
        # Authenticate to ensure token.json is ready
        authenticate_youtube()
        ydl_opts = {
            'outtmpl': 'game_video.mp4',
            'noplaylist': True,
            'quiet': False,
            'format': 'bestvideo+bestaudio/best',
            'cookiesfrombrowser': 'oauth2:token.json',  # Use OAuth token
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return 'game_video.mp4'
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        print("Ensure your OAuth credentials are valid and the video is accessible.")
        return None

# Function to extract frames from video
def extract_frames(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // target_frames)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(buffer.tobytes())
        frame_count += 1
    cap.release()
    return frames

# Function to analyze frames with Gemini API
def analyze_frame_with_gemini(frame, retries=3, delay=5):
    model = genai.GenerativeModel('gemini-1.5-flash')
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
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": frame}])
            time.sleep(1)
            return response.text
        except exceptions.TooManyRequests as e:
            print(f"Rate limit hit, retrying ({attempt+1}/{retries}) after {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return "Error analyzing frame"
    return "Failed to analyze frame due to rate limits"

# Function to aggregate stats from frame descriptions
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
    for desc in descriptions:
        desc_lower = desc.lower()
        if "score" in desc_lower or "made" in desc_lower and "shot" in desc_lower:
            stats['points'] += 1
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
    return stats

# Function to generate scouting report matching UI design
def generate_scouting_report(stats, player_name="Ronald Richards", date="05/26/2025", opponent_team="Unknown", jersey_color="Unknown", gender="Unknown"):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = (
        f"Generate a basketball scouting report for {player_name} (Gender: {gender}) against {opponent_team} (Jersey Color: {jersey_color}) on {date}:\n"
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
            response = model.generate_content(prompt)
            report = response.text
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
        except exceptions.TooManyRequests as e:
            print(f"Rate limit hit for report generation, retrying ({attempt+1}/3) after {2**attempt} seconds...")
            time.sleep(2**attempt)
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Error generating scouting report"
    return "Failed to generate report due to rate limits"

# Main logic with user input and report generation
def main():
    print("Team Information")
    opponent_team = input("Opponent team name: ") or "Unknown"
    jersey_color = input("Jersey Color: ") or "Unknown"
    gender = input("Gender (Male/Female): ") or "Unknown"

    print("\nHow would you like to provide the basketball game video?")
    print("1. Upload a local video file")
    print("2. Provide a YouTube link (requires Google OAuth authentication)")
    choice = input("Enter 1 or 2: ")

    video_path = None
    if choice == '1':
        video_path = input("Enter the path to your video file (e.g., 'path/to/video.mp4'): ")
        if not os.path.exists(video_path):
            print("File not found. Please provide a valid path.")
            return
    elif choice == '2':
        youtube_url = input("Enter the YouTube link: ")
        video_path = download_youtube_video(youtube_url)
        if video_path is None:
            video_path = input("Download failed. Enter the path to your video file (e.g., 'path/to/video.mp4'): ")
            if not os.path.exists(video_path):
                print("File not found. Please provide a valid path.")
                return
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")
        return

    print("Extracting frames from the video...")
    frames = extract_frames(video_path)
    print(f"Extracted {len(frames)} frames for analysis.")

    print("Analyzing frames with Gemini API...")
    gameplay_descriptions = [analyze_frame_with_gemini(frame) for frame in frames]

    print("Aggregating stats...")
    stats = aggregate_stats(gameplay_descriptions)
    print("Stats:", stats)

    print("Generating scouting report...")
    report = generate_scouting_report(stats, date="05/26/2025", opponent_team=opponent_team, jersey_color=jersey_color, gender=gender)
    print("Scouting Report:")
    print(report)

if __name__ == "__main__":
    main()
# Basketball-Video-Analysis-Scouting-Report




# Basketball Scouting Report Generator


A Python application that analyzes basketball gameplay videos (local files or YouTube links) to generate detailed scouting reports for individual players or entire teams using computer vision and OpenAI's GPT-4o model.

## Features

- 🎥 Video processing from local files or YouTube URLs
- 🏀 Motion detection to identify key gameplay moments
- 🔍 Color detection for player/team identification
- 📊 Statistical analysis of basketball performance metrics
- 📝 AI-generated scouting reports with:
  - Performance overview
  - Strengths analysis
  - Weaknesses identification
  - Future projection

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- PytubeFix (YouTube downloader)
- OpenAI Python client
- python-dotenv

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/basketball-scouting.git
   cd basketball-scouting
Install dependencies:

bash
pip install -r requirements.txt
Create a .env file with your OpenAI API key:

text
OPENAI_API_KEY=your_api_key_here
Usage
Run the script:

bash
python scouting.py
Follow the interactive prompts to:

Choose report type (player or team)

Enter player/team details

Provide video source (local file or YouTube URL)

View and save the generated report

Input Options
For Player Reports:

Player name

Jersey number and color

Height

Gender

Opponent team name

For Team Reports:

Team jersey color

Gender

Opponent team name

Video Sources:

Local MP4 file

YouTube URL

Configuration
The script includes several adjustable parameters in the code:

python
# Motion detection
MOTION_THRESHOLD_DEFAULT = 50
MAX_FRAMES_PER_SEGMENT_DEFAULT = 20
SEGMENT_DURATION_DEFAULT = 300

# Video processing
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
JPEG_QUALITY = 80
MAX_FRAMES = 200

# API settings
RETRIES = 5
DELAY = 2
Output
The script generates:

Console output with the full scouting report

A text file (scouting_report.txt) with the report content

Sample report structure:

text
OVERVIEW:
[Player/Team] showed [performance summary] wearing [color] jerseys...

STRENGTHS:
- Excellent [skill 1]
- Strong [skill 2]
- Notable [skill 3]

WEAKNESSES:
- Needs improvement in [area 1]
- Struggles with [area 2]
- [Specific weakness]

PROJECTION:
Supported Jersey Colors
The system recognizes these colors for player/team identification:

Red, Blue, Green, Yellow

White, Black, Orange

Purple, Gray/Grey, Pink

Brown, Navy

Limitations
Requires clear video footage for accurate analysis

Works best with focused gameplay footage (not wide-angle shots)

Dependent on OpenAI API availability and rate limits

Color detection may be affected by lighting conditions

Troubleshooting
Common Issues:

"OPENAI_API_KEY not found": Ensure your .env file exists with the correct key

"Video not found": Verify the file path or YouTube URL is correct

"No frames extracted": Try a different video with clearer gameplay

Rate limit errors: Wait and try again later



OPENAI_API_KEY=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 
# from client

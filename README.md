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

2. Install dependencies:
``` bash
pip install -r requirements.txt
```
3. Create a .env file with your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

## Usage
Run the script:

```bash
python scouting.py
```

## Follow the interactive prompts to:
1. Choose report type (player or team)
2. Enter player/team details
3. Provide video source (local file or YouTube URL)
4. View and save the generated report

## Input Options
1. For Player Reports:
- Player name
- Jersey number and color
- Height
- Gender
- Opponent team name

2. For Team Reports:
- Team jersey color
- Gender
- Opponent team name

## Video Sources:
1. Local MP4 file
2. YouTube URL

## Configuration
The script includes several adjustable parameters in the code:

```python
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
```

## Output
The script generates:
1. Console output with the full scouting report
2. A text file (scouting_report.txt) with the report content

### Sample report structure:

```text
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
```

### Supported Jersey Colors
The system recognizes these colors for player/team identification:
- Red, Blue, Green, Yellow
- White, Black, Orange
- Purple, Gray/Grey, Pink
- Brown, Navy

### Limitations
- Requires clear video footage for accurate analysis
- Works best with focused gameplay footage (not wide-angle shots)
- Dependent on OpenAI API availability and rate limits
- Color detection may be affected by lighting conditions

### Troubleshooting
Common Issues:
- "OPENAI_API_KEY not found": Ensure your .env file exists with the correct key
- "Video not found": Verify the file path or YouTube URL is correct
- "No frames extracted": Try a different video with clearer gameplay
- Rate limit errors: Wait and try again later

## 🛠 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **`OPENAI_API_KEY not found`** | 1. Create `.env` file in project root<br>2. Add `OPENAI_API_KEY=your_key_here`<br>3. Ensure no trailing spaces |
| **`Video not found`** | • Local files: Use absolute paths (`/videos/game.mp4`)<br>• YouTube: Verify URL is public |
| **`No frames extracted`** | • Use 720p+ videos with clear gameplay<br>• Avoid distant/wide-angle shots|
| **Rate limit errors** | • Wait 60s between runs<br>• Reduce video length<br>• Upgrade OpenAI plan if needed |

### Sample Test Videos
Test with these basketball gameplay videos:
1. [NBA Highlights](https://www.youtube.com/watch?v=LPDnemFoqVk)
2. [Streetball 1v1](https://youtu.be/ELwNvUnm0LA) 
3. [College Game](https://youtu.be/VvuQ-hynlEg)

> **Tip**: For best results:
> - Use videos where jersey colors contrast strongly (red vs blue)
> - Camera should follow main action
> - Avoid overexposed/backlit footage

### 📥 Video Download Options
1. Use from device
2. Download using [Python video downloader]([https://youtu.be/VvuQ-hynlEg](https://github.com/taniajasmin/Video-Downloader-Web-App/blob/main/downloader_v2.py)

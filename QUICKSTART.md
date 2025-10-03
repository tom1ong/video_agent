# Quick Start Guide

Get your Video Editing Agent up and running in 5 minutes!

This agent uses MCP-inspired tool abstraction with LangChain and Google Gemini.

## 1. Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- FFmpeg installed on your system (for video processing)

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## 2. Installation

```bash
# Clone or navigate to the project directory
cd /Users/tom/pg/hackathon

# Install Python dependencies
pip install -r requirements.txt
```

## 3. Configuration

Create a `.env.local` file in the project root:

```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env.local
```

Replace `your-api-key-here` with your actual Gemini API key.

> **Note:** We use `.env.local` instead of `.env` to avoid conflicts with Python virtual environments.

## 4. Prepare Your Videos

Place your video files in the `workspace/` directory (it will be created automatically):

```bash
mkdir -p workspace
cp /path/to/your/video1.mp4 workspace/
cp /path/to/your/video2.mp4 workspace/
```

Or use absolute paths when referencing videos in commands.

## 5. Run the Agent

```bash
python main.py
```

## 6. Try Your First Edit

Once the agent starts, try these commands:

```
You: Show me info about workspace/video1.mp4
```

```
You: Cut the first 10 seconds from workspace/video1.mp4
```

```
You: Join workspace/video1.mp4 and workspace/video2.mp4 together
```

```
You: Create a storyline from workspace/video1.mp4, workspace/video2.mp4, and workspace/video3.mp4 with 3 second segments
```

## 7. Advanced Usage

### Programmatic API

You can also use the agent programmatically:

```python
from agent import VideoEditingAgent
import os

agent = VideoEditingAgent(gemini_api_key=os.getenv("GEMINI_API_KEY"))
response = agent.chat("Cut the first 5 seconds from video1.mp4")
print(response)
```

See `example_usage.py` for more examples.

### Complex Multi-Step Edits

The agent can handle complex requests with multiple operations:

```
You: I want to take the first 5 seconds from video1.mp4, 
     the middle 10 seconds from video2.mp4, and the last 
     3 seconds from video3.mp4, join them together, and 
     add text "My Compilation" at the beginning
```

The LLM will automatically figure out the sequence of operations!

## Troubleshooting

### "GEMINI_API_KEY not found"
Make sure your `.env.local` file is in the project root and contains your API key.

### "MoviePy error" or "FFmpeg not found"
Install FFmpeg (see Prerequisites section).

### "Video file not found"
Make sure your video paths are correct. Use absolute paths or place files in the `workspace/` directory.

### Agent is slow
Video processing is computationally intensive. The speed depends on:
- Video file size and resolution
- Number of operations
- Your system's CPU/GPU capabilities

## What's Next?

- Add more videos to your workspace
- Try complex multi-step edits
- Experiment with different text overlays and effects
- Create storylines from multiple clips

Enjoy your AI-powered video editing! 🎬🤖


# Video Editing Agent (MCP)

AI-powered video editing agent using Google Gemini and the Model Context Protocol (MCP).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
echo "GEMINI_API_KEY=your-key-here" > .env.local

# 3. Run
python3 main.py
```

## Requirements

- Python 3.10+
- Google Gemini API key
- FFmpeg (for video processing)

## Features

- Cut, trim, and concatenate videos
- Add text overlays
- Speed adjustment
- Audio extraction and merging
- AI-powered video analysis and summaries
- MCP-based architecture for extensibility

## Architecture

```
main.py → agent.py (MCP Client) → [MCP Protocol] → mcp_server.py (MCP Server) → video_tools.py
```

## Configuration

Environment variables in `.env.local`:
- `GEMINI_API_KEY` - Required
- `WORKSPACE_DIR` - Default: `./workspace`

## Usage

Place video files in the `workspace/` directory, then interact with the agent:

```
You: Show me info about my_video.mp4
You: Cut the first 30 seconds from my_video.mp4
You: Add text "Hello World" at 10 seconds
```

## License

MIT

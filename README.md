# Video Editing Agent with LangChain & MCP

An intelligent video editing agent that uses LangChain for orchestration, **MCP-inspired tool abstraction**, and Google Gemini as the decision-making LLM.

All video operations are abstracted as tools following MCP (Model Context Protocol) design patterns.

## Features

- 🎬 **AI-Powered Video Editing**: Natural language commands to edit videos
- 🔧 **MCP Tool Abstraction**: All video operations exposed as MCP tools
- 🤖 **LLM Decision Making**: Gemini decides which tools to use and in what order
- 💬 **Conversational Interface**: Chat with the agent about your video editing needs
- 🎯 **Generic & Extensible**: Easy to add new video editing capabilities

## Video Editing Tools Available

- `cut_video`: Extract a segment from a video (start_time to end_time)
- `concatenate_videos`: Join multiple videos together
- `trim_video`: Remove unwanted parts from beginning/end
- `add_text_overlay`: Add text overlays to videos
- `speed_change`: Speed up or slow down videos
- `extract_audio`: Extract audio from video
- `merge_audio_video`: Combine audio and video files
- `get_video_info`: Get duration, resolution, and other video metadata

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env.local` file:
```
GEMINI_API_KEY=your-api-key-here
```

> **Note:** We use `.env.local` to avoid conflicts with Python virtual environments.

3. Run the agent:
```bash
python main.py
```

## Usage

```bash
# Start the agent
python main.py

# Example prompts:
# - "Cut the first 10 seconds from video1.mp4"
# - "Take video1.mp4 and video2.mp4, cut the first 5 seconds from each, then concatenate them"
# - "Add text 'Hello World' at 3 seconds into video.mp4"
# - "Create a fast-paced edit: speed up video1.mp4 by 2x and add it after video2.mp4"

# Files in workspace/ can be referenced by just their name!
# Example: "Show me info about myvideo.mp4" automatically looks in workspace/
```

## Architecture

1. **User Input**: Videos + natural language prompt
2. **LangChain Agent** (`agent.py`): Orchestrates the workflow
3. **Gemini LLM**: Decides which tools to call and when
4. **MCP-style Tool Server** (`mcp_server.py`): Exposes tools with standardized schemas
5. **Video Tools** (`video_tools.py`): Actual video processing with MoviePy
6. **Output**: Final edited video(s) in workspace/

### Key Benefits

✅ **Tool Abstraction**: All operations exposed as well-defined tools  
✅ **No Predefined Workflow**: Gemini decides everything dynamically  
✅ **Workspace Auto-resolution**: Just say "video.mp4" to reference files  
✅ **Conversational**: Natural language interface  
✅ **Extensible**: Easy to add new video editing capabilities

## How It Works

The agent operates with no predefined control flow. The LLM:
- Analyzes the user's request
- Determines which video editing operations are needed
- Decides the order of operations
- Calls the appropriate MCP tools
- Returns the final edited videos

All video operations are abstracted as MCP tools, making the system highly extensible.

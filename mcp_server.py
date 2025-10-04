"""
Real MCP Server for Video Editing Tools.
This is a proper MCP server that can be used with MCP clients.
"""

import asyncio
import json
import subprocess
import platform
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from video_tools import VideoTools
from pathlib import Path
import os


# Initialize the MCP server
app = Server("video-editing-server")

# Initialize video tools (will be set up on startup)
video_tools: VideoTools = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available video editing tools."""
    return [
        Tool(
            name="get_video_info",
            description="Get information about a video file including duration, resolution, fps, and audio presence",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file (relative paths look in workspace)"
                    }
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="cut_video",
            description="Extract a specific segment from a video by specifying start and end times",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to input video"},
                    "start_time": {"type": "number", "description": "Start time in seconds"},
                    "end_time": {"type": "number", "description": "End time in seconds"}
                },
                "required": ["video_path", "start_time", "end_time"]
            }
        ),
        Tool(
            name="trim_video",
            description="Remove unwanted parts from the beginning or end of a video",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "trim_start": {"type": "number", "description": "Seconds to trim from start (default: 0)"},
                    "trim_end": {"type": "number", "description": "Seconds to trim from end (default: 0)"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="concatenate_videos",
            description="Join multiple videos together in sequence",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of video paths to concatenate"
                    }
                },
                "required": ["video_paths"]
            }
        ),
        Tool(
            name="add_text_overlay",
            description="Add text overlay to a video at a specific time and position",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "text": {"type": "string"},
                    "start_time": {"type": "number", "description": "When to start (default: 0)"},
                    "duration": {"type": "number", "description": "How long (default: until end)"},
                    "position": {"type": "string", "enum": ["center", "top", "bottom"], "description": "Text position (default: center)"},
                    "fontsize": {"type": "integer", "description": "Font size (default: 50)"},
                    "color": {"type": "string", "description": "Color name (default: white)"}
                },
                "required": ["video_path", "text"]
            }
        ),
        Tool(
            name="speed_change",
            description="Change the playback speed of a video",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "speed_factor": {"type": "number", "description": "2.0 = 2x faster, 0.5 = half speed"}
                },
                "required": ["video_path", "speed_factor"]
            }
        ),
        Tool(
            name="extract_audio",
            description="Extract the audio track from a video file",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="merge_audio_video",
            description="Combine an audio file with a video file. Automatically trims audio if it's longer than the video, or loops it if shorter to match video duration perfectly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "audio_path": {"type": "string"}
                },
                "required": ["video_path", "audio_path"]
            }
        ),
        Tool(
            name="create_storyline",
            description="Create a storyline by taking short segments from multiple videos",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_paths": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "segment_duration": {"type": "number", "description": "Duration per segment (default: 3.0)"}
                },
                "required": ["video_paths"]
            }
        ),
        Tool(
            name="send_cache",
            description="Cache video(s) in Gemini for cost-effective repeated analysis. Supports both single video and multiple videos (caches in parallel).",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_paths": {
                        "oneOf": [
                            {"type": "string", "description": "Single video file path"},
                            {"type": "array", "items": {"type": "string"}, "description": "List of video file paths"}
                        ],
                        "description": "Path to video file OR array of video paths"
                    },
                    "ttl_minutes": {"type": "integer", "description": "Cache duration in minutes (default: 60)"}
                },
                "required": ["video_paths"]
            }
        ),
        Tool(
            name="get_video_summary",
            description="Get an AI-generated summary of a video using Gemini. Automatically caches the video if not already cached.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to video file"},
                    "custom_prompt": {"type": "string", "description": "Optional custom prompt for the summary"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="query_cached_video",
            description="Ask questions about video(s) using Gemini AI. Supports both single video and multiple videos. Automatically caches if needed. For multiple videos, returns edit plan with video filenames.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_paths": {
                        "oneOf": [
                            {"type": "string", "description": "Single video file path"},
                            {"type": "array", "items": {"type": "string"}, "description": "List of video file paths"}
                        ],
                        "description": "Path to video file OR array of video paths"
                    },
                    "query": {"type": "string", "description": "Question or query about the video(s)"}
                },
                "required": ["video_paths", "query"]
            }
        ),
        Tool(
            name="get_cache_info",
            description="Check if a video is cached and get cache information",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to video file"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="list_files",
            description="List files in the workspace directory or a subdirectory. Similar to 'ls' command.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional subdirectory path within workspace (default: workspace root)"}
                }
            }
        ),
        Tool(
            name="delete_file",
            description="Delete a file from the workspace. Similar to 'rm' command. Use with caution!",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to delete"}
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="play_video",
            description="Open and play a video file using the system's default video player. Similar to 'open' command.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to the video file to play"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="search_music",
            description="Search for music tracks on Epidemic Sound. Returns track information including title, artist, BPM, genres, moods, and track ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search query (e.g., 'cozy vlog', 'energetic workout', 'cinematic trailer')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, max: 50)"
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="download_music",
            description="Download a music track from Epidemic Sound using the track ID obtained from search_music. The audio file will be saved to the workspace as an MP3 file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "string",
                        "description": "The track ID from search_music results"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (without extension). If not provided, uses track ID."
                    }
                },
                "required": ["track_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a tool with the given arguments."""
    global video_tools
    
    try:
        if name == "get_video_info":
            result = await video_tools.get_video_info(arguments["video_path"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "cut_video":
            result = await video_tools.cut_video(
                arguments["video_path"],
                arguments["start_time"],
                arguments["end_time"]
            )
            return [TextContent(type="text", text=f"Video cut successfully. Output: {result}")]
        
        elif name == "trim_video":
            result = await video_tools.trim_video(
                arguments["video_path"],
                arguments.get("trim_start", 0),
                arguments.get("trim_end", 0)
            )
            return [TextContent(type="text", text=f"Video trimmed successfully. Output: {result}")]
        
        elif name == "concatenate_videos":
            result = await video_tools.concatenate_videos(arguments["video_paths"])
            return [TextContent(type="text", text=f"Videos concatenated successfully. Output: {result}")]
        
        elif name == "add_text_overlay":
            result = await video_tools.add_text_overlay(
                arguments["video_path"],
                arguments["text"],
                arguments.get("start_time", 0),
                arguments.get("duration"),
                arguments.get("position", "center"),
                arguments.get("fontsize", 50),
                arguments.get("color", "white")
            )
            return [TextContent(type="text", text=f"Text overlay added successfully. Output: {result}")]
        
        elif name == "speed_change":
            result = await video_tools.speed_change(
                arguments["video_path"],
                arguments["speed_factor"]
            )
            return [TextContent(type="text", text=f"Video speed changed successfully. Output: {result}")]
        
        elif name == "extract_audio":
            result = await video_tools.extract_audio(arguments["video_path"])
            return [TextContent(type="text", text=f"Audio extracted successfully. Output: {result}")]
        
        elif name == "merge_audio_video":
            result = await video_tools.merge_audio_video(
                arguments["video_path"],
                arguments["audio_path"]
            )
            return [TextContent(type="text", text=f"Audio and video merged successfully. Output: {result}")]
        
        elif name == "create_storyline":
            result = await video_tools.create_storyline(
                arguments["video_paths"],
                arguments.get("segment_duration", 3.0)
            )
            return [TextContent(type="text", text=f"Storyline created successfully. Output: {result}")]
        
        elif name == "send_cache":
            # Handle both old parameter name (video_path) and new unified name (video_paths)
            if "video_path" in arguments and "video_paths" not in arguments:
                arguments["video_paths"] = arguments["video_path"]
            elif "video_file" in arguments and "video_paths" not in arguments:
                arguments["video_paths"] = arguments["video_file"]
            
            if "video_paths" not in arguments:
                return [TextContent(type="text", text="Error: Missing required parameter 'video_paths'. Please provide video file path(s).")]
            
            result = await video_tools.send_cache(
                arguments["video_paths"],
                arguments.get("ttl_minutes", 60)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_video_summary":
            # Check for common parameter naming mistakes
            if "video_file" in arguments and "video_path" not in arguments:
                arguments["video_path"] = arguments["video_file"]
            
            result = await video_tools.get_video_summary(
                arguments["video_path"],
                arguments.get("custom_prompt")
            )
            return [TextContent(type="text", text=result)]
        
        elif name == "query_cached_video":
            # Handle both old parameter name (video_path) and new unified name (video_paths)
            if "video_path" in arguments and "video_paths" not in arguments:
                arguments["video_paths"] = arguments["video_path"]
            elif "video_file" in arguments and "video_paths" not in arguments:
                arguments["video_paths"] = arguments["video_file"]
            
            if "video_paths" not in arguments:
                return [TextContent(type="text", text="Error: Missing required parameter 'video_paths'. Please provide video file path(s).")]
            if "query" not in arguments:
                return [TextContent(type="text", text="Error: Missing required parameter 'query'. Please provide a query.")]
            
            result = await video_tools.query_cached_video(
                arguments["video_paths"],
                arguments["query"]
            )
            return [TextContent(type="text", text=result)]
        
        elif name == "get_cache_info":
            # Check for common parameter naming mistakes
            if "video_file" in arguments and "video_path" not in arguments:
                arguments["video_path"] = arguments["video_file"]
            
            result = video_tools.get_cache_info(arguments["video_path"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "list_files":
            # List files in workspace directory
            workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
            target_path = Path(workspace_dir) / (arguments.get("path", "") or "")
            
            if not target_path.exists():
                return [TextContent(type="text", text=f"Path not found: {target_path}")]
            
            if not target_path.is_dir():
                return [TextContent(type="text", text=f"Not a directory: {target_path}")]
            
            # List all files and directories
            items = []
            for item in sorted(target_path.iterdir()):
                size = ""
                if item.is_file():
                    file_size = item.stat().st_size
                    if file_size < 1024:
                        size = f"{file_size}B"
                    elif file_size < 1024 * 1024:
                        size = f"{file_size / 1024:.1f}KB"
                    elif file_size < 1024 * 1024 * 1024:
                        size = f"{file_size / (1024 * 1024):.1f}MB"
                    else:
                        size = f"{file_size / (1024 * 1024 * 1024):.2f}GB"
                    items.append(f"  📄 {item.name} ({size})")
                else:
                    items.append(f"  📁 {item.name}/")
            
            if not items:
                return [TextContent(type="text", text=f"Directory is empty: {target_path}")]
            
            result = f"Files in {target_path}:\n" + "\n".join(items)
            return [TextContent(type="text", text=result)]
        
        elif name == "delete_file":
            # Delete a file from workspace
            workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
            file_path = Path(workspace_dir) / arguments["file_path"]
            
            if not file_path.exists():
                return [TextContent(type="text", text=f"File not found: {file_path}")]
            
            if not file_path.is_file():
                return [TextContent(type="text", text=f"Not a file (use caution with directories): {file_path}")]
            
            try:
                file_path.unlink()
                return [TextContent(type="text", text=f"✅ Deleted: {file_path.name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error deleting file: {str(e)}")]
        
        elif name == "play_video":
            # Play video using system's default player
            workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
            video_path = Path(workspace_dir) / arguments["video_path"]
            
            if not video_path.exists():
                return [TextContent(type="text", text=f"Video file not found: {video_path}")]
            
            try:
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.Popen(["open", str(video_path)])
                elif system == "Windows":
                    os.startfile(str(video_path))
                elif system == "Linux":
                    subprocess.Popen(["xdg-open", str(video_path)])
                else:
                    return [TextContent(type="text", text=f"Unsupported platform: {system}")]
                
                return [TextContent(type="text", text=f"🎬 Opening video: {video_path.name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error playing video: {str(e)}")]
        
        elif name == "search_music":
            result = await video_tools.search_music(
                arguments["search_term"],
                arguments.get("limit", 10)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "download_music":
            result = await video_tools.download_music(
                arguments["track_id"],
                arguments.get("filename")
            )
            return [TextContent(type="text", text=f"Music downloaded successfully. File: {result}")]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


async def main():
    """Run the MCP server."""
    global video_tools
    
    # Get workspace, API key, and model name from environment
    workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Initialize video tools
    video_tools = VideoTools(
        workspace_dir=workspace_dir,
        gemini_api_key=gemini_api_key,
        model_name=model_name
    )
    
    # Create workspace directory
    Path(workspace_dir).mkdir(exist_ok=True)
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())


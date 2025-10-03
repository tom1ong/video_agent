"""
Real MCP Server for Video Editing Tools.
This is a proper MCP server that can be used with MCP clients.
"""

import asyncio
import json
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
            description="Combine an audio file with a video file",
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
            description="Cache a video in Gemini for cost-effective repeated analysis. Use this when user wants to analyze a video multiple times.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to video file"},
                    "ttl_minutes": {"type": "integer", "description": "Cache duration in minutes (default: 60)"}
                },
                "required": ["video_path"]
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
            description="Ask questions about a video using Gemini AI. Automatically caches the video if not already cached.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to video file"},
                    "query": {"type": "string", "description": "Question or query about the video"}
                },
                "required": ["video_path", "query"]
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
            result = await video_tools.send_cache(
                arguments["video_path"],
                arguments.get("ttl_minutes", 60)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_video_summary":
            result = await video_tools.get_video_summary(
                arguments["video_path"],
                arguments.get("custom_prompt")
            )
            return [TextContent(type="text", text=result)]
        
        elif name == "query_cached_video":
            result = await video_tools.query_cached_video(
                arguments["video_path"],
                arguments["query"]
            )
            return [TextContent(type="text", text=result)]
        
        elif name == "get_cache_info":
            result = video_tools.get_cache_info(arguments["video_path"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


async def main():
    """Run the MCP server."""
    global video_tools
    
    # Get workspace and API key from environment
    workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize video tools
    video_tools = VideoTools(workspace_dir, gemini_api_key=gemini_api_key)
    
    # Create workspace directory
    Path(workspace_dir).mkdir(exist_ok=True)
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())


"""
Async wrapper for MCP Video Server.
This provides a simple interface for LangChain while keeping MCP concepts.
"""

import json
from typing import Dict, Any, List
from video_tools import VideoTools


class VideoEditingMCPServer:
    """
    MCP-style server for video editing tools.
    Uses direct calls but follows MCP tool definition patterns.
    """
    
    def __init__(self, workspace_dir: str = "./workspace", gemini_api_key: str = None):
        self.video_tools = VideoTools(workspace_dir, gemini_api_key=gemini_api_key)
        self.tools = self._define_mcp_tools()
    
    def _define_mcp_tools(self) -> List[Dict[str, Any]]:
        """Define tools following MCP specification format."""
        return [
            {
                "name": "get_video_info",
                "description": "Get information about a video file including duration, resolution, fps, and audio presence",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to the video file (relative paths look in workspace)"
                        }
                    },
                    "required": ["video_path"]
                }
            },
            {
                "name": "cut_video",
                "description": "Extract a specific segment from a video by specifying start and end times",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Path to input video"},
                        "start_time": {"type": "number", "description": "Start time in seconds"},
                        "end_time": {"type": "number", "description": "End time in seconds"}
                    },
                    "required": ["video_path", "start_time", "end_time"]
                }
            },
            {
                "name": "trim_video",
                "description": "Remove unwanted parts from the beginning or end of a video",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"},
                        "trim_start": {"type": "number", "description": "Seconds to trim from start"},
                        "trim_end": {"type": "number", "description": "Seconds to trim from end"}
                    },
                    "required": ["video_path"]
                }
            },
            {
                "name": "concatenate_videos",
                "description": "Join multiple videos together in sequence",
                "inputSchema": {
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
            },
            {
                "name": "add_text_overlay",
                "description": "Add text overlay to a video at a specific time and position",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"},
                        "text": {"type": "string"},
                        "start_time": {"type": "number"},
                        "duration": {"type": "number"},
                        "position": {"type": "string", "enum": ["center", "top", "bottom"]},
                        "fontsize": {"type": "integer"},
                        "color": {"type": "string"}
                    },
                    "required": ["video_path", "text"]
                }
            },
            {
                "name": "speed_change",
                "description": "Change the playback speed of a video",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"},
                        "speed_factor": {"type": "number", "description": "2.0 = 2x faster, 0.5 = half speed"}
                    },
                    "required": ["video_path", "speed_factor"]
                }
            },
            {
                "name": "extract_audio",
                "description": "Extract the audio track from a video file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"}
                    },
                    "required": ["video_path"]
                }
            },
            {
                "name": "merge_audio_video",
                "description": "Combine an audio file with a video file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"},
                        "audio_path": {"type": "string"}
                    },
                    "required": ["video_path", "audio_path"]
                }
            },
            {
                "name": "create_storyline",
                "description": "Create a storyline by taking short segments from multiple videos",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_paths": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "segment_duration": {"type": "number"}
                    },
                    "required": ["video_paths"]
                }
            },
            {
                "name": "send_cache",
                "description": "Cache a video in Gemini for cost-effective repeated analysis. Use this when user says they want to start working on a video or analyze it multiple times.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Path to video file"},
                        "ttl_minutes": {"type": "integer", "description": "Cache duration in minutes (default: 60)"}
                    },
                    "required": ["video_path"]
                }
            },
            {
                "name": "get_video_summary",
                "description": "Get an AI-generated summary of a video using Gemini. Automatically caches the video if not already cached.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Path to video file"},
                        "custom_prompt": {"type": "string", "description": "Optional custom prompt for the summary"}
                    },
                    "required": ["video_path"]
                }
            },
            {
                "name": "query_cached_video",
                "description": "Ask questions about a video using Gemini AI. Automatically caches the video if not already cached. Use this for any AI-powered video analysis questions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Path to video file"},
                        "query": {"type": "string", "description": "Question or query about the video"}
                    },
                    "required": ["video_path", "query"]
                }
            },
            {
                "name": "get_cache_info",
                "description": "Check if a video is cached and get cache information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Path to video file"}
                    },
                    "required": ["video_path"]
                }
            }
        ]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools (MCP-style)."""
        return self.tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool (MCP-style)."""
        try:
            if name == "get_video_info":
                result = await self.video_tools.get_video_info(arguments["video_path"])
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            elif name == "cut_video":
                result = await self.video_tools.cut_video(
                    arguments["video_path"],
                    arguments["start_time"],
                    arguments["end_time"]
                )
                return {"content": [{"type": "text", "text": f"Video cut successfully. Output: {result}"}]}
            
            elif name == "trim_video":
                result = await self.video_tools.trim_video(
                    arguments["video_path"],
                    arguments.get("trim_start", 0),
                    arguments.get("trim_end", 0)
                )
                return {"content": [{"type": "text", "text": f"Video trimmed successfully. Output: {result}"}]}
            
            elif name == "concatenate_videos":
                result = await self.video_tools.concatenate_videos(arguments["video_paths"])
                return {"content": [{"type": "text", "text": f"Videos concatenated successfully. Output: {result}"}]}
            
            elif name == "add_text_overlay":
                result = await self.video_tools.add_text_overlay(
                    arguments["video_path"],
                    arguments["text"],
                    arguments.get("start_time", 0),
                    arguments.get("duration"),
                    arguments.get("position", "center"),
                    arguments.get("fontsize", 50),
                    arguments.get("color", "white")
                )
                return {"content": [{"type": "text", "text": f"Text overlay added successfully. Output: {result}"}]}
            
            elif name == "speed_change":
                result = await self.video_tools.speed_change(
                    arguments["video_path"],
                    arguments["speed_factor"]
                )
                return {"content": [{"type": "text", "text": f"Video speed changed successfully. Output: {result}"}]}
            
            elif name == "extract_audio":
                result = await self.video_tools.extract_audio(arguments["video_path"])
                return {"content": [{"type": "text", "text": f"Audio extracted successfully. Output: {result}"}]}
            
            elif name == "merge_audio_video":
                result = await self.video_tools.merge_audio_video(
                    arguments["video_path"],
                    arguments["audio_path"]
                )
                return {"content": [{"type": "text", "text": f"Audio and video merged successfully. Output: {result}"}]}
            
            elif name == "create_storyline":
                result = await self.video_tools.create_storyline(
                    arguments["video_paths"],
                    arguments.get("segment_duration", 3.0)
                )
                return {"content": [{"type": "text", "text": f"Storyline created successfully. Output: {result}"}]}
            
            elif name == "send_cache":
                result = await self.video_tools.send_cache(
                    arguments["video_path"],
                    arguments.get("ttl_minutes", 60)
                )
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            elif name == "get_video_summary":
                result = await self.video_tools.get_video_summary(
                    arguments["video_path"],
                    arguments.get("custom_prompt")
                )
                return {"content": [{"type": "text", "text": result}]}
            
            elif name == "query_cached_video":
                result = await self.video_tools.query_cached_video(
                    arguments["video_path"],
                    arguments["query"]
                )
                return {"content": [{"type": "text", "text": result}]}
            
            elif name == "get_cache_info":
                result = self.video_tools.get_cache_info(arguments["video_path"])
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
        
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error executing {name}: {str(e)}"}], "isError": True}


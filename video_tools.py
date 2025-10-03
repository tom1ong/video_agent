"""
Video editing tools implementation using MoviePy.
These tools are exposed via MCP for the LangChain agent.
"""

import os
import asyncio
import time
from typing import Dict, Any, List, Optional
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
    AudioFileClip
)
from pathlib import Path
from google import genai
from google.genai import types


class VideoTools:
    """Video editing tools that will be exposed via MCP."""
    
    def __init__(self, workspace_dir: str = "./workspace", gemini_api_key: Optional[str] = None):
        """Initialize video tools with a workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.operation_counter = 0
        
        # Initialize Gemini client for caching
        self.genai_client = None
        if gemini_api_key:
            self.genai_client = genai.Client(api_key=gemini_api_key)
        
        # Store cache information for videos
        self.video_caches: Dict[str, Dict[str, Any]] = {}  # video_path -> {cache_name, file_uri, etc}
    
    def _resolve_video_path(self, video_path: str) -> str:
        """
        Resolve video path - if relative, look in workspace directory.
        
        Args:
            video_path: Path to video (can be absolute or relative)
            
        Returns:
            Resolved absolute path to video
        """
        path = Path(video_path)
        
        # If it's already an absolute path, return as is
        if path.is_absolute():
            return str(path)
        
        # Otherwise, look in workspace directory
        workspace_path = self.workspace_dir / video_path
        return str(workspace_path)
    
    def _generate_output_filename(self, prefix: str = "output", extension: str = ".mp4") -> str:
        """Generate a unique output filename."""
        self.operation_counter += 1
        filename = f"{prefix}_{self.operation_counter}{extension}"
        return str(self.workspace_dir / filename)
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file (relative paths look in workspace)
            
        Returns:
            Dictionary with video information (duration, fps, size, resolution)
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _get_info():
                clip = VideoFileClip(resolved_path)
                info = {
                    "duration": clip.duration,
                    "fps": clip.fps,
                    "size": clip.size,
                    "width": clip.w,
                    "height": clip.h,
                    "has_audio": clip.audio is not None,
                    "filename": video_path
                }
                clip.close()
                return info
            
            return await loop.run_in_executor(None, _get_info)
        except Exception as e:
            return {"error": str(e)}
    
    async def cut_video(self, video_path: str, start_time: float, end_time: float) -> str:
        """
        Cut a segment from a video.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path to the output video
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            loop = asyncio.get_event_loop()
            
            def _cut():
                clip = VideoFileClip(resolved_path)
                cut_clip = clip.subclip(start_time, end_time)
                
                output_path = self._generate_output_filename("cut")
                cut_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                clip.close()
                cut_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _cut)
        except Exception as e:
            return f"Error cutting video: {str(e)}"
    
    async def trim_video(self, video_path: str, trim_start: float = 0, trim_end: float = 0) -> str:
        """
        Trim video from beginning and/or end.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            trim_start: Seconds to trim from start
            trim_end: Seconds to trim from end
            
        Returns:
            Path to the output video
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            loop = asyncio.get_event_loop()
            
            def _trim():
                clip = VideoFileClip(resolved_path)
                duration = clip.duration
                
                start = trim_start
                end = duration - trim_end
                
                trimmed_clip = clip.subclip(start, end)
                output_path = self._generate_output_filename("trimmed")
                trimmed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                clip.close()
                trimmed_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _trim)
        except Exception as e:
            return f"Error trimming video: {str(e)}"
    
    async def concatenate_videos(self, video_paths: List[str]) -> str:
        """
        Concatenate multiple videos together.
        
        Args:
            video_paths: List of video file paths to concatenate (relative paths look in workspace)
            
        Returns:
            Path to the output video
        """
        try:
            resolved_paths = [self._resolve_video_path(path) for path in video_paths]
            loop = asyncio.get_event_loop()
            
            def _concatenate():
                clips = [VideoFileClip(path) for path in resolved_paths]
                final_clip = concatenate_videoclips(clips, method="compose")
                
                output_path = self._generate_output_filename("concatenated")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                for clip in clips:
                    clip.close()
                final_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _concatenate)
        except Exception as e:
            return f"Error concatenating videos: {str(e)}"
    
    async def add_text_overlay(
        self, 
        video_path: str, 
        text: str, 
        start_time: float = 0,
        duration: float = None,
        position: str = "center",
        fontsize: int = 50,
        color: str = "white"
    ) -> str:
        """
        Add text overlay to a video.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            text: Text to overlay
            start_time: When to start showing text (seconds)
            duration: How long to show text (seconds, None = until end)
            position: Text position ('center', 'top', 'bottom')
            fontsize: Font size
            color: Text color
            
        Returns:
            Path to the output video
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            loop = asyncio.get_event_loop()
            
            def _add_text():
                clip = VideoFileClip(resolved_path)
                
                video_duration = clip.duration
                text_duration = video_duration - start_time if duration is None else duration
                
                # Create text clip
                txt_clip = TextClip(
                    text, 
                    fontsize=fontsize, 
                    color=color,
                    font='Arial'
                )
                txt_clip = txt_clip.set_start(start_time).set_duration(text_duration)
                
                # Position the text
                if position == "center":
                    txt_clip = txt_clip.set_position("center")
                elif position == "top":
                    txt_clip = txt_clip.set_position(("center", 50))
                elif position == "bottom":
                    txt_clip = txt_clip.set_position(("center", clip.h - 100))
                
                # Composite video
                final_clip = CompositeVideoClip([clip, txt_clip])
                
                output_path = self._generate_output_filename("text_overlay")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                clip.close()
                final_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _add_text)
        except Exception as e:
            return f"Error adding text overlay: {str(e)}"
    
    async def speed_change(self, video_path: str, speed_factor: float) -> str:
        """
        Change the speed of a video.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            speed_factor: Speed multiplier (2.0 = 2x faster, 0.5 = half speed)
            
        Returns:
            Path to the output video
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            loop = asyncio.get_event_loop()
            
            def _change_speed():
                clip = VideoFileClip(resolved_path)
                
                # Change speed
                final_clip = clip.fx(lambda c: c.speedx(speed_factor))
                
                output_path = self._generate_output_filename(f"speed_{speed_factor}x")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                clip.close()
                final_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _change_speed)
        except Exception as e:
            return f"Error changing speed: {str(e)}"
    
    async def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from a video.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            
        Returns:
            Path to the output audio file
        """
        try:
            resolved_path = self._resolve_video_path(video_path)
            loop = asyncio.get_event_loop()
            
            def _extract():
                clip = VideoFileClip(resolved_path)
                
                output_path = self._generate_output_filename("audio", ".mp3")
                clip.audio.write_audiofile(output_path)
                
                clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _extract)
        except Exception as e:
            return f"Error extracting audio: {str(e)}"
    
    async def merge_audio_video(self, video_path: str, audio_path: str) -> str:
        """
        Merge an audio file with a video file.
        
        Args:
            video_path: Path to the input video (relative paths look in workspace)
            audio_path: Path to the input audio (relative paths look in workspace)
            
        Returns:
            Path to the output video
        """
        try:
            resolved_video_path = self._resolve_video_path(video_path)
            resolved_audio_path = self._resolve_video_path(audio_path)
            loop = asyncio.get_event_loop()
            
            def _merge():
                video_clip = VideoFileClip(resolved_video_path)
                audio_clip = AudioFileClip(resolved_audio_path)
                
                final_clip = video_clip.set_audio(audio_clip)
                
                output_path = self._generate_output_filename("merged")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _merge)
        except Exception as e:
            return f"Error merging audio and video: {str(e)}"
    
    async def create_storyline(self, video_paths: List[str], segment_duration: float = 3.0) -> str:
        """
        Create a storyline by taking short segments from multiple videos and concatenating them.
        
        Args:
            video_paths: List of video paths (relative paths look in workspace)
            segment_duration: Duration of each segment (seconds)
            
        Returns:
            Path to the output video
        """
        try:
            resolved_paths = [self._resolve_video_path(path) for path in video_paths]
            loop = asyncio.get_event_loop()
            
            def _create_storyline():
                segments = []
                
                for resolved_path in resolved_paths:
                    clip = VideoFileClip(resolved_path)
                    # Take a segment from the beginning
                    duration = min(segment_duration, clip.duration)
                    segment = clip.subclip(0, duration)
                    segments.append(segment)
                
                final_clip = concatenate_videoclips(segments, method="compose")
                
                output_path = self._generate_output_filename("storyline")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                for segment in segments:
                    segment.close()
                final_clip.close()
                
                return output_path
            
            return await loop.run_in_executor(None, _create_storyline)
        except Exception as e:
            return f"Error creating storyline: {str(e)}"
    
    async def send_cache(self, video_path: str, ttl_minutes: int = 60) -> Dict[str, Any]:
        """
        Upload video and create a cache for it in Gemini.
        This caches the visual tokens for cost savings on repeated queries.
        
        Args:
            video_path: Path to the video file (relative paths look in workspace)
            ttl_minutes: Time to live for the cache in minutes (default: 60)
            
        Returns:
            Dictionary with cache information
        """
        try:
            if not self.genai_client:
                return {"error": "Gemini API client not initialized. Please provide API key."}
            
            resolved_path = self._resolve_video_path(video_path)
            
            # Check if cache already exists for this video
            if video_path in self.video_caches:
                return {
                    "status": "already_cached",
                    "video_path": video_path,
                    "cache_name": self.video_caches[video_path]["cache_name"],
                    "message": "Video is already cached. Using existing cache."
                }
            
            loop = asyncio.get_event_loop()
            
            def _upload_and_cache():
                # Upload the video using the Files API
                video_file = self.genai_client.files.upload(file=resolved_path)
                
                # Wait for the file to finish processing
                while video_file.state.name == 'PROCESSING':
                    print(f'⏳ Waiting for video {video_path} to be processed...')
                    time.sleep(2)
                    video_file = self.genai_client.files.get(name=video_file.name)
                
                print(f'✅ Video processing complete: {video_file.uri}')
                
                # Create a cache with specified TTL
                cache = self.genai_client.caches.create(
                    model='models/gemini-2.0-flash-001',
                    config=types.CreateCachedContentConfig(
                        display_name=f'video_cache_{Path(video_path).stem}',
                        system_instruction=(
                            'You are an expert video analyzer. Your job is to analyze '
                            'and answer questions about the video content you have access to. '
                            'Provide detailed, accurate information based on what you observe in the video.'
                        ),
                        contents=[video_file],
                        ttl=f"{ttl_minutes * 60}s",
                    )
                )
                
                # Store cache information
                cache_info = {
                    "cache_name": cache.name,
                    "file_uri": video_file.uri,
                    "file_name": video_file.name,
                    "expire_time": cache.expire_time.isoformat() if hasattr(cache.expire_time, 'isoformat') else str(cache.expire_time),
                    "token_count": cache.usage_metadata.total_token_count if hasattr(cache, 'usage_metadata') else None
                }
                
                return cache_info
            
            cache_info = await loop.run_in_executor(None, _upload_and_cache)
            
            # Store cache info for future use
            self.video_caches[video_path] = cache_info
            
            return {
                "status": "cached",
                "video_path": video_path,
                "cache_name": cache_info["cache_name"],
                "file_uri": cache_info["file_uri"],
                "expire_time": cache_info["expire_time"],
                "token_count": cache_info["token_count"],
                "message": f"Video cached successfully. Cache will expire at {cache_info['expire_time']}"
            }
            
        except Exception as e:
            return {"error": f"Error caching video: {str(e)}"}
    
    async def get_video_summary(self, video_path: str, custom_prompt: Optional[str] = None) -> str:
        """
        Get a summary of the video using cached content.
        Automatically caches the video if not already cached.
        
        Args:
            video_path: Path to the video file (relative paths look in workspace)
            custom_prompt: Optional custom prompt for the summary
            
        Returns:
            Video summary text
        """
        try:
            if not self.genai_client:
                return "Error: Gemini API client not initialized. Please provide API key."
            
            # Check if video is cached, if not, cache it
            if video_path not in self.video_caches:
                print(f"📦 Video not cached yet. Caching {video_path}...")
                cache_result = await self.send_cache(video_path)
                if "error" in cache_result:
                    return f"Error: {cache_result['error']}"
            
            cache_info = self.video_caches[video_path]
            loop = asyncio.get_event_loop()
            
            def _generate_summary():
                # Default prompt if none provided
                prompt = custom_prompt or (
                    "Please provide a comprehensive summary of this video including:\n"
                    "1. Main subjects or characters\n"
                    "2. Key events or actions\n"
                    "3. Setting and environment\n"
                    "4. Overall theme or purpose\n"
                    "5. Notable timestamps for important moments"
                )
                
                # Generate content using the cached video
                response = self.genai_client.models.generate_content(
                    model='models/gemini-2.0-flash-001',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_info["cache_name"]
                    )
                )
                
                # Print usage metadata to show cache savings
                if hasattr(response, 'usage_metadata'):
                    print(f"📊 Token usage - Total: {response.usage_metadata.total_token_count}, "
                          f"Cached: {response.usage_metadata.cached_content_token_count}, "
                          f"New prompt: {response.usage_metadata.prompt_token_count}")
                
                return response.text
            
            summary = await loop.run_in_executor(None, _generate_summary)
            return summary
            
        except Exception as e:
            return f"Error generating video summary: {str(e)}"
    
    async def query_cached_video(self, video_path: str, query: str) -> str:
        """
        Query a cached video with a custom question.
        Automatically caches the video if not already cached.
        
        Args:
            video_path: Path to the video file (relative paths look in workspace)
            query: Question or query about the video
            
        Returns:
            Response text from Gemini
        """
        try:
            if not self.genai_client:
                return "Error: Gemini API client not initialized. Please provide API key."
            
            # Check if video is cached, if not, cache it
            if video_path not in self.video_caches:
                print(f"📦 Video not cached yet. Caching {video_path}...")
                cache_result = await self.send_cache(video_path)
                if "error" in cache_result:
                    return f"Error: {cache_result['error']}"
            
            cache_info = self.video_caches[video_path]
            loop = asyncio.get_event_loop()
            
            def _query_video():
                # Generate content using the cached video
                response = self.genai_client.models.generate_content(
                    model='models/gemini-2.0-flash-001',
                    contents=query,
                    config=types.GenerateContentConfig(
                        cached_content=cache_info["cache_name"]
                    )
                )
                
                return response.text
            
            result = await loop.run_in_executor(None, _query_video)
            return result
            
        except Exception as e:
            return f"Error querying video: {str(e)}"
    
    def get_cache_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a cached video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Cache information or error message
        """
        if video_path in self.video_caches:
            return {
                "status": "cached",
                "cache_info": self.video_caches[video_path]
            }
        else:
            return {
                "status": "not_cached",
                "message": f"Video {video_path} is not cached yet."
            }


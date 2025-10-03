"""
Main entry point for the Video Editing Agent.
Uses REAL MCP (Model Context Protocol) with LangChain and Gemini.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from agent import VideoEditingAgent


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       🎬 Video Editing Agent with MCP & Gemini 🤖             ║
║                                                               ║
║  Powered by: LangChain | Google Gemini | Real MCP Protocol   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Welcome! I'm your AI video editing assistant.

I can help you:
  • Cut and trim videos
  • Concatenate multiple videos
  • Add text overlays
  • Change video speed
  • Extract and merge audio
  • Create storylines from multiple clips
  • List, delete, and play video files
  • And much more!

Just tell me what you want to do in natural language!
"""
    print(banner)


def print_examples():
    """Print example commands."""
    examples = """
Example commands you can try:
  • "Show me info about IMG_5437.MOV"
  • "Cut the first 10 seconds from video1.mp4"
  • "Take video1.mp4 and video2.mp4 and join them together"
  • "Add text 'Hello World' at 5 seconds into my_video.mp4"
  • "Speed up video1.mp4 by 2x"
  • "Create a storyline from video1.mp4, video2.mp4, video3.mp4"
  • "Extract audio from video1.mp4"
  
File management commands:
  • "List files in the workspace" (ls command)
  • "Delete output.mp4" (rm command)
  • "Play IMG_5437.MOV" (open command)
  
Special commands:
  • 'examples' - Show these examples again
  • 'clear' - Clear conversation history (start fresh)
  • 'history' - Show conversation history
  • 'exit' or 'quit' - Exit the program
"""
    print(examples)


def setup_workspace():
    """Create workspace directory if it doesn't exist."""
    workspace_dir = Path("./workspace")
    workspace_dir.mkdir(exist_ok=True)
    return workspace_dir


async def main():
    """Main async function to run the agent."""
    # Load environment variables
    load_dotenv('.env.local')
    
    # Get API key and model name
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if not gemini_api_key:
        print("\n❌ Error: GEMINI_API_KEY not found!")
        print("\nPlease set your Gemini API key:")
        print("  1. Create a .env.local file in the current directory")
        print("  2. Add: GEMINI_API_KEY=your-api-key-here")
        print("\nOr set it as an environment variable:")
        print("  export GEMINI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Setup workspace
    workspace_dir = setup_workspace()
    
    # Print banner
    print_banner()
    print(f"📁 Workspace directory: {workspace_dir.absolute()}")
    print(f"🤖 Using model: {model_name}")
    print(f"📍 Place your video files in the workspace directory.\n")
    
    # Initialize agent
    print("🔄 Initializing video editing agent with MCP...")
    agent = None
    try:
        agent = VideoEditingAgent(
            gemini_api_key=gemini_api_key, 
            workspace_dir=str(workspace_dir),
            model_name=model_name
        )
        await agent.connect()
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Show examples
    print_examples()
    
    # Main conversation loop
    print("\n" + "="*60)
    print("Start chatting! (type 'exit' to quit)")
    print("="*60 + "\n")
    
    try:
        while True:
            try:
                # Get user input (use asyncio to not block)
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("You: ").strip())
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thanks for using Video Editing Agent! Goodbye!")
                    break
                
                if user_input.lower() == 'examples':
                    print_examples()
                    continue
                
                if user_input.lower() == 'clear':
                    agent.clear_history()
                    print("\n✅ Conversation history cleared! Starting fresh.\n")
                    continue
                
                if user_input.lower() == 'history':
                    history = agent.get_history()
                    if history:
                        print(f"\n📜 Conversation History:\n{history}\n")
                    else:
                        print("\n📜 No conversation history yet.\n")
                    continue
                
                # Send to agent
                print("\n🤖 Agent: Thinking...\n")
                response = await agent.chat(user_input)
                
                print(f"\n🤖 Agent: {response}\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using Video Editing Agent! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
                continue
    finally:
        # Clean up MCP connection
        if agent:
            print("\n🔌 Disconnecting from MCP server...")
            await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

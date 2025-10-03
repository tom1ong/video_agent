"""
Test script to verify the Video Editing Agent setup.
Run this to check if all dependencies are properly installed.
"""

import sys
import os


def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    
    required_packages = [
        ("langchain", "LangChain"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("moviepy", "MoviePy"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "Pydantic"),
    ]
    
    failed = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} NOT installed")
            failed.append(name)
    
    return len(failed) == 0


def test_ffmpeg():
    """Test if FFmpeg is installed."""
    print("\nTesting FFmpeg...")
    
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  ✅ FFmpeg installed")
            return True
        else:
            print("  ❌ FFmpeg NOT working properly")
            return False
    except FileNotFoundError:
        print("  ❌ FFmpeg NOT installed")
        return False
    except Exception as e:
        print(f"  ❌ Error testing FFmpeg: {e}")
        return False


def test_api_key():
    """Test if Gemini API key is configured."""
    print("\nTesting API key configuration...")
    
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  ✅ GEMINI_API_KEY found ({masked_key})")
        return True
    else:
        print("  ❌ GEMINI_API_KEY NOT found")
        print("     Create a .env.local file with: GEMINI_API_KEY=your-key-here")
        return False


def test_workspace():
    """Test if workspace directory exists."""
    print("\nTesting workspace directory...")
    
    workspace = "./workspace"
    if os.path.exists(workspace):
        print(f"  ✅ Workspace directory exists: {os.path.abspath(workspace)}")
        return True
    else:
        print(f"  ⚠️  Workspace directory doesn't exist (will be created automatically)")
        return True


def test_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    modules = [
        ("video_tools", "VideoTools"),
        ("mcp_server", "MCP Tool Server"),
        ("agent", "Video Editing Agent"),
    ]
    
    failed = []
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✅ {name} module loads correctly")
        except Exception as e:
            print(f"  ❌ {name} module failed: {e}")
            failed.append(name)
    
    return len(failed) == 0


def main():
    """Run all tests."""
    print("="*60)
    print("Video Editing Agent - Setup Test")
    print("="*60)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Package imports", test_imports()))
    results.append(("FFmpeg", test_ffmpeg()))
    results.append(("API key", test_api_key()))
    results.append(("Workspace", test_workspace()))
    results.append(("Custom modules", test_modules()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to use the Video Editing Agent.")
        print("\nRun: python main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nSee QUICKSTART.md for setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


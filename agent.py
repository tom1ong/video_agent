"""
LangChain Agent with Real MCP Integration.
Connects to the MCP server as a client.
Uses structured ReAct format with response_schema for guaranteed valid JSON.
"""

import os
import json
import ast
import asyncio
from typing import List, Dict, Any, Optional, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, create_model
from google import genai
from google.genai import types


# Pydantic models for agent steps with mode indicators
from typing import Literal

class ThinkingStep(BaseModel):
    """Agent is thinking - free text reasoning"""
    type: Literal["thinking"] = "thinking"
    thought: str = Field(description="Your reasoning, analysis, or planning in natural language")


class ActionStep(BaseModel):
    """Agent is taking an action - structured tool call"""
    type: Literal["action"] = "action"
    tool: str = Field(description="The name of the tool to use")
    args: Dict[str, Any] = Field(description="The arguments for the tool as a JSON object with correct parameter names")


class FinalStep(BaseModel):
    """Agent has the final answer"""
    type: Literal["final"] = "final"
    answer: str = Field(description="The final answer to give to the user")


# Union type for agent step - model chooses which mode
AgentStep = Union[ThinkingStep, ActionStep, FinalStep]


class VideoEditingAgent:
    """LangChain Agent that connects to a real MCP server."""
    
    def __init__(self, gemini_api_key: str = None, workspace_dir: str = "./workspace", model_name: str = "gemini-2.5-flash"):
        """
        Initialize the video editing agent with MCP client.
        
        Args:
            gemini_api_key: Google Gemini API key
            workspace_dir: Directory for storing temporary and output files
            model_name: Gemini model name to use (default: gemini-2.5-flash)
        """
        # Set up API key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        self.gemini_api_key = gemini_api_key
        self.workspace_dir = workspace_dir
        self.model_name = model_name
        
        # MCP session will be initialized in async context
        self.session: ClientSession = None
        self.exit_stack = None
        
        # Initialize direct Gemini client for structured output
        self.genai_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        
        # Store tool schemas for structured parameter generation
        self.tool_schemas: Dict[str, Any] = {}
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
            output_key="output"
        )
        
        # Tools and agent will be created after MCP connection
        self.tools = []
        self.agent_executor = None
    
    async def connect(self):
        """Connect to the MCP server."""
        import sys
        from contextlib import AsyncExitStack
        
        # Set up MCP server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", "mcp_server.py"],
            env={
                "WORKSPACE_DIR": self.workspace_dir,
                "GEMINI_API_KEY": self.gemini_api_key or "",
                "GEMINI_MODEL": self.model_name,
                # Epidemic Sound credentials
                "EPIDEMIC_ACCESS_KEY_ID": os.getenv("EPIDEMIC_ACCESS_KEY_ID", ""),
                "EPIDEMIC_ACCESS_KEY_SECRET": os.getenv("EPIDEMIC_ACCESS_KEY_SECRET", ""),
                "EPIDEMIC_USER_ID": os.getenv("EPIDEMIC_USER_ID", "default_user")
            }
        )
        
        # Create exit stack to manage cleanup
        self.exit_stack = AsyncExitStack()
        
        # Connect to MCP server via stdio
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        
        # Initialize MCP session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        # Initialize the session
        await self.session.initialize()
        
        # Get tools from MCP server
        tools_result = await self.session.list_tools()
        
        # Create LangChain tools from MCP tools
        self.tools = self._create_langchain_tools(tools_result.tools)
        
        # Create the agent
        self.agent_executor = self._create_agent()
        
        print(f"✅ Connected to MCP server with {len(self.tools)} tools!")
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.exit_stack:
            await self.exit_stack.aclose()
    
    async def _generate_agent_step(self, prompt: str, conversation_history: str = "") -> Union[AgentAction, AgentFinish, str]:
        """
        Generate ONE agent step using response_schema with mode detection.
        Model self-indicates: thinking (free text) | action (structured) | final
        Returns: AgentAction, AgentFinish, or thinking text
        """
        if not self.genai_client:
            raise ValueError("Gemini client not initialized")
        
        full_prompt = f"""{conversation_history}

{prompt}

Output ONE step. Choose the appropriate type:
- "thinking": if you need to reason, analyze, or plan (use free text)
- "action": if you're ready to use a tool (provide tool name and args)
- "final": if you have the final answer for the user"""
        
        # DEBUG: Print the prompt we're sending
        print("\n" + "="*80)
        print("🔵 LLM PROMPT:")
        print("="*80)
        print(full_prompt)
        print("="*80)
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            response = self.genai_client.models.generate_content(
                model=f'models/{self.model_name}',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=AgentStep  # Union type - model picks
                )
            )
            # DEBUG: Print raw response
            print("\n" + "="*80)
            print("🟢 LLM RAW RESPONSE:")
            print("="*80)
            print(response.text)
            print("="*80)
            
            # DEBUG: Print usage metadata if available
            if hasattr(response, 'usage_metadata'):
                print("\n📊 Token Usage:")
                print(f"  - Total tokens: {response.usage_metadata.total_token_count}")
                print(f"  - Prompt tokens: {response.usage_metadata.prompt_token_count}")
                print(f"  - Response tokens: {response.usage_metadata.candidates_token_count}")
            
            return json.loads(response.text)
        
        try:
            step = await loop.run_in_executor(None, _generate)
            step_type = step.get('type')
            
            # DEBUG: Print parsed structure
            print("\n" + "="*80)
            print("🟡 PARSED STEP:")
            print("="*80)
            print(f"Type: {step_type}")
            print(f"Full structure: {json.dumps(step, indent=2)}")
            print("="*80)
            
            if step_type == 'thinking':
                # Return thinking text to be added to scratchpad
                thought = step.get('thought', '')
                print(f"\n💭 THINKING MODE: {thought}\n")
                return thought
            
            elif step_type == 'action':
                # Return structured action with guaranteed valid JSON args
                tool_name = step.get('tool', '')
                tool_args = step.get('args', {})
                print(f"\n🎬 ACTION MODE")
                print(f"  Tool: {tool_name}")
                print(f"  Args: {json.dumps(tool_args, indent=2)}\n")
                
                return AgentAction(
                    tool=tool_name,
                    tool_input=tool_args,
                    log=f"Using tool: {tool_name}"
                )
            
            elif step_type == 'final':
                # Return final answer
                answer = step.get('answer', '')
                print(f"\n✅ FINAL MODE: {answer}\n")
                return AgentFinish(
                    return_values={"output": answer},
                    log="Task completed"
                )
            
            else:
                # Unknown type, treat as finish
                print(f"\n⚠️  UNKNOWN TYPE: {step_type}")
                return AgentFinish(
                    return_values={"output": str(step)},
                    log=""
                )
                
        except Exception as e:
            print(f"\n❌ ERROR in agent step generation: {e}")
            import traceback
            traceback.print_exc()
            return AgentFinish(
                return_values={"output": f"Error: {str(e)}"},
                log=""
            )
    
    async def _generate_structured_params(self, tool_name: str, user_request: str) -> Dict[str, Any]:
        """
        Use Gemini's response_schema to generate structured JSON parameters.
        This ensures 100% valid JSON for tool inputs.
        """
        if not self.genai_client or tool_name not in self.tool_schemas:
            return {}
        
        schema = self.tool_schemas[tool_name]
        
        print(f"\n🔧 FALLBACK: Generating structured params for '{tool_name}'")
        
        # Create Pydantic model from schema
        fields = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type", "string")
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in schema.get("required", [])
            
            # Map JSON types to Python types
            if prop_type == "string":
                py_type = str
            elif prop_type == "number":
                py_type = float
            elif prop_type == "integer":
                py_type = int
            elif prop_type == "boolean":
                py_type = bool
            elif prop_type == "array":
                items_type = prop_schema.get("items", {}).get("type", "string")
                if items_type == "string":
                    py_type = list[str]
                else:
                    py_type = list
            else:
                py_type = str
            
            if is_required:
                fields[prop_name] = (py_type, Field(description=prop_desc))
            else:
                fields[prop_name] = (py_type, Field(default=None, description=prop_desc))
        
        # Create dynamic model
        ParamsModel = create_model(f"{tool_name}_params", **fields)
        
        # Generate structured output using Gemini
        prompt = f"""Based on this user request, generate the parameters for the {tool_name} tool:

User request: {user_request}

Generate the appropriate parameters as a JSON object."""
        
        print("\n" + "-"*80)
        print("🔵 STRUCTURED PARAMS PROMPT:")
        print("-"*80)
        print(prompt)
        print("-"*80)
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            response = self.genai_client.models.generate_content(
                model=f'models/{self.model_name}',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=ParamsModel
                )
            )
            print("\n" + "-"*80)
            print("🟢 STRUCTURED PARAMS RESPONSE:")
            print("-"*80)
            print(response.text)
            print("-"*80)
            return json.loads(response.text)
        
        params = await loop.run_in_executor(None, _generate)
        
        # Remove None values for optional parameters
        result = {k: v for k, v in params.items() if v is not None}
        print(f"\n✅ Final params: {json.dumps(result, indent=2)}\n")
        return result
    
    def _create_langchain_tools(self, mcp_tools) -> List[Tool]:
        """Convert MCP tools to LangChain Tools with robust JSON parsing."""
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            tool_name = mcp_tool.name
            tool_description = mcp_tool.description
            input_schema = mcp_tool.inputSchema
            
            # Store schema for structured parameter generation
            if input_schema:
                self.tool_schemas[tool_name] = input_schema
            
            # Create tool function that calls MCP
            # Handles both string (legacy) and dict (structured ReAct) inputs
            def create_tool_func(name: str):
                async def tool_func(input_data: Union[str, Dict[str, Any]]) -> str:
                    try:
                        params = None
                        
                        print(f"\n{'='*80}")
                        print(f"🔧 TOOL EXECUTION: {name}")
                        print(f"{'='*80}")
                        print(f"Input data type: {type(input_data).__name__}")
                        print(f"Input data: {input_data}")
                        
                        # Handle dict input (from structured ReAct)
                        if isinstance(input_data, dict):
                            params = input_data
                            print(f"✅ Received structured dict input")
                        # Handle string input (legacy or fallback)
                        elif isinstance(input_data, str):
                            input_str = input_data
                            # Try to parse as JSON first
                            if input_str.strip().startswith('{') or input_str.strip().startswith('['):
                                try:
                                    # Try standard JSON (double quotes)
                                    params = json.loads(input_str)
                                    print(f"✅ Parsed JSON from string")
                                except json.JSONDecodeError:
                                    try:
                                        # Try Python literal eval (single quotes)
                                        params = ast.literal_eval(input_str)
                                        print(f"✅ Parsed with literal_eval")
                                    except (ValueError, SyntaxError):
                                        # JSON parsing failed - fall back to structured output
                                        print(f"⚠️  JSON parsing failed, using structured output as fallback")
                                        params = None
                            
                            # If parsing failed OR no JSON detected, use structured output
                            if params is None and self.genai_client and name in self.tool_schemas:
                                print(f"🎯 Generating structured params with response_schema...")
                                params = await self._generate_structured_params(name, input_str)
                            elif params is None:
                                # Fallback: simple string for single-parameter tools
                                params = {"video_path": input_str}
                                print(f"ℹ️  Using fallback single-parameter format")
                        
                        print(f"\n📤 Calling MCP tool '{name}' with params:")
                        print(f"   {json.dumps(params, indent=2)}")
                        
                        # Call MCP server
                        result = await self.session.call_tool(name, params)
                        
                        # Extract text from MCP response
                        if result.content:
                            texts = [c.text for c in result.content if hasattr(c, 'text')]
                            response_text = "\n".join(texts)
                        else:
                            response_text = str(result)
                        
                        print(f"\n📥 Tool response:")
                        print(f"   {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                        print(f"{'='*80}\n")
                        
                        return response_text
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        print(f"\n❌ Tool execution error: {error_msg}")
                        print(f"{'='*80}\n")
                        return error_msg
                
                return tool_func
            
            # Create LangChain tool with async support
            tool = Tool(
                name=tool_name,
                func=lambda x, name=tool_name: asyncio.run(create_tool_func(name)(x)),
                coroutine=create_tool_func(tool_name),
                description=tool_description
            )
            langchain_tools.append(tool)
        
        return langchain_tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with explicit JSON formatting instructions."""
        
        template = """You are a friendly video editing assistant with reasoning capabilities.

You have access to these tools:
{tools}

Available tool names: {tool_names}

IMPORTANT GUIDELINES:
- Only use tools when the user asks you to DO something
- For greetings or casual questions, just respond directly
- You can have multiple reasoning steps before taking actions
- Tool arguments are automatically validated with JSON schemas

WORKFLOW FOR CREATING VIRAL/HIGHLIGHT CLIPS:

IMPORTANT: When user says "from my videos" (plural) or mentions multiple files, create a compilation using clips from MULTIPLE videos, not just one!

1. CACHE: Use send_cache to cache videos (works with one or more videos)
   - Tool: send_cache
   - One video: {{"video_paths": "video.mp4"}}
   - Many videos: {{"video_paths": ["video1.mp4", "video2.mp4", "video3.mp4"]}}
   - Automatically caches in parallel when given multiple videos

2. QUERY: Use query_cached_video to get an edit plan
   - Tool: query_cached_video
   - Parameters: {{"video_paths": "video.mp4" or ["video1.mp4", "..."], "query": "your question here"}}
   - Response format for one video: {{'clip_1': {{'start': X.X, 'end': Y.Y, 'reason': '...'}}, ...}}
   - Response format with videos array: {{'clip_1': {{'video': 'video1.mp4', 'start': X.X, 'end': Y.Y, 'reason': '...'}}, ...}}
   - When querying multiple videos, response includes 'video' field indicating source
   - CRITICAL: When querying multiple videos for a compilation, ask for MULTIPLE clips from DIFFERENT videos!
   - Example query for 5-second compilation from 3 videos: "Create an edit plan for a 5-second viral compilation. Select 1-2 second clips from at least 2-3 different videos showing the most exciting/emotional moments. Return JSON with 'video' field for each clip."

3. EXECUTE: Use cut_video for each segment in the edit plan
   - Tool: cut_video
   - Parameters: {{"video_path": "video.mp4", "start_time": X.X, "end_time": Y.Y}}
   - Use exact start/end times from the JSON response
   - NEVER default to 0-5s or start-end without analysis!

4. CONCATENATE (if needed): Use concatenate_videos to combine clips
   - Tool: concatenate_videos
   - Parameters: {{"video_paths": ["cut_1.mp4", "cut_2.mp4", ...]}}
   - IMPORTANT: Use just filenames without "workspace/" prefix
   - The cut_video tool returns paths like "workspace/cut_1.mp4", extract just "cut_1.mp4"

EXAMPLES:

Example A - Make viral clip from one video:
- User: "Make a 5 second viral clip from video.mp4"
- Step 1: query_cached_video with {{"video_paths": "video.mp4", "query": "Provide an edit plan for a 5-second viral clip with the most emotionally impactful segment. Return JSON: {{'clip_1': {{'start': X.X, 'end': Y.Y, 'reason': '...'}}}}"}}
- Step 2: Parse response to get timestamps (e.g., start: 10.5, end: 15.5)
- Step 3: cut_video with {{"video_path": "video.mp4", "start_time": 10.5, "end_time": 15.5}}

Example B - Create storyline from one video:
- User: "Create a storyline with five 1-second clips from video.mp4"
- Step 1: query_cached_video to get edit plan for 5 segments
- Step 2-6: cut_video for each clip
- Step 7: concatenate_videos with {{"video_paths": ["cut_1.mp4", "cut_2.mp4", "cut_3.mp4", "cut_4.mp4", "cut_5.mp4"]}}

Example C - Highlight reel from multiple videos:
- User: "Create a 10-second highlight reel from video1.mp4, video2.mp4, and video3.mp4"
- Step 1: send_cache with {{"video_paths": ["video1.mp4", "video2.mp4", "video3.mp4"]}}
- Step 2: query_cached_video with {{"video_paths": ["video1.mp4", "video2.mp4", "video3.mp4"], "query": "Create edit plan for 10-second highlight reel using clips from ALL 3 videos. Select 3-4 clips of 2-3 seconds each from DIFFERENT videos showing the most exciting moments. Return JSON with 'video' field for each clip: {{'clip_1': {{'video': 'videoX.mp4', 'start': X.X, 'end': Y.Y, 'reason': '...'}}, ...}}"}}
- Step 3: Parse response - clips will have 'video' field indicating source (MUST have clips from multiple videos)
- Step 4-N: cut_video from each source video using timestamps
- Final: concatenate_videos to combine all clips

Example D - Viral video from multiple videos:
- User: "make a 5s viral video from my videos" (where workspace has video1.mp4, video2.mp4, video3.mp4)
- Step 1: send_cache with {{"video_paths": ["video1.mp4", "video2.mp4", "video3.mp4"]}}
- Step 2: query_cached_video asking for "5-second viral compilation using 1-2 second clips from at least 2-3 different videos"
- Step 3: cut_video for each clip from different videos
- Step 4: concatenate_videos to combine into final 5-second viral video

PARAMETER NAMES:
- send_cache: "video_paths" (string or array), "ttl_minutes" (optional)
- query_cached_video: "video_paths" (string or array), "query" (NOT "question")
- cut_video: "video_path" (string), "start_time", "end_time" (numbers in seconds)
- concatenate_videos: "video_paths" (array)

Use the ReAct format:

Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (JSON format with double quotes)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT FORMATTING RULES:
- After EVERY "Thought:" you MUST write either "Action:" (with Action Input) OR "Final Answer:"
- NEVER write just "Thought:" alone without following up
- For simple tasks (like deleting files), provide Final Answer immediately after completion
- Example: After deleting files, write "Thought: The files have been deleted successfully\nFinal Answer: I have successfully deleted the requested files."

FLEXIBLE INPUT FORMAT:
Action Input can be:
- Valid JSON: {{"video_path": "file.mp4"}}
- Plain text: cache IMG_5437.MOV
- Simple name: IMG_5437.MOV

The system will auto-correct invalid JSON using response_schema.

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def chat(self, message: str) -> str:
        """
        Send a message to the agent.
        
        Args:
            message: User message/prompt
            
        Returns:
            Agent's response
        """
        try:
            response = await self.agent_executor.ainvoke({"input": message})
            return response.get("output", "No response generated")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        
    def get_history(self) -> str:
        """Get the conversation history."""
        return self.memory.load_memory_variables({}).get("chat_history", "")


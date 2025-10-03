"""
LangChain Agent with Real MCP Integration.
Connects to the MCP server as a client.
"""

import os
import json
import asyncio
from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager


class VideoEditingAgent:
    """LangChain Agent that connects to a real MCP server."""
    
    def __init__(self, gemini_api_key: str = None, workspace_dir: str = "./workspace"):
        """
        Initialize the video editing agent with MCP client.
        
        Args:
            gemini_api_key: Google Gemini API key
            workspace_dir: Directory for storing temporary and output files
        """
        # Set up API key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        self.gemini_api_key = gemini_api_key
        self.workspace_dir = workspace_dir
        
        # MCP session will be initialized in async context
        self.session: ClientSession = None
        self.exit_stack = None
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=1,
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
                "GEMINI_API_KEY": self.gemini_api_key or ""
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
    
    def _create_langchain_tools(self, mcp_tools) -> List[Tool]:
        """Convert MCP tools to LangChain Tools."""
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            tool_name = mcp_tool.name
            tool_description = mcp_tool.description
            
            # Create tool function that calls MCP
            def create_tool_func(name: str):
                async def tool_func(input_str: str) -> str:
                    try:
                        # Parse JSON input
                        if input_str.strip().startswith('{'):
                            params = json.loads(input_str)
                        else:
                            # Simple string for single-parameter tools
                            params = {"video_path": input_str}
                        
                        # Call MCP server
                        result = await self.session.call_tool(name, params)
                        
                        # Extract text from MCP response
                        if result.content:
                            texts = [c.text for c in result.content if hasattr(c, 'text')]
                            return "\n".join(texts)
                        
                        return str(result)
                    except Exception as e:
                        return f"Error: {str(e)}"
                
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
        """Create the LangChain agent."""
        
        template = """You are a friendly video editing assistant. You can have conversations AND use tools to edit videos.

IMPORTANT: Only use tools when the user asks you to DO something with a video (edit, cut, analyze, etc).
For greetings, questions, or general chat - just respond directly without using any tools.

You have access to these tools:
{tools}

WHEN TO USE TOOLS:
- User asks to edit/cut/trim/modify a video
- User asks for video information or analysis
- User wants to create/merge/process videos

WHEN NOT TO USE TOOLS:
- Greetings (hi, hello, etc.) - just respond friendly
- General questions about capabilities - just explain what you can do
- Casual conversation - just chat normally

Use this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For simple conversations (like greetings), skip straight to the Final Answer without using any tools.

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
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


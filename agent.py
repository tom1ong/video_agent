"""
LangChain Agent with MCP-style Tool Integration.
Uses async MCP-inspired tool abstraction for reliability.
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
from mcp_server import VideoEditingMCPServer


class VideoEditingAgent:
    """LangChain Agent with MCP-style video editing tools."""
    
    def __init__(self, gemini_api_key: str = None, workspace_dir: str = "./workspace"):
        """
        Initialize the video editing agent.
        
        Args:
            gemini_api_key: Google Gemini API key
            workspace_dir: Directory for storing temporary and output files
        """
        # Set up API key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize MCP-style server with API key for caching
        self.mcp_server = VideoEditingMCPServer(workspace_dir, gemini_api_key=gemini_api_key)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=1,
            convert_system_message_to_human=True
        )
        
        # Create tools from MCP server
        self.tools = self._create_langchain_tools()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
            output_key="output"
        )
        
        # Create the agent
        self.agent_executor = self._create_agent()
    
    def _create_langchain_tools(self) -> List[Tool]:
        """Convert MCP tools to LangChain Tools."""
        langchain_tools = []
        
        for mcp_tool in self.mcp_server.list_tools():
            tool_name = mcp_tool["name"]
            tool_description = mcp_tool["description"]
            
            # Create tool function
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
                        result = await self.mcp_server.call_tool(name, params)
                        
                        # Extract text from MCP response
                        if "content" in result:
                            texts = [c["text"] for c in result["content"] if c.get("type") == "text"]
                            return "\n".join(texts)
                        
                        return str(result)
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                return tool_func
            
            # Create LangChain tool with async support
            tool = Tool(
                name=tool_name,
                func=lambda x: asyncio.run(create_tool_func(tool_name)(x)),  # Sync wrapper for compatibility
                coroutine=create_tool_func(tool_name),  # Async version
                description=tool_description
            )
            langchain_tools.append(tool)
        
        return langchain_tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""
        
        template = """You are an intelligent video editing assistant. You help users edit their videos by understanding their natural language requests and using the available video editing tools.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Users can reference videos by just their filename (e.g. "video.mp4") - the system automatically looks in the workspace directory
- If a user provides just a filename, use it as-is - DO NOT add "workspace/" prefix
- For complex edits, break them down into multiple tool calls
- Keep track of intermediate output files as inputs for subsequent operations
- Output videos are saved to the workspace directory
- Always provide the final output file path to the user
- IMPORTANT: Remember the conversation history! If you asked the user for information (like a filename) and they provide it, use that information to complete the original request

Previous conversation:
{chat_history}

Be conversational and helpful.

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

#!/usr/bin/env python3
"""
LangChain Interactive Multi-Agent System (LangChain v3)

This example demonstrates an interactive multi-agent system using three key concepts:
1. chat_history: Conversation context tracking
2. messages: Structured agent communication
3. tool_calling: Agent capabilities and interactions

It provides an interactive system where users can work with specialized agents.
"""

import os
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
    FunctionMessage
)
from langchain_core.tools import Tool, BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.render import render_text_description

# Load environment variables
load_dotenv(".env")

class AgentRole(str, Enum):
    """Agent specialization roles."""
    RESEARCH = "research"
    WRITING = "writing"
    EDITING = "editing"
    ORCHESTRATOR = "orchestrator"

class TaskStatus(str, Enum):
    """Task progress states."""
    NEW = "new"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETE = "complete"
    ERROR = "error"

class WorkflowTask(BaseModel):
    """Task assignment for agents."""
    title: str = Field(description="Task title")
    description: str = Field(description="Task details")
    assigned_to: AgentRole = Field(description="Agent responsible for task")
    status: TaskStatus = Field(description="Current task status")
    result: Optional[str] = Field(description="Task output", default=None)

# Research Tools
def search_topic(query: str) -> str:
    """Search for information about a topic."""
    return f"Research results for: {query}"

def find_references(topic: str) -> List[str]:
    """Find relevant references for a topic."""
    return [f"Reference {i} about {topic}" for i in range(1, 4)]

def verify_fact(statement: str) -> bool:
    """Verify if a statement is factually correct."""
    return True  # Simulated verification

# Writing Tools
def generate_outline(topic: str) -> List[str]:
    """Create a structured outline for the topic."""
    return [f"Section {i}: {topic} aspect {i}" for i in range(1, 4)]

def expand_section(outline: str) -> str:
    """Expand an outline section into detailed content."""
    return f"Detailed content for: {outline}"

def suggest_improvements(text: str) -> List[str]:
    """Suggest improvements for the text."""
    return [f"Improvement suggestion {i}" for i in range(1, 4)]

# Editing Tools
def grammar_check(text: str) -> List[str]:
    """Check text for grammar issues."""
    return [f"Grammar suggestion {i}" for i in range(1, 3)]

def style_check(text: str) -> List[str]:
    """Check text for style consistency."""
    return [f"Style suggestion {i}" for i in range(1, 3)]

def format_content(text: str) -> str:
    """Format content according to style guidelines."""
    return f"Formatted: {text}"

class SpecializedAgent:
    """Base class for specialized agents."""
    
    def __init__(self, role: AgentRole, tools: List[BaseTool], llm):
        """Initialize agent with role and tools."""
        self.role = role
        self.messages = []  # Chat history
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a specialized {role.value} agent.
            Use your tools to help complete tasks.
            Maintain professional communication.
            Ask for clarification when needed."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent with tools
        self.agent = create_openai_functions_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools)
    
    def add_message(self, message: Any) -> None:
        """Add message to chat history."""
        self.messages.append(message)
    
    async def process_task(self, task: WorkflowTask) -> WorkflowTask:
        """Handle assigned task."""
        try:
            # Add task to history
            self.add_message(
                SystemMessage(content=f"New task: {task.title}")
            )
            
            # Process task
            result = await self.executor.ainvoke(
                {
                    "input": task.description,
                    "chat_history": self.messages
                }
            )
            
            # Update task
            task.status = TaskStatus.COMPLETE
            task.result = result["output"]
            
            # Record result
            self.add_message(
                AIMessage(content=f"Task completed: {task.result}")
            )
            
            return task
            
        except Exception as e:
            # Handle error
            task.status = TaskStatus.ERROR
            task.result = f"Error: {str(e)}"
            
            # Record error
            self.add_message(
                SystemMessage(content=f"Error: {task.result}")
            )
            
            return task

class OrchestratorAgent:
    """Coordinates specialized agents and user interaction."""
    
    def __init__(self):
        """Initialize orchestrator and specialized agents."""
        # Initialize Azure OpenAI
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Create specialized agents
        self.agents = {
            AgentRole.RESEARCH: SpecializedAgent(
                AgentRole.RESEARCH,
                [
                    Tool(name="search_topic", func=search_topic, description="Search for topic information"),
                    Tool(name="find_references", func=find_references, description="Find topic references"),
                    Tool(name="verify_fact", func=verify_fact, description="Verify statement accuracy")
                ],
                llm
            ),
            AgentRole.WRITING: SpecializedAgent(
                AgentRole.WRITING,
                [
                    Tool(name="generate_outline", func=generate_outline, description="Create content outline"),
                    Tool(name="expand_section", func=expand_section, description="Expand outline sections"),
                    Tool(name="suggest_improvements", func=suggest_improvements, description="Suggest content improvements")
                ],
                llm
            ),
            AgentRole.EDITING: SpecializedAgent(
                AgentRole.EDITING,
                [
                    Tool(name="grammar_check", func=grammar_check, description="Check grammar"),
                    Tool(name="style_check", func=style_check, description="Check style"),
                    Tool(name="format_content", func=format_content, description="Format content")
                ],
                llm
            )
        }
        
        self.messages = []  # Orchestrator history
    
    def add_message(self, message: Any) -> None:
        """Add message to chat history."""
        self.messages.append(message)
    
    def get_agent_capabilities(self, role: AgentRole) -> str:
        """Get available tools for an agent."""
        agent = self.agents.get(role)
        if not agent:
            return f"No agent available for role: {role}"
        
        return render_text_description(agent.executor.tools)
    
    async def process_task(self, task: WorkflowTask) -> WorkflowTask:
        """Route task to appropriate agent."""
        # Get specialized agent
        agent = self.agents.get(task.assigned_to)
        if not agent:
            task.status = TaskStatus.ERROR
            task.result = f"No agent available for role: {task.assigned_to}"
            return task
        
        # Update history
        self.add_message(
            SystemMessage(content=f"Routing task to {task.assigned_to} agent")
        )
        
        try:
            # Process task with agent
            task.status = TaskStatus.IN_PROGRESS
            result = await agent.process_task(task)
            
            # Record completion
            self.add_message(
                SystemMessage(content=f"Task completed by {task.assigned_to} agent")
            )
            
            return result
            
        except Exception as e:
            # Handle error
            task.status = TaskStatus.ERROR
            task.result = f"Error processing task: {str(e)}"
            
            # Record error
            self.add_message(
                SystemMessage(content=f"Error in {task.assigned_to} agent: {str(e)}")
            )
            
            return task

async def demonstrate_system():
    """Demonstrate the multi-agent system."""
    print("\nInteractive Multi-Agent System Demo")
    print("=================================\n")
    
    # Create orchestrator
    orchestrator = OrchestratorAgent()
    
    # Show agent capabilities
    for role in AgentRole:
        if role != AgentRole.ORCHESTRATOR:
            print(f"\n{role.value.title()} Agent Capabilities:")
            print("-" * 40)
            print(orchestrator.get_agent_capabilities(role))
            print("-" * 40)
    
    # Test tasks
    tasks = [
        WorkflowTask(
            title="Research AI Impact",
            description="Research the impact of AI on modern workplaces",
            assigned_to=AgentRole.RESEARCH,
            status=TaskStatus.NEW
        ),
        WorkflowTask(
            title="Write Summary",
            description="Write a summary of AI workplace impact findings",
            assigned_to=AgentRole.WRITING,
            status=TaskStatus.NEW
        ),
        WorkflowTask(
            title="Edit Report",
            description="Edit and format the AI impact report",
            assigned_to=AgentRole.EDITING,
            status=TaskStatus.NEW
        )
    ]
    
    # Process tasks
    for task in tasks:
        print(f"\nProcessing Task: {task.title}")
        print("-" * 40)
        
        result = await orchestrator.process_task(task)
        print(f"Status: {result.status}")
        print(f"Result: {result.result}")
        print("-" * 40)
    
    # Show final history
    print("\nWorkflow History:")
    print("-" * 40)
    for msg in orchestrator.messages:
        print(f"{msg.type}: {msg.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_system())
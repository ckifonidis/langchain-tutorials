"""
LangChain Smart Research Assistant Example

This example demonstrates how to combine multiple agent types with different memory systems
to create a sophisticated research assistant that can gather information, analyze data,
make recommendations, and coordinate complex tasks.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Define agent template with default empty chat history
AGENT_TEMPLATE = """You are a highly capable AI agent.

You have access to the following tools:
{tools}

Use the following format:
Thought: Think about what to do
Action: Choose a tool from [{tool_names}]
Action Input: Input for the tool
Observation: Tool output
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: The final response to the input question

Previous Chat History:
{chat_history}

Task: {input}

{agent_scratchpad}"""

class ResearchTask(BaseModel):
    """Schema for research tasks."""
    task_id: str = Field(description="Unique task identifier")
    topic: str = Field(description="Research topic")
    requirements: List[str] = Field(description="Research requirements")
    deadline: datetime = Field(description="Task deadline")

class ResearchResults(BaseModel):
    """Schema for research results."""
    task_id: str = Field(description="Task identifier")
    findings: List[Dict[str, Any]] = Field(description="Research findings")
    analysis: Dict[str, Any] = Field(description="Data analysis")
    recommendations: List[str] = Field(description="Recommendations")
    sources: List[str] = Field(description="Information sources")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def setup_agent_memory() -> ConversationBufferMemory:
    """Initialize memory for an agent."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def create_research_agent(
    llm: AzureChatOpenAI,
    memory: ConversationBufferMemory
) -> AgentExecutor:
    """Create an agent for gathering information."""
    tools = [
        DuckDuckGoSearchRun(name="web_search"),
        Tool(
            name="check_sources",
            func=lambda x: "Source verification completed: authentic",
            description="Verify the authenticity of sources"
        )
    ]
    
    prompt = PromptTemplate(
        template=AGENT_TEMPLATE,
        input_variables=["chat_history", "input", "agent_scratchpad", "tools", "tool_names"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_analysis_agent(
    llm: AzureChatOpenAI,
    memory: ConversationBufferMemory
) -> AgentExecutor:
    """Create an agent for analyzing data."""
    tools = [
        Tool(
            name="analyze_patterns",
            func=lambda x: "Pattern analysis completed: trends identified",
            description="Analyze patterns in research data"
        ),
        Tool(
            name="statistical_analysis",
            func=lambda x: "Analysis completed: significant patterns found",
            description="Perform statistical analysis"
        ),
        Tool(
            name="validate_findings",
            func=lambda x: "Validation completed: findings confirmed",
            description="Validate analysis results"
        )
    ]
    
    prompt = PromptTemplate(
        template=AGENT_TEMPLATE,
        input_variables=["chat_history", "input", "agent_scratchpad", "tools", "tool_names"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_decision_agent(
    llm: AzureChatOpenAI,
    memory: ConversationBufferMemory
) -> AgentExecutor:
    """Create an agent for making recommendations."""
    tools = [
        Tool(
            name="evaluate_options",
            func=lambda x: "Options evaluated: recommendation ready",
            description="Evaluate different options"
        ),
        Tool(
            name="risk_assessment",
            func=lambda x: "Risk assessment completed: minimal risks identified",
            description="Assess potential risks"
        )
    ]
    
    prompt = PromptTemplate(
        template=AGENT_TEMPLATE,
        input_variables=["chat_history", "input", "agent_scratchpad", "tools", "tool_names"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_coordination_agent(
    llm: AzureChatOpenAI,
    agents: Dict[str, AgentExecutor]
) -> AgentExecutor:
    """Create an agent for coordinating other agents."""
    memory = setup_agent_memory()
    
    tools = [
        Tool(
            name="assign_task",
            func=lambda x: f"Task assigned to {x}",
            description="Assign tasks to specific agents"
        ),
        Tool(
            name="check_progress",
            func=lambda x: "Progress checked: on schedule",
            description="Check task progress"
        )
    ]
    
    prompt = PromptTemplate(
        template=AGENT_TEMPLATE,
        input_variables=["chat_history", "input", "agent_scratchpad", "tools", "tool_names"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def demonstrate_research_assistant():
    """Demonstrate the Smart Research Assistant capabilities."""
    try:
        print("\nInitializing Smart Research Assistant...\n")
        
        # Initialize components
        llm = create_chat_model()
        
        # Create agents with individual memories
        research_agent = create_research_agent(llm, setup_agent_memory())
        analysis_agent = create_analysis_agent(llm, setup_agent_memory())
        decision_agent = create_decision_agent(llm, setup_agent_memory())
        
        agents = {
            "research": research_agent,
            "analysis": analysis_agent,
            "decision": decision_agent
        }
        
        coordination_agent = create_coordination_agent(llm, agents)
        
        # Example research task
        task = ResearchTask(
            task_id="TASK001",
            topic="Renewable Energy Trends",
            requirements=[
                "Current market analysis",
                "Technology comparison",
                "Future projections"
            ],
            deadline=datetime.now()
        )
        
        print("Research Task:")
        print(f"ID: {task.task_id}")
        print(f"Topic: {task.topic}")
        print("Requirements:")
        for req in task.requirements:
            print(f"- {req}")
        print("-" * 50)
        
        # Initialize empty chat history for each invocation
        empty_history = {"chat_history": ""}
        
        # Execute research workflow
        # 1. Research Agent gathers information
        research_results = research_agent.invoke({
            "input": f"Research {task.topic} focusing on: {', '.join(task.requirements)}",
            **empty_history
        })
        print("\nResearch Findings:")
        print(research_results)
        print("-" * 50)
        
        # 2. Analysis Agent processes data
        analysis_results = analysis_agent.invoke({
            "input": f"Analyze research findings for {task.topic}",
            **empty_history
        })
        print("\nAnalysis Results:")
        print(analysis_results)
        print("-" * 50)
        
        # 3. Decision Agent makes recommendations
        decision_results = decision_agent.invoke({
            "input": f"Make recommendations based on analysis of {task.topic}",
            **empty_history
        })
        print("\nRecommendations:")
        print(decision_results)
        print("-" * 50)
        
        # 4. Coordination Agent manages overall process
        coordination_results = coordination_agent.invoke({
            "input": f"Coordinate completion of research task {task.task_id}",
            **empty_history
        })
        print("\nCoordination Summary:")
        print(coordination_results)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Smart Research Assistant...")
    demonstrate_research_assistant()

if __name__ == "__main__":
    main()

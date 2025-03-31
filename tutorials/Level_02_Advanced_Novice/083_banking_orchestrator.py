#!/usr/bin/env python3
"""
LangChain Banking Orchestrator (LangChain v3)

This example demonstrates a multi-agent banking system using three key concepts:
1. agents: Specialized banking agents coordinated by an orchestrator
2. output_parsers: Structured communication between agents
3. tools: Specific banking operations for each agent

It provides coordinated financial services through specialized agents.
"""

import os
from typing import Dict, List, Any, Optional, Annotated, Type
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool, Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv(".env")

# Agent template
AGENT_TEMPLATE = """You are a specialized banking agent for {agent_type} operations.
Use the available tools to help customers with their banking needs.
If a task is complex or requires special attention, mention needing a supervisor.

Available tools:
{tools}

Tool names: {tool_names}

Remember:
1. Use tools to perform banking operations
2. Keep responses professional
3. Escalate complex cases
4. Maintain security and compliance

Task: {input}
{agent_scratchpad}
"""

class TaskType(str, Enum):
    """Banking task categories."""
    ACCOUNT = "account"
    INVESTMENT = "investment"
    LOAN = "loan"
    SUPPORT = "support"

class AgentTask(BaseModel):
    """Task assignment for agents."""
    task_type: TaskType = Field(description="Type of banking task")
    priority: int = Field(description="Task priority (1-5)")
    description: str = Field(description="Task details")
    requires_human: bool = Field(description="Whether human intervention is needed")

class AgentResponse(BaseModel):
    """Structured agent response."""
    status: str = Field(description="Task status")
    message: str = Field(description="Response message")
    next_action: Optional[str] = Field(description="Recommended next action", default=None)
    requires_escalation: bool = Field(description="Whether escalation is needed")

# Banking Operations
def check_balance(account_id: str) -> str:
    """Check account balance and recent transactions."""
    return "Balance: $5,000, Last transaction: Deposit $1,000 on 2025-03-26"

def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts."""
    return f"Transferred ${amount} from {from_account} to {to_account}"

def analyze_portfolio(portfolio_id: str) -> str:
    """Analyze investment portfolio performance."""
    return "Portfolio performance: +8.5% YTD, Risk level: Moderate"

def process_loan(application_id: str, loan_type: str) -> str:
    """Process loan applications."""
    return f"Loan {application_id} ({loan_type}) preliminary approval, pending verification"

# Tool Creation
BANKING_TOOLS = {
    TaskType.ACCOUNT: [
        Tool(
            name="check_balance",
            description="Check account balance and recent transactions",
            func=check_balance
        ),
        Tool(
            name="transfer_funds",
            description="Transfer funds between accounts",
            func=transfer_funds
        )
    ],
    TaskType.INVESTMENT: [
        Tool(
            name="analyze_portfolio",
            description="Analyze investment portfolio performance",
            func=analyze_portfolio
        )
    ],
    TaskType.LOAN: [
        Tool(
            name="process_loan",
            description="Process loan applications",
            func=process_loan
        )
    ]
}

def format_error_response(error: Exception) -> AgentResponse:
    """Create a standardized error response."""
    return AgentResponse(
        status="error",
        message=f"Error processing task: {str(error)}",
        next_action="Contact system administrator",
        requires_escalation=True
    )

class SpecializedAgent:
    """Base class for specialized banking agents."""
    
    def __init__(self, agent_type: TaskType, tools: List[BaseTool]):
        """Initialize agent with tools and LLM."""
        self.agent_type = agent_type
        
        # Initialize Azure OpenAI
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-2"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-agent-swarm-1.openai.azure.com/"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "979b84bde7c04d8784208309bcdac5d0"),
            temperature=0.7
        )
        
        # Create prompt template with required variables
        prompt = ChatPromptTemplate.from_template(AGENT_TEMPLATE)
        
        # Create agent
        self.agent = create_structured_chat_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools)
    
    def handle_task(self, task: AgentTask) -> AgentResponse:
        """Process assigned task using agent executor."""
        try:
            # Execute task
            result = self.executor.invoke({
                "input": task.description,
                "agent_type": self.agent_type.value
            })
            
            # Parse result
            is_error = "error" in result["output"].lower()
            needs_supervisor = "supervisor" in result["output"].lower()
            
            return AgentResponse(
                status="error" if is_error else "success",
                message=result["output"],
                next_action="Contact supervisor" if needs_supervisor else None,
                requires_escalation=needs_supervisor or is_error
            )
            
        except Exception as e:
            return format_error_response(e)

class OrchestratorAgent:
    """Coordinates specialized banking agents."""
    
    def __init__(self):
        """Initialize orchestrator with specialized agents."""
        # Initialize specialized agents with tools
        self.agents = {
            task_type: SpecializedAgent(task_type, tools)
            for task_type, tools in BANKING_TOOLS.items()
        }
    
    def process_task(self, task: AgentTask) -> AgentResponse:
        """Route task to appropriate agent and process response."""
        # Get specialized agent
        agent = self.agents.get(task.task_type)
        if not agent:
            return AgentResponse(
                status="error",
                message=f"No agent available for task type: {task.task_type}",
                next_action="Contact system administrator",
                requires_escalation=True
            )
        
        try:
            # Process task with specialized agent
            response = agent.handle_task(task)
            
            # Handle escalation if needed
            if response.requires_escalation:
                print(f"\n[ALERT] Task escalated: {task.description}")
            
            return response
            
        except Exception as e:
            return format_error_response(e)

def demonstrate_orchestrator():
    """Demonstrate the banking orchestrator."""
    print("\nBanking Orchestrator Demo")
    print("=======================\n")
    
    # Create orchestrator
    orchestrator = OrchestratorAgent()
    
    # Test tasks
    tasks = [
        AgentTask(
            task_type=TaskType.ACCOUNT,
            priority=1,
            description="Check balance and recent transactions for account ACC123",
            requires_human=False
        ),
        AgentTask(
            task_type=TaskType.INVESTMENT,
            priority=2,
            description="Analyze portfolio PORT456 performance and provide recommendations",
            requires_human=False
        ),
        AgentTask(
            task_type=TaskType.LOAN,
            priority=3,
            description="Process mortgage application LOAN789 for complex property purchase",
            requires_human=True
        )
    ]
    
    # Process tasks
    for task in tasks:
        print(f"\nTask: {task.description}")
        print("-" * 40)
        
        response = orchestrator.process_task(task)
        print(f"Status: {response.status}")
        print(f"Message: {response.message}")
        if response.next_action:
            print(f"Next Action: {response.next_action}")
        if response.requires_escalation:
            print("* Requires escalation *")
        
        print("-" * 40)

if __name__ == "__main__":
    demonstrate_orchestrator()
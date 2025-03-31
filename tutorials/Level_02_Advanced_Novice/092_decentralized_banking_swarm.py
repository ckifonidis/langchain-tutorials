#!/usr/bin/env python3
"""
LangChain Decentralized Banking Swarm (LangChain v3)

This example demonstrates a decentralized multi-agent swarm using three key concepts:
1. messages: Peer-to-peer agent communication
2. memory: Agent state and interaction tracking
3. lcel: Composable agent behaviors

It provides decentralized banking operations through self-organizing agents.
"""

import os
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage,
    AIMessage,
    FunctionMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Load environment variables
load_dotenv(".env")

class AgentRole(str, Enum):
    """Agent specialization roles."""
    ACCOUNT = "account_specialist"
    LOAN = "loan_specialist"
    INVESTMENT = "investment_specialist"
    COMPLIANCE = "compliance_specialist"
    RISK = "risk_specialist"

class MessageType(str, Enum):
    """Inter-agent message types."""
    TASK_REQUEST = "task_request"
    TASK_OFFER = "task_offer"
    TASK_ACCEPT = "task_accept"
    TASK_COMPLETE = "task_complete"
    INFO_REQUEST = "info_request"
    INFO_RESPONSE = "info_response"

class AgentMessage(BaseModel):
    """Message format for agent communication."""
    msg_id: str = Field(description="Unique message identifier")
    msg_type: MessageType = Field(description="Message type")
    sender_id: str = Field(description="Sender agent ID")
    recipient_id: Optional[str] = Field(description="Recipient agent ID if direct")
    content: Dict = Field(description="Message content")
    timestamp: str = Field(description="Message timestamp")
    thread_id: Optional[str] = Field(description="Conversation thread ID")

class TaskStatus(str, Enum):
    """Task processing states."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class BankingTask(BaseModel):
    """Banking operation task."""
    task_id: str = Field(description="Task identifier")
    description: str = Field(description="Task description")
    required_roles: List[AgentRole] = Field(description="Required agent roles")
    assigned_agents: List[str] = Field(description="Assigned agent IDs")
    status: TaskStatus = Field(description="Task status")
    deadline: Optional[str] = Field(description="Task deadline")
    priority: int = Field(description="Task priority (1-5)")

class SwarmAgent:
    """Autonomous banking agent in the swarm."""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        peers: Optional[List[str]] = None
    ):
        """Initialize agent."""
        self.agent_id = agent_id
        self.role = role
        self.peers = peers or []
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.2
        )
        
        # Initialize chat history
        self.memory = ChatMessageHistory(
        )
        
        # Message queue
        self.message_queue: List[AgentMessage] = []
        
        # Active tasks
        self.active_tasks: Dict[str, BankingTask] = {}
        
        # Task history (for confidence calculation)
        self.completed_tasks: List[BankingTask] = []
        
        # Create role-specific chain
        self.chain = (
            RunnablePassthrough.assign(
                timestamp=lambda _: datetime.now().isoformat()
            )
            | self._create_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_prompt(self, inputs: Dict) -> List[Any]:
        """Create role-specific prompt."""
        base_prompt = f"""You are a {self.role.value} in a decentralized banking system.
        Your agent ID is {self.agent_id}.
        
        Analyze the following information and determine appropriate actions:
        - Message type: {inputs.get('message_type')}
        - Content: {inputs.get('content')}
        - Current tasks: {self.active_tasks}
        
        Respond with appropriate actions based on your role and expertise.
        """
        
        return [
            SystemMessage(content=base_prompt),
            HumanMessage(content=str(inputs))
        ]
    
    def _calculate_confidence(self, task: BankingTask) -> float:
        """Calculate confidence score for handling a task."""
        # Base confidence 
        if self.role not in task.required_roles:
            return 0.0
        
        confidence_score = 0.0
        
        # Role-specific base confidence (0.5-0.7)
        if self.role in task.required_roles:
            # Higher confidence for primary role tasks
            if task.required_roles[0] == self.role:
                confidence_score += 0.7
            else:
                confidence_score += 0.5
        
        # Adjust based on current workload (0-0.2)
        workload_factor = max(0, 0.2 * (1 - len(self.active_tasks) / 3))
        confidence_score += workload_factor
        
        # Adjust based on task priority (0-0.1)
        priority_factor = 0.1 * (task.priority / 5)
        confidence_score += priority_factor
        
        # Experience factor based on completed tasks (0-0.2)
        similar_tasks = sum(
            1 for t in self.completed_tasks 
            if any(role == self.role for role in t.required_roles)
        )
        experience_factor = min(0.2, 0.05 * similar_tasks)
        confidence_score += experience_factor
        
        # Consider memory/history (0-0.1)
        try:
            memory_messages = self.memory.messages
            relevant_history = [
                msg for msg in memory_messages
                if isinstance(msg, FunctionMessage)
                and msg.name.startswith(self.role.value)
            ]
            history_factor = min(0.1, len(relevant_history) * 0.02)
            confidence_score += history_factor
        except Exception:
            pass
        
        return round(min(confidence_score, 1.0), 2)
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and generate response."""
        try:
            # Add to chat history
            self.memory.add_message(
                FunctionMessage(
                    content=str(message.content),
                    name=f"{message.sender_id}_{message.msg_type}"
                )
            )
            
            # Process based on message type
            if message.msg_type == MessageType.TASK_REQUEST:
                return await self._handle_task_request(message)
            elif message.msg_type == MessageType.TASK_OFFER:
                return await self._handle_task_offer(message)
            elif message.msg_type == MessageType.INFO_REQUEST:
                return await self._handle_info_request(message)
            
            return None
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return None
    
    async def _handle_task_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming task request."""
        task = BankingTask(**message.content)
        
        # Calculate confidence
        confidence = self._calculate_confidence(task)
        
        # Respond if confident enough
        if confidence > 0.5 and len(self.active_tasks) < 3:
            return AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.TASK_OFFER,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content={
                    "task_id": task.task_id,
                    "confidence": confidence
                },
                timestamp=datetime.now().isoformat(),
                thread_id=message.thread_id
            )
        
        return None
    
    async def _handle_task_offer(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle task offer from another agent."""
        if message.content["task_id"] in self.active_tasks:
            task = self.active_tasks[message.content["task_id"]]
            
            # Accept if we need this agent's role
            if message.sender_id not in task.assigned_agents:
                task.assigned_agents.append(message.sender_id)
                self.active_tasks[task.task_id] = task
                
                return AgentMessage(
                    msg_id=str(uuid.uuid4()),
                    msg_type=MessageType.TASK_ACCEPT,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    content={"task_id": task.task_id},
                    timestamp=datetime.now().isoformat(),
                    thread_id=message.thread_id
                )
        
        return None
    
    async def _handle_info_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle information request based on expertise."""
        # Get expert response using chain
        response = await self.chain.ainvoke({
            "message_type": "info_request",
            "content": message.content
        })
        
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.INFO_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            content={"response": response},
            timestamp=datetime.now().isoformat(),
            thread_id=message.thread_id
        )
    
    async def send_message(self, message: AgentMessage) -> None:
        """Add message to queue for processing."""
        self.message_queue.append(message)
    
    async def run(self) -> None:
        """Main agent loop."""
        while True:
            # Process queued messages
            while self.message_queue:
                message = self.message_queue.pop(0)
                response = await self.process_message(message)
                if response:
                    print(f"Agent {self.agent_id} responding to {message.sender_id}")
                    print(f"Response: {response.content}")
            
            # Brief pause
            await asyncio.sleep(0.1)

class BankingSwarm:
    """Decentralized banking agent swarm."""
    
    def __init__(self, num_agents: int = 5):
        """Initialize the swarm."""
        # Create agents
        self.agents: Dict[str, SwarmAgent] = {}
        
        # Create one of each role
        roles = list(AgentRole)
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            role = roles[i % len(roles)]
            self.agents[agent_id] = SwarmAgent(
                agent_id=agent_id,
                role=role,
                peers=[f"agent_{j}" for j in range(num_agents) if j != i]
            )
    
    async def submit_task(self, description: str, required_roles: List[AgentRole], priority: int = 3) -> None:
        """Submit task to the swarm."""
        task = BankingTask(
            task_id=str(uuid.uuid4()),
            description=description,
            required_roles=required_roles,
            assigned_agents=[],
            status=TaskStatus.PENDING,
            deadline=(datetime.now().isoformat()),
            priority=priority
        )
        
        # Broadcast task request to all agents
        message = AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.TASK_REQUEST,
            sender_id="system",
            recipient_id=None,  # Broadcast
            content=task.model_dump(),
            timestamp=datetime.now().isoformat(),
            thread_id=str(uuid.uuid4())
        )
        
        # Send to all agents
        for agent in self.agents.values():
            await agent.send_message(message)
    
    async def run(self) -> None:
        """Run the swarm."""
        # Start all agents
        agent_tasks = [
            agent.run()
            for agent in self.agents.values()
        ]
        
        # Run agents concurrently
        await asyncio.gather(*agent_tasks)

async def demonstrate_swarm():
    """Demonstrate the decentralized banking swarm."""
    print("\nDecentralized Banking Swarm Demo")
    print("================================\n")
    
    # Create swarm
    swarm = BankingSwarm(num_agents=5)
    
    # Create demo tasks
    tasks = [
        {
            "description": "Review and approve loan application for $50,000",
            "roles": [AgentRole.LOAN, AgentRole.RISK, AgentRole.COMPLIANCE],
            "priority": 4
        },
        {
            "description": "Create investment portfolio diversification strategy",
            "roles": [AgentRole.INVESTMENT, AgentRole.RISK],
            "priority": 3
        },
        {
            "description": "Process large account transfer with compliance check",
            "roles": [AgentRole.ACCOUNT, AgentRole.COMPLIANCE],
            "priority": 5
        }
    ]
    
    try:
        # Start swarm
        swarm_task = asyncio.create_task(swarm.run())
        
        # Submit tasks
        for task in tasks:
            print(f"\nSubmitting task: {task['description']}")
            print(f"Required roles: {[role.value for role in task['roles']]}")
            print(f"Priority: {task['priority']}")
            print("-" * 40)
            
            await swarm.submit_task(
                description=task["description"],
                required_roles=task["roles"],
                priority=task["priority"]
            )
            
            # Wait to see interactions
            await asyncio.sleep(2)
        
        # Let swarm process for a bit
        await asyncio.sleep(5)
        
        # Cancel swarm task
        swarm_task.cancel()
        
        try:
            await swarm_task
        except asyncio.CancelledError:
            print("\nSwarm demonstration completed")
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demonstrate_swarm())
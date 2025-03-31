#!/usr/bin/env python3
"""
LangChain ML Workflow Swarm (093) (LangChain v3)

This example demonstrates a machine learning workflow using three key concepts:
1. Tools: ML operations and data processing
2. Structured Output: ML parameters and results
3. Tracing: Workflow monitoring and metrics

It provides a comprehensive solution for managing and optimizing ML workflows in banking/fintech applications.
"""

import os
import uuid
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Load environment variables from .env file
load_dotenv()

# Define agent roles
class AgentRole(str, Enum):
    DATA_CLEANER = "data_cleaner"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"

# Define message types
class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_OFFER = "task_offer"
    TASK_ACCEPT = "task_accept"
    TASK_COMPLETE = "task_complete"
    INFO_REQUEST = "info_request"
    INFO_RESPONSE = "info_response"

# Define agent message format
class AgentMessage(BaseModel):
    msg_id: str = Field(description="Unique message identifier")
    msg_type: MessageType = Field(description="Message type")
    sender_id: str = Field(description="Sender agent ID")
    recipient_id: Optional[str] = Field(description="Recipient agent ID if direct")
    content: Dict = Field(description="Message content")
    timestamp: str = Field(description="Message timestamp")
    thread_id: Optional[str] = Field(description="Conversation thread ID")

# Define task status
class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# Define ML task
class MLTask(BaseModel):
    task_id: str = Field(description="Task identifier")
    description: str = Field(description="Task description")
    required_roles: List[AgentRole] = Field(description="Required agent roles")
    assigned_agents: List[str] = Field(description="Assigned agent IDs")
    status: TaskStatus = Field(description="Task status")
    priority: int = Field(description="Task priority (1-5)")

# Define swarm agent
class SwarmAgent:
    def __init__(self, agent_id: str, role: AgentRole, peers: Optional[List[str]] = None):
        self.agent_id = agent_id
        self.role = role
        self.peers = peers or []
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.2
        )
        self.memory = ChatMessageHistory()
        self.message_queue: List[AgentMessage] = []
        self.active_tasks: Dict[str, MLTask] = {}
        self.completed_tasks: List[MLTask] = []
        self.chain = (
            RunnablePassthrough.assign(
                timestamp=lambda _: datetime.now().isoformat()
            )
            | self._create_prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_prompt(self, inputs: Dict) -> List[Any]:
        base_prompt = f"""You are a {self.role.value} in a machine learning workflow.
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

    def _calculate_confidence(self, task: MLTask) -> float:
        """Calculate confidence score for handling a task."""
        confidence_score = 0.0
        if self.role in task.required_roles:
            confidence_score += 0.7 if task.required_roles[0] == self.role else 0.5
        workload_factor = max(0, 0.2 * (1 - len(self.active_tasks) / 3))
        confidence_score += workload_factor
        confidence_score += 0.1 * (task.priority / 5)
        return confidence_score

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        try:
            self.memory.add_message(
                FunctionMessage(
                    content=str(message.content),
                    name=f"{message.sender_id}_{message.msg_type}"
                )
            )
            if message.msg_type == MessageType.TASK_REQUEST:
                print(f"Agent {self.agent_id} received task request: {message.content}")
                return await self._handle_task_request(message)
            elif message.msg_type == MessageType.TASK_OFFER:
                print(f"Agent {self.agent_id} received task offer: {message.content}")
                return await self._handle_task_offer(message)
            elif message.msg_type == MessageType.INFO_REQUEST:
                print(f"Agent {self.agent_id} received info request: {message.content}")
                return await self._handle_info_request(message)
            return None
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return None

    async def _handle_task_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        task = MLTask(**message.content)
        confidence = self._calculate_confidence(task)
        if confidence > 0.5 and len(self.active_tasks) < 3:
            print(f"Agent {self.agent_id} offering to take task: {task.task_id} with confidence {confidence}")
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
        if message.content["task_id"] in self.active_tasks:
            task = self.active_tasks[message.content["task_id"]]
            if message.sender_id not in task.assigned_agents:
                task.assigned_agents.append(message.sender_id)
                self.active_tasks[task.task_id] = task
                print(f"Agent {self.agent_id} accepted task: {task.task_id}")
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
        response = await self.chain.ainvoke({
            "message_type": "info_request",
            "content": message.content
        })
        print(f"Agent {self.agent_id} responding to info request with: {response}")
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
        self.message_queue.append(message)

    async def run(self) -> None:
        while True:
            while self.message_queue:
                message = self.message_queue.pop(0)
                response = await self.process_message(message)
                if response:
                    print(f"Agent {self.agent_id} responding to {message.sender_id}")
                    print(f"Response: {response.content}")
            await asyncio.sleep(0.1)

# Define ML swarm
class MLWorkflowSwarm:
    def __init__(self, num_agents: int = 4):
        self.agents: Dict[str, SwarmAgent] = {}
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
        task = MLTask(
            task_id=str(uuid.uuid4()),
            description=description,
            required_roles=required_roles,
            assigned_agents=[],
            status=TaskStatus.PENDING,
            deadline=(datetime.now().isoformat()),
            priority=priority
        )
        message = AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.TASK_REQUEST,
            sender_id="system",
            recipient_id=None,
            content=task.model_dump(),
            timestamp=datetime.now().isoformat(),
            thread_id=str(uuid.uuid4())
        )
        for agent in self.agents.values():
            await agent.send_message(message)

    async def run(self) -> None:
        agent_tasks = [agent.run() for agent in self.agents.values()]
        await asyncio.gather(*agent_tasks)

async def demonstrate_swarm():
    print("\nML Workflow Swarm Demo")
    print("======================\n")
    swarm = MLWorkflowSwarm(num_agents=4)
    tasks = [
        {
            "description": "Clean and preprocess loan data",
            "roles": [AgentRole.DATA_CLEANER],
            "priority": 5
        },
        {
            "description": "Perform feature engineering on cleaned data",
            "roles": [AgentRole.FEATURE_ENGINEER],
            "priority": 4
        },
        {
            "description": "Train model on engineered features",
            "roles": [AgentRole.MODEL_TRAINER],
            "priority": 3
        },
        {
            "description": "Evaluate trained model on test data",
            "roles": [AgentRole.EVALUATOR],
            "priority": 2
        }
    ]
    try:
        swarm_task = asyncio.create_task(swarm.run())
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
            await asyncio.sleep(2)
        await asyncio.sleep(5)
        swarm_task.cancel()
        try:
            await swarm_task
        except asyncio.CancelledError:
            print("\nSwarm demonstration completed")
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demonstrate_swarm())
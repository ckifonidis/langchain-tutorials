# Decentralized Banking Swarm with LangChain: Complete Guide

## Introduction

This implementation demonstrates a decentralized multi-agent banking system by combining three key LangChain v3 concepts:
1. Messages: Inter-agent communication and coordination
2. Memory: Agent state and interaction history
3. LCEL: Composable agent behaviors

The system implements a self-organizing swarm of specialized banking agents that collaborate without central coordination.

### Real-World Application Value
- Decentralized operations
- Dynamic task allocation
- Expert collaboration
- Confidence-based decisions
- Workload balancing
- Experience tracking

### System Architecture Overview
```
Task → BankingSwarm → Agent Network
  ↓         ↓             ↓
Request  Distribution  Peer-to-Peer
  ↓         ↓             ↓
Roles    Messages     Collaboration
```

## Core LangChain Concepts

### 1. Messages
```python
class AgentMessage(BaseModel):
    msg_id: str = Field(description="Unique message identifier")
    msg_type: MessageType = Field(description="Message type")
    sender_id: str = Field(description="Sender agent ID")
    recipient_id: Optional[str] = Field(description="Recipient agent ID")
    content: Dict = Field(description="Message content")
    thread_id: Optional[str] = Field(description="Conversation thread ID")

message = AgentMessage(
    msg_id=str(uuid.uuid4()),
    msg_type=MessageType.TASK_REQUEST,
    sender_id="system",
    content={"task_id": task.task_id, "confidence": 0.96}
)
```

Features:
- Structured communication
- Message threading
- Role-based routing
- Content organization

### 2. Memory
```python
from langchain_community.chat_message_histories import ChatMessageHistory

self.memory = ChatMessageHistory()

# Record interaction
self.memory.add_message(
    FunctionMessage(
        content=str(message.content),
        name=f"{message.sender_id}_{message.msg_type}"
    )
)

# Access history
memory_messages = self.memory.messages
relevant_history = [
    msg for msg in memory_messages
    if isinstance(msg, FunctionMessage)
    and msg.name.startswith(self.role.value)
]
```

Benefits:
- Interaction tracking
- Experience building
- Context awareness
- History-based decisions

### 3. LCEL
```python
self.chain = (
    RunnablePassthrough.assign(
        timestamp=lambda _: datetime.now().isoformat()
    )
    | self._create_prompt
    | self.llm
    | StrOutputParser()
)

response = await self.chain.ainvoke({
    "message_type": "info_request",
    "content": message.content
})
```

Advantages:
- Pipeline composition
- Data transformation
- Async processing
- Error handling

## Implementation Components

### 1. Agent Roles and Confidence
```python
def _calculate_confidence(self, task: BankingTask) -> float:
    """Calculate confidence score for handling a task."""
    confidence_score = 0.0
    
    # Role-specific base confidence (0.5-0.7)
    if self.role in task.required_roles:
        if task.required_roles[0] == self.role:
            confidence_score += 0.7  # Primary role
        else:
            confidence_score += 0.5  # Secondary role
    
    # Workload factor (0-0.2)
    workload_factor = max(0, 0.2 * (1 - len(self.active_tasks) / 3))
    confidence_score += workload_factor
    
    # Priority factor (0-0.1)
    confidence_score += 0.1 * (task.priority / 5)
```

Key elements:
- Role specialization
- Workload consideration
- Priority weighting
- Experience factor

### 2. Task Management
```python
class BankingTask(BaseModel):
    task_id: str = Field(description="Task identifier")
    description: str = Field(description="Task description")
    required_roles: List[AgentRole] = Field(description="Required agent roles")
    assigned_agents: List[str] = Field(description="Assigned agent IDs")
    status: TaskStatus = Field(description="Task status")
    priority: int = Field(description="Task priority (1-5)")
```

Features:
- Clear structure
- Role requirements
- Priority levels
- Assignment tracking

### 3. Communication Protocol
```python
class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_OFFER = "task_offer"
    TASK_ACCEPT = "task_accept"
    TASK_COMPLETE = "task_complete"
    INFO_REQUEST = "info_request"
    INFO_RESPONSE = "info_response"
```

Capabilities:
- Message typing
- Protocol definition
- Flow control
- Status updates

## Advanced Features

### 1. Dynamic Task Distribution
```python
async def _handle_task_request(self, message: AgentMessage) -> Optional[AgentMessage]:
    task = BankingTask(**message.content)
    confidence = self._calculate_confidence(task)
    
    if confidence > 0.5 and len(self.active_tasks) < 3:
        return AgentMessage(
            msg_type=MessageType.TASK_OFFER,
            content={
                "task_id": task.task_id,
                "confidence": confidence
            }
        )
```

Implementation:
- Confidence scoring
- Workload limits
- Role matching
- Priority handling

### 2. Experience-Based Decisions
```python
# Experience factor based on completed tasks (0-0.2)
similar_tasks = sum(
    1 for t in self.completed_tasks 
    if any(role == self.role for role in t.required_roles)
)
experience_factor = min(0.2, 0.05 * similar_tasks)
confidence_score += experience_factor

# History consideration (0-0.1)
history_factor = min(0.1, len(relevant_history) * 0.02)
confidence_score += history_factor
```

Features:
- Task history
- Role expertise
- Interaction learning
- Adaptive confidence

### 3. Message Processing
```python
async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
    # Record in history
    self.memory.add_message(
        FunctionMessage(
            content=str(message.content),
            name=f"{message.sender_id}_{message.msg_type}"
        )
    )
    
    # Process by type
    if message.msg_type == MessageType.TASK_REQUEST:
        return await self._handle_task_request(message)
```

Strategies:
- History recording
- Type-based routing
- Async handling
- Error management

## Expected Output

### 1. Task Assignment
```text
Submitting task: Review and approve loan application for $50,000
Required roles: ['loan_specialist', 'risk_specialist', 'compliance_specialist']
Priority: 4
----------------------------------------
Agent agent_1 responding to system
Response: {'task_id': 'a2065996-442d-4e13-9ec9-aa750a75bc00', 'confidence': 0.98}
```

### 2. Role-Based Response
```text
Agent agent_2 responding to system
Response: {'task_id': '7a87accf-7c51-4fa7-804a-b8702fc0bfe0', 'confidence': 0.96}
Agent agent_4 responding to system
Response: {'task_id': '7a87accf-7c51-4fa7-804a-b8702fc0bfe0', 'confidence': 0.76}
```

## Best Practices

### 1. Agent Design
- Clear specialization
- Dynamic confidence
- Experience tracking
- Workload management

### 2. Communication
- Structured messages
- History recording
- Type handling
- Error recovery

### 3. Task Management
- Priority-based
- Role-matched
- Load-balanced
- Experience-weighted

## References

### 1. LangChain Core Concepts
- [Messages Guide](https://python.langchain.com/docs/modules/model_io/messages)
- [Chat History](https://python.langchain.com/docs/integrations/memory/chat_message_history)
- [LCEL](https://python.langchain.com/docs/expression_language)

### 2. Implementation Guides
- [Multi-Agent Systems](https://python.langchain.com/docs/use_cases/agent_simulations)
- [Async Operations](https://python.langchain.com/docs/expression_language/cookbook/async_parallel)
- [Message Handling](https://python.langchain.com/docs/modules/model_io/messages)

### 3. Additional Resources
- [Agent Communication](https://python.langchain.com/docs/modules/agents/agent_types)
- [Error Handling](https://python.langchain.com/docs/guides/debugging)
- [Chat Management](https://python.langchain.com/docs/modules/model_io/chat)
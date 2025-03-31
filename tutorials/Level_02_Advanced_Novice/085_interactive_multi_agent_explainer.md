# Interactive Multi-Agent System with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated multi-agent system by combining three key LangChain v3 concepts:
1. Chat History: Message tracking with custom history
2. Messages: Typed agent communication
3. Tool Calling: Role-specific capabilities

The system provides an interactive environment where specialized agents collaborate on complex tasks.

### Real-World Application Value
- Task specialization
- Message tracking
- Tool organization
- Error resilience
- Clear workflow

### System Architecture Overview
```
User → Orchestrator → Specialized Agents → Tool Actions → Results
         ↓                  ↓          ↓           ↑
      History        Message Types   Tools      Feedback
```

## Core LangChain Concepts

### 1. Message History
```python
class SpecializedAgent:
    def __init__(self, role: AgentRole, tools: List[BaseTool], llm):
        self.messages = []  # Chat history
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a specialized {role} agent..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def add_message(self, message: Any) -> None:
        """Add message to chat history."""
        self.messages.append(message)
```

Features:
- Simple storage
- Message tracking
- History access
- Clean interface

### 2. Message Types
```python
# Record task initiation
self.add_message(
    SystemMessage(content=f"New task: {task.title}")
)

# Record completion
self.add_message(
    AIMessage(content=f"Task completed: {task.result}")
)

# Record errors
self.add_message(
    SystemMessage(content=f"Error: {task.result}")
)
```

Benefits:
- Typed messages
- Clear roles
- Event tracking
- Error logging

### 3. Tool Implementation
```python
# Research Tools
def search_topic(query: str) -> str:
    """Search for information about a topic."""
    return f"Research results for: {query}"

# Create agents with tools
self.agents = {
    AgentRole.RESEARCH: SpecializedAgent(
        AgentRole.RESEARCH,
        [
            Tool(name="search_topic", func=search_topic, ...),
            Tool(name="find_references", func=find_references, ...),
            Tool(name="verify_fact", func=verify_fact, ...)
        ],
        llm
    )
}
```

Advantages:
- Role-specific tools
- Clear interfaces
- Tool documentation
- Function typing

## Implementation Components

### 1. Specialized Agents
```python
class SpecializedAgent:
    def __init__(self, role: AgentRole, tools: List[BaseTool], llm):
        self.role = role
        self.messages = []
        
        # Create agent with tools
        self.agent = create_openai_functions_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools)

    async def process_task(self, task: WorkflowTask) -> WorkflowTask:
        try:
            result = await self.executor.ainvoke({
                "input": task.description,
                "chat_history": self.messages
            })
            return task
        except Exception as e:
            task.status = TaskStatus.ERROR
            return task
```

Key elements:
- Role definition
- Tool assignment
- History tracking
- Error handling

### 2. Task Management
```python
class WorkflowTask(BaseModel):
    """Task assignment for agents."""
    title: str = Field(description="Task title")
    description: str = Field(description="Task details")
    assigned_to: AgentRole = Field(description="Agent responsible")
    status: TaskStatus = Field(description="Current status")
    result: Optional[str] = Field(description="Task output")

class TaskStatus(str, Enum):
    """Task progress states."""
    NEW = "new"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETE = "complete"
    ERROR = "error"
```

Features:
- Status tracking
- Clear assignment
- Result capture
- Error states

### 3. Orchestration
```python
class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            AgentRole.RESEARCH: SpecializedAgent(...),
            AgentRole.WRITING: SpecializedAgent(...),
            AgentRole.EDITING: SpecializedAgent(...)
        }
        self.messages = []

    async def process_task(self, task: WorkflowTask) -> WorkflowTask:
        agent = self.agents.get(task.assigned_to)
        if not agent:
            task.status = TaskStatus.ERROR
            return task
        
        try:
            result = await agent.process_task(task)
            return result
        except Exception as e:
            task.status = TaskStatus.ERROR
            return task
```

Capabilities:
- Agent coordination
- Task routing
- History tracking
- Error handling

## Advanced Features

### 1. Tool Organization
```python
# Research Tools
def search_topic(query: str) -> str:
    """Search for information about a topic."""
    return f"Research results for: {query}"

# Writing Tools
def generate_outline(topic: str) -> List[str]:
    """Create a structured outline for the topic."""
    return [f"Section {i}: {topic} aspect {i}" for i in range(1, 4)]

# Editing Tools
def grammar_check(text: str) -> List[str]:
    """Check text for grammar issues."""
    return [f"Grammar suggestion {i}" for i in range(1, 3)]
```

Implementation:
- Role grouping
- Clear interfaces
- Documentation
- Return typing

### 2. Message Flow
```python
# Task initiation
self.add_message(
    SystemMessage(content=f"Routing task to {task.assigned_to} agent")
)

# Task completion
self.add_message(
    SystemMessage(content=f"Task completed by {task.assigned_to} agent")
)

# Error handling
self.add_message(
    SystemMessage(content=f"Error in {task.assigned_to} agent: {str(e)}")
)
```

Features:
- Event logging
- Status updates
- Error tracking
- Clear flow

### 3. Error Management
```python
try:
    # Process task
    result = await agent.process_task(task)
    task.status = TaskStatus.COMPLETE
    return task
except Exception as e:
    task.status = TaskStatus.ERROR
    task.result = f"Error: {str(e)}"
    return task
```

Strategies:
- Status updates
- Error capture
- Message logging
- Clean recovery

## Expected Output

### 1. Agent Capabilities
```text
Research Agent Capabilities:
----------------------------------------
search_topic: Search for topic information
find_references: Find topic references
verify_fact: Verify statement accuracy
----------------------------------------
```

### 2. Task Processing
```text
Processing Task: Research AI Impact
----------------------------------------
Status: complete
Result: Research findings about AI workplace impact...
----------------------------------------
```

### 3. History Log
```text
Workflow History:
----------------------------------------
system: Routing task to research agent
system: Task completed by research agent
system: Routing task to writing agent
system: Task completed by writing agent
----------------------------------------
```

## Best Practices

### 1. Message Management
- Type safety
- Clear roles
- Event tracking
- Error logging

### 2. Tool Design
- Role grouping
- Clear interfaces
- Documentation
- Error handling

### 3. Error Handling
- Status tracking
- Message logging
- Clean recovery
- Clear feedback

## References

### 1. LangChain Core Concepts
- [Messages Guide](https://python.langchain.com/docs/modules/model_io/messages/)
- [Tool Creation](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Agent Types](https://python.langchain.com/docs/modules/agents/agent_types/)

### 2. Implementation Guides
- [Message Types](https://python.langchain.com/docs/modules/model_io/messages/message_types)
- [Tool Interface](https://python.langchain.com/docs/modules/agents/tools/how_to/custom_tools)
- [Error Handling](https://python.langchain.com/docs/guides/debugging)

### 3. Additional Resources
- [Multi-Agent Systems](https://python.langchain.com/docs/use_cases/multi_agent_systems)
- [Tool Guidelines](https://python.langchain.com/docs/modules/agents/tools/how_to/)
- [Azure Integration](https://python.langchain.com/docs/integrations/llms/azure_openai)
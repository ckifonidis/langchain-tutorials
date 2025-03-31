# Banking Orchestrator with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated banking orchestrator by combining three key LangChain v3 concepts:
1. Agents: Specialized banking agents with structured chat support
2. Output Parsers: Type-safe communication with standardized responses
3. Tools: Function-based banking operations with rich descriptions

The system provides coordinated financial services through specialized agents.

### Real-World Application Value
- Task specialization
- Error resilience
- Type safety
- Azure integration
- Clean escalation

### System Architecture Overview
```
Task → Orchestrator → Specialized Agent → Banking Tools → Standard Response
                   ↓                   ↓              ↑
           Error Handling      Agent Template    Tool Results
```

## Core LangChain Concepts

### 1. Agent System
```python
# Agent template with required variables
AGENT_TEMPLATE = """You are a specialized banking agent for {agent_type} operations.
Available tools: {tools}
Tool names: {tool_names}
Task: {input}
{agent_scratchpad}
"""

class SpecializedAgent:
    def __init__(self, agent_type: TaskType, tools: List[BaseTool]):
        llm = AzureChatOpenAI(...)
        prompt = ChatPromptTemplate.from_template(AGENT_TEMPLATE)
        self.agent = create_structured_chat_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools)
```

Features:
- Structured templates
- Tool integration
- Azure support
- Clean execution

### 2. Response Models
```python
class AgentResponse(BaseModel):
    status: str = Field(description="Task status")
    message: str = Field(description="Response message")
    next_action: Optional[str] = Field(
        description="Recommended next action",
        default=None
    )
    requires_escalation: bool = Field(
        description="Whether escalation is needed"
    )

def format_error_response(error: Exception) -> AgentResponse:
    return AgentResponse(
        status="error",
        message=f"Error processing task: {str(error)}",
        next_action="Contact system administrator",
        requires_escalation=True
    )
```

Benefits:
- Type validation
- Standard formatting
- Default values
- Error handling

### 3. Banking Tools
```python
def check_balance(account_id: str) -> str:
    """Check account balance and recent transactions."""
    return "Balance: $5,000, Last transaction: Deposit $1,000..."

BANKING_TOOLS = {
    TaskType.ACCOUNT: [
        Tool(
            name="check_balance",
            description="Check account balance and recent transactions",
            func=check_balance
        )
    ]
}
```

Advantages:
- Function-based tools
- Rich descriptions
- Task grouping
- Clean interfaces

## Implementation Components

### 1. Agent Creation
```python
def __init__(self, agent_type: TaskType, tools: List[BaseTool]):
    # Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.7
    )
    
    # Create prompt with template
    prompt = ChatPromptTemplate.from_template(AGENT_TEMPLATE)
    
    # Create agent and executor
    self.agent = create_structured_chat_agent(llm, tools, prompt)
    self.executor = AgentExecutor(agent=self.agent, tools=tools)
```

Key elements:
- Azure setup
- Template usage
- Agent creation
- Tool assignment

### 2. Response Processing
```python
def handle_task(self, task: AgentTask) -> AgentResponse:
    try:
        result = self.executor.invoke({
            "input": task.description,
            "agent_type": self.agent_type.value
        })
        
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
```

Features:
- Error detection
- Supervisor checks
- Clean formatting
- Standard responses

### 3. Task Routing
```python
def process_task(self, task: AgentTask) -> AgentResponse:
    agent = self.agents.get(task.task_type)
    if not agent:
        return AgentResponse(
            status="error",
            message=f"No agent available for task type: {task.task_type}",
            next_action="Contact system administrator",
            requires_escalation=True
        )
    
    try:
        response = agent.handle_task(task)
        if response.requires_escalation:
            print(f"\n[ALERT] Task escalated: {task.description}")
        return response
    except Exception as e:
        return format_error_response(e)
```

Capabilities:
- Agent validation
- Task handling
- Error management
- Escalation alerts

## Advanced Features

### 1. Error Management
```python
def format_error_response(error: Exception) -> AgentResponse:
    """Create a standardized error response."""
    return AgentResponse(
        status="error",
        message=f"Error processing task: {str(error)}",
        next_action="Contact system administrator",
        requires_escalation=True
    )
```

Implementation:
- Standard format
- Clear messages
- Action guidance
- Proper escalation

### 2. Task Specialization
```python
BANKING_TOOLS = {
    TaskType.ACCOUNT: [
        Tool(name="check_balance", ...),
        Tool(name="transfer_funds", ...)
    ],
    TaskType.INVESTMENT: [
        Tool(name="analyze_portfolio", ...)
    ]
}

self.agents = {
    task_type: SpecializedAgent(task_type, tools)
    for task_type, tools in BANKING_TOOLS.items()
}
```

Features:
- Tool grouping
- Agent assignment
- Clear boundaries
- Easy extension

### 3. Response Validation
```python
is_error = "error" in result["output"].lower()
needs_supervisor = "supervisor" in result["output"].lower()

return AgentResponse(
    status="error" if is_error else "success",
    message=result["output"],
    next_action="Contact supervisor" if needs_supervisor else None,
    requires_escalation=needs_supervisor or is_error
)
```

Strategies:
- Content analysis
- Status detection
- Action determination
- Escalation rules

## Expected Output

### 1. Successful Query
```text
Task: Check balance for account ACC123
----------------------------------------
Status: success
Message: Balance: $5,000, Last transaction: Deposit $1,000...
----------------------------------------
```

### 2. Complex Case
```text
Task: Process mortgage LOAN789
----------------------------------------
Status: success
Message: Complex loan scenario requires supervisor...
Next Action: Contact supervisor
* Requires escalation *
----------------------------------------
```

### 3. Error Case
```text
Task: Unknown operation
----------------------------------------
Status: error
Message: Error processing task: No agent available...
Next Action: Contact system administrator
* Requires escalation *
----------------------------------------
```

## Best Practices

### 1. Error Handling
- Standard responses
- Clear messages
- Action guidance
- Clean recovery

### 2. Agent Design
- Tool specialization
- Template usage
- Error management
- Clear boundaries

### 3. Response Processing
- Type validation
- Content analysis
- Status tracking
- Proper escalation

## References

### 1. LangChain Core Concepts
- [Structured Chat Agents](https://python.langchain.com/docs/modules/agents/agent_types/structured_chat)
- [Tool Creation](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Azure Integration](https://python.langchain.com/docs/integrations/llms/azure_openai)

### 2. Implementation Guides
- [Agent Executors](https://python.langchain.com/docs/modules/agents/executor/quick_start)
- [Response Validation](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Error Management](https://python.langchain.com/docs/guides/debugging)

### 3. Additional Resources
- [Tool Guidelines](https://python.langchain.com/docs/modules/agents/tools/how_to/)
- [Agent Templates](https://python.langchain.com/docs/modules/agents/agent_types/)
- [Pydantic Models](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)
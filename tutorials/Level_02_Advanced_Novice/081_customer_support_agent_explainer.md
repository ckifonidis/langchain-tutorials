# Customer Support Agent with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated banking customer support agent by combining three key LangChain v3 concepts:
1. Simple memory management for maintaining conversation context
2. Chat models (with mock for testing) for natural language interaction
3. Callbacks for monitoring and logging

The system provides context-aware customer support while maintaining audit trails and performance metrics.

### Real-World Application Value
- Context-aware customer support
- Conversation history tracking
- Real-time performance monitoring
- Supervisor escalation detection
- Clean error handling

### System Architecture Overview
```
User Input → Simple Memory → Chat Model (with callbacks) → Response → Memory Update
                               ↓
                         Monitoring System
                        (Metrics & Logging)
```

## Core LangChain Concepts

### 1. Simple Memory System
```python
class SimpleMemory:
    def __init__(self):
        self.chat_history: List[BaseMessage] = []
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"chat_history": self.chat_history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self.chat_history.extend([
            HumanMessage(content=inputs["input"]),
            AIMessage(content=outputs["output"])
        ])
```

Features:
- Direct message storage
- Simple state management
- Clear interface
- Type safety

### 2. Chat Models (FakeListChatModel)
```python
llm = FakeListChatModel(
    responses=[
        "I'll help you check your balance...",
        "I understand your card was declined... Let me connect you with a supervisor.",
        # More predefined responses...
    ],
    callbacks=[handler]
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="..."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])
```

Benefits:
- Predictable testing
- Scenario coverage
- Callback integration
- Context awareness

### 3. Callback System (SupportCallbackHandler)
```python
class SupportCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        if not self.start_time:
            self.start_time = datetime.now()
            
    def on_llm_end(self, response, *args, **kwargs):
        if hasattr(response, 'generations'):
            message = response.generations[0][0].text
            # Process message and log metrics
```

Features:
- Event monitoring
- Real-time metrics
- Supervisor alerts
- Error tracking

## Implementation Components

### 1. Support Agent Structure
```python
def create_support_agent():
    # Components
    handler = SupportCallbackHandler()
    llm = FakeListChatModel(..., callbacks=[handler])
    memory = SimpleMemory()
    
    # Chain
    support_chain = (
        RunnablePassthrough.assign(chat_history=...)
        | prompt 
        | llm
        | (lambda x: {"response": x.content})
    )
```

Key elements:
- Simple memory management
- Direct callback integration
- Clean chain composition
- Type-safe interfaces

### 2. Memory Management
```python
def process_input(user_input: Dict[str, Any]) -> Dict[str, Any]:
    result = support_chain.invoke(user_input)
    memory.save_context(
        {"input": user_input["input"]},
        {"output": result["response"]}
    )
    return result
```

Features:
- Explicit updates
- Message tracking
- Simple state management
- Error resilience

### 3. Monitoring System
```python
def on_llm_end(self, response, *args, **kwargs):
    message = response.generations[0][0].text
    self.messages.append(message)
    
    # Comprehensive metrics
    duration = datetime.now() - self.start_time
    print(f"\nConversation metrics:")
    print(f"- Duration: {duration}")
    print(f"- Messages: {len(self.messages)}")
```

Capabilities:
- Message tracking
- Time monitoring
- Alert generation
- Error logging

## Expected Output

### 1. Standard Interaction
```text
Conversation started at: 2025-03-27 12:15:00
Customer: Hi, I need help with my account balance
Agent: I'll help you check your balance. First, I need to verify your identity...

Conversation metrics:
- Duration: 0:00:00.000487
- Messages: 1
```

### 2. Supervisor Required
```text
Customer: My card was declined abroad

[ALERT] Supervisor assistance may be needed!

Conversation metrics:
- Duration: 0:00:00.014473
- Messages: 1
- Supervisor flag raised!

Agent: I understand your card was declined... Let me connect you with a supervisor.
```

## Best Practices

### 1. Memory Management
- Keep it simple
- Track messages directly
- Clear interfaces
- Type safety

### 2. Testing
- Mock responses
- Scenario coverage
- Error simulation
- Metric verification

### 3. Monitoring
- Event tracking
- Clear metrics
- Alert handling
- Error logging

## References

### 1. LangChain Core Concepts
- [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
- [Message Types](https://python.langchain.com/docs/concepts/messages/)
- [Callbacks](https://python.langchain.com/docs/concepts/callbacks/)

### 2. Implementation Guides
- [Mock Models](https://python.langchain.com/docs/guides/testing)
- [Chain Composition](https://python.langchain.com/docs/expression_language/)
- [Error Handling](https://python.langchain.com/docs/guides/errors)

### 3. Additional Resources
- [Testing Best Practices](https://python.langchain.com/docs/guides/testing)
- [Message Handling](https://python.langchain.com/docs/modules/model_io/messages/)
- [Chain Monitoring](https://python.langchain.com/docs/modules/callbacks/)
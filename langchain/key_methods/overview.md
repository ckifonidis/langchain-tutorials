# Key Methods in LangChain

## Overview
LangChain provides several key methods for interacting with language models and tools. These methods form the core API for building AI applications.

## Primary Methods

### 1. invoke
```python
def invoke(self, input):
    """Primary method for model interaction."""
    return self.generate_response(input)
```
- Basic interaction method
- Synchronous operation
- Single input/output
- Full response

### 2. stream
```python
def stream(self, input):
    """Stream responses token by token."""
    for token in self.generate_tokens(input):
        yield token
```
- Token-by-token output
- Real-time updates
- Generator pattern
- Progress feedback

### 3. batch
```python
def batch(self, inputs):
    """Process multiple inputs efficiently."""
    return [self.generate_response(inp) for inp in inputs]
```
- Multiple inputs
- Parallel processing
- Resource optimization
- Bulk operations

### 4. bind_tools
```python
def bind_tools(self, tools):
    """Attach tools to the model."""
    return self.with_tools(tools)
```
- Tool integration
- Function calling
- Capability extension
- Context binding

## Implementation Details

### 1. Invoke Pattern
```python
# Basic invoke
result = llm.invoke("What is AI?")

# With messages
result = llm.invoke([
    SystemMessage(content="Be helpful"),
    HumanMessage(content="What is AI?")
])

# With structured input
result = llm.invoke({
    "question": "What is AI?",
    "context": "..."
})
```

### 2. Streaming Pattern
```python
# Sync streaming
for chunk in llm.stream("Query"):
    process_chunk(chunk)

# Async streaming
async for chunk in llm.astream("Query"):
    await process_chunk(chunk)

# Event streaming
async for event in llm.astream_events("Query"):
    handle_event(event)
```

### 3. Batch Pattern
```python
# Simple batch
results = llm.batch(["Q1", "Q2", "Q3"])

# Async batch
results = await llm.abatch(["Q1", "Q2", "Q3"])

# With configuration
results = llm.batch(
    ["Q1", "Q2", "Q3"],
    config={"max_concurrency": 2}
)
```

### 4. Tool Binding Pattern
```python
# Basic tool binding
llm_with_tools = llm.bind_tools([tool1, tool2])

# With configuration
llm_with_tools = llm.bind_tools(
    tools=[tool1, tool2],
    tool_choice="auto"
)

# With callbacks
llm_with_tools = llm.bind_tools(
    tools=[tool1, tool2],
    callbacks=[handler1, handler2]
)
```

## Common Patterns

### 1. Chain Composition
```python
chain = (
    prompt 
    | llm.bind_tools([tool1, tool2]) 
    | parser
)
```

### 2. Async Operations
```python
async def process():
    async for chunk in llm.astream(input):
        await handle_chunk(chunk)
```

### 3. Error Handling
```python
try:
    result = llm.invoke(input)
except Exception as e:
    handle_error(e)
```

## Best Practices

### 1. Method Selection
- Use invoke for simple queries
- Use stream for real-time feedback
- Use batch for multiple inputs
- Use bind_tools for extensibility

### 2. Error Handling
- Implement proper try/catch
- Handle timeout errors
- Manage rate limits
- Log issues appropriately

### 3. Resource Management
- Control concurrency
- Monitor token usage
- Manage memory
- Clean up resources

## Advanced Usage

### 1. Custom Configuration
```python
result = llm.invoke(
    input,
    config={
        "timeout": 30,
        "max_tokens": 500,
        "temperature": 0.7
    }
)
```

### 2. Callback Integration
```python
result = llm.invoke(
    input,
    callbacks=[
        StreamingCallback(),
        LoggingCallback()
    ]
)
```

### 3. Tool Orchestration
```python
result = llm.bind_tools(
    tools=[tool1, tool2],
    tool_choice="auto",
    config={"max_iterations": 3}
)
```

## Security Considerations

### 1. Input Validation
- Sanitize inputs
- Validate parameters
- Check tool inputs
- Handle edge cases

### 2. Output Protection
- Filter sensitive data
- Validate responses
- Handle errors securely
- Mask internal details

### 3. Resource Protection
- Implement rate limiting
- Set timeouts
- Monitor usage
- Handle overload

## Related Concepts

### 1. Message Types
- SystemMessage
- HumanMessage
- AIMessage
- FunctionMessage

### 2. Tool Types
- Standard tools
- Custom tools
- Async tools
- Chain tools

### 3. Output Types
- String output
- JSON output
- Structured output
- Stream output
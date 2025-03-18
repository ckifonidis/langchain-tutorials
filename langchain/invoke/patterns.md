# LangChain Invoke Patterns Documentation

## Overview
The `invoke` method is a fundamental pattern in LangChain used across various components. It provides a consistent way to interact with LLMs, chat models, chains, and other components.

## Core Patterns

### 1. Direct String Input
```python
result = llm.invoke("Your question here")
```
- Simplest form of invocation
- String automatically converted to HumanMessage
- Good for one-off queries

### 2. Message List Input
```python
messages = [
    SystemMessage("Context..."),
    HumanMessage("Question..."),
    AIMessage("Previous response...")
]
result = llm.invoke(messages)
```
- Maintains conversation context
- Supports multi-turn interactions
- Can include system instructions

### 3. Prompt Template Input
```python
prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | llm
result = chain.invoke({"var": "value"})
```
- Reusable prompt patterns
- Variable substitution
- Clean separation of template and values

### 4. Chain Composition
```python
chain = prompt | llm | output_parser
result = chain.invoke(input_dict)
```
- Pipeline style processing
- Multiple components working together
- Clean data flow

## Best Practices

1. Error Handling
- Always handle potential API errors
- Consider implementing retries
- Validate inputs before invocation

2. Context Management
- Use appropriate message types
- Maintain conversation history when needed
- Clear context when starting new conversations

3. Performance
- Use batch methods for multiple inputs
- Consider caching for repeated queries
- Monitor token usage

## Common Use Cases

1. Chat Applications
- Message history management
- System role definition
- Interactive conversations

2. Structured Output
- Output parsing
- Format validation
- Error recovery

3. Complex Chains
- Multi-step processing
- Data transformation
- Parallel operations
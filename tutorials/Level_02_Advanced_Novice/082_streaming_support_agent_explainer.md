# Streaming Support Agent with LangChain: Complete Guide

## Introduction

This example demonstrates an intelligent banking support system by combining three key LangChain v3 concepts:
1. Retrieval with BM25 for finding relevant policies
2. Few-shot prompting with example selection for response consistency
3. Streaming with mock responses for testing

The implementation provides responsive, knowledge-based customer support with immediate feedback.

### Real-World Application Value
- Policy-based responses
- Consistent support format
- Real-time interaction
- Testing capabilities
- Example-based learning

### System Architecture Overview
```
Query → Retrieval Chain → Few-Shot Prompt → Mock Streaming
       (Context)        (Example-Based)    (Test Responses)
```

## Core LangChain Concepts

### 1. Retrieval Chain
```python
# Set up retriever
retriever = BM25Retriever.from_texts(
    texts=BANKING_POLICIES,
    preprocess_func=lambda x: x.lower()
)

# Create retrieval chain
retrieval_chain = RunnablePassthrough.assign(
    context = lambda x: "\n".join(doc.page_content for doc in retriever.invoke(x["query"]))
)
```

Features:
- Modern invoke pattern
- Context building
- Clean composition
- Simple preprocessing

### 2. Mock Chat Model
```python
llm = FakeListChatModel(
    responses=[
        "Based on our account access policy, I'll help you check your balance...",
        "I understand your card was declined while traveling...",
        "Regarding mortgage rates, let me assist you..."
    ],
    streaming=True
)
```

Benefits:
- Predictable testing
- Streaming support
- Realistic responses
- No API dependency

### 3. Few-Shot Learning
```python
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a helpful banking support agent...",
    suffix="Query: {query}\nRelevant Policies: {context}\nResponse:",
    input_variables=["query", "context"]
)
```

Advantages:
- Example selection
- Format consistency
- Context integration
- Clean structure

## Implementation Components

### 1. Mock Responses
```python
responses=[
    "Based on our account access policy...",  # Balance query
    "I understand your card was declined...",  # Card issue
    "Regarding mortgage rates..."             # Loan inquiry
]
```

Purpose:
- Testing scenarios
- Predictable output
- Policy references
- Natural flow

### 2. Context Building
```python
retrieval_chain = RunnablePassthrough.assign(
    context = lambda x: "\n".join(
        doc.page_content for doc in retriever.invoke(x["query"])
    )
)
```

Features:
- Policy retrieval
- Context assembly
- Clean composition
- Error handling

### 3. Chain Composition
```python
chain = (
    retrieval_chain      # Get policies
    | few_shot_prompt   # Format response
    | llm              # Generate output
)
```

Steps:
1. Policy lookup
2. Response formatting
3. Stream generation
4. Error management

## Advanced Features

### 1. Testing Support
```python
# Initialize mock LLM
llm = FakeListChatModel(
    responses=[...],
    streaming=True
)

# Stream response
for chunk in chain.stream(inputs):
    yield chunk
```

Capabilities:
- Mock streaming
- Predictable output
- Error simulation
- Clean testing

### 2. Error Management
```python
try:
    for chunk in agent({"query": query}):
        content = chunk.content if hasattr(chunk, 'content') else chunk
        print(content, end="", flush=True)
except Exception as e:
    print(f"\nError processing query: {str(e)}")
```

Strategies:
- Exception handling
- Content validation
- User feedback
- Clean recovery

### 3. Response Building
- Policy integration
- Example selection
- Stream management
- Format consistency

## Expected Output

### 1. Account Query
```text
Customer: How do I check my account balance?
Agent: Based on our account access policy, I'll help you check your balance. 
First, please verify your identity for security purposes...
```

### 2. Card Issue
```text
Customer: My card was declined while traveling
Agent: I understand your card was declined while traveling. 
This is often due to our security measures...
```

## Best Practices

### 1. Mock Testing
- Realistic responses
- Streaming support
- Error scenarios
- Clear output

### 2. Chain Composition
- Clean structure
- Context handling
- Variable management
- Error resilience

### 3. Stream Handling
- Content validation
- Progressive display
- Error catching
- User feedback

## References

### 1. LangChain Core Concepts
- [Chain Composition](https://python.langchain.com/docs/expression_language/how_to/compose)
- [Mock Models](https://python.langchain.com/docs/guides/testing)
- [Streaming Guide](https://python.langchain.com/docs/expression_language/streaming/)

### 2. Implementation Guides
- [Retrieval Chain](https://python.langchain.com/docs/expression_language/cookbook/retrieval)
- [Testing Setup](https://python.langchain.com/docs/guides/testing)
- [Variable Assignment](https://python.langchain.com/docs/expression_language/how_to/map)

### 3. Additional Resources
- [Mock Testing](https://python.langchain.com/docs/guides/testing)
- [Chain Building](https://python.langchain.com/docs/expression_language/why)
- [Error Handling](https://python.langchain.com/docs/guides/errors/)
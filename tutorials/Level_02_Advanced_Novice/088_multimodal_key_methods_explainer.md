# Key Methods with Azure OpenAI: Complete Guide

## Introduction

This implementation demonstrates sophisticated data processing by combining three key LangChain v3 concepts:
1. Key Methods: Efficient method patterns (invoke, stream, batch)
2. Azure OpenAI: Model interaction and embeddings
3. Structured Output: Type-safe response handling

The system provides robust data processing capabilities for banking/fintech applications, demonstrating practical patterns for handling financial data analysis with different processing modes.

### Real-World Application Value
- Multiple processing modes for different use cases
- Real-time streaming for interactive analysis
- Batch processing for multiple perspectives
- Secure Azure OpenAI integration
- Type-safe data handling
- Financial domain expertise

### System Architecture Overview
```
Request → DataProcessor → Azure OpenAI Services
           ↓                ↓
    Processing Modes    Model Types
    - Invoke           - Chat (regular)
    - Stream           - Chat (streaming)
    - Batch           - Embeddings
           ↓                ↓
    Structured Output ← Results Processing
```

## Core LangChain Concepts

### 1. Key Methods Pattern
```python
# Regular chat model for invoke/batch
self.chat_model = AzureChatOpenAI(
    streaming=False,
    temperature=0.7
)

# Streaming chat model
self.streaming_chat_model = AzureChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

Features:
- Specialized model instances
- Configurable streaming
- Callback integration
- Temperature control

### 2. Azure OpenAI Integration
```python
# Embeddings configuration
self.embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    model=os.getenv("AZURE_MODEL_NAME")
)
```

Benefits:
- Secure configuration
- Environment variable support
- Model flexibility
- Separate endpoints

### 3. Structured Output
```python
class ProcessingResult(BaseModel):
    content: str = Field(description="Processed content")
    mode: ProcessingMode = Field(description="Processing mode used")
    metadata: Dict = Field(description="Processing metadata")
    embedding: Optional[List[float]] = Field(
        description="Content embedding"
    )
```

Advantages:
- Type validation
- Optional fields
- Clear documentation
- Consistent structure

## Implementation Components

### 1. Processing Modes
```python
async def process_invoke(self, request: ProcessingRequest):
    """Regular synchronous processing."""
    response = self.chat_model.invoke(messages)
    embedding = self.embeddings.embed_query(request.content)
    return ProcessingResult(...)

async def process_stream(self, request: ProcessingRequest):
    """Real-time streaming response."""
    await self.streaming_chat_model.ainvoke(messages)

async def process_batch(self, request: ProcessingRequest):
    """Batch processing for multiple analyses."""
    responses = await self.chat_model.abatch(messages_batch)
    embeddings = self.embeddings.embed_documents(texts)
    return [ProcessingResult(...) for response, embedding in zip(responses, embeddings)]
```

Key elements:
- Mode specialization
- Async support
- Error handling
- Result formatting

### 2. System Template
```python
SYSTEM_TEMPLATE = """You are a specialized financial analyst AI assistant. Your role is to:
1. Analyze financial transaction data
2. Identify patterns and anomalies
3. Provide actionable insights
4. Use precise financial terminology
5. Consider risk and compliance"""
```

Features:
- Clear role definition
- Specific instructions
- Domain expertise
- Structured output guidance

### 3. Transaction Analysis
```python
SAMPLE_TRANSACTION_DATA = """
Transaction Data Sample:
- Date: 2025-03-27
- Amount: $15,750
- Type: Wire Transfer
- Source Account: Business Checking #1234
- Destination: Investment Account #5678
"""
```

Capabilities:
- Structured data format
- Essential transaction details
- Clear categorization
- Account tracking

## Advanced Features

### 1. Error Management
```python
try:
    response = self.chat_model.invoke(messages)
    embedding = self.embeddings.embed_query(request.content)
except Exception as e:
    raise ValueError(f"Error in invoke processing: {str(e)}")
```

Implementation:
- Specific error messages
- Exception handling
- Error propagation
- Clean recovery

### 2. Batch Analysis
```python
analyses = [
    "Analyze this transaction for potential fraud indicators.",
    "Provide a risk assessment of this transaction.",
    "Suggest compliance checks for this transaction."
]
```

Features:
- Multiple perspectives
- Specialized analysis
- Comprehensive coverage
- Parallel processing

### 3. Streaming Support
```python
self.streaming_chat_model = AzureChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

Strategies:
- Real-time output
- Callback handlers
- Async streaming
- Progress tracking

## Expected Output

### 1. Invoke Pattern
```text
1. Invoke Pattern:
----------------------------------------
Response: [Detailed financial analysis]
Embedding Size: 1536
```

### 2. Stream Pattern
```text
2. Stream Pattern:
----------------------------------------
[Real-time streaming response with financial analysis]
```

### 3. Batch Pattern
```text
3. Batch Pattern:
----------------------------------------
Batch 1: Fraud Analysis
Batch 2: Risk Assessment
Batch 3: Compliance Checks
```

## Best Practices

### 1. Configuration Management
- Separate model instances
- Environment variables
- Secure credentials
- Clear configuration

### 2. Error Handling
- Specific exceptions
- Clear messages
- Error recovery
- Proper propagation

### 3. Performance
- Async operations
- Batch processing
- Streaming support
- Resource optimization

## References

### 1. LangChain Core Concepts
- [Azure Chat Models](https://python.langchain.com/docs/integrations/chat/azure_openai)
- [Azure Embeddings](https://python.langchain.com/docs/integrations/text_embedding/azure_openai)
- [Streaming Support](https://python.langchain.com/docs/modules/model_io/streaming)

### 2. Implementation Guides
- [Async Operations](https://python.langchain.com/docs/expression_language/cookbook/async_parallel)
- [Batch Processing](https://python.langchain.com/docs/expression_language/cookbook/multiple_chains)
- [Callbacks](https://python.langchain.com/docs/modules/callbacks/)

### 3. Additional Resources
- [Environment Setup](https://python.langchain.com/docs/guides/deployments/azure_container_instances)
- [Embeddings Guide](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [Type Hints](https://python.langchain.com/docs/guides/safety)
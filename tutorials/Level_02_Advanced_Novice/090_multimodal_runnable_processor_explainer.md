# Multimodal Document Processor with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated document processing system by combining three key LangChain v3 concepts:
1. Multimodality: Handle both text and image content
2. Runnable Interface: Composable processing pipeline
3. Key Methods: Flexible processing patterns (invoke, stream, batch)

The system provides robust document processing capabilities for banking/fintech applications with support for various document types.

### Real-World Application Value
- Document analysis
- Image processing
- Data extraction
- Content validation
- Batch processing
- Real-time streaming

### System Architecture Overview
```
Document → MultimodalProcessor → Content Parsing
    ↓            ↓                   ↓
 Text/Image   Processing Chain    Data Extraction
    ↓            ↓                   ↓
   Base64    Runnable Pipeline   JSON Response
```

## Core LangChain Concepts

### 1. Multimodal Processing
```python
content = []

# Add text content if available
if doc.text:
    content.append({
        "type": "text",
        "text": f"Text content:\n{doc.text}"
    })

# Add image content if available
if doc.image_data:
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{doc.image_data}"
        }
    })
```

Features:
- Combined text/image handling
- Base64 image encoding
- Content type detection
- Flexible message structure

### 2. Runnable Interface
```python
self.chain = (
    RunnablePassthrough.assign(
        timestamp=lambda _: datetime.now().isoformat()
    )
    | self._create_prompt
    | self.llm
    | self._parse_response
)
```

Benefits:
- Pipeline composition
- Data transformation
- Function chaining
- Error handling

### 3. Key Methods
```python
async def process(self, document: DocumentContent) -> ProcessingResult:
    """Process document using invoke pattern."""

async def stream_process(self, document: DocumentContent) -> AsyncIterator[Dict]:
    """Process document using stream pattern."""

async def batch_process(self, documents: List[DocumentContent]) -> List[ProcessingResult]:
    """Process documents using batch pattern."""
```

Advantages:
- Multiple processing modes
- Async support
- Stream capabilities
- Batch handling

## Implementation Components

### 1. Document Models
```python
class DocumentContent(BaseModel):
    text: Optional[str] = Field(description="Text content", default=None)
    image_data: Optional[str] = Field(description="Base64 encoded image", default=None)
    doc_type: DocumentType = Field(description="Document type")
    metadata: Dict = Field(description="Document metadata")
```

Key elements:
- Optional content types
- Document categorization
- Metadata support
- Validation rules

### 2. Content Templates
```python
self.check_template = """Analyze this check image and extract:
1. Check amount
2. Date
3. Payee
4. Bank name
Provide the information in a structured format."""

self.receipt_template = """Analyze this receipt and extract:
1. Total amount
2. Date
3. Merchant
4. Items purchased
Provide the information in a structured format."""
```

Features:
- Specialized analysis
- Structured extraction
- Clear instructions
- Format guidelines

### 3. Response Parsing
```python
def _parse_response(self, response: AIMessage) -> Dict:
    content = response.content.strip()
    try:
        # Try parsing as JSON first
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback to key-value parsing
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip().strip(',"')
```

Capabilities:
- JSON parsing
- Fallback handling
- Clean formatting
- Error recovery

## Advanced Features

### 1. Message Construction
```python
messages = [
    SystemMessage(content=template)
]

content = []
if doc.text:
    content.append({
        "type": "text",
        "text": f"Text content:\n{doc.text}"
    })
```

Implementation:
- Dynamic content
- Message structuring
- Type handling
- Format validation

### 2. Stream Processing
```python
async for chunk in self.chain.astream(
    {
        "document": document,
        "run_id": run_id or datetime.now().isoformat()
    }
):
    yield {
        "chunk": chunk.content if hasattr(chunk, "content") else str(chunk),
        "timestamp": datetime.now().isoformat()
    }
```

Features:
- Real-time output
- Progress tracking
- Timestamp logging
- Format handling

### 3. Batch Processing
```python
for doc in documents:
    try:
        result = await self.process(doc)
        results.append(result)
    except Exception as e:
        results.append(
            ProcessingResult(
                doc_id=datetime.now().isoformat(),
                extracted_data={"error": str(e)},
                confidence=0.0,
                processing_time=0.0
            )
        )
```

Strategies:
- Parallel handling
- Error management
- Result aggregation
- Status tracking

## Expected Output

### 1. Individual Document
```text
Document ID: 2025-03-28T00:41:15.546416
Extracted Data: {
    "Amount": "$1,500.00",
    "Date": "2025-03-27",
    "Payee": "John Smith",
    "Bank": "Example Bank"
}
Confidence: 0.85
Processing Time: 2.05s
```

### 2. Stream Output
```text
Chunk: {
    "content": "Processing receipt...",
    "timestamp": "2025-03-28T00:41:16.546578"
}
```

## Best Practices

### 1. Content Handling
- Type validation
- Format checking
- Base64 encoding
- Error protection

### 2. Processing Pipeline
- Clear composition
- Error handling
- Performance tracking
- Status monitoring

### 3. Response Management
- JSON parsing
- Fallback strategies
- Clean formatting
- Error recovery

## References

### 1. LangChain Core Concepts
- [Multimodal Chat Models](https://python.langchain.com/docs/integrations/chat/azure_openai)
- [Runnable Interface](https://python.langchain.com/docs/expression_language/interface)
- [Processing Patterns](https://python.langchain.com/docs/expression_language/cookbook)

### 2. Implementation Guides
- [Azure Vision Integration](https://python.langchain.com/docs/integrations/chat/azure_openai)
- [Image Processing](https://python.langchain.com/docs/guides/vision)
- [Async Operations](https://python.langchain.com/docs/expression_language/cookbook/async_parallel)

### 3. Additional Resources
- [Error Handling](https://python.langchain.com/docs/guides/debugging)
- [Response Parsing](https://python.langchain.com/docs/modules/model_io/output_parsers)
- [Base64 Images](https://python.langchain.com/docs/guides/vision)
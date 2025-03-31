# Async Document Analyzer with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated document analysis system by combining three key LangChain v3 concepts:
1. Async: Concurrent processing with proper chain composition
2. Evaluation: Custom quality assessment with async scoring
3. Structured Output: Type-safe validation with modern Pydantic

The system provides efficient, validated document analysis with quality metrics.

### Real-World Application Value
- Concurrent processing
- Quality scoring
- Type validation
- Error resilience
- Clear output

### System Architecture Overview
```
Documents → Async Chain → Analysis → Concurrent Evaluation → Validated Results
            ↓           ↓         ↓                     ↑
         JSON Output  Types     Quality Metrics      Error Handling
```

## Core LangChain Concepts

### 1. Async Chain Composition
```python
def __init__(self, llm: BaseLLM, criteria: str):
    self.eval_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert document analyst..."),
            ("user", "Document: {input}\nAnalysis: {prediction}")
        ])
        | self.llm
    )

async def aevaluate_strings(
    self, prediction: str, input: str, **kwargs
) -> dict:
    result = await self.eval_chain.ainvoke(
        {"input": input, "prediction": prediction},
        config=RunnableConfig(callbacks=None)
    )
```

Features:
- Clean composition
- Proper async
- Error handling
- Config options

### 2. Quality Evaluation
```python
class CustomEvaluator(StringEvaluator):
    async def aevaluate_strings(
        self, prediction: str, input: str, **kwargs
    ) -> dict:
        result = await self.eval_chain.ainvoke(...)
        score = float(result.strip())
        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
        return {"score": score}
```

Benefits:
- Score normalization
- Error handling
- Clear interface
- Default values

### 3. Modern Validation
```python
class DocumentAnalysis(BaseModel):
    doc_type: DocumentType = Field(description="Type of document")
    summary: str = Field(description="Brief document summary")
    key_points: List[str] = Field(description="Main points")
    sentiment: str = Field(description="Overall sentiment")
    metrics: Optional[AnalysisMetrics] = Field(
        description="Analysis quality metrics",
        default=None
    )
    
    @field_validator("key_points")
    @classmethod
    def validate_key_points(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Must have at least one key point")
        return v
```

Advantages:
- Modern validation
- Optional fields
- Clear descriptions
- Custom checks

## Implementation Components

### 1. Document Processor
```python
class DocumentProcessor:
    def __init__(self):
        self.llm = AzureChatOpenAI(...)
        self.relevance_evaluator = CustomEvaluator(self.llm, "relevance")
        self.clarity_evaluator = CustomEvaluator(self.llm, "clarity")
        
        # Create analysis chain
        self.analyze_chain = (
            ChatPromptTemplate.from_messages([...])
            | self.llm
            | JsonOutputParser()
        )
```

Key elements:
- Azure integration
- Custom evaluators
- Chain composition
- JSON parsing

### 2. Concurrent Evaluation
```python
async def evaluate_analysis(self, document: str, analysis: DocumentAnalysis) -> AnalysisMetrics:
    # Run evaluations concurrently
    relevance_task = asyncio.create_task(
        self.relevance_evaluator.aevaluate_strings(...)
    )
    clarity_task = asyncio.create_task(
        self.clarity_evaluator.aevaluate_strings(...)
    )
    
    # Wait for results
    relevance_result, clarity_result = await asyncio.gather(
        relevance_task,
        clarity_task
    )
```

Features:
- Task creation
- Concurrent execution
- Result gathering
- Score calculation

### 3. Output Processing
```python
async def analyze_document(self, document: str) -> DocumentAnalysis:
    try:
        # Get initial analysis
        result = await self.analyze_chain.ainvoke({"document": document})
        
        # Create analysis object
        analysis = DocumentAnalysis.model_validate(result)
        
        # Evaluate analysis
        metrics = await self.evaluate_analysis(document, analysis)
        analysis.metrics = metrics
        
        return analysis
    except Exception as e:
        raise ValueError(f"Error analyzing document: {str(e)}")
```

Capabilities:
- JSON parsing
- Type validation
- Quality scoring
- Error handling

## Advanced Features

### 1. System Prompt
```python
SYSTEM_PROMPT = """You are a document analysis expert.
Return ONLY a JSON object with these fields:
{
    "doc_type": "email|contract|report|policy",
    "summary": "brief summary here",
    "key_points": ["point 1", "point 2", "point 3"],
    "sentiment": "positive|neutral|negative"
}"""
```

Implementation:
- Clear instructions
- JSON schema
- Example values
- Format guidance

### 2. Batch Processing
```python
# Process documents concurrently
tasks = [processor.analyze_document(doc) for doc in documents]
results = await asyncio.gather(*tasks)

# Display results
for i, analysis in enumerate(results, 1):
    print(f"\nDocument {i} Analysis")
    print(f"Type: {analysis.doc_type}")
    print(f"Summary: {analysis.summary}")
```

Features:
- Task creation
- Concurrent processing
- Result handling
- Clean output

### 3. Error Management
```python
try:
    # Process documents
    tasks = [processor.analyze_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    
    # Display results
    for analysis in results:
        print(f"Type: {analysis.doc_type}")
except Exception as e:
    print(f"Error during demonstration: {str(e)}")
```

Strategies:
- Task handling
- Result validation
- Error catching
- Clear messages

## Expected Output

### 1. Sales Report Analysis
```text
Document 1 Analysis
----------------------------------------
Type: email
Summary: Q1 sales performance review showing positive results
Key Points:
- Exceeded targets by 15%
- New client acquisition up 20%
- Customer retention at 95%
Sentiment: positive

Quality Metrics:
- Relevance: 0.95
- Clarity: 0.92
- Confidence: 0.93
```

### 2. Contract Analysis
```text
Document 3 Analysis
----------------------------------------
Type: contract
Summary: Service agreement outlining terms and conditions
Key Points:
- Service delivery specifications
- 30-day payment terms
- 12-month duration
Sentiment: neutral

Quality Metrics:
- Relevance: 0.88
- Clarity: 0.90
- Confidence: 0.89
```

## Best Practices

### 1. Chain Composition
- Proper async
- Clean structure
- Error handling
- Config options

### 2. Evaluation Logic
- Score normalization
- Concurrent checks
- Default values
- Error resilience

### 3. Type Safety
- Modern validators
- Optional fields
- Custom checks
- Clear errors

## References

### 1. LangChain Core Concepts
- [Async Chain](https://python.langchain.com/docs/expression_language/how_to/async)
- [Custom Evaluation](https://python.langchain.com/docs/guides/evaluation/custom)
- [Modern Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)

### 2. Implementation Guides
- [Chain Composition](https://python.langchain.com/docs/expression_language/cookbook/async_chain)
- [Modern Validation](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)
- [Error Handling](https://python.langchain.com/docs/guides/debugging)

### 3. Additional Resources
- [Task Management](https://docs.python.org/3/library/asyncio-task.html)
- [Pydantic v2](https://docs.pydantic.dev/latest/migration/)
- [Azure Integration](https://python.langchain.com/docs/integrations/llms/azure_openai)
# Understanding the Transaction Anomaly Detector in LangChain

Welcome to this comprehensive guide on building a transaction anomaly detection system using LangChain! This example demonstrates how to combine memory management with evaluation capabilities to create an intelligent system for detecting unusual transaction patterns and potential fraud indicators.

## Complete Code Walkthrough

### 1. Required Imports and Technical Foundation

```python
import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory
```

Let's understand each import's technical role in our system:

`typing` provides type hints that enhance code reliability:
- `List`: Used for collections of transactions or patterns
- `Dict`: For structured transaction data
- `Optional`: For nullable fields like location
- Type hints help catch errors at development time

The LangChain imports create our analysis framework:
- `ConversationSummaryMemory`: Maintains transaction history
- `PydanticOutputParser`: Ensures structured analysis output
- `ChatPromptTemplate`: Templates for analysis prompts

### 2. Transaction Schema Implementation

```python
class TransactionDetails(BaseModel):
    """Schema for transaction details."""
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    timestamp: datetime = Field(description="Transaction timestamp")
    merchant: str = Field(description="Merchant name")
    category: str = Field(description="Transaction category")
    payment_method: str = Field(description="Payment method used")
    location: Optional[str] = Field(description="Transaction location")
```

Technical aspects of the schema:
1. Data Validation:
   - Type enforcement for each field
   - Optional field handling
   - Automatic timestamp conversion
   - Field descriptions for documentation

2. Schema Design:
   - Immutable field definitions
   - Clear type hierarchies
   - Null safety with Optional
   - Proper datetime handling

### 3. Pattern Recognition Schema

```python
class TransactionPattern(BaseModel):
    """Schema for transaction patterns."""
    avg_amount: float = Field(description="Average transaction amount")
    common_merchants: List[str] = Field(description="Frequently visited merchants")
    usual_categories: List[str] = Field(description="Common transaction categories")
    typical_locations: List[str] = Field(description="Usual transaction locations")
    active_hours: List[int] = Field(description="Typically active hours")
    payment_preferences: List[str] = Field(description="Preferred payment methods")
```

Implementation details:
1. Pattern Storage:
   - Numerical averages
   - List-based frequency tracking
   - Time-based patterns
   - Location clustering

2. Data Structures:
   - Efficient list implementations
   - Numeric precision handling
   - String normalization
   - Time zone awareness

### 4. Anomaly Analysis Schema

```python
class AnomalyAnalysis(BaseModel):
    """Schema for anomaly detection results."""
    transaction: TransactionDetails = Field(description="Transaction being analyzed")
    patterns: TransactionPattern = Field(description="Established transaction patterns")
    anomaly_score: float = Field(description="Anomaly score (0-100)")
    risk_level: str = Field(description="Risk level assessment")
    anomaly_factors: List[str] = Field(description="Factors contributing to anomaly")
    recommendations: List[str] = Field(description="Suggested actions")
    timestamp: datetime = Field(default_factory=datetime.now)
```

Technical implementation details:
1. Score Calculation:
   - Normalized scoring (0-100)
   - Multi-factor analysis
   - Weighted risk assessment
   - Time-sensitivity handling

2. Data Integration:
   - Nested schema relationships
   - Cross-reference capabilities
   - Timestamp synchronization
   - Recommendation generation

### 5. Memory Management Integration

```python
def analyze_transaction(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    memory: ConversationSummaryMemory,
    transaction_data: Dict
) -> AnomalyAnalysis:
```

Technical aspects of memory management:

1. Memory Storage:
```python
memory.save_context(
    {"input": f"New transaction: {transaction_data['transaction_id']}"},
    {"output": f"Amount: ${transaction_data['amount']}, "
              f"Merchant: {transaction_data['merchant']}"}
)
```

2. History Retrieval:
```python
history = memory.load_memory_variables({}).get("history", "")
```

### 6. Analysis Implementation

```python
system_text = (
    "You are a transaction anomaly detection system. Analyze the current transaction "
    "considering the transaction history and identify any unusual patterns or potential "
    "fraud indicators.\n\n"
    f"Previous transaction history:\n{history}\n\n"
    "Respond with a JSON object that exactly follows this schema (no additional text):\n\n"
    f"{format_instructions}\n"
)
```

Technical components:
1. Prompt Engineering:
   - Context integration
   - Schema enforcement
   - History incorporation
   - Response formatting

2. Analysis Pipeline:
   - Data validation
   - Pattern matching
   - Score calculation
   - Recommendation generation

### 7. Example Usage

```python
# Save transaction history
normal_transactions = [
    {
        "transaction_id": "TX001",
        "amount": 25.50,
        "merchant": "Local Coffee Shop"
    }
]

for tx in normal_transactions:
    memory.save_context(
        {"input": f"Transaction: {tx['transaction_id']}"},
        {"output": f"Amount: ${tx['amount']}, Merchant: {tx['merchant']}"}
    )
```

## Resources

### Memory Management Documentation
Understanding transaction history:
https://python.langchain.com/docs/concepts/memory/

Memory implementation patterns:
https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types

### Evaluation Documentation
Anomaly detection techniques:
https://python.langchain.com/docs/guides/evaluation/

Pattern recognition:
https://python.langchain.com/docs/guides/evaluation/metrics

## Best Practices

### 1. Technical Implementation
When implementing the anomaly detector:
- Validate all input data thoroughly
- Handle edge cases in calculations
- Implement proper error handling
- Monitor memory usage
- Log important operations
- Test boundary conditions

### 2. Memory Management
For efficient history tracking:
- Implement cleanup strategies
- Monitor memory growth
- Optimize storage patterns
- Handle concurrent access
- Implement backup strategies

### 3. Analysis Pipeline
For reliable detection:
- Calibrate scoring algorithms
- Validate pattern recognition
- Test with diverse data
- Monitor false positives
- Implement feedback loops

Remember: When implementing transaction monitoring:
- Ensure data accuracy
- Maintain privacy standards
- Document all assumptions
- Test thoroughly
- Monitor performance
- Update patterns regularly
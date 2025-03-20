# Understanding Async Transaction Processing in LangChain

Welcome to this comprehensive guide on building an async transaction processor using LangChain! This example demonstrates how to combine output parsing with async programming to create an efficient system for processing multiple financial transactions concurrently.

## Complete Code Walkthrough

### 1. Required Imports
```python
import os
import json
import asyncio
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
```

Let's understand each import:
- `os`: For environment variable handling
- `json`: For transaction data serialization
- `asyncio`: Core library for async operations
- `typing`: Type hints for better code clarity
- `datetime`: For transaction timestamps
- `pydantic`: For data validation and serialization
- `langchain` components: For model interaction and output parsing

### 2. Environment Setup
```python
# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
```

This section:
- Loads environment variables from .env file
- Checks for required Azure OpenAI settings
- Raises clear error if any are missing

### 3. Transaction Schema Definition
```python
class TransactionAnalysis(BaseModel):
    """Schema for transaction analysis output."""
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    category: str = Field(description="Transaction category")
    risk_level: str = Field(description="Risk assessment (Low, Medium, High)")
    flags: List[str] = Field(description="Potential risk flags")
    recommendation: str = Field(description="Processing recommendation")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "transaction_id": "TX123456",
                "amount": 1500.00,
                "category": "International Transfer",
                "risk_level": "Medium",
                "flags": [
                    "International transaction",
                    "Above average amount"
                ],
                "recommendation": "Proceed with standard verification",
                "timestamp": "2025-03-19T01:06:36.100849"
            }]
        }
    }
```

This schema:
- Defines the structure of transaction analysis
- Includes field descriptions for clarity
- Provides validation rules
- Includes example data for reference
- Uses ISO 8601 format for timestamps

### 4. Chat Model Creation
```python
def create_chat_model() -> AzureChatOpenAI:
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0  # Use 0 for consistent outputs
    )
```

This function:
- Creates the AI model instance
- Uses environment variables for configuration
- Sets temperature to 0 for deterministic results
- Returns configured Azure ChatOpenAI instance

### 5. Transaction Processing Function
```python
async def process_transaction(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    transaction: dict,
    delay: float = 0
) -> TransactionAnalysis:
    """Process a single transaction asynchronously."""
    # Simulate processing delay if specified
    if delay > 0:
        await asyncio.sleep(delay)
    
    # Convert transaction to minified JSON
    tx_str = json.dumps(transaction, separators=(",", ":"))
    
    # Construct prompt with explicit schema
    prompt_text = (
        "You are a financial transaction analyzer. Analyze the provided transaction data "
        "and output a JSON object that strictly follows this schema:\n\n"
        '{"transaction_id": "<string>", "amount": <number>, "category": "<string>", '
        '"risk_level": "<Low, Medium, High>", "flags": ["<string>", ...], '
        '"recommendation": "<string>", "timestamp": "<ISO 8601 datetime string>"}\n\n'
        "Ensure that the JSON contains exactly these keys with appropriate values "
        "and no additional text.\n\n"
        "Transaction data: " + tx_str
    )
    
    # Use single system message
    messages = [SystemMessage(content=prompt_text)]
    
    # Process asynchronously
    response = await chat_model.ainvoke(messages)
    
    # Parse and return
    return parser.parse(response.content)
```

This function:
- Takes model, parser, and transaction data
- Supports optional processing delay
- Minifies JSON for efficiency
- Uses clear prompt template
- Processes asynchronously
- Returns parsed result

### 6. Batch Processing Function
```python
async def process_batch(transactions: List[dict]) -> List[TransactionAnalysis]:
    """Process a batch of transactions concurrently."""
    # Initialize components
    chat_model = create_chat_model()
    parser = PydanticOutputParser(pydantic_object=TransactionAnalysis)
    
    # Create tasks for all transactions
    tasks = [
        process_transaction(
            chat_model=chat_model,
            parser=parser,
            transaction=tx,
            delay=0.5  # Simulated processing delay
        )
        for tx in transactions
    ]
    
    # Execute all tasks concurrently
    return await asyncio.gather(*tasks)
```

This function:
- Takes a list of transactions
- Creates processing components
- Generates tasks for each transaction
- Executes all tasks concurrently
- Returns list of results

### 7. Demonstration Function
```python
async def demonstrate_async_processing():
    """Demonstrate async transaction processing capabilities."""
    try:
        print("\nDemonstrating Async Transaction Processing...\n")
        
        # Example 1: Process Multiple Transactions
        transactions = [
            {
                "transaction_id": "TX001",
                "amount": 1500.00,
                "category": "domestic_payment",
                # ... more fields
            },
            # ... more transactions
        ]
        
        results = await process_batch(transactions)
        
        # Display results
        for result in results:
            print(f"\nTransaction: {result.transaction_id}")
            print(f"Amount: ${result.amount:,.2f}")
            # ... more output
        
        # Example 2: Different Processing Times
        tasks = [
            process_transaction(
                chat_model=chat_model,
                parser=parser,
                transaction=transactions[0],
                delay=delay
            )
            for delay in [1.0, 0.5, 0.2]
        ]
        
        timed_results = await asyncio.gather(*tasks)
        # ... display results
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
```

This function:
- Shows practical usage examples
- Demonstrates batch processing
- Shows varying processing times
- Includes error handling
- Formats output clearly

### 8. Main Entry Point
```python
def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Async Transaction Processing...")
    asyncio.run(demonstrate_async_processing())

if __name__ == "__main__":
    main()
```

This section:
- Provides entry point
- Runs async demonstration
- Uses asyncio.run for async execution

## Resources

1. **Output Parsing Documentation**
   - **Schema Definition**: https://python.langchain.com/docs/concepts/structured_outputs/#schema-definition
   - **JSON Mode**: https://python.langchain.com/docs/concepts/structured_outputs/#json-mode
   - **Output Methods**: https://python.langchain.com/docs/concepts/structured_outputs/#structured-output-method

2. **Async Programming Documentation**
   - **Async Guide**: https://python.langchain.com/docs/guides/async_programming/
   - **Performance**: https://python.langchain.com/docs/guides/performance
   - **Best Practices**: https://python.langchain.com/docs/guides/best_practices

## Key Takeaways

1. **Efficiency**
   - Use async for concurrent processing
   - Minify JSON for better performance
   - Use single system messages
   - Implement batch processing

2. **Best Practices**
   - Clear schema definitions
   - Explicit error handling
   - Proper async patterns
   - Clean code organization

3. **Real-World Applications**
   - Payment processing
   - Risk assessment
   - Transaction monitoring
   - Financial analysis
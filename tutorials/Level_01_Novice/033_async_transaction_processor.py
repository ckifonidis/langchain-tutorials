"""
LangChain Async Transaction Processor Example

This example demonstrates how to combine output parsing and async programming
to create an efficient transaction processing system that can handle multiple
financial operations concurrently.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

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

def create_chat_model() -> AzureChatOpenAI:
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

async def process_transaction(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    transaction: dict,
    delay: float = 0
) -> TransactionAnalysis:
    """
    Process a single transaction asynchronously.
    
    Args:
        chat_model: The chat model to use.
        parser: The output parser.
        transaction: Transaction data.
        delay: Optional delay to simulate processing time.
    
    Returns:
        TransactionAnalysis: Processed transaction data.
    """
    # Simulate processing delay if specified.
    if delay > 0:
        await asyncio.sleep(delay)
    
    # Convert the transaction to a minified JSON string.
    tx_str = json.dumps(transaction, separators=(",", ":"))
    
    # Manually construct the prompt string.
    prompt_text = (
        "You are a financial transaction analyzer. Analyze the provided transaction data "
        "and output a JSON object that strictly follows this schema:\n\n"
        '{"transaction_id": "<string>", "amount": <number>, "category": "<string>", "risk_level": "<Low, Medium, High>", '
        '"flags": ["<string>", ...], "recommendation": "<string>", "timestamp": "<ISO 8601 datetime string>"}\n\n'
        "Ensure that the JSON contains exactly these keys with appropriate values and no additional text.\n\n"
        "Transaction data: " + tx_str
    )
    
    # Create a single SystemMessage with the full prompt.
    messages = [SystemMessage(content=prompt_text)]
    
    # Process with the chat model asynchronously.
    response = await chat_model.ainvoke(messages)
    
    # Parse and return the result.
    return parser.parse(response.content)

async def process_batch(transactions: List[dict]) -> List[TransactionAnalysis]:
    """Process a batch of transactions concurrently."""
    # Initialize components.
    chat_model = create_chat_model()
    parser = PydanticOutputParser(pydantic_object=TransactionAnalysis)
    
    # Create tasks for all transactions.
    tasks = [
        process_transaction(
            chat_model=chat_model,
            parser=parser,
            transaction=tx,
            delay=0.5  # Simulated processing delay.
        )
        for tx in transactions
    ]
    
    # Execute all tasks concurrently.
    return await asyncio.gather(*tasks)

async def demonstrate_async_processing():
    """Demonstrate async transaction processing capabilities."""
    try:
        print("\nDemonstrating Async Transaction Processing...\n")
        
        # Example 1: Process Multiple Transactions.
        print("Example 1: Processing Multiple Transactions")
        print("-" * 50)
        
        # Sample transactions.
        transactions = [
            {
                "transaction_id": "TX001",
                "amount": 1500.00,
                "category": "domestic_payment",
                "risk_level": "Low",
                "flags": [],
                "recommendation": "Proceed with transaction",
                "source": "Account A",
                "destination": "Account B",
                "currency": "USD"
            },
            {
                "transaction_id": "TX002",
                "amount": 5000.00,
                "category": "international_transfer",
                "risk_level": "High",
                "flags": ["International transaction", "High amount"],
                "recommendation": "Further verification required",
                "source": "Account C",
                "destination": "Account D",
                "currency": "EUR"
            },
            {
                "transaction_id": "TX003",
                "amount": 750.00,
                "category": "wire_transfer",
                "risk_level": "Medium",
                "flags": ["Above average amount"],
                "recommendation": "Proceed with caution",
                "source": "Account E",
                "destination": "Account F",
                "currency": "USD"
            }
        ]
        
        print("\nProcessing transactions concurrently...")
        results = await process_batch(transactions)
        
        for result in results:
            print(f"\nTransaction: {result.transaction_id}")
            print(f"Amount: ${result.amount:,.2f}")
            print(f"Category: {result.category}")
            print(f"Risk Level: {result.risk_level}")
            print("Flags:", ", ".join(result.flags))
            print(f"Recommendation: {result.recommendation}")
            print(f"Timestamp: {result.timestamp}")
        
        # Example 2: Processing with Different Delays.
        print("\nExample 2: Processing with Varying Times")
        print("-" * 50)
        
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=TransactionAnalysis)
        
        tasks = [
            process_transaction(
                chat_model=chat_model,
                parser=parser,
                transaction=transactions[0],
                delay=delay
            )
            for delay in [1.0, 0.5, 0.2]  # Different processing times.
        ]
        
        print("\nProcessing with varying delays...")
        timed_results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(timed_results):
            print(f"\nProcessor {i+1}")
            print(f"Transaction: {result.transaction_id}")
            print(f"Risk Level: {result.risk_level}")
            print(f"Timestamp: {result.timestamp}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Async Transaction Processing...")
    asyncio.run(demonstrate_async_processing())

if __name__ == "__main__":
    main()

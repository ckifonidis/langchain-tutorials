"""
LangChain Transaction Anomaly Detector Example

This example demonstrates how to combine memory management and evaluation capabilities
to create a system that monitors transactions and detects anomalous patterns while
maintaining transaction history.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

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

class TransactionDetails(BaseModel):
    """Schema for transaction details."""
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    timestamp: datetime = Field(description="Transaction timestamp")
    merchant: str = Field(description="Merchant name")
    category: str = Field(description="Transaction category")
    payment_method: str = Field(description="Payment method used")
    location: Optional[str] = Field(description="Transaction location")

class TransactionPattern(BaseModel):
    """Schema for transaction patterns."""
    avg_amount: float = Field(description="Average transaction amount")
    common_merchants: List[str] = Field(description="Frequently visited merchants")
    usual_categories: List[str] = Field(description="Common transaction categories")
    typical_locations: List[str] = Field(description="Usual transaction locations")
    active_hours: List[int] = Field(description="Typically active hours")
    payment_preferences: List[str] = Field(description="Preferred payment methods")

class AnomalyAnalysis(BaseModel):
    """Schema for anomaly detection results."""
    transaction: TransactionDetails = Field(description="Transaction being analyzed")
    patterns: TransactionPattern = Field(description="Established transaction patterns")
    anomaly_score: float = Field(description="Anomaly score (0-100)")
    risk_level: str = Field(description="Risk level assessment")
    anomaly_factors: List[str] = Field(description="Factors contributing to anomaly")
    recommendations: List[str] = Field(description="Suggested actions")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "transaction": {
                    "transaction_id": "TX123456",
                    "amount": 1500.00,
                    "timestamp": "2024-03-19T15:30:00",
                    "merchant": "Tech Store X",
                    "category": "Electronics",
                    "payment_method": "Credit Card",
                    "location": "New York, NY"
                },
                "patterns": {
                    "avg_amount": 200.00,
                    "common_merchants": ["Local Market", "Coffee Shop"],
                    "usual_categories": ["Groceries", "Dining"],
                    "typical_locations": ["Boston, MA"],
                    "active_hours": [9, 10, 11, 12, 13, 14],
                    "payment_preferences": ["Debit Card"]
                },
                "anomaly_score": 85.5,
                "risk_level": "High",
                "anomaly_factors": [
                    "Unusual amount",
                    "New location",
                    "Uncommon merchant category"
                ],
                "recommendations": [
                    "Verify transaction with cardholder",
                    "Request additional authentication",
                    "Monitor for similar patterns"
                ]
            }]
        }
    }

def create_chat_model():
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def analyze_transaction(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    memory: ConversationSummaryMemory,
    transaction_data: Dict
) -> AnomalyAnalysis:
    """
    Analyze a transaction for anomalies.
    
    Args:
        chat_model: The chat model to use
        parser: The output parser for structured analysis
        memory: Conversation memory for transaction history
        transaction_data: Current transaction data
        
    Returns:
        AnomalyAnalysis: Structured analysis of the transaction
    """
    # Get format instructions
    format_instructions = parser.get_format_instructions()
    
    # Load transaction history
    history = memory.load_memory_variables({}).get("history", "")
    
    # Build system message
    system_text = (
        "You are a transaction anomaly detection system. Analyze the current transaction "
        "considering the transaction history and identify any unusual patterns or potential "
        "fraud indicators.\n\n"
        f"Previous transaction history:\n{history}\n\n"
        "Respond with a JSON object that exactly follows this schema (no additional text):\n\n"
        f"{format_instructions}\n"
    )
    
    # Create messages
    system_msg = SystemMessage(content=system_text)
    human_msg = HumanMessage(
        content=f"Analyze this transaction for anomalies: {json.dumps(transaction_data)}"
    )
    
    # Get analysis
    response = chat_model.invoke([system_msg, human_msg])
    analysis = parser.parse(response.content)
    
    # Update memory with current transaction
    memory.save_context(
        {"input": f"New transaction: {transaction_data['transaction_id']}"},
        {"output": f"Amount: ${transaction_data['amount']}, "
                  f"Merchant: {transaction_data['merchant']}, "
                  f"Category: {transaction_data['category']}"}
    )
    
    return analysis

def demonstrate_anomaly_detection():
    """Demonstrate transaction anomaly detection capabilities."""
    try:
        print("\nDemonstrating Transaction Anomaly Detection...\n")
        
        # Initialize components
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=AnomalyAnalysis)
        memory = ConversationSummaryMemory(llm=chat_model, memory_key="history")
        
        # Example 1: Normal Transaction Pattern
        print("Example 1: Normal Transaction")
        print("-" * 50)
        
        # Save some transaction history
        normal_transactions = [
            {
                "transaction_id": "TX001",
                "amount": 25.50,
                "merchant": "Local Coffee Shop",
                "category": "Dining",
                "payment_method": "Debit Card",
                "location": "Boston, MA"
            },
            {
                "transaction_id": "TX002",
                "amount": 85.75,
                "merchant": "Local Market",
                "category": "Groceries",
                "payment_method": "Debit Card",
                "location": "Boston, MA"
            }
        ]
        
        for tx in normal_transactions:
            memory.save_context(
                {"input": f"Transaction: {tx['transaction_id']}"},
                {"output": f"Amount: ${tx['amount']}, Merchant: {tx['merchant']}"}
            )
        
        # Analyze new normal transaction
        normal_tx = {
            "transaction_id": "TX003",
            "amount": 32.50,
            "timestamp": datetime.now().isoformat(),
            "merchant": "Local Coffee Shop",
            "category": "Dining",
            "payment_method": "Debit Card",
            "location": "Boston, MA"
        }
        
        normal_analysis = analyze_transaction(
            chat_model, parser, memory, normal_tx
        )
        
        print("\nTransaction Analysis:")
        print(f"Transaction ID: {normal_analysis.transaction.transaction_id}")
        print(f"Amount: ${normal_analysis.transaction.amount:.2f}")
        print(f"Merchant: {normal_analysis.transaction.merchant}")
        print(f"Risk Level: {normal_analysis.risk_level}")
        print(f"Anomaly Score: {normal_analysis.anomaly_score:.1f}")
        
        if normal_analysis.anomaly_factors:
            print("\nAnomaly Factors:")
            for factor in normal_analysis.anomaly_factors:
                print(f"- {factor}")
        
        # Example 2: Anomalous Transaction
        print("\nExample 2: Unusual Transaction")
        print("-" * 50)
        
        # Analyze unusual transaction
        unusual_tx = {
            "transaction_id": "TX004",
            "amount": 1500.00,
            "timestamp": datetime.now().isoformat(),
            "merchant": "Tech Store X",
            "category": "Electronics",
            "payment_method": "Credit Card",
            "location": "New York, NY"
        }
        
        unusual_analysis = analyze_transaction(
            chat_model, parser, memory, unusual_tx
        )
        
        print("\nTransaction Analysis:")
        print(f"Transaction ID: {unusual_analysis.transaction.transaction_id}")
        print(f"Amount: ${unusual_analysis.transaction.amount:.2f}")
        print(f"Merchant: {unusual_analysis.transaction.merchant}")
        print(f"Risk Level: {unusual_analysis.risk_level}")
        print(f"Anomaly Score: {unusual_analysis.anomaly_score:.1f}")
        
        print("\nAnomaly Factors:")
        for factor in unusual_analysis.anomaly_factors:
            print(f"- {factor}")
        
        print("\nRecommendations:")
        for rec in unusual_analysis.recommendations:
            print(f"- {rec}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Transaction Anomaly Detector...")
    demonstrate_anomaly_detection()

if __name__ == "__main__":
    main()
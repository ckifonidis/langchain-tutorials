#!/usr/bin/env python3
"""
LangChain Fraud Detection (LangChain v3)

This example demonstrates building a fraud detection system using LangChain's
retrieval and structured output capabilities. It can identify potential fraud
in financial transactions and provide alerts.

Key concepts demonstrated:
1. Retrieval: Accessing and processing transaction data
2. Structured Output: Generating organized fraud detection reports
"""

import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain_openai import AzureChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Check for required Azure OpenAI configuration
missing_config = []
if not AZURE_ENDPOINT:
    missing_config.append("AZURE_OPENAI_ENDPOINT")
if not AZURE_API_KEY:
    missing_config.append("AZURE_OPENAI_API_KEY")
if not AZURE_API_VERSION:
    missing_config.append("AZURE_OPENAI_API_VERSION")
if not AZURE_DEPLOYMENT:
    missing_config.append("AZURE_OPENAI_DEPLOYMENT_NAME")

if missing_config:
    raise ValueError(f"Missing required Azure OpenAI configuration in .env file: {', '.join(missing_config)}")
print("\nChecking Azure OpenAI configuration...")

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Let the base class default method handle other types
        return super().default(obj)

class TransactionType(str, Enum):
    """Types of financial transactions."""
    TRANSFER = "transfer"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"

class FraudRiskLevel(str, Enum):
    """Fraud risk assessment levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FraudDetection(BaseModel):
    """Detection results for a transaction."""
    fraud_risk_level: FraudRiskLevel = Field(description="Assessed fraud risk level")
    fraud_indicators: List[str] = Field(description="Identified fraud indicators")
    alerts: List[str] = Field(description="Fraud alerts and recommendations")
    
class TransactionData(BaseModel):
    """Metadata for transaction data."""
    transaction_id: str = Field(description="Transaction ID")
    type: TransactionType = Field(description="Type of transaction")
    amount: float = Field(description="Transaction amount")
    currency: str = Field(description="Currency code")
    source: str = Field(description="Source account/entity")
    destination: str = Field(description="Destination account/entity")
    timestamp: datetime = Field(description="Transaction timestamp")
    description: Optional[str] = Field(description="Transaction description")
    
    def validate_amount(self) -> bool:
        """Validate transaction amount."""
        return self.amount > 0
    
    def validate_currency(self) -> bool:
        """Validate currency code."""
        VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}
        return self.currency in VALID_CURRENCIES

class FraudDetector:
    """Detect potential fraud using LangChain."""
    
    def __init__(self):
        """Initialize the fraud detector."""
        # Setup LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.0
        )
        
        self.parser = PydanticOutputParser(pydantic_object=FraudDetection)
        self.format_instructions = self.parser.get_format_instructions()
        
        # Setup analysis chain
        self.detector = self._create_detector()
    
    def _create_detector(self) -> Runnable:
        """Create the fraud detection chain."""
        template = """You are an expert fraud detector. Evaluate the following transaction and provide a fraud risk assessment.

TRANSACTION DETAILS
==================
Type: {type}
Amount: {amount} {currency}
Source: {source}
Destination: {destination}

DETECTION GUIDELINES
==================
Consider:
1. Transaction amount and patterns
2. Cross-border implications
3. Account identifiers and patterns
4. Known fraud patterns
5. Regulatory requirements

Fraud Risk Levels:
- HIGH: Immediate alert required
- MEDIUM: Monitor closely
- LOW: Standard processing

REQUIRED OUTPUT FORMAT
====================
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.parser
        
        return chain

    def _handle_error(self, e: Exception, transaction_id: str) -> Dict:
        """Create standardized error response."""
        error_info = {
            "transaction_id": transaction_id,
            "processed_at": datetime.now(),
            "result": {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "details": self._get_error_details(e)
                }
            },
        }
        
        if isinstance(e, ValidationError):
            error_info["validation_errors"] = e.errors()
            error_info["result"]["category"] = "validation"
            error_info["result"]["error"]["suggestion"] = "Check input data format and constraints"
        elif isinstance(e, ValueError):
            error_info["result"]["category"] = "validation"
            error_info["result"]["error"]["suggestion"] = "Ensure all values meet requirements"
        else:
            error_info["result"]["category"] = "system"
            error_info["result"]["error"]["suggestion"] = "Contact system administrator"
        
        return error_info

    def _get_error_details(self, error: Exception) -> Dict:
        """Extract detailed error information."""
        details = {
            "error_chain": [],
            "traceback": None
        }
        
        # Build error chain
        current = error
        while current is not None:
            details["error_chain"].append({
                "type": type(current).__name__,
                "message": str(current)
            })
            current = current.__cause__

        # Get traceback if available
        if hasattr(error, "__traceback__"):
            import traceback
            tb_lines = traceback.format_tb(error.__traceback__)
            details["traceback"] = "".join(tb_lines)

        return details

    
    def detect_fraud(self, transaction: TransactionData) -> Dict:
        """Detect fraud for a single transaction."""
        try:
            # Basic validation
            if not transaction.validate_amount():
                raise ValueError(f"Invalid transaction amount: {transaction.amount}")
            
            if not transaction.validate_currency():
                raise ValueError(f"Unsupported currency: {transaction.currency}")
            
            # Detect fraud
            detection = self.detector.invoke({
                "type": transaction.type,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "source": transaction.source,
                "destination": transaction.destination,
                "format_instructions": self.format_instructions
            })

            # Create result
            result = {
                "result": {
                    "success": True,
                    "processed_at": datetime.now(),
                    "fraud_risk_level": detection.fraud_risk_level,
                    "fraud_indicators": detection.fraud_indicators,
                    "alerts": detection.alerts
                },
                "transaction_id": transaction.transaction_id,
                "detection": detection.model_dump(),
                "details": transaction.model_dump()
            }

            return result
            
        except Exception as e:
            return self._handle_error(e, transaction.transaction_id)

def demonstrate_detector():
    """Demonstrate the fraud detector."""
    print("\nFraud Detector Demo")
    print("Using Azure deployment:", AZURE_DEPLOYMENT)
    print("=" * 50)
    print("\nInitializing Fraud Detector...")
    
    # Create sample transactions
    transactions = [
        TransactionData(
            transaction_id="TRX001",
            type=TransactionType.TRANSFER,
            amount=5000.00,
            currency="USD",
            source="account_123",
            destination="account_456",
            timestamp=datetime.now(),
            description="Regular monthly transfer"
        ),
        # High risk international transfer
        TransactionData(
            transaction_id="TRX002",
            type=TransactionType.TRANSFER,
            amount=75000.00,
            currency="EUR",
            source="account_789",
            destination="FOREIGN_ACC_456",
            timestamp=datetime.now(),
            description="Large international wire transfer"
        ),
        # Medium risk payment
        TransactionData(
            transaction_id="TRX003",
            type=TransactionType.PAYMENT,
            amount=15000.00,
            currency="USD",
            source="account_999",
            destination="vendor_account_888",
            timestamp=datetime.now(),
            description="Vendor payment - new beneficiary"
        ),
        TransactionData(
            transaction_id="TRX004",
            type=TransactionType.PAYMENT,
            amount=1500.00,
            currency="GBP",
            source="retail_acc_123",
            destination="merchant_456",
            timestamp=datetime.now(),
            description="Regular merchant payment"
        ),
        # Invalid transaction for error handling test
        TransactionData(
            transaction_id="TRX005",
            type=TransactionType.PAYMENT,
            amount=-100.00,  # Invalid amount
            currency="XYZ",  # Invalid currency
            source="acc_1",  # Too short
            destination="acc_1",  # Same as source
            timestamp=datetime.now(),
            description="Invalid transaction test"
        )
    ]
    
    print("\nPreparing test transactions...")
    print(f"\nPrepared {len(transactions)} test transactions")
    print("Including both valid and invalid cases for testing")
    print("\nTest Cases:")
    
    # Initialize detector
    try:
        detector = FraudDetector()
    except Exception as e:
        print("\nError initializing detector:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return
    
    print("\nDetector initialized successfully")
    
    # Detect fraud in transactions
    for idx, txn in enumerate(transactions, 1):
        print(f"\nTest Case {idx}/{len(transactions)}")
        print("=" * 30)
        print(f"Transaction ID: {txn.transaction_id}")
        print(f"Type: {txn.type}")
        print(f"Amount: {txn.amount} {txn.currency}")
        print(f"From: {txn.source}")
        print(f"To: {txn.destination}")
        print("-" * 30 + "\n")

        result = detector.detect_fraud(txn)
        
        print("\nResult Details:")
        print("----------------")
        print(json.dumps(result, indent=2, cls=DateTimeEncoder))
        
        # Print summary
        if result.get("result", {}).get("success", False):
            fraud_risk_level = result.get("result", {}).get("fraud_risk_level", "unknown")
            fraud_indicators = result.get("result", {}).get("fraud_indicators", [])
            alerts = result.get("result", {}).get("alerts", [])
            
            print("\nStatus: ✅ Success")
            print(f"Fraud Risk Level: {fraud_risk_level.upper()}")
            print(f"Fraud Indicators: {', '.join(fraud_indicators) if fraud_indicators else 'None identified'}")
            print(f"Alerts: {', '.join(alerts) if alerts else 'None provided'}")
        else:
            error = result.get("result", {}).get("error", {})
            category = result.get("result", {}).get("category", "unknown")
            print("\nStatus: ❌ Failed")
            print(f"Error Type: {error.get('type', 'Unknown error')}")
            print(f"Message: {error.get('message', 'No error message')}")
            print(f"Category: {category.upper()}")
        print("=" * 50)

if __name__ == "__main__":
    demonstrate_detector()
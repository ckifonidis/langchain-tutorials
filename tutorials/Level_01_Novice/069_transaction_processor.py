#!/usr/bin/env python3
"""
LangChain Transaction Processor (LangChain v3)

This example demonstrates smart transaction validation and processing using LCEL and 
the Runnable interface. It provides financial transaction validation and routing
for banking applications.

Key concepts demonstrated:
1. LCEL: Building composable chains for transaction analysis
2. Runnable interface: Creating reusable transaction processors
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

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

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")
print("\nChecking Azure OpenAI configuration...")

from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class TransactionType(str, Enum):
    """Types of financial transactions."""
    TRANSFER = "transfer"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"

class RiskLevel(str, Enum):
    """Transaction risk assessment levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TransactionAnalysis(BaseModel):
    """Analysis results for a transaction."""
    risk_level: RiskLevel = Field(description="Assessed risk level")
    risk_factors: List[str] = Field(description="Identified risk factors")
    requires_review: bool = Field(description="Whether manual review is needed")
    routing: str = Field(description="Suggested processing route")
    
class Transaction(BaseModel):
    """Financial transaction details."""
    id: str = Field(description="Transaction ID")
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

class TransactionProcessor:
    """Process and validate financial transactions using LangChain."""
    
    def __init__(self):
        """Initialize the transaction processor."""
        # Setup LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.0
        )

        self.trace_enabled = True
    
    def _log(self, message: str, level: str = "info"):
        """Log a message if tracing is enabled."""
        if self.trace_enabled:
            timestamp = datetime.now().isoformat()
            if level == "warning":
                print(f"\nWARNING [{timestamp}]: {message}")
            else:
                print(f"\n[{timestamp}] {message}")
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=TransactionAnalysis)
        self.format_instructions = self.parser.get_format_instructions()
        
        # Setup analysis chain
        self.analyzer = self._create_analyzer()
    
    def _create_analyzer(self) -> Runnable:
        """Create the transaction analysis chain."""
        template = """You are an expert financial transaction analyzer. Analyze the following transaction for risks and provide routing recommendations.

TRANSACTION DETAILS
==================
Type: {type}
Amount: {amount} {currency}
Source: {source}
Destination: {destination}

ANALYSIS GUIDELINES
==================
Consider:
1. Transaction amount and patterns
2. Cross-border implications
3. Account identifiers and patterns
4. Known fraud patterns
5. Regulatory requirements

Risk Levels:
- HIGH: Requires immediate manual review
- MEDIUM: Enhanced monitoring needed
- LOW: Standard processing acceptable

Routing Options:
- manual_review: For high-risk transactions
- enhanced_verification: For medium-risk transactions
- standard_processing: For low-risk transactions
- blocked: For suspicious or invalid transactions

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
            self._log(f"Unexpected error type: {type(e).__name__}", "warning")
        
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

    
    def process_transaction(self, transaction: Transaction) -> Dict:
        """Process a single transaction."""
        try:
            # Basic validation
            if not transaction.validate_amount():
                raise ValueError(f"Invalid transaction amount: {transaction.amount}")
            
            if not transaction.validate_currency():
                raise ValueError(f"Unsupported currency: {transaction.currency}")
            
            if transaction.source == transaction.destination:
                raise ValueError("Source and destination accounts must be different")
            
            if len(transaction.source) < 5 or len(transaction.destination) < 5:
                raise ValueError("Account identifiers must be at least 5 characters")
            
            self._log(f"Processing {transaction.type} transaction")
            self._log(f"Amount: {transaction.amount} {transaction.currency}")

            # High-value transaction warning
            HIGH_AMOUNT_THRESHOLD = 10000.00
            if transaction.amount > HIGH_AMOUNT_THRESHOLD:
                self._log(
                    f"High-value transaction detected: {transaction.amount} {transaction.currency}",
                    "warning")

            # Analyze transaction
            analysis = self.analyzer.invoke({
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
                    "status": "processed" if analysis.risk_level != "high" else "review_required",
                    "risk_level": analysis.risk_level,
                    "requires_review": analysis.requires_review,
                    "routing": analysis.routing
                },
                "transaction_id": transaction.id,
                "analysis": analysis.model_dump(),
                "details": transaction.model_dump()
            }

            # Add risk warning if needed
            if analysis.risk_level == "high":
                result["warning"] = (
                    "HIGH RISK TRANSACTION - Manual review required\n"
                    f"Risk factors:\n" +
                    "\n".join(f"- {factor}" for factor in analysis.risk_factors)
                )

            self._log(f"Successfully processed transaction {transaction.id}")
            return result
            
        except Exception as e:
            return self._handle_error(e, transaction.id)

def demonstrate_processor():
    """Demonstrate the transaction processor."""
    print("\nTransaction Processor Demo")
    print("Using Azure deployment:", AZURE_DEPLOYMENT)
    print("=" * 50)
    print("\nInitializing Transaction Processor...")
    
    # Create sample transactions
    transactions = [
        Transaction(
            id="TRX001",
            type=TransactionType.TRANSFER,
            amount=5000.00,
            currency="USD",
            source="account_123",
            destination="account_456",
            timestamp=datetime.now(),
            description="Regular monthly transfer"
        ),
        # High risk international transfer
        Transaction(
            id="TRX002",
            type=TransactionType.TRANSFER,
            amount=75000.00,
            currency="EUR",
            source="account_789",
            destination="FOREIGN_ACC_456",
            timestamp=datetime.now(),
            description="Large international wire transfer"
        ),
        # Medium risk payment
        Transaction(
            id="TRX003",
            type=TransactionType.PAYMENT,
            amount=15000.00,
            currency="USD",
            source="account_999",
            destination="vendor_account_888",
            timestamp=datetime.now(),
            description="Vendor payment - new beneficiary"
        ),
        Transaction(
            id="TRX004",
            type=TransactionType.PAYMENT,
            amount=1500.00,
            currency="GBP",
            source="retail_acc_123",
            destination="merchant_456",
            timestamp=datetime.now(),
            description="Regular merchant payment"
        ),
        # Invalid transaction for error handling test
        Transaction(
            id="TRX005",
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
    
    # Initialize processor
    try:
        processor = TransactionProcessor()  # Initialize with tracing enabled
    except Exception as e:
        print("\nError initializing processor:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return
    
    print("\nProcessor initialized successfully")
    
    # Process transactions
    for idx, txn in enumerate(transactions, 1):
        print(f"\nTest Case {idx}/{len(transactions)}")
        print("=" * 30)
        print(f"Transaction ID: {txn.id}")
        print(f"Type: {txn.type}")
        print(f"Amount: {txn.amount} {txn.currency}")
        print(f"From: {txn.source}")
        print(f"To: {txn.destination}")
        print("-" * 30 + "\n")

        result = processor.process_transaction(txn)
        
        print("\nResult Details:")
        print("----------------")
        print(json.dumps(result, indent=2, cls=DateTimeEncoder))
        
        # Print summary
        if result.get("result", {}).get("success", False):
            risk_level = result.get("result", {}).get("risk_level", "unknown")
            status = result.get("result", {}).get("status", "unknown")
            routing = result.get("result", {}).get("routing", "unknown")
            
            print("\nStatus: ✅ Success")
            print(f"Processing Status: {status.upper()}")
            print(f"Risk Level: {risk_level.upper()}")
            print(f"Routing Decision: {routing}")
            
            risk_factors = result.get("analysis", {}).get("risk_factors", [])
            print(f"\nRisk Factors ({len(risk_factors)}):", *[f"\n- {factor}" for factor in risk_factors] if risk_factors else "\nNone identified")
            if result.get('warning'):
                print("\nWarnings:")
                print(f"⚠️  {result['warning']}")
        else:
            error = result.get("result", {}).get("error", {})
            category = result.get("result", {}).get("category", "unknown")
            print("\nStatus: ❌ Failed")
            print(f"Error Type: {error.get('type', 'Unknown error')}")
            print(f"Message: {error.get('message', 'No error message')}")
            print(f"Category: {category.upper()}")
        print("=" * 50)

if __name__ == "__main__":
    demonstrate_processor()
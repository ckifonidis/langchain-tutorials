#!/usr/bin/env python3
"""
LangChain Transaction Validator (099) (LangChain v3)

This example demonstrates a transaction validation system using three key concepts:
1. Few Shot Prompting: Guide validation decisions
2. Key Methods: Flexible processing patterns
3. Output Parsers: Structured validation results

It provides accurate and consistent transaction validation for banking applications.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define validation result model
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Indicates if the transaction is valid")
    confidence_score: float = Field(description="Confidence score of the validation")
    risk_flags: List[str] = Field(description="List of identified risk flags")
    recommendation: str = Field(description="Recommendation for handling the transaction")

# Define transaction model
class Transaction(BaseModel):
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    type: str = Field(description="Type of transaction")
    timestamp: str = Field(description="Transaction timestamp")
    details: Dict[str, str] = Field(description="Additional transaction details")

class TransactionValidator:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ValidationResult)

    async def validate(self, transaction: Transaction) -> ValidationResult:
        """Validate a transaction using invoke pattern."""
        try:
            # Create messages for validation
            messages = [
                SystemMessage(content=f"""You are a transaction validator. Analyze the transaction and provide a structured validation result.
                For high-risk transactions (large amounts, unusual locations, or patterns), set is_valid to false and list appropriate risk flags.
                
                Format your response exactly as shown:
                {self.output_parser.get_format_instructions()}"""),
                HumanMessage(content=f"""Please validate this transaction:
                Amount: ${transaction.amount:.2f}
                Type: {transaction.type}
                Details: {transaction.details}

                Your response must follow the specified JSON format.""")
            ]
            
            # Get model's response
            response = await self.llm.ainvoke(messages)
            
            # Parse the response
            return self.output_parser.parse(response.content)
            
        except Exception as e:
            # Return a default validation result on error
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                risk_flags=["Error during validation"],
                recommendation=f"Manual review required - {str(e)}"
            )

async def demonstrate_validation():
    print("\nTransaction Validator Demo")
    print("=========================\n")

    validator = TransactionValidator()

    # Test transactions
    transactions = [
        Transaction(
            transaction_id="tx_001",
            amount=750.00,
            type="transfer",
            timestamp=datetime.now().isoformat(),
            details={"recipient": "Jane Smith", "purpose": "Consulting fee"}
        ),
        Transaction(
            transaction_id="tx_002",
            amount=15000.00,
            type="withdrawal",
            timestamp=datetime.now().isoformat(),
            details={"location": "Foreign IP", "device": "New device"}
        )
    ]

    for transaction in transactions:
        print(f"Validating transaction: {transaction.transaction_id}")
        print(f"Amount: ${transaction.amount:.2f}")
        print(f"Type: {transaction.type}")
        print(f"Details: {transaction.details}")
        
        result = await validator.validate(transaction)
        
        print("\nValidation Result:")
        print(f"Valid: {result.is_valid}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Risk Flags: {', '.join(result.risk_flags) if result.risk_flags else 'None'}")
        print(f"Recommendation: {result.recommendation}\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_validation())
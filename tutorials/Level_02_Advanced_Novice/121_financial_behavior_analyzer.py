#!/usr/bin/env python3
"""
Financial Behavior Analyzer (121) (LangChain v3)

This example demonstrates customer financial behavior analysis using:
1. Document Processing: Transaction data handling
2. Pattern Analysis: Behavior detection
3. Report Generation: Clear insights

It helps data science teams understand and predict customer financial patterns.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TransactionType(str, Enum):
    """Types of financial transactions."""
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    INVESTMENT = "investment"

class TransactionCategory(str, Enum):
    """Transaction categories."""
    RETAIL = "retail"
    DINING = "dining"
    TRAVEL = "travel"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    SAVINGS = "savings"
    INVESTMENT = "investment"

class Transaction(BaseModel):
    """Individual transaction details."""
    timestamp: str = Field(description="Transaction time")
    type: TransactionType = Field(description="Transaction type")
    category: TransactionCategory = Field(description="Category")
    amount: float = Field(description="Amount")
    merchant: str = Field(description="Merchant name")
    location: str = Field(description="Transaction location")

class CustomerProfile(BaseModel):
    """Customer financial profile."""
    customer_id: str = Field(description="Customer ID")
    age_group: str = Field(description="Age group")
    income_bracket: str = Field(description="Income level")
    transactions: List[Transaction] = Field(description="Transaction history")
    metadata: Dict = Field(default_factory=dict)

class BehaviorAnalyzer:
    """Financial behavior analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting behavior analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Create analysis chain
        analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial behavior analyst. Given customer transaction data:
1. Identify spending patterns
2. Detect financial behaviors
3. Predict future trends
4. Suggest personalized recommendations

Your analysis should include:

1. Spending Analysis
- Category breakdown
- Frequency patterns
- Amount distributions
- Location insights

2. Behavioral Patterns
- Regular behaviors
- Unusual activities
- Risk indicators
- Savings habits

3. Financial Health
- Income utilization
- Savings rate
- Investment activity
- Risk exposure

4. Recommendations
- Saving opportunities
- Investment options
- Risk management
- Financial goals

Format with clear sections and insights."""),
            ("human", """Analyze this customer profile:

Customer ID: {customer_id}
Age Group: {age_group}
Income Level: {income_bracket}

Transactions:
{transactions}

Provide a detailed analysis.""")
        ])
        
        self.chain = (
            {"customer_id": RunnablePassthrough(), 
             "age_group": RunnablePassthrough(),
             "income_bracket": RunnablePassthrough(),
             "transactions": RunnablePassthrough()} 
            | analyzer_prompt 
            | self.llm 
            | StrOutputParser()
        )
        logger.info("Analysis chain ready")

    def format_transactions(self, transactions: List[Transaction]) -> str:
        """Format transactions for analysis."""
        lines = []
        for tx in transactions:
            lines.append(
                f"[{tx.timestamp}] {tx.type} - {tx.category}"
                f"\n  Amount: ${tx.amount:.2f}"
                f"\n  Merchant: {tx.merchant}"
                f"\n  Location: {tx.location}\n"
            )
        return "\n".join(lines)

    async def analyze_behavior(self, profile: CustomerProfile) -> str:
        """Analyze customer financial behavior."""
        logger.info(f"Analyzing customer: {profile.customer_id}")
        
        try:
            # Run analysis
            result = await self.chain.ainvoke({
                "customer_id": profile.customer_id,
                "age_group": profile.age_group,
                "income_bracket": profile.income_bracket,
                "transactions": self.format_transactions(profile.transactions)
            })
            logger.info("Analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting financial behavior demo...")
    
    try:
        # Create analyzer
        analyzer = BehaviorAnalyzer()
        
        # Example customer
        customer = CustomerProfile(
            customer_id="CUST001",
            age_group="25-34",
            income_bracket="75K-100K",
            transactions=[
                Transaction(
                    timestamp="2025-04-01 09:15",
                    type=TransactionType.DEPOSIT,
                    category=TransactionCategory.SAVINGS,
                    amount=4500.00,
                    merchant="Direct Deposit",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-02 10:30",
                    type=TransactionType.PAYMENT,
                    category=TransactionCategory.UTILITIES,
                    amount=150.00,
                    merchant="City Power Co",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-02 12:45",
                    type=TransactionType.PURCHASE,
                    category=TransactionCategory.DINING,
                    amount=25.50,
                    merchant="Lunch Spot",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-02 17:30",
                    type=TransactionType.TRANSFER,
                    category=TransactionCategory.SAVINGS,
                    amount=1000.00,
                    merchant="Savings Account",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-03 09:15",
                    type=TransactionType.INVESTMENT,
                    category=TransactionCategory.INVESTMENT,
                    amount=500.00,
                    merchant="Trading Account",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-03 15:20",
                    type=TransactionType.PURCHASE,
                    category=TransactionCategory.RETAIL,
                    amount=89.99,
                    merchant="Fashion Store",
                    location="NYC"
                ),
                Transaction(
                    timestamp="2025-04-03 20:15",
                    type=TransactionType.PURCHASE,
                    category=TransactionCategory.ENTERTAINMENT,
                    amount=45.00,
                    merchant="Cinema World",
                    location="NYC"
                )
            ]
        )
        
        print("\nAnalyzing Financial Behavior")
        print("==========================")
        print(f"Customer ID: {customer.customer_id}")
        print(f"Age Group: {customer.age_group}")
        print(f"Income Bracket: {customer.income_bracket}")
        print(f"Transactions: {len(customer.transactions)}\n")
        
        try:
            # Get analysis
            result = await analyzer.analyze_behavior(customer)
            print("\nBehavior Analysis:")
            print("=================")
            print(result)
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
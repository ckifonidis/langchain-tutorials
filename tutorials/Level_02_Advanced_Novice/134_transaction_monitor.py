#!/usr/bin/env python3
"""
Transaction Monitor (134) (LangChain v3)

This example demonstrates transaction monitoring using:
1. Sequential Chains: Multi-step analysis
2. Output Parsers: Structured results
3. Chat History: Context tracking

It helps compliance teams monitor banking transactions.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TransactionType(str, Enum):
    """Transaction types."""
    WIRE = "wire_transfer"
    ACH = "ach_transfer"
    CARD = "card_payment"
    CASH = "cash_deposit"
    CHECK = "check_payment"
    CRYPTO = "crypto_transfer"

class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_risk"

class Transaction(BaseModel):
    """Transaction details."""
    transaction_id: str = Field(description="Transaction ID")
    type: TransactionType = Field(description="Transaction type")
    amount: float = Field(description="Transaction amount")
    currency: str = Field(description="Currency code")
    country: str = Field(description="Country code")
    metrics: Dict[str, float] = Field(description="Risk metrics")
    metadata: Dict = Field(default_factory=dict)

class TransactionMonitor:
    """Transaction monitoring system."""

    def __init__(self):
        """Initialize monitor."""
        logger.info("Starting transaction monitor...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup memory
        self.memory = ChatMessageHistory()
        logger.info("Chat memory ready")
        
        # Setup risk chain
        self.risk = ChatPromptTemplate.from_messages([
            ("system", """You are a transaction risk analyst.
Review transactions and identify potential risks.

Format your response exactly like this:

RISK ASSESSMENT
-------------
Transaction: ID
Type: Category
Risk Level: HIGH/MEDIUM/LOW

Risk Factors:
1. Factor Name
   Level: Risk level
   Reason: Description
   Action: Required step

2. Factor Name
   Level: Risk level
   Reason: Description
   Action: Required step

Required Actions:
1. Action item
2. Action item

Consider:
- Amount patterns
- Location risks
- Type patterns
- Time factors"""),
            ("human", """Assess this transaction:

ID: {transaction_id}
Type: {type}
Amount: {amount} {currency}
Country: {country}

Risk Metrics:
{metrics}

Previous Analysis:
{history}

Provide a complete risk assessment.""")
        ])
        
        # Setup fraud chain
        self.fraud = ChatPromptTemplate.from_messages([
            ("system", """Review transaction patterns for fraud.

Format your response like this:

FRAUD ANALYSIS
------------
Status: ALERT/CLEAR
Level: HIGH/MEDIUM/LOW

Patterns Found:
1. Pattern Name
   - Description
   - Indicators
   - Similar cases

2. Pattern Name
   - Description
   - Indicators
   - Similar cases

Required Actions:
1. Action step
2. Action step"""),
            ("human", """Review for fraud:
Transaction: {transaction_id}
Risk Level: {risk_level}

Assessment:
{risk_details}

Provide fraud analysis.""")
        ])
        
        # Setup parser
        self.parser = StrOutputParser()
        logger.info("Analysis chains ready")

    def add_analysis(self, analysis: str) -> None:
        """Add analysis to history."""
        self.memory.add_user_message(analysis)
        logger.debug("Added to history")

    def get_history(self) -> List[str]:
        """Get analysis history."""
        messages = self.memory.messages
        return [msg.content for msg in messages[-5:]]

    async def analyze_transaction(self, transaction: Transaction) -> str:
        """Analyze transaction risk."""
        logger.info(f"Analyzing transaction: {transaction.transaction_id}")
        
        try:
            # Format metrics
            metrics = "\n".join(
                f"{k}: {v:.2f}" 
                for k, v in transaction.metrics.items()
            )
            
            # Get history
            history = "\n".join(self.get_history())
            
            # Get risk assessment
            messages = self.risk.format_messages(
                transaction_id=transaction.transaction_id,
                type=transaction.type.value,
                amount=transaction.amount,
                currency=transaction.currency,
                country=transaction.country,
                metrics=metrics,
                history=history
            )
            
            response = await self.llm.ainvoke(messages)
            risk_result = self.parser.parse(response.content)
            logger.debug("Risk assessment complete")
            
            # Add to history
            self.add_analysis(risk_result)
            
            # Get fraud analysis
            messages = self.fraud.format_messages(
                transaction_id=transaction.transaction_id,
                risk_level="HIGH",  # Example level
                risk_details=risk_result
            )
            
            response = await self.llm.ainvoke(messages)
            fraud_result = self.parser.parse(response.content)
            logger.info("Analysis complete")
            
            return f"{risk_result}\n\n{fraud_result}"
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting monitoring demo...")
    
    try:
        # Create monitor
        monitor = TransactionMonitor()
        
        # Example transaction
        transaction = Transaction(
            transaction_id="TXN-2025-001",
            type=TransactionType.WIRE,
            amount=50000.00,
            currency="USD",
            country="RO",
            metrics={
                "velocity": 3.5,
                "frequency": 2.0,
                "pattern_match": 0.85,
                "location_risk": 0.65,
                "amount_variance": 2.5,
                "profile_match": 0.55
            }
        )
        
        print("\nAnalyzing Transaction")
        print("===================")
        print(f"Transaction: {transaction.transaction_id}")
        print(f"Type: {transaction.type.value}")
        print(f"Amount: {transaction.currency} {transaction.amount:,.2f}")
        print(f"Country: {transaction.country}\n")
        
        print("Risk Metrics:")
        for name, value in transaction.metrics.items():
            print(f"{name}: {value:.2f}")
        
        try:
            # Get analysis
            result = await monitor.analyze_transaction(transaction)
            print("\nAnalysis Results:")
            print("================")
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
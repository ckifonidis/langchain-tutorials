#!/usr/bin/env python3
"""
Customer Segmenter (125) (LangChain v3)

This example demonstrates banking customer analysis using:
1. Chat Models: Profile evaluation and segmentation
2. Message Templates: Structured prompts and examples
3. RunnablePassthrough: Data mapping and flow control

It helps marketing teams understand and target customer segments.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
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

class ProductType(str, Enum):
    """Banking products."""
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit_card"
    LOAN = "personal_loan"
    MORTGAGE = "mortgage"
    INVESTMENT = "investment"

class CustomerProfile(BaseModel):
    """Customer profile details."""
    customer_id: str = Field(description="Customer ID")
    age: int = Field(description="Customer age")
    income: float = Field(description="Annual income")
    products: List[ProductType] = Field(description="Active products")
    balance: Dict[str, float] = Field(description="Account balances")
    transactions: int = Field(description="Monthly transactions")
    digital: bool = Field(description="Digital banking user")
    tenure: int = Field(description="Months with bank")

class SegmentReport(BaseModel):
    """Segment analysis report."""
    segment: str = Field(description="Segment name")
    description: str = Field(description="Segment details")
    value_score: int = Field(description="Value score (0-100)")
    products: List[str] = Field(description="Product matches")
    channels: List[str] = Field(description="Best channels")
    actions: List[str] = Field(description="Next actions")

class CustomerSegmenter:
    """Customer segmentation system."""

    def __init__(self):
        """Initialize segmenter."""
        logger.info("Starting customer segmenter...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup parser
        self.parser = StrOutputParser()
        
        # Setup analysis prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a customer segmentation analyst.
Analyze customer profiles and provide segmentation details.

Format your response exactly like this example:

CUSTOMER SEGMENT
---------------
Segment: segment_name
Description: Brief description

Value Analysis:
Score: 85/100
- Key value factor
- Key value factor

Product Matches:
- Product name (reason)
- Product name (reason)

Best Channels:
- Channel (why)
- Channel (why)

Next Actions:
1. Action detail
2. Action detail"""),
            ("human", """Analyze this customer:

ID: {customer_id}
Age: {age}
Income: ${income:,.2f}
Products: {products}
Balances: {balances}
Transactions: {transactions}/month
Digital: {digital}
Tenure: {tenure} months

Provide a complete segment analysis.""")
        ])
        logger.info("Analysis chain ready")

    async def analyze_customer(self, profile: CustomerProfile) -> str:
        """Analyze a customer profile."""
        logger.info(f"Analyzing customer: {profile.customer_id}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                customer_id=profile.customer_id,
                age=profile.age,
                income=profile.income,
                products=", ".join(p.value for p in profile.products),
                balances=", ".join(f"{k}: ${v:,.2f}" for k, v in profile.balance.items()),
                transactions=profile.transactions,
                digital=profile.digital,
                tenure=profile.tenure
            )
            logger.debug("Request formatted")
            
            # Get analysis
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting segmentation demo...")
    
    try:
        # Create segmenter
        segmenter = CustomerSegmenter()
        
        # Example customer
        customer = CustomerProfile(
            customer_id="CUST001",
            age=32,
            income=95000.00,
            products=[
                ProductType.CHECKING,
                ProductType.SAVINGS,
                ProductType.CREDIT,
                ProductType.INVESTMENT
            ],
            balance={
                "checking": 3500.00,
                "savings": 25000.00,
                "investment": 50000.00
            },
            transactions=45,
            digital=True,
            tenure=48
        )
        
        print("\nAnalyzing Customer")
        print("=================")
        print(f"Customer: {customer.customer_id}")
        print(f"Age: {customer.age}")
        print(f"Income: ${customer.income:,.2f}")
        print(f"Products: {len(customer.products)}")
        print(f"Digital: {'Yes' if customer.digital else 'No'}\n")
        
        try:
            # Get analysis
            result = await segmenter.analyze_customer(customer)
            print("\nSegmentation Results:")
            print("====================")
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
#!/usr/bin/env python3
"""
Banking Risk Checker (128) (LangChain v3)

This example demonstrates risk assessment using:
1. String Output: Simple text parsing
2. Message Control: Clear prompts
3. Error Recovery: Step-by-step handling

It helps risk teams assess banking operations.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCase(BaseModel):
    """Risk case details."""
    case_id: str = Field(description="Case identifier")
    title: str = Field(description="Case title")
    details: str = Field(description="Case details")
    context: Dict = Field(default_factory=dict)

class RiskChecker:
    """Banking risk assessment system."""

    def __init__(self):
        """Initialize checker."""
        logger.info("Starting risk checker...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup output parser
        self.parser = StrOutputParser()
        logger.info("Output parser ready")
        
        # Setup analysis prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking risk analyst. Review cases and provide structured assessments.

Format your response exactly like this example:

RISK ASSESSMENT
--------------
Overall Risk: MEDIUM

Key Factors:
1. Credit History
   Level: Medium
   Issue: Limited history available
   Action: Request additional records

2. Cash Flow
   Level: High
   Issue: Irregular patterns
   Action: Monitor monthly trends

Required Steps:
1. Step one details
2. Step two details
3. Step three details

Next Review: YYYY-MM-DD

Use clear sections and bullet points."""),
            ("human", """Analyze this case:

ID: {case_id}
Title: {title}

Details:
{details}

Provide a complete risk assessment.""")
        ])
        logger.info("Analysis prompt ready")

    async def check_case(self, case: RiskCase) -> str:
        """Check a risk case."""
        logger.info(f"Checking case: {case.case_id}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                case_id=case.case_id,
                title=case.title,
                details=case.details
            )
            logger.debug("Request formatted")
            
            # Get assessment
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Assessment complete")
            return result
            
        except Exception as e:
            logger.error(f"Assessment failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting risk assessment demo...")
    
    try:
        # Create checker
        checker = RiskChecker()
        
        # Example case
        case = RiskCase(
            case_id="RISK-2025-001",
            title="Startup Credit Line Increase",
            details="""Application Details:
Current Status:
- Tech startup, 3 years old
- Current credit line: $100,000
- Utilization: 85%
- Payment history: Good
- Request: Increase to $250,000

Financial Position:
- Annual revenue: $2M
- Growth rate: 40% YoY
- Profit margin: 15%
- Cash reserves: $150,000
- Monthly burn: $100,000

Business Health:
- Customer base: Growing
- Market share: 5%
- Industry outlook: Positive
- Competition: High
- Product maturity: Medium

Risk Indicators:
- High utilization
- Fast growth
- High burn rate
- Limited history
- Market competition

Positive Factors:
- Strong growth
- Good payments
- Solid margins
- Market potential
- Team experience"""
        )
        
        print("\nProcessing Risk Case")
        print("===================")
        print(f"Case: {case.case_id}")
        print(f"Title: {case.title}\n")
        
        try:
            # Get assessment
            result = await checker.check_case(case)
            print("\nAssessment Results:")
            print("==================")
            print(result)
            
        except Exception as e:
            print(f"\nAssessment failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
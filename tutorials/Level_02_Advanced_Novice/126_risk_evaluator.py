#!/usr/bin/env python3
"""
Risk Evaluator (126) (LangChain v3)

This example demonstrates banking risk evaluation using:
1. Chat Models: Case analysis and scoring
2. Prompt Templates: Evaluation format
3. Output Parsing: Structured results

It helps risk teams evaluate banking operations.
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

class RiskType(str, Enum):
    """Banking risk types."""
    CREDIT = "credit_risk"
    MARKET = "market_risk"
    OPERATIONAL = "operational_risk"
    LIQUIDITY = "liquidity_risk"
    COMPLIANCE = "compliance_risk"

class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCase(BaseModel):
    """Risk evaluation case."""
    case_id: str = Field(description="Case ID")
    type: RiskType = Field(description="Risk type")
    details: str = Field(description="Case details")
    context: Dict = Field(default_factory=dict)

class RiskEvaluator:
    """Banking risk evaluation system."""

    def __init__(self):
        """Initialize evaluator."""
        logger.info("Starting risk evaluator...")
        
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
        
        # Setup evaluation prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking risk analyst.
Evaluate cases and provide structured assessment.

Format your response exactly like this:

RISK EVALUATION
-------------
Overall Risk: HIGH/MEDIUM/LOW
Risk Score: 85/100

Analysis:
Brief summary of key findings and major concerns.

Risk Factors:
1. Factor Name
   Level: HIGH/MEDIUM/LOW
   Impact: Description of potential impact
   Action: Required mitigation steps

2. Factor Name
   Level: HIGH/MEDIUM/LOW
   Impact: Description of potential impact
   Action: Required mitigation steps

Required Steps:
1. Action step with timeline
2. Action step with timeline
3. Action step with timeline"""),
            ("human", """Evaluate this case:

Case: {case_id}
Type: {risk_type}
Details:
{details}

Provide a complete risk assessment.""")
        ])
        logger.info("Evaluation prompt ready")

    async def evaluate_case(self, case: RiskCase) -> str:
        """Evaluate a risk case."""
        logger.info(f"Evaluating case: {case.case_id}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                case_id=case.case_id,
                risk_type=case.type.value,
                details=case.details
            )
            logger.debug("Request formatted")
            
            # Get evaluation
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Evaluation complete")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting risk evaluation demo...")
    
    try:
        # Create evaluator
        evaluator = RiskEvaluator()
        
        # Example case
        case = RiskCase(
            case_id="RISK-2025-001",
            type=RiskType.CREDIT,
            details="""Small Business Loan Review

Business Profile:
- Tech startup, 2 years old
- Monthly revenue: $50,000
- Growth rate: 15% monthly
- Burn rate: $40,000/month
- Cash reserves: $100,000

Loan Details:
- Amount: $500,000
- Purpose: Market expansion
- Term: 36 months
- Interest: 8% APR

Risk Factors:
- Limited operating history
- High burn rate
- B2B concentration risk
- Market competition high
- No hard assets

Mitigants:
- Strong growth trend
- Good margins (80%)
- Key contracts in place
- Experienced team
- Angel funding history"""
        )
        
        print("\nEvaluating Risk Case")
        print("===================")
        print(f"Case: {case.case_id}")
        print(f"Type: {case.type.value}\n")
        
        try:
            # Get evaluation
            result = await evaluator.evaluate_case(case)
            print("\nRisk Assessment:")
            print("===============")
            print(result)
            
        except Exception as e:
            print(f"\nEvaluation failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
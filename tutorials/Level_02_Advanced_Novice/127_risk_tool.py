#!/usr/bin/env python3
"""
Banking Risk Tool (127) (LangChain v3)

This example demonstrates risk assessment using:
1. Structured Output: Pydantic parsing
2. Response Formatting: Output templates
3. Error Handling: Clean recovery

It helps risk teams assess banking operations.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
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

class RiskFactor(BaseModel):
    """Risk factor analysis."""
    name: str = Field(description="Factor name")
    level: RiskLevel = Field(description="Risk level")
    details: str = Field(description="Risk details")
    actions: List[str] = Field(description="Required actions")

class RiskReport(BaseModel):
    """Risk assessment report."""
    case_id: str = Field(description="Case identifier")
    overview: str = Field(description="Brief overview")
    factors: List[RiskFactor] = Field(description="Risk factors found")
    overall_risk: RiskLevel = Field(description="Overall risk level")
    next_steps: List[str] = Field(description="Required steps")
    review_date: str = Field(description="Next review date")

class RiskCase(BaseModel):
    """Case details for analysis."""
    case_id: str = Field(description="Case identifier")
    summary: str = Field(description="Case summary")
    details: str = Field(description="Full details")
    metadata: Dict = Field(default_factory=dict)

class RiskAnalyzer:
    """Banking risk analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting risk analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup parser and prompt
        self.parser = PydanticOutputParser(pydantic_object=RiskReport)
        
        # Get format instructions
        format_instructions = self.parser.get_format_instructions()
        logger.debug(f"Format instructions:\n{format_instructions}")
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking risk analyst.
Analyze risks and provide structured assessments.

Use clear language and specific examples.
Focus on actionable insights and practical steps.
Format as Pydantic JSON with proper types.

Instructions:
{format_instructions}"""),
            ("human", """Analyze this case:

Case ID: {case_id}
Summary: {summary}

Details:
{details}

Provide a complete risk assessment.""")
        ])
        logger.info("Analysis chain ready")

    async def analyze_case(self, case: RiskCase) -> RiskReport:
        """Analyze a risk case."""
        logger.info(f"Analyzing case: {case.case_id}")
        
        try:
            # Format request
            logger.debug("Formatting request...")
            messages = self.prompt.format(
                format_instructions=self.parser.get_format_instructions(),
                case_id=case.case_id,
                summary=case.summary,
                details=case.details
            )
            
            # Get analysis
            logger.debug("Getting analysis...")
            response = await self.llm.ainvoke(messages)
            logger.debug(f"Raw response:\n{response.content}")
            
            # Parse result
            logger.debug("Parsing response...")
            result = self.parser.parse(response.content)
            logger.info(f"Analysis complete - Risk: {result.overall_risk}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting risk analysis demo...")
    
    try:
        # Create analyzer
        analyzer = RiskAnalyzer()
        
        # Example case
        case = RiskCase(
            case_id="RISK-2025-001",
            summary="Small business loan default risk",
            details="""Business Profile:
- Tech startup, 2 years old 
- B2B SaaS product
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
        
        print("\nAnalyzing Risk Case")
        print("=================")
        print(f"Case: {case.case_id}")
        print(f"Summary: {case.summary}\n")
        
        try:
            # Get analysis
            result = await analyzer.analyze_case(case)
            
            # Show results
            print("\nRisk Assessment:")
            print("===============")
            print(f"Overview: {result.overview}")
            print(f"Overall Risk: {result.overall_risk.upper()}\n")
            
            print("Risk Factors:")
            for i, factor in enumerate(result.factors, 1):
                print(f"\n{i}. {factor.name}")
                print(f"   {'❗' if factor.level in ['high', 'critical'] else '⚠️'} Level: {factor.level}")
                print(f"   Details: {factor.details}")
                print("\n   Actions:")
                for action in factor.actions:
                    print(f"   - {action}")
            
            print("\nNext Steps:")
            for i, step in enumerate(result.next_steps, 1):
                print(f"{i}. {step}")
                
            print(f"\nNext Review: {result.review_date}")
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
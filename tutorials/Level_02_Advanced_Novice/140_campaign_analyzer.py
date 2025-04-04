#!/usr/bin/env python3
"""
Marketing Campaign Analysis (140) (LangChain v3)

This example demonstrates:
1. Text Generation: Chat model for analysis
2. Structured Output: JSON response formatting
3. Chain Composition: Sequential processing

It helps marketing teams assess campaign performance.
"""

import os
import json
import logging
from typing import Dict, List
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CampaignType(str, Enum):
    """Marketing campaign types."""
    MOBILE = "mobile_launch"
    WEB = "web_promotion"
    EMAIL = "email_outreach"
    BRANCH = "branch_event"

class TargetSegment(str, Enum):
    """Customer segment types."""
    RETAIL = "retail_banking"
    WEALTH = "wealth_management"
    BUSINESS = "small_business"
    STUDENT = "student_banking"

class Campaign(BaseModel):
    """Campaign details."""
    name: str = Field(description="Campaign name")
    type: CampaignType = Field(description="Campaign type")
    segment: TargetSegment = Field(description="Target segment")
    budget: float = Field(description="Campaign budget")
    objectives: List[str] = Field(description="Campaign objectives")
    metrics: Dict = Field(description="Performance metrics")

class AnalysisResult(BaseModel):
    """Analysis result structure."""
    strengths: List[Dict[str, str]] = Field(description="Campaign strengths")
    weaknesses: List[Dict[str, str]] = Field(description="Areas to improve")
    recommendations: List[Dict[str, str]] = Field(description="Action items")

class CampaignAnalyzer:
    """Marketing campaign analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Initializing campaign analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a marketing campaign analyst for a bank.
Analyze the campaign performance and provide insights.

Output your analysis in this exact format:
{{
    "strengths": [{{
        "aspect": "Area of success",
        "details": "What went well",
        "impact": "Business effect"
    }}],
    "weaknesses": [{{
        "aspect": "Area to improve",
        "details": "What needs work",
        "impact": "Business effect"
    }}],
    "recommendations": [{{
        "action": "What to do",
        "rationale": "Why do it",
        "timeline": "When to do it"
    }}]
}}"""),
            ("human", """Analyze this campaign:
Name: {name}
Type: {campaign_type}
Segment: {segment}
Budget: ${budget:,.2f}

Objectives:
{objectives}

Performance Metrics:
{metrics}

Provide a complete analysis.""")
        ])
        logger.info("Analysis template ready")
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=AnalysisResult)
        logger.info("Output parser ready")
        
        # Create analysis chain
        self.chain = (
            self.template 
            | self.llm 
            | self.parser
        )
        logger.info("Analysis chain ready")

    async def analyze_campaign(self, campaign: Campaign) -> AnalysisResult:
        """Analyze campaign performance."""
        logger.info(f"Analyzing campaign: {campaign.name}")
        
        try:
            # Format objectives
            objectives = "\n".join(f"- {obj}" for obj in campaign.objectives)
            
            # Format metrics
            metrics = "\n".join(
                f"- {key}: {value}" 
                for key, value in campaign.metrics.items()
            )
            
            # Run analysis
            result = await self.chain.ainvoke({
                "name": campaign.name,
                "campaign_type": campaign.type.value,
                "segment": campaign.segment.value,
                "budget": campaign.budget,
                "objectives": objectives,
                "metrics": metrics
            })
            logger.info("Analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting campaign analysis demo...")
    
    try:
        # Create analyzer
        analyzer = CampaignAnalyzer()
        
        # Example campaign
        campaign = Campaign(
            name="Mobile Banking App Launch",
            type=CampaignType.MOBILE,
            segment=TargetSegment.RETAIL,
            budget=150000.00,
            objectives=[
                "Drive app downloads",
                "Increase digital engagement",
                "Reduce branch transactions",
                "Improve customer satisfaction"
            ],
            metrics={
                "Downloads": "25,000",
                "Active Users": "18,500",
                "Engagement Rate": "73%",
                "App Rating": "4.5/5.0",
                "Digital Transactions": "+45%",
                "Customer Satisfaction": "92%",
                "Cost per Download": "$6.00"
            }
        )
        
        print("\nAnalyzing Marketing Campaign")
        print("==========================")
        print(f"Name: {campaign.name}")
        print(f"Type: {campaign.type.value}")
        print(f"Segment: {campaign.segment.value}")
        print(f"Budget: ${campaign.budget:,.2f}\n")
        
        print("Objectives:")
        for objective in campaign.objectives:
            print(f"- {objective}")
        print()
        
        print("Performance Metrics:")
        for key, value in campaign.metrics.items():
            print(f"- {key}: {value}")
        print()
        
        try:
            # Get analysis
            analysis = await analyzer.analyze_campaign(campaign)
            
            print("\nAnalysis Results:")
            print("================")
            print(json.dumps(analysis, indent=2))
            logger.info("Analysis displayed")
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
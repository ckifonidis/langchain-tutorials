#!/usr/bin/env python3
"""
Campaign Analyzer (132) (LangChain v3)

This example demonstrates marketing analytics using:
1. Document Loading: Campaign data ingestion
2. Aggregation: Performance metrics analysis
3. Prompt Templates: Insight generation

It helps marketing teams optimize banking campaigns.
"""

import os
import csv
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChannelType(str, Enum):
    """Marketing channels."""
    EMAIL = "email"
    MOBILE = "mobile_app"
    WEB = "website"
    SOCIAL = "social_media"
    BRANCH = "branch_network"
    SMS = "sms"

class ProductType(str, Enum):
    """Banking products."""
    SAVINGS = "savings_account"
    CHECKING = "checking_account"
    CREDIT = "credit_card"
    LOAN = "personal_loan"
    MORTGAGE = "mortgage"
    INVESTMENT = "investment_product"

class CampaignData(BaseModel):
    """Campaign performance data."""
    campaign_id: str = Field(description="Campaign ID")
    channel: ChannelType = Field(description="Marketing channel")
    product: ProductType = Field(description="Product offered")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    segments: List[str] = Field(description="Target segments")
    metadata: Dict = Field(default_factory=dict)

class CampaignAnalyzer:
    """Marketing campaign analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting campaign analyzer...")
        
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
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a marketing analytics expert.
Analyze campaign data and provide optimization insights.

Format your response exactly like this:

CAMPAIGN ANALYSIS
---------------
Campaign: ID
Channel: Type
Product: Name

Performance Summary:
- Key metrics
- Comparisons
- Trends

Target Analysis:
1. Segment Name
   - Performance
   - Response
   - Potential

Optimization Plan:
1. Action item
   Impact: Expected result
   Timeline: Implementation time

2. Action item
   Impact: Expected result
   Timeline: Implementation time

Next Steps:
1. Required action
2. Required action

Focus on actionable insights and ROI improvement."""),
            ("human", """Analyze this campaign:
ID: {campaign_id}
Channel: {channel}
Product: {product}

Metrics:
{metrics}

Segments:
{segments}

Provide optimization recommendations.""")
        ])
        logger.info("Analysis template ready")

    def load_campaign_data(self, file_path: str) -> List[Dict]:
        """Load campaign data from CSV."""
        logger.info(f"Loading data from: {file_path}")
        
        try:
            # Load CSV data
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"'
                }
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} records")
            return [doc.page_content for doc in documents]
            
        except Exception as e:
            logger.error(f"Loading failed: {str(e)}")
            raise

    def aggregate_metrics(self, data: List[Dict]) -> Dict[str, float]:
        """Aggregate campaign metrics."""
        logger.info("Aggregating metrics")
        
        try:
            metrics = {}
            for record in data:
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = metrics.get(key, 0) + value
            
            logger.info(f"Aggregated {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            raise

    async def analyze_campaign(self, campaign: CampaignData) -> str:
        """Analyze campaign performance."""
        logger.info(f"Analyzing campaign: {campaign.campaign_id}")
        
        try:
            # Format metrics
            metrics = "\n".join(
                f"{k}: {v:.2f}" 
                for k, v in campaign.metrics.items()
            )
            
            # Format segments
            segments = "\n".join(
                f"- {s}" for s in campaign.segments
            )
            
            # Format request
            messages = self.template.format_messages(
                campaign_id=campaign.campaign_id,
                channel=campaign.channel.value,
                product=campaign.product.value,
                metrics=metrics,
                segments=segments
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
    logger.info("Starting marketing demo...")
    
    try:
        # Create analyzer
        analyzer = CampaignAnalyzer()
        
        # Example campaign
        campaign = CampaignData(
            campaign_id="CAMP-2025-001",
            channel=ChannelType.EMAIL,
            product=ProductType.CREDIT,
            metrics={
                "sent": 100000,
                "delivered": 98500,
                "opened": 35460,
                "clicked": 8865,
                "converted": 443,
                "revenue": 221500.00,
                "cost": 15000.00,
                "roi": 1376.67
            },
            segments=[
                "young_professionals",
                "high_income",
                "credit_active",
                "digital_first"
            ]
        )
        
        print("\nAnalyzing Campaign")
        print("=================")
        print(f"Campaign: {campaign.campaign_id}")
        print(f"Channel: {campaign.channel.value}")
        print(f"Product: {campaign.product.value}\n")
        
        print("Performance Metrics:")
        for name, value in campaign.metrics.items():
            if name == "roi":
                print(f"{name}: {value:.2f}%")
            elif name in ["revenue", "cost"]:
                print(f"{name}: ${value:,.2f}")
            else:
                print(f"{name}: {value:,.0f}")
        
        print("\nTarget Segments:")
        for segment in campaign.segments:
            print(f"- {segment}")
        
        try:
            # Get analysis
            result = await analyzer.analyze_campaign(campaign)
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
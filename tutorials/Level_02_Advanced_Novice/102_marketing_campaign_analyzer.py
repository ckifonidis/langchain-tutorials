#!/usr/bin/env python3
"""
LangChain Marketing Campaign Analyzer (102) (LangChain v3)

This example demonstrates a marketing campaign analysis system using three key concepts:
1. Multimodality: Analyze text and image content
2. Parallel Processing: Efficient tool execution
3. Tool Calling: Integration with marketing tools

It provides comprehensive campaign analysis for marketing departments in banking.
"""

import os
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from PIL import Image
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CampaignAsset(BaseModel):
    """Schema for marketing campaign assets."""
    asset_id: str = Field(description="Asset identifier")
    asset_type: str = Field(description="Type of asset (image/text)")
    content: str = Field(description="Text content or image path")
    metadata: Dict = Field(description="Asset metadata")

class CampaignMetrics(BaseModel):
    """Schema for campaign performance metrics."""
    impressions: int = Field(description="Number of impressions")
    clicks: int = Field(description="Number of clicks")
    conversions: int = Field(description="Number of conversions")
    engagement_rate: float = Field(description="Engagement rate")

class AnalysisResult(BaseModel):
    """Schema for campaign analysis results."""
    asset_id: str = Field(description="Asset identifier")
    content_score: float = Field(description="Content quality score")
    performance_metrics: CampaignMetrics = Field(description="Performance metrics")
    recommendations: List[str] = Field(description="Improvement recommendations")

class MarketingAnalyzer:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=True
        )
        self.setup_tools()

    def setup_tools(self):
        """Set up marketing analysis tools."""
        self.tools = [
            Tool(
                name="analyze_image",
                description="Analyze image content and branding",
                func=self._analyze_image
            ),
            Tool(
                name="get_metrics",
                description="Get campaign performance metrics",
                func=self._get_metrics
            ),
            Tool(
                name="generate_recommendations",
                description="Generate campaign improvement recommendations",
                func=self._generate_recommendations
            )
        ]

    async def _analyze_image(self, image_path: str) -> Dict:
        """Analyze image content (simplified for demo)."""
        return {
            "branding_score": 0.85,
            "visual_appeal": 0.78,
            "message_clarity": 0.92
        }

    async def _get_metrics(self, asset_id: str) -> CampaignMetrics:
        """Get campaign metrics (simplified for demo)."""
        return CampaignMetrics(
            impressions=10000,
            clicks=450,
            conversions=50,
            engagement_rate=4.5
        )

    async def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        return [
            "Enhance brand visibility in visual elements",
            "Optimize call-to-action placement",
            "Test alternative messaging variants"
        ]

    async def analyze_campaign(self, asset: CampaignAsset) -> Dict[str, str]:
        """Analyze a marketing campaign asset."""
        try:
            # Execute tools in parallel
            image_analysis, metrics, recommendations = await asyncio.gather(
                self._analyze_image(asset.content) if asset.asset_type == "image" else asyncio.sleep(0, {}),
                self._get_metrics(asset.asset_id),
                self._generate_recommendations({})
            )

            messages = [
                SystemMessage(content="""You are a marketing campaign analyzer for a bank. 
                Analyze the campaign assets and provide insights."""),
                HumanMessage(content=f"""Analyze this campaign asset: {json.dumps(asset.model_dump())}
                Image Analysis: {json.dumps(image_analysis)}
                Performance Metrics: {json.dumps(metrics.model_dump())}
                Recommendations: {json.dumps(recommendations)}""")
            ]

            response = await self.llm.ainvoke(messages)
            
            return {
                "analysis": response.content,
                "metrics": metrics.model_dump(),
                "recommendations": recommendations
            }

        except Exception as e:
            return {"error": f"Error during analysis: {str(e)}"}

async def demonstrate_marketing_analyzer():
    print("\nMarketing Campaign Analyzer Demo")
    print("===============================\n")

    analyzer = MarketingAnalyzer()

    # Example campaign assets
    assets = [
        CampaignAsset(
            asset_id="camp_001",
            asset_type="image",
            content="path/to/credit_card_promo.jpg",
            metadata={"campaign": "Premium Credit Card Launch", "target": "High-net-worth"}
        ),
        CampaignAsset(
            asset_id="camp_002",
            asset_type="text",
            content="Experience seamless digital banking with our new mobile app",
            metadata={"campaign": "Digital Banking", "target": "Millennials"}
        )
    ]

    # Process assets
    for asset in assets:
        print(f"Analyzing Asset: {asset.asset_id}")
        print(f"Type: {asset.asset_type}")
        print(f"Campaign: {asset.metadata['campaign']}\n")

        result = await analyzer.analyze_campaign(asset)
        
        print("Analysis Results:")
        print(f"Analysis: {result['analysis']}")
        print(f"\nMetrics:")
        for key, value in result['metrics'].items():
            print(f"- {key}: {value}")
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_marketing_analyzer())
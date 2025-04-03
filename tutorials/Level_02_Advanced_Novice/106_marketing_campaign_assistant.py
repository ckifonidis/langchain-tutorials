#!/usr/bin/env python3
"""
LangChain Marketing Campaign Assistant (106) (LangChain v3)

This example demonstrates a marketing campaign assistant using three key concepts:
1. Multimodality: Handle text and image content
2. Streaming: Real-time campaign updates
3. Tool Calling: Marketing automation integration

It provides comprehensive campaign management support for marketing teams in banking.
"""

import os
import json
from typing import List, Dict, Optional, Union, Any, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CampaignAsset(BaseModel):
    """Schema for marketing campaign assets."""
    asset_id: str = Field(description="Asset identifier")
    asset_type: str = Field(description="Type (image/text/video)")
    content: Union[str, bytes] = Field(description="Asset content")
    metadata: Dict = Field(description="Asset metadata")

class CampaignTarget(BaseModel):
    """Schema for campaign targeting."""
    segment: str = Field(description="Target customer segment")
    channel: str = Field(description="Marketing channel")
    timing: Dict = Field(description="Campaign timing")
    metrics: List[str] = Field(description="Success metrics")

class CampaignAnalysis(BaseModel):
    """Schema for campaign analysis results."""
    recommendations: List[str] = Field(description="Campaign recommendations")
    risk_factors: List[str] = Field(description="Identified risks")
    compliance_check: Dict = Field(description="Compliance status")
    feedback: str = Field(description="Detailed feedback")

class ComplianceTool(BaseTool):
    """Tool for analyzing campaign compliance."""
    name: str = Field(default="compliance_analyzer")
    description: str = Field(default="Analyzes campaign content for regulatory compliance.")
    
    def _run(self, input: str) -> str:
        """Run compliance analysis."""
        compliant_terms = ["secure", "regulated", "terms apply"]
        risky_terms = ["guaranteed", "risk-free", "best"]
        
        result = {"compliant": True, "warnings": []}
        
        # Check for required terms
        for term in compliant_terms:
            if term not in input.lower():
                result["warnings"].append(f"Missing required term: {term}")
        
        # Check for risky terms
        for term in risky_terms:
            if term in input.lower():
                result["warnings"].append(f"Contains risky term: {term}")
                result["compliant"] = False
        
        return json.dumps(result)

    async def _arun(self, input: str) -> str:
        """Run compliance analysis asynchronously."""
        return self._run(input)

class SocialMediaTool(BaseTool):
    """Tool for social media scheduling."""
    name: str = Field(default="social_scheduler")
    description: str = Field(default="Schedules social media posts.")
    
    def _run(self, input: str) -> str:
        """Schedule a social media post."""
        try:
            # Parse input as JSON if it's a string
            if isinstance(input, str):
                post_data = json.loads(input)
            else:
                post_data = input
            
            # Validate required fields
            if not all(k in post_data for k in ["platform", "datetime"]):
                raise ValueError("Missing required fields: platform and datetime")
            
            result = {
                "status": "scheduled",
                "platform": post_data["platform"],
                "datetime": post_data["datetime"],
                "content": post_data.get("content"),
                "media": post_data.get("media")
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    async def _arun(self, input: str) -> str:
        """Schedule a social media post asynchronously."""
        return self._run(input)

class ImageAnalysisTool(BaseTool):
    """Tool for analyzing marketing images."""
    name: str = Field(default="image_analyzer")
    description: str = Field(default="Analyzes marketing image content for compliance and quality.")
    
    def _run(self, input: str) -> str:
        """Run image analysis."""
        try:
            image_data = base64.b64decode(input)
            image = Image.open(io.BytesIO(image_data))
            
            result = {
                "dimensions": f"{image.width}x{image.height}",
                "format": image.format,
                "mode": image.mode,
                "compliance": {
                    "has_disclaimer": True,
                    "logo_placement": "valid",
                    "text_legible": True
                }
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "status": "failed"
            })

    async def _arun(self, input: str) -> str:
        """Run image analysis asynchronously."""
        return self._run(input)

class MarketingCampaignAssistant:
    def __init__(self):
        # Initialize Azure OpenAI with streaming
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize tools
        self.compliance_tool = ComplianceTool()
        self.social_media_tool = SocialMediaTool()
        self.image_tool = ImageAnalysisTool()

    async def analyze_campaign_assets(self, assets: List[CampaignAsset], target: CampaignTarget) -> CampaignAnalysis:
        """Analyze campaign assets and provide recommendations."""
        try:
            analysis_results = []
            
            # Process each asset
            for asset in assets:
                if asset.asset_type == "text":
                    # Check text compliance
                    compliance_result = await self.compliance_tool.ainvoke(input=asset.content)
                    analysis_results.append({
                        "asset_id": asset.asset_id,
                        "compliance": json.loads(compliance_result)
                    })
                elif asset.asset_type == "image":
                    # Analyze image content
                    if isinstance(asset.content, bytes):
                        image_result = await self.image_tool.ainvoke(
                            input=base64.b64encode(asset.content).decode()
                        )
                        analysis_results.append({
                            "asset_id": asset.asset_id,
                            "image_analysis": json.loads(image_result)
                        })

            # Generate recommendations
            recommendations_prompt = f"""
            Analyze the marketing campaign for {target.segment} through {target.channel}.
            Campaign timing: {json.dumps(target.timing)}
            Asset analysis results: {json.dumps(analysis_results)}
            
            Provide recommendations for:
            1. Content optimization
            2. Timing adjustments
            3. Risk mitigation
            4. Compliance improvements
            """
            
            stream_response = await self.llm.ainvoke(
                [
                    SystemMessage(content="You are a banking marketing expert."),
                    HumanMessage(content=recommendations_prompt)
                ]
            )
            
            # Extract recommendations from response
            recommendations = [
                line.strip() for line in stream_response.content.split("\n")
                if line.strip().startswith("-")
            ]
            
            # Schedule social media posts if needed
            schedule_results = []
            if target.channel == "social_media":
                for timing in target.timing.get("posts", []):
                    # Find matching text content
                    content = next(
                        (asset.content for asset in assets 
                         if asset.asset_type == "text" and 
                         asset.metadata.get("platform") == timing["platform"]),
                        None
                    )
                    
                    # Create post data and convert to JSON
                    post_data = {
                        "platform": timing["platform"],
                        "datetime": timing["datetime"],
                        "content": content
                    }
                    
                    # Schedule post with JSON string input
                    schedule_result = await self.social_media_tool.ainvoke(
                        input=json.dumps(post_data)
                    )
                    schedule_results.append(json.loads(schedule_result))
            
            return CampaignAnalysis(
                recommendations=recommendations or ["No automated recommendations generated."],
                risk_factors=[
                    warning for result in analysis_results 
                    if "compliance" in result 
                    for warning in result["compliance"].get("warnings", [])
                ],
                compliance_check={
                    "status": "reviewed",
                    "timestamp": datetime.now().isoformat(),
                    "schedule_status": schedule_results
                },
                feedback=stream_response.content
            )
            
        except Exception as e:
            print(f"Error analyzing campaign: {str(e)}")
            return None

async def demonstrate_campaign_assistant():
    print("\nMarketing Campaign Assistant Demo")
    print("================================\n")

    assistant = MarketingCampaignAssistant()

    # Example campaign assets
    assets = [
        CampaignAsset(
            asset_id="text_001",
            asset_type="text",
            content="""Discover our new Premium Banking Services:
            - Secure digital transactions
            - 24/7 personal banker access
            - Regulated investment options
            Terms apply. Contact us to learn more.""",
            metadata={"type": "social_post", "platform": "linkedin"}
        ),
        CampaignAsset(
            asset_id="img_001",
            asset_type="image",
            content=b"sample_image_bytes",  # Simulated image content
            metadata={"type": "banner", "dimensions": "1200x628"}
        )
    ]

    # Example campaign target
    target = CampaignTarget(
        segment="affluent_professionals",
        channel="social_media",
        timing={
            "start_date": "2025-04-01",
            "end_date": "2025-04-30",
            "posts": [
                {"platform": "linkedin", "datetime": "2025-04-01T09:00:00"},
                {"platform": "instagram", "datetime": "2025-04-02T12:00:00"}
            ]
        },
        metrics=["engagement", "lead_generation", "conversion"]
    )

    print("Analyzing Campaign Assets...")
    print(f"Target Segment: {target.segment}")
    print(f"Channel: {target.channel}\n")

    analysis = await assistant.analyze_campaign_assets(assets, target)
    if analysis:
        print("\nCampaign Analysis:")
        print("\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"- {rec}")
        
        print("\nRisk Factors:")
        for risk in analysis.risk_factors:
            print(f"- {risk}")
        
        print("\nCompliance Status:")
        for key, value in analysis.compliance_check.items():
            if key == "schedule_status":
                print("\nSchedule Status:")
                for status in value:
                    print(f"- Platform: {status['platform']}")
                    print(f"  Time: {status['datetime']}")
                    print(f"  Status: {status['status']}")
            else:
                print(f"- {key}: {value}")
        
        print(f"\nDetailed Feedback:\n{analysis.feedback}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_campaign_assistant())
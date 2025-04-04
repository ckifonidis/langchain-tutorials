#!/usr/bin/env python3
"""
LangChain Campaign Testing Assistant (111) (LangChain v3)

This example demonstrates a marketing campaign testing system using three key concepts:
1. Key Methods: Core campaign validation
2. Retrieval: Find similar campaigns
3. Testing: Campaign verification

It provides comprehensive campaign testing support for marketing teams in banking.
"""

import os
import sys
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChannelType(str, Enum):
    """Marketing channel types."""
    EMAIL = "email"
    MOBILE = "mobile"
    SOCIAL = "social"
    WEB = "web"
    BRANCH = "branch"

class AudienceSegment(str, Enum):
    """Banking customer segments."""
    RETAIL = "retail"
    BUSINESS = "business"
    WEALTH = "wealth"
    STUDENT = "student"
    SENIOR = "senior"

class ComplianceCheck(BaseModel):
    """Schema for compliance check results."""
    passed: bool = Field(description="Check pass status")
    issues: List[str] = Field(description="Identified issues")
    regulations: List[str] = Field(description="Relevant regulations")
    severity: str = Field(description="Issue severity level")

class CampaignTest(BaseModel):
    """Schema for campaign test results."""
    campaign_id: str = Field(description="Campaign identifier")
    channel: ChannelType = Field(description="Marketing channel")
    segment: AudienceSegment = Field(description="Target audience")
    test_date: str = Field(description="Test execution date")
    compliance: ComplianceCheck = Field(description="Compliance results")
    success_metrics: Dict[str, float] = Field(description="Performance metrics")
    recommendations: List[str] = Field(description="Improvement suggestions")

class CampaignTestingAssistant:
    def __init__(self):
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Setup components
        self.setup_retriever()
        self.setup_evaluators()

    def setup_retriever(self):
        """Initialize campaign retriever."""
        campaigns = [
            {
                "campaign": "Student Account Promotion",
                "channel": "social",
                "segment": "student",
                "compliance_score": 0.95,
                "content": """
                🎓 Student Banking Made Easy!
                - No monthly fees
                - Mobile banking app
                - Student discounts
                - 24/7 support
                Apply now with your student ID.
                """
            },
            {
                "campaign": "Business Credit Card Launch",
                "channel": "email",
                "segment": "business",
                "compliance_score": 0.98,
                "content": """
                Introducing Business Edge Credit Card
                - 2% cashback on business expenses
                - Travel insurance included
                - Account management tools
                - Dedicated support line
                Apply today for premium benefits.
                """
            }
        ]
        
        # Create vector store
        texts = [c["content"] for c in campaigns]
        metadatas = campaigns
        
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Initialize time-weighted retriever
        self.retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vectorstore,
            decay_rate=0.01,
            k=2
        )

    def setup_evaluators(self):
        """Initialize campaign evaluators."""
        # Define evaluation criteria
        compliance_criteria = {
            "regulatory_alignment": "Content follows banking regulations",
            "data_privacy": "Customer data handling follows GDPR",
            "disclosure": "Financial terms clearly disclosed",
            "fairness": "Non-discriminatory language and practices"
        }
        
        performance_criteria = {
            "engagement": "Message is engaging and clear",
            "targeting": "Content is well-targeted to audience",
            "call_to_action": "Clear and compelling call to action",
            "value_proposition": "Clear value proposition"
        }
        
        # Initialize evaluators
        self.compliance_evaluator = LabeledCriteriaEvalChain.from_llm(
            llm=self.llm,
            criteria=compliance_criteria
        )
        
        self.performance_evaluator = LabeledCriteriaEvalChain.from_llm(
            llm=self.llm,
            criteria=performance_criteria
        )

    async def test_campaign(self, campaign_id: str, content: str, 
                          channel: ChannelType, segment: AudienceSegment) -> CampaignTest:
        """Test a marketing campaign."""
        # Find similar campaigns using invoke method
        try:
            similar = await self.retriever.ainvoke(content)
            if not isinstance(similar, list):
                similar = []
        except Exception as e:
            print(f"Warning: Retrieval error - using empty results: {str(e)}")
            similar = []
        
        # Check compliance
        compliance_result = self.compliance_evaluator.evaluate_strings(
            prediction="Campaign Content: " + content,
            reference="Compliance Guidelines: Banks must clearly disclose terms, avoid discrimination, protect data privacy, and comply with banking regulations.",
            input="Evaluate if this banking campaign follows compliance guidelines."
        )
        
        # Evaluate performance
        performance_result = self.performance_evaluator.evaluate_strings(
            prediction="Campaign Content: " + content,
            reference="Performance Standards: Campaigns should be engaging, targeted, have clear call-to-action, and strong value proposition.",
            input="Evaluate this banking campaign's marketing effectiveness."
        )
        
        # Generate recommendations
        context = "\n".join(doc.page_content for doc in similar) if similar else "No similar campaigns found"
        prompt = f"""
        Analyze this banking marketing campaign and provide recommendations:

        Campaign:
        {content}

        Channel: {channel}
        Segment: {segment}

        Similar Campaigns:
        {context}

        Compliance Score: {compliance_result['score']}
        Performance Score: {performance_result['score']}

        Provide specific recommendations to improve the campaign while maintaining compliance.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert banking marketing advisor."),
            HumanMessage(content=prompt)
        ])
        
        # Return test results
        return CampaignTest(
            campaign_id=campaign_id,
            channel=channel,
            segment=segment,
            test_date=datetime.now().isoformat(),
            compliance=ComplianceCheck(
                passed=compliance_result['score'] >= 0.8,
                issues=compliance_result.get('reasoning', "").split("\n"),
                regulations=["GDPR", "Banking Act", "Fair Advertising"],
                severity="high" if compliance_result['score'] < 0.8 else "low"
            ),
            success_metrics={
                "compliance_score": compliance_result['score'],
                "performance_score": performance_result['score'],
                "relevance_score": (
                    sum(doc.metadata.get("compliance_score", 0) for doc in similar) / len(similar)
                    if similar else 0.0
                )
            },
            recommendations=[r.strip() for r in response.content.split("\n") 
                           if r.strip()]
        )

async def demonstrate_campaign_testing():
    print("\nCampaign Testing Assistant Demo")
    print("==============================\n")

    try:
        # Initialize assistant
        assistant = CampaignTestingAssistant()

        # Example campaign
        campaign_content = """
        Introducing NextGen Student Banking!
        
        🎓 Zero monthly fees
        💳 Instant virtual card
        📱 Modern banking app
        💰 5% cashback on books
        🎯 Student-exclusive deals
        🔒 Secure banking
        
        Apply now with your student ID and get a $50 welcome bonus!
        Terms and conditions apply. Member FDIC. Banking services 
        provided by NextGen Bank, N.A.
        """

        print("Testing campaign...")
        results = await assistant.test_campaign(
            campaign_id="CAMP-2025-001",
            content=campaign_content,
            channel=ChannelType.SOCIAL,
            segment=AudienceSegment.STUDENT
        )

        # Display results
        print("\nTest Results:")
        print(f"Campaign: {results.campaign_id}")
        print(f"Channel: {results.channel}")
        print(f"Segment: {results.segment}")
        
        print("\nCompliance Check:")
        print(f"Passed: {results.compliance.passed}")
        if results.compliance.issues:
            print("\nIssues:")
            for issue in results.compliance.issues:
                print(f"- {issue}")
        
        print("\nMetrics:")
        for metric, score in results.success_metrics.items():
            print(f"{metric}: {score:.2f}")
        
        print("\nRecommendations:")
        for rec in results.recommendations:
            print(f"- {rec}")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_campaign_testing())
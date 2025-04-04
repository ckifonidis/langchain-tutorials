#!/usr/bin/env python3
"""
Performance Analyzer (139) (LangChain v3)

This example demonstrates HR review analysis using:
1. Document Loading: Review data ingestion
2. Structured Output: Defined response format
3. RAG Chains: Context-enhanced reasoning

It helps HR teams analyze banking employee performance reviews.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ReviewType(str, Enum):
    """Review categories."""
    ANNUAL = "annual_review"
    QUARTERLY = "quarterly_review"
    MIDYEAR = "midyear_review"
    PROBATION = "probation_review"
    PROJECT = "project_review"

class ReviewStatus(str, Enum):
    """Review status levels."""
    DRAFT = "draft"
    PENDING = "pending"
    COMPLETE = "completed"
    SIGNED = "signed"
    ARCHIVED = "archived"

class Performance(BaseModel):
    """Performance details."""
    overall_rating: float = Field(description="Overall rating (1-5)")
    strengths: List[str] = Field(description="Key strengths")
    areas_for_improvement: List[str] = Field(description="Areas to improve")
    goals_met: List[str] = Field(description="Goals achieved")
    goals_pending: List[str] = Field(description="Goals in progress")

class Review(BaseModel):
    """Review details."""
    review_id: str = Field(description="Review ID")
    employee_id: str = Field(description="Employee ID")
    type: ReviewType = Field(description="Review type")
    status: ReviewStatus = Field(description="Review status")
    period: str = Field(description="Review period")
    performance: Performance = Field(description="Performance details")
    metadata: Dict = Field(default_factory=dict)

class AnalysisResult(BaseModel):
    """Analysis result format."""
    highlights: List[str] = Field(description="Key achievements")
    concerns: List[str] = Field(description="Areas of concern")
    trends: List[Dict[str, str]] = Field(description="Performance trends")
    recommendations: List[Dict[str, str]] = Field(description="Action items")
    next_steps: List[str] = Field(description="Next steps")

def generate_example_reviews():
    """Generate example review data."""
    return [
        """Employee Review - Q4 2024
Overall: Excellent performance in project delivery
Strengths: Technical expertise, team collaboration
Areas to Improve: Documentation standards
Goals: Exceeded customer satisfaction targets""",
        
        """Employee Review - Q3 2024
Overall: Strong leadership demonstrated
Strengths: Strategic thinking, mentoring
Areas to Improve: Time management
Goals: Successfully led system upgrade""",
        
        """Employee Review - Q2 2024
Overall: Consistent high performance
Strengths: Problem-solving, innovation
Areas to Improve: Cross-team communication
Goals: Delivered ahead of schedule"""
    ]

class PerformanceAnalyzer:
    """Performance analysis system."""

    def __init__(self, reviews_dir: str):
        """Initialize analyzer."""
        logger.info("Starting performance analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup document loader if directory exists
        if os.path.exists(reviews_dir):
            self.loader = DirectoryLoader(
                reviews_dir,
                glob="**/*.txt",
                show_progress=True
            )
            logger.info("Document loader ready")
            
            # Setup text splitter
            self.splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            logger.info("Text splitter ready")
            
            try:
                # Load and process documents
                docs = self.loader.load()
                texts = self.splitter.split_documents(docs)
                self.retriever = TFIDFRetriever.from_documents(texts)
                logger.info(f"Loaded {len(texts)} text chunks")
            except Exception as e:
                logger.warning(f"Error loading documents: {str(e)}. Using example data.")
                self._setup_example_retriever()
        else:
            logger.warning(f"Directory not found: '{reviews_dir}'. Using example data.")
            self._setup_example_retriever()
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=AnalysisResult)
        logger.info("Output parser ready")
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are an HR performance analyst.
Review employee performance data and provide insights.

Use this exact JSON format (do not modify the structure):
{{
    "highlights": [
        "Achievement 1",
        "Achievement 2"
    ],
    "concerns": [
        "Concern 1",
        "Concern 2"
    ],
    "trends": [
        {{
            "pattern": "Pattern name",
            "details": "Description",
            "impact": "Effect"
        }}
    ],
    "recommendations": [
        {{
            "action": "Action item",
            "reason": "Justification",
            "timeline": "When"
        }}
    ],
    "next_steps": [
        "Step 1",
        "Step 2"
    ]
}}"""),
            ("human", """Analyze this employee review:

Review ID: {review_id}
Type: {review_type}
Period: {period}

Current Performance:
{current_details}

Historical Context:
{historical_context}

Provide a complete analysis.""")
        ])
        logger.info("Analysis template ready")

    def _setup_example_retriever(self):
        """Setup retriever with example data."""
        example_texts = generate_example_reviews()
        self.retriever = TFIDFRetriever.from_texts(example_texts)
        logger.info("Example retriever ready")

    async def analyze_review(self, review: Review) -> AnalysisResult:
        """Analyze employee review."""
        logger.info(f"Analyzing review: {review.review_id}")
        
        try:
            # Get historical context
            query = f"reviews for employee {review.employee_id}"
            docs = await self.retriever.ainvoke(query)  # Use ainvoke for async
            historical = "\n\n".join(doc.page_content for doc in docs)
            logger.info("Retrieved historical context")
            
            # Format performance details
            current = (
                f"Overall Rating: {review.performance.overall_rating}\n\n"
                f"Strengths:\n" + "\n".join(f"- {s}" for s in review.performance.strengths) + "\n\n"
                f"Areas for Improvement:\n" + "\n".join(f"- {a}" for a in review.performance.areas_for_improvement) + "\n\n"
                f"Goals Met:\n" + "\n".join(f"- {g}" for g in review.performance.goals_met) + "\n\n"
                f"Goals Pending:\n" + "\n".join(f"- {g}" for g in review.performance.goals_pending)
            )
            
            # Create chain
            chain = (
                self.template
                | self.llm
                | self.parser
            )
            
            # Run analysis
            result = await chain.ainvoke({
                "review_id": review.review_id,
                "review_type": review.type.value,
                "period": review.period,
                "current_details": current,
                "historical_context": historical
            })
            logger.info("Analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting performance analysis demo...")
    
    try:
        # Create analyzer with example data path
        analyzer = PerformanceAnalyzer("./example_reviews")
        
        # Example review
        review = Review(
            review_id="REV-2025-001",
            employee_id="EMP-001",
            type=ReviewType.ANNUAL,
            status=ReviewStatus.COMPLETE,
            period="2024",
            performance=Performance(
                overall_rating=4.2,
                strengths=[
                    "Strong leadership of cross-functional teams",
                    "Excellent customer relationship management",
                    "Consistent high-quality deliverables",
                    "Proactive problem-solving approach"
                ],
                areas_for_improvement=[
                    "Documentation could be more detailed",
                    "Time management in peak periods",
                    "Delegation of routine tasks"
                ],
                goals_met=[
                    "Led digital transformation project on schedule",
                    "Achieved 98% customer satisfaction rating",
                    "Mentored three junior team members",
                    "Reduced processing time by 25%"
                ],
                goals_pending=[
                    "Complete advanced certifications",
                    "Develop team succession plan",
                    "Optimize resource allocation framework"
                ]
            )
        )
        
        print("\nAnalyzing Performance Review")
        print("==========================")
        print(f"Review: {review.review_id}")
        print(f"Type: {review.type.value}")
        print(f"Period: {review.period}\n")
        
        print("Current Performance:")
        print(f"Overall Rating: {review.performance.overall_rating}\n")
        
        print("Strengths:")
        for strength in review.performance.strengths:
            print(f"- {strength}")
        print()
        
        print("Areas for Improvement:")
        for area in review.performance.areas_for_improvement:
            print(f"- {area}")
        print()
        
        print("Goals Met:")
        for goal in review.performance.goals_met:
            print(f"- {goal}")
        print()
        
        print("Goals Pending:")
        for goal in review.performance.goals_pending:
            print(f"- {goal}")
        print()
        
        try:
            # Get analysis
            result = await analyzer.analyze_review(review)
            print("\nAnalysis Results:")
            print("================")
            # Convert output to dict (result is already a dict from JsonOutputParser)
            print(json.dumps(result, indent=2))
            logger.info("Output formatted")
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
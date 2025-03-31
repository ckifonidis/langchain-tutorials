#!/usr/bin/env python3
"""
LangChain Async Document Analyzer (LangChain v3)

This example demonstrates an asynchronous document analysis system using three key concepts:
1. async: Concurrent document processing
2. evaluation: Quality assessment of model outputs
3. structured_output: Type-safe response handling

It provides efficient and validated document analysis with quality metrics.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, TypeVar
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.language_models import BaseLLM
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.evaluation import StringEvaluator

# Load environment variables
load_dotenv(".env")

class DocumentType(str, Enum):
    """Document categories."""
    EMAIL = "email"
    CONTRACT = "contract"
    REPORT = "report"
    POLICY = "policy"

class AnalysisMetrics(BaseModel):
    """Quality metrics for analysis."""
    relevance_score: float = Field(
        description="How relevant the analysis is (0-1)",
        ge=0.0, le=1.0
    )
    clarity_score: float = Field(
        description="How clear the analysis is (0-1)",
        ge=0.0, le=1.0
    )
    confidence_score: float = Field(
        description="Model's confidence in analysis (0-1)",
        ge=0.0, le=1.0
    )

class DocumentAnalysis(BaseModel):
    """Structured document analysis."""
    doc_type: DocumentType = Field(description="Type of document")
    summary: str = Field(description="Brief document summary")
    key_points: List[str] = Field(description="Main points from document")
    sentiment: str = Field(description="Overall document sentiment")
    metrics: Optional[AnalysisMetrics] = Field(
        description="Analysis quality metrics",
        default=None
    )
    
    @field_validator("key_points")
    @classmethod
    def validate_key_points(cls, v: List[str]) -> List[str]:
        """Ensure key points are not empty."""
        if not v:
            raise ValueError("Must have at least one key point")
        return v

class CustomEvaluator(StringEvaluator):
    """Custom evaluator for document analysis."""
    
    def __init__(self, llm: BaseLLM, criteria: str):
        """Initialize with LLM and evaluation criteria."""
        self.llm = llm
        self.criteria = criteria
        self.eval_chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert document analyst.
                Evaluate the quality of the analysis based on {criteria}.
                Provide a score between 0.0 and 1.0, where:
                - 1.0: Perfect {criteria}
                - 0.0: Poor {criteria}
                
                Respond with just the numeric score."""),
                ("user", "Document: {input}\nAnalysis: {prediction}")
            ])
            | self.llm
        )
    
    def _evaluate_strings(
        self, prediction: str, input: str, **kwargs
    ) -> dict:
        """Required synchronous evaluation method."""
        # Default implementation
        return {"score": 0.5}
    
    async def aevaluate_strings(
        self, prediction: str, input: str, **kwargs
    ) -> dict:
        """Evaluate analysis quality."""
        # Get score from LLM
        result = await self.eval_chain.ainvoke(
            {"input": input, "prediction": prediction},
            config=RunnableConfig(callbacks=None)
        )
        
        try:
            score = float(result.strip())
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
        except:
            score = 0.5  # Default score on error
        
        return {"score": score}

class DocumentProcessor:
    """Asynchronous document processor with evaluation."""
    
    def __init__(self):
        """Initialize processor with LLM and evaluators."""
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Initialize custom evaluators
        self.relevance_evaluator = CustomEvaluator(self.llm, "relevance")
        self.clarity_evaluator = CustomEvaluator(self.llm, "clarity")
        
        # Create system prompt
        SYSTEM_PROMPT = """You are a document analysis expert.
        Analyze the given document and provide:
        1. Document type: One of [email, contract, report, policy]
        2. Brief summary of the content
        3. At least 3 key points from the document
        4. Overall sentiment: positive/neutral/negative

        Return ONLY a JSON object with these fields:
        {{
            "doc_type": "email|contract|report|policy",
            "summary": "brief summary here",
            "key_points": ["point 1", "point 2", "point 3"],
            "sentiment": "positive|neutral|negative"
        }}"""
        
        # Create analysis chain
        self.analyze_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "{document}")
            ])
            | self.llm
            | JsonOutputParser()
        )
    
    async def evaluate_analysis(self, document: str, analysis: DocumentAnalysis) -> AnalysisMetrics:
        """Evaluate analysis quality."""
        # Run evaluations concurrently
        relevance_task = asyncio.create_task(
            self.relevance_evaluator.aevaluate_strings(
                prediction=analysis.summary,
                input=document
            )
        )
        
        clarity_task = asyncio.create_task(
            self.clarity_evaluator.aevaluate_strings(
                prediction="\n".join(analysis.key_points),
                input=document
            )
        )
        
        # Wait for results
        relevance_result, clarity_result = await asyncio.gather(
            relevance_task,
            clarity_task
        )
        
        # Calculate confidence from evaluations
        confidence = (relevance_result["score"] + clarity_result["score"]) / 2.0
        
        return AnalysisMetrics(
            relevance_score=relevance_result["score"],
            clarity_score=clarity_result["score"],
            confidence_score=confidence
        )
    
    async def analyze_document(self, document: str) -> DocumentAnalysis:
        """Process document and evaluate results."""
        try:
            # Get initial analysis
            result = await self.analyze_chain.ainvoke({"document": document})
            
            # Create analysis object
            analysis = DocumentAnalysis.model_validate(result)
            
            # Evaluate analysis
            metrics = await self.evaluate_analysis(document, analysis)
            analysis.metrics = metrics
            
            return analysis
            
        except Exception as e:
            raise ValueError(f"Error analyzing document: {str(e)}")

async def demonstrate_analyzer():
    """Demonstrate the document analyzer."""
    print("\nAsync Document Analyzer Demo")
    print("==========================\n")
    
    # Create processor
    processor = DocumentProcessor()
    
    # Test documents
    documents = [
        """
        Subject: Q1 Sales Performance Review
        
        Team,
        
        Our Q1 sales exceeded targets by 15%. Key achievements:
        - New client acquisition up 20%
        - Customer retention at 95%
        - Product line expansion successful
        
        Great work everyone!
        
        Best regards,
        Sales Director
        """,
        
        """
        Privacy Policy Update
        
        Effective Date: March 27, 2025
        
        1. Data Collection
        We collect user data to improve services.
        
        2. Data Usage
        Your data is used only for service improvement.
        
        3. Security
        We employ industry-standard security measures.
        """,
        
        """
        CONTRACT AGREEMENT
        
        This agreement, dated March 27, 2025, establishes:
        
        1. Service Terms
        Provider will deliver specified services.
        
        2. Payment Terms
        Client agrees to pay within 30 days.
        
        3. Duration
        12-month initial term with renewal option.
        """
    ]
    
    try:
        # Process documents concurrently
        tasks = [processor.analyze_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        # Display results
        for i, analysis in enumerate(results, 1):
            print(f"\nDocument {i} Analysis")
            print("-" * 40)
            print(f"Type: {analysis.doc_type}")
            print(f"Summary: {analysis.summary}")
            print("\nKey Points:")
            for point in analysis.key_points:
                print(f"- {point}")
            print(f"\nSentiment: {analysis.sentiment}")
            print("\nQuality Metrics:")
            print(f"- Relevance: {analysis.metrics.relevance_score:.2f}")
            print(f"- Clarity: {analysis.metrics.clarity_score:.2f}")
            print(f"- Confidence: {analysis.metrics.confidence_score:.2f}")
            print("-" * 40)
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demonstrate_analyzer())
#!/usr/bin/env python3
"""
LangChain Investment Pipeline Builder (LangChain v3)

This example demonstrates how to create a flexible investment analysis pipeline
using LCEL (LangChain Expression Language) and the Runnable interface. The system
allows dynamic construction and modification of investment analysis workflows.

Key concepts demonstrated:
1. LCEL: Building complex analysis chains with LangChain Expression Language
2. Runnable Interface: Creating composable and reusable investment analysis components
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration from environment variables
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration. Please check your .env file.")

def format_json_response(response: Any) -> str:
    """Ensure response is properly formatted JSON."""
    try:
        # Handle AIMessage objects
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)
            
        # First try to parse as is
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # If not JSON, create a structured response
        return json.dumps({
            "analysis": text.strip(),
            "timestamp": datetime.now().isoformat()
        })

class InvestmentData(BaseModel):
    """Investment analysis data schema."""
    asset_type: str = Field(description="Type of investment asset")
    price_history: List[float] = Field(description="Historical price data")
    market_cap: float = Field(description="Market capitalization")
    sector: str = Field(description="Business sector")
    risk_metrics: Dict[str, float] = Field(description="Risk analysis metrics")

class AnalysisResult(BaseModel):
    """Analysis result schema."""
    recommendation: str = Field(description="Investment recommendation")
    confidence: float = Field(description="Confidence score")
    rationale: List[str] = Field(description="Analysis rationale")
    risk_level: str = Field(description="Risk assessment")

def validate_deployments() -> bool:
    """Validate Azure OpenAI deployments."""
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        
        response = llm.invoke("Respond with 'OK' if you can read this.")
        if "OK" not in response.content:
            raise ValueError("Model response validation failed")
            
        print("Azure OpenAI deployments validated successfully")
        return True
        
    except Exception as e:
        print(f"Deployment validation failed: {str(e)}")
        return False

def create_chat_model() -> AzureChatOpenAI:
    """Initialize Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        openai_api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0
    )

class InvestmentAnalyzer:
    """Investment analysis components using LCEL and Runnable interface."""
    
    def __init__(self):
        if not validate_deployments():
            raise RuntimeError("Failed to validate Azure OpenAI deployments")
            
        self.llm = create_chat_model()
        self.analysis_components = self._create_components()
    
    def _create_components(self) -> Dict[str, RunnableLambda]:
        """Create analysis pipeline components."""
        
        # Technical Analysis
        technical_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in technical analysis. Analyze the provided data and return a JSON response.

Example response format:
{{
    "analysis": "Detailed technical analysis",
    "risk_factors": ["factor1", "factor2"],
    "confidence": 0.85,
    "recommendation": "Buy/Sell recommendation"
}}"""),
            ("human", "Analyze this price data: {price_history}")
        ])
        
        technical_chain = (
            technical_prompt 
            | self.llm 
            | RunnableLambda(format_json_response)
        )
        
        # Fundamental Analysis
        fundamental_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in fundamental analysis. Analyze the provided data and return a JSON response.

Example response format:
{{
    "analysis": "Detailed fundamental analysis",
    "metrics": {{"key_metric": "value"}},
    "confidence": 0.88,
    "recommendation": "Investment recommendation"
}}"""),
            ("human", """
            Analyze these metrics:
            Market Cap: {market_cap}
            Sector: {sector}
            """)
        ])
        
        fundamental_chain = (
            fundamental_prompt 
            | self.llm 
            | RunnableLambda(format_json_response)
        )
        
        # Risk Analysis
        risk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in risk assessment. Analyze the provided data and return a JSON response.

Example response format:
{{
    "risk_level": "HIGH/MEDIUM/LOW",
    "factors": ["risk1", "risk2"],
    "confidence": 0.92,
    "action": "Recommended action"
}}"""),
            ("human", "Evaluate these risk metrics: {risk_metrics}")
        ])
        
        risk_chain = (
            risk_prompt 
            | self.llm 
            | RunnableLambda(format_json_response)
        )
        
        return {
            "technical": technical_chain,
            "fundamental": fundamental_chain,
            "risk": risk_chain
        }
    
    def create_analysis_pipeline(
        self,
        components: List[str]
    ) -> RunnableLambda:
        """Create a custom analysis pipeline."""
        
        def analyze(inputs: Dict[str, Any]) -> Dict[str, Any]:
            results = {}
            
            try:
                for component in components:
                    if component in self.analysis_components:
                        result = self.analysis_components[component].invoke(inputs)
                        results[component] = json.loads(result)
                
                # Combine results
                combined_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an investment advisor. Create a final recommendation based on the analyses provided.

Return a JSON response with this format:
{{
    "status": "success",
    "recommendation": "Final recommendation",
    "confidence": 0.95,
    "rationale": ["reason1", "reason2"],
    "risk_level": "RISK_LEVEL"
}}"""),
                    ("human", "Analyze these results:\n{results}")
                ])
                
                final_analysis = (
                    combined_prompt 
                    | self.llm 
                    | RunnableLambda(format_json_response)
                ).invoke({"results": json.dumps(results, indent=2)})
                
                return json.loads(final_analysis)
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return RunnableLambda(analyze)

def demonstrate_investment_pipeline():
    """Demonstrate the investment pipeline builder."""
    print("\nInitializing Investment Pipeline Builder...\n")
    
    try:
        # Sample investment data
        data = {
            "asset_type": "stock",
            "price_history": [100, 102, 98, 103, 105],
            "market_cap": 5000000000,
            "sector": "Technology",
            "risk_metrics": {
                "volatility": 0.15,
                "beta": 1.2,
                "sharpe_ratio": 1.8
            }
        }
        
        # Create analyzer
        analyzer = InvestmentAnalyzer()
        
        # Create different pipeline configurations
        pipelines = [
            ["technical", "risk"],
            ["fundamental", "risk"],
            ["technical", "fundamental", "risk"]
        ]
        
        # Test each pipeline
        for components in pipelines:
            print(f"\nTesting pipeline with components: {components}")
            pipeline = analyzer.create_analysis_pipeline(components)
            result = pipeline.invoke(data)
            
            print("\nAnalysis Results:")
            print(json.dumps(result, indent=2))
            print("\n" + "="*50)
            
    except Exception as e:
        print(f"\nError in demonstration: {str(e)}")

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Investment Pipeline Builder...")
    demonstrate_investment_pipeline()

if __name__ == "__main__":
    main()
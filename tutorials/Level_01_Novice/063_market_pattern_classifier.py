#!/usr/bin/env python3
"""
LangChain Market Pattern Classifier (LangChain v3)

This example demonstrates pattern recognition using few-shot learning.
It analyzes price and volume data to identify common market patterns.

Key concepts demonstrated:
1. Few Shot Prompting: Using examples to guide pattern recognition
2. Example-Based Analysis: Learning from known patterns
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

class MarketPattern(BaseModel):
    """Market pattern analysis result."""
    pattern: str = Field(description="Name of the identified pattern")
    description: str = Field(description="Pattern description")
    trend: str = Field(description="Bullish/Bearish/Neutral")
    confidence: float = Field(description="Pattern confidence score")
    suggested_action: str = Field(description="Suggested trading action")

PROMPT_TEMPLATE = """You are an expert market pattern analyst. Analyze the given price and volume data to identify trading patterns.

Examples:

Example 1:
Input:
Prices: [100, 102, 101, 104, 103, 106]
Volumes: [1000, 1200, 900, 1300, 1100, 1400]

Analysis:
{{
    "pattern": "Bullish Channel",
    "description": "Higher highs and higher lows with increasing volume",
    "trend": "Bullish",
    "confidence": 0.85,
    "suggested_action": "Consider long position"
}}

Example 2:
Input:
Prices: [100, 98, 99, 96, 97, 94]
Volumes: [1000, 1100, 900, 1200, 950, 1300]

Analysis:
{{
    "pattern": "Bearish Channel",
    "description": "Lower highs and lower lows with increasing volume",
    "trend": "Bearish",
    "confidence": 0.82,
    "suggested_action": "Consider short position"
}}

Now analyze this new pattern:
Input:
Prices: {prices}
Volumes: {volumes}

Return ONLY a JSON object with your analysis."""

class PatternAnalyzer:
    """Market pattern analyzer using few-shot learning."""
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.parser = StrOutputParser()
    
    def analyze(
        self,
        prices: List[float],
        volumes: List[float]
    ) -> MarketPattern:
        """Analyze price and volume data for patterns."""
        try:
            # Create analysis chain
            chain = self.prompt | self.llm | self.parser
            
            # Get pattern analysis
            result = chain.invoke({
                "prices": str(prices),
                "volumes": str(volumes)
            })
            
            # Extract JSON
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                pattern_data = json.loads(json_str)
            else:
                raise ValueError("No valid pattern found in response")
            
            return MarketPattern(**pattern_data)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def demonstrate_pattern_analyzer():
    """Demonstrate the pattern analyzer."""
    print("\nAnalyzing Market Patterns")
    print("=" * 50)
    
    # Test patterns
    test_data = [
        {
            "name": "Uptrend Pattern",
            "prices": [100, 102, 101, 104, 103, 106],
            "volumes": [1000, 1200, 900, 1300, 1100, 1400]
        },
        {
            "name": "Downtrend Pattern",
            "prices": [100, 98, 99, 96, 97, 94],
            "volumes": [1000, 1100, 900, 1200, 950, 1300]
        },
        {
            "name": "Sideways Pattern",
            "prices": [100, 102, 99, 101, 98, 100],
            "volumes": [1000, 1100, 1050, 900, 950, 1000]
        }
    ]
    
    # Create analyzer
    analyzer = PatternAnalyzer()
    
    for i, data in enumerate(test_data, 1):
        print(f"\nAnalyzing {data['name']}:")
        print(f"Prices: {data['prices']}")
        print(f"Volumes: {data['volumes']}")
        
        try:
            result = analyzer.analyze(data["prices"], data["volumes"])
            
            print("\nIdentified Pattern:")
            print(f"Pattern: {result.pattern}")
            print(f"Description: {result.description}")
            print(f"Trend: {result.trend}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Suggested Action: {result.suggested_action}")
            
        except Exception as e:
            print(f"Could not analyze pattern: {str(e)}")
        
        if i < len(test_data):
            print("\n" + "-" * 50)
        else:
            print("\n" + "=" * 50)

if __name__ == "__main__":
    demonstrate_pattern_analyzer()
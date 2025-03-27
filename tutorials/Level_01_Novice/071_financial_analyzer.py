#!/usr/bin/env python3
"""
LangChain Financial Analyzer (LangChain v3)

This example demonstrates building a financial analyzer using structured output
and retrieval. It can process financial data from various sources and provide
insights into trends and anomalies.

Key concepts demonstrated:
1. Structured Output: Generating organized data outputs
2. Retrieval: Accessing and processing data from multiple sources
"""

import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Let the base class default method handle other types
        return super().default(obj)

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")
print("\nChecking Azure OpenAI configuration...")

from langchain_openai import AzureChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

class DataType(str, Enum):
    """Types of financial data."""
    STOCK = "stock"
    MARKET_TRENDS = "market_trends"
    FINANCIAL_REPORTS = "financial_reports"

class DataAnalysis(BaseModel):
    """Analysis results for financial data."""
    summary: str = Field(description="Summary of the data")
    key_insights: List[str] = Field(description="Key insights extracted from the data")
    requires_review: bool = Field(description="Whether manual review is needed")
    
class FinancialData(BaseModel):
    """Metadata for financial data."""
    data_id: str = Field(description="Data ID")
    type: DataType = Field(description="Type of data")
    content: str = Field(description="Data content")
    timestamp: datetime = Field(description="Data timestamp")
    
    def validate_content(self) -> bool:
        """Validate data content."""
        return len(self.content) > 0

class FinancialAnalyzer:
    """Analyze financial data using LangChain."""
    
    def __init__(self):
        """Initialize the financial analyzer."""
        # Setup LLMs
        self.data_llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.0
        )
        
        self.parser = PydanticOutputParser(pydantic_object=DataAnalysis)
        self.format_instructions = self.parser.get_format_instructions()
        
        # Setup analysis chain
        self.analyzer = self._create_analyzer()
    
    def _create_analyzer(self) -> Runnable:
        """Create the financial data analysis chain."""
        template = """You are an expert financial data analyzer. Analyze the following data and provide a summary and key insights.

DATA DETAILS
==================
Type: {type}
Content: {content}

ANALYSIS GUIDELINES
==================
Consider:
1. Data type and content
2. Key financial information
3. Compliance requirements
4. Known patterns

REQUIRED OUTPUT FORMAT
====================
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.data_llm | self.parser
        
        return chain

    def _handle_error(self, e: Exception, data_id: str) -> Dict:
        """Create standardized error response."""
        error_info = {
            "data_id": data_id,
            "processed_at": datetime.now(),
            "result": {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "details": self._get_error_details(e)
                }
            },
        }
        
        if isinstance(e, ValidationError):
            error_info["validation_errors"] = e.errors()
            error_info["result"]["category"] = "validation"
            error_info["result"]["error"]["suggestion"] = "Check input data format and constraints"
        elif isinstance(e, ValueError):
            error_info["result"]["category"] = "validation"
            error_info["result"]["error"]["suggestion"] = "Ensure all values meet requirements"
        else:
            error_info["result"]["category"] = "system"
            error_info["result"]["error"]["suggestion"] = "Contact system administrator"
        
        return error_info

    def _get_error_details(self, error: Exception) -> Dict:
        """Extract detailed error information."""
        details = {
            "error_chain": [],
            "traceback": None
        }
        
        # Build error chain
        current = error
        while current is not None:
            details["error_chain"].append({
                "type": type(current).__name__,
                "message": str(current)
            })
            current = current.__cause__

        # Get traceback if available
        if hasattr(error, "__traceback__"):
            import traceback
            tb_lines = traceback.format_tb(error.__traceback__)
            details["traceback"] = "".join(tb_lines)

        return details

    
    def analyze_data(self, data: FinancialData) -> Dict:
        """Analyze a single data set."""
        try:
            # Basic validation
            if not data.validate_content():
                raise ValueError("Data content cannot be empty")
            
            # Analyze data
            analysis = self.analyzer.invoke({
                "type": data.type,
                "content": data.content,
                "format_instructions": self.format_instructions
            })

            # Create result
            result = {
                "result": {
                    "success": True,
                    "processed_at": datetime.now(),
                    "summary": analysis.summary,
                    "key_insights": analysis.key_insights,
                    "requires_review": analysis.requires_review
                },
                "data_id": data.data_id,
                "analysis": analysis.model_dump(),
                "details": data.model_dump()
            }

            return result
            
        except Exception as e:
            return self._handle_error(e, data.data_id)

def demonstrate_analyzer():
    """Demonstrate the financial analyzer."""
    print("\nFinancial Analyzer Demo")
    print("Using Azure deployment:", AZURE_DEPLOYMENT)
    print("=" * 50)
    print("\nInitializing Financial Analyzer...")
    
    # Create sample data sets
    data_sets = [
        FinancialData(
            data_id="DATA001",
            type=DataType.STOCK,
            content="Stock data content with financial details",
            timestamp=datetime.now()
        ),
        FinancialData(
            data_id="DATA002",
            type=DataType.MARKET_TRENDS,
            content="Market trends data content with itemized charges",
            timestamp=datetime.now()
        ),
        FinancialData(
            data_id="DATA003",
            type=DataType.FINANCIAL_REPORTS,
            content="Financial reports content with transaction history",
            timestamp=datetime.now()
        ),
        # Invalid data for error handling test
        FinancialData(
            data_id="DATA004",
            type=DataType.FINANCIAL_REPORTS,
            content="",  # Empty content
            timestamp=datetime.now()
        )
    ]
    
    print("\nPreparing test data sets...")
    print(f"\nPrepared {len(data_sets)} test data sets")
    print("Including both valid and invalid cases for testing")
    print("\nTest Cases:")
    
    # Initialize analyzer
    try:
        analyzer = FinancialAnalyzer()
    except Exception as e:
        print("\nError initializing analyzer:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return
    
    print("\nAnalyzer initialized successfully")
    
    # Analyze data sets
    for idx, data in enumerate(data_sets, 1):
        print(f"\nTest Case {idx}/{len(data_sets)}")
        print("=" * 30)
        print(f"Data ID: {data.data_id}")
        print(f"Type: {data.type}")
        print("-" * 30 + "\n")

        result = analyzer.analyze_data(data)
        
        print("\nResult Details:")
        print("----------------")
        print(json.dumps(result, indent=2, cls=DateTimeEncoder))
        
        # Print summary
        if result.get("result", {}).get("success", False):
            summary = result.get("result", {}).get("summary", "unknown")
            key_insights = result.get("result", {}).get("key_insights", [])
            
            print("\nStatus: ✅ Success")
            print(f"Summary: {summary}")
            print(f"Key Insights: {', '.join(key_insights) if key_insights else 'None identified'}")
        else:
            error = result.get("result", {}).get("error", {})
            category = result.get("result", {}).get("category", "unknown")
            print("\nStatus: ❌ Failed")
            print(f"Error Type: {error.get('type', 'Unknown error')}")
            print(f"Message: {error.get('message', 'No error message')}")
            print(f"Category: {category.upper()}")
        print("=" * 50)

if __name__ == "__main__":
    demonstrate_analyzer()
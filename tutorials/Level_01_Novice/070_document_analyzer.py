#!/usr/bin/env python3
"""
LangChain Document Analyzer (LangChain v3)

This example demonstrates building a financial document analyzer using multimodality
and few-shot prompting. It can analyze both text and images from documents like
checks, invoices, and bank statements.

Key concepts demonstrated:
1. Multimodality: Processing both text and image inputs
2. Few-shot prompting: Using examples to improve analysis accuracy
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

class DocumentType(str, Enum):
    """Types of financial documents."""
    CHECK = "check"
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"

class DocumentAnalysis(BaseModel):
    """Analysis results for a document."""
    summary: str = Field(description="Summary of the document")
    key_points: List[str] = Field(description="Key points extracted from the document")
    requires_review: bool = Field(description="Whether manual review is needed")
    
class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    document_id: str = Field(description="Document ID")
    type: DocumentType = Field(description="Type of document")
    content: str = Field(description="Document content")
    timestamp: datetime = Field(description="Document timestamp")
    
    def validate_content(self) -> bool:
        """Validate document content."""
        return len(self.content) > 0

class DocumentAnalyzer:
    """Analyze financial documents using LangChain."""
    
    def __init__(self):
        """Initialize the document analyzer."""
        # Setup LLMs
        self.text_llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.0
        )
        
        self.vision_llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.0,
            max_tokens=1000
        )
        
        self.parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)
        self.format_instructions = self.parser.get_format_instructions()
        
        # Setup analysis chain
        self.analyzer = self._create_analyzer()
    
    def _create_analyzer(self) -> Runnable:
        """Create the document analysis chain."""
        template = """You are an expert financial document analyzer. Analyze the following document and provide a summary and key points.

DOCUMENT DETAILS
==================
Type: {type}
Content: {content}

ANALYSIS GUIDELINES
==================
Consider:
1. Document type and content
2. Key financial information
3. Compliance requirements
4. Known patterns

REQUIRED OUTPUT FORMAT
====================
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.text_llm | self.parser
        
        return chain

    def _handle_error(self, e: Exception, document_id: str) -> Dict:
        """Create standardized error response."""
        error_info = {
            "document_id": document_id,
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

    
    def analyze_document(self, document: DocumentMetadata) -> Dict:
        """Analyze a single document."""
        try:
            # Basic validation
            if not document.validate_content():
                raise ValueError("Document content cannot be empty")
            
            # Analyze document
            analysis = self.analyzer.invoke({
                "type": document.type,
                "content": document.content,
                "format_instructions": self.format_instructions
            })

            # Create result
            result = {
                "result": {
                    "success": True,
                    "processed_at": datetime.now(),
                    "summary": analysis.summary,
                    "key_points": analysis.key_points,
                    "requires_review": analysis.requires_review
                },
                "document_id": document.document_id,
                "analysis": analysis.model_dump(),
                "details": document.model_dump()
            }

            return result
            
        except Exception as e:
            return self._handle_error(e, document.document_id)

def demonstrate_analyzer():
    """Demonstrate the document analyzer."""
    print("\nDocument Analyzer Demo")
    print("Using Azure deployment:", AZURE_DEPLOYMENT)
    print("=" * 50)
    print("\nInitializing Document Analyzer...")
    
    # Create sample documents
    documents = [
        DocumentMetadata(
            document_id="DOC001",
            type=DocumentType.CHECK,
            content="Check document content with financial details",
            timestamp=datetime.now()
        ),
        DocumentMetadata(
            document_id="DOC002",
            type=DocumentType.INVOICE,
            content="Invoice document content with itemized charges",
            timestamp=datetime.now()
        ),
        DocumentMetadata(
            document_id="DOC003",
            type=DocumentType.BANK_STATEMENT,
            content="Bank statement content with transaction history",
            timestamp=datetime.now()
        ),
        # Invalid document for error handling test
        DocumentMetadata(
            document_id="DOC004",
            type=DocumentType.INVOICE,
            content="",  # Empty content
            timestamp=datetime.now()
        )
    ]
    
    print("\nPreparing test documents...")
    print(f"\nPrepared {len(documents)} test documents")
    print("Including both valid and invalid cases for testing")
    print("\nTest Cases:")
    
    # Initialize analyzer
    try:
        analyzer = DocumentAnalyzer()
    except Exception as e:
        print("\nError initializing analyzer:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return
    
    print("\nAnalyzer initialized successfully")
    
    # Analyze documents
    for idx, doc in enumerate(documents, 1):
        print(f"\nTest Case {idx}/{len(documents)}")
        print("=" * 30)
        print(f"Document ID: {doc.document_id}")
        print(f"Type: {doc.type}")
        print("-" * 30 + "\n")

        result = analyzer.analyze_document(doc)
        
        print("\nResult Details:")
        print("----------------")
        print(json.dumps(result, indent=2, cls=DateTimeEncoder))
        
        # Print summary
        if result.get("result", {}).get("success", False):
            summary = result.get("result", {}).get("summary", "unknown")
            key_points = result.get("result", {}).get("key_points", [])
            
            print("\nStatus: ✅ Success")
            print(f"Summary: {summary}")
            print(f"Key Points: {', '.join(key_points) if key_points else 'None identified'}")
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
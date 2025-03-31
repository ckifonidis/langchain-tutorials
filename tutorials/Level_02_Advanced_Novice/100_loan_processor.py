#!/usr/bin/env python3
"""
LangChain Loan Processor (100) (LangChain v3)

This example demonstrates a loan document processing system using three key concepts:
1. Document Loaders: Handle various loan document formats
2. Retrievers: Access relevant loan policies and requirements
3. Testing: Validate document processing accuracy

It provides comprehensive loan document processing and validation for banking applications.
"""

import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LoanDocument(BaseModel):
    """Schema for a loan document."""
    doc_id: str = Field(description="Document identifier")
    doc_type: str = Field(description="Type of loan document")
    content: str = Field(description="Document content")
    metadata: Dict = Field(description="Document metadata")

class ProcessingResult(BaseModel):
    """Schema for document processing results."""
    doc_id: str = Field(description="Document identifier")
    is_valid: bool = Field(description="Document validation status")
    missing_items: List[str] = Field(description="List of missing required items")
    validation_notes: str = Field(description="Validation notes")

class LoanProcessor:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )
        self.setup_document_processing()

    def setup_document_processing(self):
        """Set up document processing components."""
        # Text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Document loader
        self.loader = DirectoryLoader(
            "data/loan_documents/",
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        # Define loan requirements
        self.requirements = {
            "mortgage": [
                "Income verification must be provided",
                "Credit score report must be included",
                "Proof of employment is required",
                "Property valuation report must be attached",
                "Debt-to-income ratio calculation must be shown"
            ],
            "personal": [
                "Income verification must be provided",
                "Credit score report must be included",
                "Proof of employment is required",
                "Purpose of loan must be stated"
            ]
        }

    async def process_document(self, loan_doc: LoanDocument) -> ProcessingResult:
        """Process and validate a loan document."""
        try:
            # Split document for processing
            splits = self.text_splitter.split_text(loan_doc.content)
            docs = [Document(page_content=split) for split in splits]
            
            # Get requirements for document type
            requirements = self.requirements.get(loan_doc.doc_type, [])
            
            # Validate document against requirements
            missing_items = []
            for req in requirements:
                if not any(req.lower() in doc.page_content.lower() for doc in docs):
                    missing_items.append(req)
            
            # Create processing result
            return ProcessingResult(
                doc_id=loan_doc.doc_id,
                is_valid=len(missing_items) == 0,
                missing_items=missing_items,
                validation_notes="Document processed successfully"
            )
        except Exception as e:
            return ProcessingResult(
                doc_id=loan_doc.doc_id,
                is_valid=False,
                missing_items=["Error during processing"],
                validation_notes=f"Processing error: {str(e)}"
            )

    async def run_test_suite(self) -> Dict[str, Dict]:
        """Run test suite with predefined test cases."""
        test_results = {}
        
        # Test cases
        test_cases = [
            LoanDocument(
                doc_id="test_001",
                doc_type="mortgage",
                content="""
                Loan Application
                Income: $120,000/year verified with W2
                Credit Score: 750 from TransUnion
                Employment: Software Engineer at Tech Corp (5 years)
                Property Value: $500,000 assessed by ABC Appraisers
                Debt-to-Income: 28%
                """,
                metadata={"test_type": "complete"}
            ),
            LoanDocument(
                doc_id="test_002",
                doc_type="mortgage",
                content="""
                Loan Application
                Income: Not provided
                Credit Score: Not available
                Employment: Current position
                Property Value: Pending assessment
                """,
                metadata={"test_type": "incomplete"}
            )
        ]
        
        # Run tests
        for test_doc in test_cases:
            result = await self.process_document(test_doc)
            test_results[test_doc.doc_id] = {
                "valid": result.is_valid,
                "missing": result.missing_items
            }
        
        return test_results

async def demonstrate_loan_processor():
    print("\nLoan Document Processor Demo")
    print("===========================\n")
    
    processor = LoanProcessor()
    
    # Example loan documents
    documents = [
        LoanDocument(
            doc_id="loan_001",
            doc_type="mortgage",
            content="""
            Mortgage Application for 123 Main St
            Annual Income: $150,000 (W2 Attached)
            Current Credit Score: 780
            Employment: Senior Manager at ABC Corp (3 years)
            Property Valuation: $600,000
            Current Debt-to-Income Ratio: 25%
            """,
            metadata={"applicant": "John Doe", "loan_amount": "500000"}
        ),
        LoanDocument(
            doc_id="loan_002",
            doc_type="personal",
            content="""
            Personal Loan Application
            Purpose: Debt Consolidation
            Looking to borrow $40,000
            New job started recently
            Credit score to be provided
            """,
            metadata={"applicant": "Jane Smith", "loan_amount": "40000"}
        )
    ]
    
    # Process documents
    for doc in documents:
        print(f"Processing document: {doc.doc_id}")
        print(f"Type: {doc.doc_type}")
        print(f"Metadata: {doc.metadata}")
        
        result = await processor.process_document(doc)
        
        print("\nProcessing Result:")
        print(f"Valid: {result.is_valid}")
        print(f"Missing Items: {', '.join(result.missing_items) if result.missing_items else 'None'}")
        print(f"Notes: {result.validation_notes}\n")
        print("-" * 50 + "\n")
    
    # Run test suite
    print("Running test suite...")
    test_results = await processor.run_test_suite()
    for test_id, result in test_results.items():
        print(f"\nTest {test_id}:")
        print(f"Valid: {result['valid']}")
        print(f"Missing Items: {', '.join(result['missing']) if result['missing'] else 'None'}")
    print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_loan_processor())
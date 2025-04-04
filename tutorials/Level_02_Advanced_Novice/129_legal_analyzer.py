#!/usr/bin/env python3
"""
Legal Analyzer (129) (LangChain v3)

This example demonstrates legal document analysis using:
1. Text Splitting: Document segmentation
2. Chat Models: Content analysis
3. String Output: Structured results

It helps legal teams analyze contracts and agreements.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentType(str, Enum):
    """Legal document types."""
    LOAN = "loan_agreement"
    CREDIT = "credit_agreement"
    SERVICE = "service_agreement"
    EMPLOYMENT = "employment_contract"

class LegalDocument(BaseModel):
    """Legal document details."""
    doc_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    type: DocumentType = Field(description="Document type")
    content: str = Field(description="Document text")
    metadata: Dict = Field(default_factory=dict)

class DocumentAnalyzer:
    """Legal document analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting document analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("Text splitter ready")
        
        # Setup output parser
        self.parser = StrOutputParser()
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a legal document analyst.
Analyze documents and provide structured output.

Format your response exactly like this example:

DOCUMENT ANALYSIS
---------------
Overview: Brief summary of key points
Risk Level: HIGH/MEDIUM/LOW

Key Issues:
1. Section Name
   Risk: Level
   Issue: Description
   Fix: Required action
   Ref: Legal reference

Important Sections:
1. Section Name
   - Main point
   - Requirement
   - Implication

Required Actions:
1. Action description
2. Action description

Next Review: YYYY-MM-DD

Use exactly this format with no additional text."""),
            ("human", """Review this document:
Document: {doc_id}
Title: {title}
Type: {type}

Content:
{content}""")
        ])
        logger.info("Analysis prompt ready")

    def process_content(self, doc: LegalDocument) -> List[Document]:
        """Split document into chunks."""
        logger.info(f"Processing document: {doc.doc_id}")
        
        try:
            # Create chunks
            chunks = self.splitter.create_documents(
                texts=[doc.content],
                metadatas=[{"doc_id": doc.doc_id}]
            )
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    async def analyze_document(self, doc: LegalDocument) -> str:
        """Analyze legal document."""
        logger.info(f"Analyzing document: {doc.doc_id}")
        
        try:
            # Process content
            chunks = self.process_content(doc)
            logger.debug(f"Using {len(chunks)} chunks")
            
            # Format request
            messages = self.template.format_messages(
                doc_id=doc.doc_id,
                title=doc.title,
                type=doc.type.value,
                content=chunks[0].page_content
            )
            logger.debug("Template formatted")
            
            # Get analysis
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting document analysis demo...")
    
    try:
        # Create analyzer
        analyzer = DocumentAnalyzer()
        
        # Example document
        document = LegalDocument(
            doc_id="LOAN-2025-001",
            title="Commercial Loan Agreement",
            type=DocumentType.LOAN,
            content="""LOAN AGREEMENT

1. Loan Terms
1.1 Principal Amount: $500,000
1.2 Interest Rate: 5.5% per annum
1.3 Term: 60 months
1.4 Payment Schedule: Monthly payments

2. Security
2.1 Collateral: Commercial property
2.2 Insurance: Coverage required
2.3 Valuation: Annual assessment

3. Conditions
3.1 Use of Funds: Business only
3.2 Reporting: Quarterly required
3.3 Covenants: Maintain ratios

4. Default Events
4.1 Payment Default: Past due
4.2 Covenant Breach: Default
4.3 Material Change: Review

5. Representations
5.1 Legal Authority: Valid
5.2 Financial Status: Good
5.3 Disclosure: Complete

6. Remedies
6.1 Acceleration: Full amount
6.2 Property Seizure: After default
6.3 Legal Action: As needed"""
        )
        
        print("\nAnalyzing Document")
        print("=================")
        print(f"Document: {document.doc_id}")
        print(f"Title: {document.title}")
        print(f"Type: {document.type.value}\n")
        
        try:
            # Get analysis
            result = await analyzer.analyze_document(document)
            print("\nAnalysis Results:")
            print("================")
            print(result)
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
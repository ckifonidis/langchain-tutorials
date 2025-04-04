#!/usr/bin/env python3
"""
LangChain Legal Document Analyzer (109) (LangChain v3)

This example demonstrates a legal document analysis system using three key concepts:
1. Text Splitters: Smart document chunking
2. Output Parsers: Structured legal analysis
3. Callbacks: Progress tracking and logging

It provides comprehensive document analysis support for legal teams in banking.
"""

import os
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentType(str, Enum):
    """Types of legal documents."""
    CONTRACT = "contract"
    POLICY = "policy"
    REGULATION = "regulation"
    AGREEMENT = "agreement"
    DISCLOSURE = "disclosure"

class RiskLevel(str, Enum):
    """Risk levels for legal analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class LegalReference(BaseModel):
    """Schema for legal references."""
    section: str = Field(description="Document section or clause")
    context: str = Field(description="Relevant legal context")
    requirement: str = Field(description="Legal requirement")
    source: str = Field(description="Source regulation or law")

class ComplianceIssue(BaseModel):
    """Schema for compliance issues."""
    issue: str = Field(description="Issue description")
    risk_level: RiskLevel = Field(description="Risk level")
    impact: str = Field(description="Potential impact")
    recommendation: str = Field(description="Recommended action")
    references: List[LegalReference] = Field(description="Related legal references")

class DocumentAnalysis(BaseModel):
    """Schema for document analysis results."""
    document_id: str = Field(description="Document identifier")
    document_type: DocumentType = Field(description="Type of document")
    timestamp: str = Field(description="Analysis timestamp")
    issues: List[ComplianceIssue] = Field(description="Identified issues")
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(description="Analysis confidence score")

class AnalysisProgressCallback(BaseCallbackHandler):
    """Callback handler for analysis progress tracking."""
    def __init__(self):
        self.current_section = "Not started"
        self.total_chunks = 0
        self.processed_chunks = 0

    def on_llm_start(self, *args, **kwargs):
        print(f"\nAnalyzing section: {self.current_section}")

    def on_llm_end(self, *args, **kwargs):
        self.processed_chunks += 1
        progress = (self.processed_chunks / self.total_chunks) * 100
        print(f"Progress: {progress:.1f}% ({self.processed_chunks}/{self.total_chunks})")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], *args, **kwargs):
        print(f"\nError in section {self.current_section}: {str(error)}")

class LegalDocumentAnalyzer:
    def __init__(self):
        # Initialize LLM with callbacks
        self.callback_handler = AnalysisProgressCallback()
        self.callback_manager = CallbackManager([self.callback_handler])
        
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            callbacks=self.callback_manager
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
            keep_separator=True
        )
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ComplianceIssue)

    async def analyze_document(self, document: str, document_id: str, 
                             document_type: DocumentType) -> DocumentAnalysis:
        """Analyze a legal document."""
        try:
            # Split document into chunks
            chunks = self.text_splitter.split_text(document)
            self.callback_handler.total_chunks = len(chunks)
            self.callback_handler.processed_chunks = 0
            
            issues = []
            for i, chunk in enumerate(chunks):
                self.callback_handler.current_section = f"Section {i+1}"
                
                # Analyze chunk for compliance issues
                prompt = f"""
                Analyze this section of a {document_type.value} for compliance issues:

                {chunk}

                Identify any potential compliance issues, considering:
                1. Regulatory requirements
                2. Banking standards
                3. Legal obligations
                4. Risk factors

                {self.output_parser.get_format_instructions()}
                """
                
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an expert legal analyst specializing in banking regulations."),
                    HumanMessage(content=prompt)
                ])
                
                try:
                    issue = self.output_parser.parse(response.content)
                    if issue.risk_level != RiskLevel.LOW:
                        issues.append(issue)
                except Exception as parse_error:
                    print(f"Error parsing section {i+1}: {str(parse_error)}")
            
            # Generate summary
            summary_prompt = f"""
            Summarize the analysis of this {document_type.value}:

            Issues Found: {len(issues)}
            Risk Levels: {[issue.risk_level for issue in issues]}

            Provide a concise summary of the document analysis and overall risk assessment.
            """
            
            summary_response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert legal analyst."),
                HumanMessage(content=summary_prompt)
            ])
            
            # Calculate confidence based on analysis coverage
            confidence = self.callback_handler.processed_chunks / self.callback_handler.total_chunks
            
            return DocumentAnalysis(
                document_id=document_id,
                document_type=document_type,
                timestamp=datetime.now().isoformat(),
                issues=issues,
                summary=summary_response.content,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Error analyzing document: {str(e)}")
            raise

async def demonstrate_legal_analyzer():
    print("\nLegal Document Analyzer Demo")
    print("===========================\n")

    # Example banking policy document
    sample_document = """
    SECTION 1: CUSTOMER DATA PROTECTION POLICY
    
    1.1 Data Collection and Storage
    The bank shall collect and store customer data in accordance with GDPR and local 
    banking regulations. Personal data must be encrypted using industry-standard methods.
    
    1.2 Data Access and Processing
    Access to customer data is restricted to authorized personnel only. All data access 
    must be logged and monitored. Processing of personal data requires explicit consent.
    
    SECTION 2: TRANSACTION MONITORING
    
    2.1 Anti-Money Laundering (AML) Procedures
    All transactions exceeding €10,000 must undergo enhanced due diligence. 
    Suspicious transactions must be reported to the compliance department within 24 hours.
    
    2.2 Customer Verification
    Customer identity must be verified using two-factor authentication for online 
    transactions. Physical presence is required for account opening.
    """

    try:
        # Initialize analyzer
        analyzer = LegalDocumentAnalyzer()

        print("Analyzing document...")
        analysis = await analyzer.analyze_document(
            document=sample_document,
            document_id="POL-2025-001",
            document_type=DocumentType.POLICY
        )

        print("\nAnalysis Results:")
        print(f"Document: {analysis.document_id}")
        print(f"Type: {analysis.document_type}")
        print(f"Confidence: {analysis.confidence:.2%}")
        
        print("\nIdentified Issues:")
        for i, issue in enumerate(analysis.issues, 1):
            print(f"\nIssue {i}:")
            print(f"Description: {issue.issue}")
            print(f"Risk Level: {issue.risk_level}")
            print(f"Impact: {issue.impact}")
            print(f"Recommendation: {issue.recommendation}")
            print("\nReferences:")
            for ref in issue.references:
                print(f"- Section {ref.section}: {ref.requirement}")
                print(f"  Source: {ref.source}")

        print(f"\nSummary:")
        print(analysis.summary)
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_legal_analyzer())
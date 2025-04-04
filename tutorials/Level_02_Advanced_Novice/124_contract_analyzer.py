#!/usr/bin/env python3
"""
Contract Analyzer (124) (LangChain v3)

This example demonstrates legal document analysis using:
1. Document Processing: Content extraction
2. Text Splitting: Document segmentation
3. String Output: Clear formatting

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

class ContractType(str, Enum):
    """Contract types."""
    LOAN = "loan_agreement"
    CREDIT = "credit_agreement"
    MORTGAGE = "mortgage_contract"
    SERVICE = "service_agreement"

class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Contract(BaseModel):
    """Legal contract details."""
    contract_id: str = Field(description="Contract ID")
    type: ContractType = Field(description="Contract type")
    parties: List[str] = Field(description="Contract parties")
    content: str = Field(description="Contract text")
    metadata: Dict = Field(default_factory=dict)

class ContractAnalyzer:
    """Legal document analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting contract analyzer...")
        
        # Chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("Text splitter ready")
        
        # Output parser
        self.parser = StrOutputParser()
        
        # Analysis prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal document analyst.
Review contracts and provide structured analysis.

Format your response exactly like this example:

CONTRACT ANALYSIS
---------------
Overview: Brief summary of the contract
Risk Level: HIGH/MEDIUM/LOW

Issues Found:
1. Section Name
   Risk: Level
   Issue: Description
   Fix: Solution
   Ref: Legal reference

Key Clauses:
1. Clause Title
   - Main points
   - Requirements
   - Implications

Required Actions:
1. Action details
2. Action details

Next Review: YYYY-MM-DD"""),
            ("human", """Review this contract:

Contract: {contract_id}
Type: {contract_type}
Parties: {parties}

Content:
{content}

Provide a structured analysis.""")
        ])
        logger.info("Analysis prompt ready")

    def split_document(self, contract: Contract) -> List[Document]:
        """Process contract content."""
        logger.info(f"Processing contract: {contract.contract_id}")
        
        try:
            # Split content
            docs = self.splitter.create_documents(
                texts=[contract.content],
                metadatas=[{"contract_id": contract.contract_id}]
            )
            logger.info(f"Created {len(docs)} chunks")
            return docs
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    async def analyze_contract(self, contract: Contract) -> str:
        """Analyze contract document."""
        logger.info(f"Analyzing contract: {contract.contract_id}")
        
        try:
            # Split content
            chunks = self.split_document(contract)
            
            # Format request
            messages = self.prompt.format_messages(
                contract_id=contract.contract_id,
                contract_type=contract.type.value,
                parties=", ".join(contract.parties),
                content=chunks[0].page_content
            )
            logger.debug("Request formatted")
            
            # Get and parse response
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting contract analysis demo...")
    
    try:
        # Create analyzer
        analyzer = ContractAnalyzer()
        
        # Example contract
        contract = Contract(
            contract_id="LOAN-2025-001",
            type=ContractType.LOAN,
            parties=["Bank", "Customer"],
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
        
        print("\nAnalyzing Contract")
        print("=================")
        print(f"Contract: {contract.contract_id}")
        print(f"Type: {contract.type.value}")
        print(f"Parties: {', '.join(contract.parties)}\n")
        
        try:
            # Get analysis
            result = await analyzer.analyze_contract(contract)
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
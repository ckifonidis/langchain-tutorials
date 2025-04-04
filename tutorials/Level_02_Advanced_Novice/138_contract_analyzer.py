#!/usr/bin/env python3
"""
Contract Analyzer (138) (LangChain v3)

This example demonstrates legal document analysis using:
1. Text Splitters: Document chunking
2. Embeddings: Semantic search
3. Example Selection: Pattern matching

It helps legal teams analyze banking contracts.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ContractType(str, Enum):
    """Contract categories."""
    LOAN = "loan_agreement"
    CREDIT = "credit_facility"
    ACCOUNT = "account_terms"
    MORTGAGE = "mortgage_agreement"
    INVESTMENT = "investment_terms"
    SERVICE = "service_agreement"

class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_risk"

class Contract(BaseModel):
    """Contract details."""
    contract_id: str = Field(description="Contract ID")
    type: ContractType = Field(description="Contract type")
    content: str = Field(description="Contract text")
    metadata: Dict = Field(default_factory=dict)

class ContractAnalyzer:
    """Contract analysis system."""

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Starting contract analyzer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("AZURE_MODEL_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )
        logger.info("Embeddings ready")
        
        # Setup text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("Text splitter ready")
        
        # Setup analysis template with examples
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a legal document analyst.
Review contracts and identify risks and requirements.

Consider these example analyses:

Example 1 - Loan Agreement:
RISK ANALYSIS
-------------
- Variable interest rates need caps
- Early termination needs fair penalties
- Default triggers need clear definition
- Covenant ratios need monitoring

Example 2 - Account Terms:
COMPLIANCE CHECK
---------------
- Privacy terms meet GDPR
- Fee disclosure complete
- Dispute resolution clear
- Notice periods defined

Format your response like this:

CONTRACT ANALYSIS
---------------
Document: ID
Type: Category
Section: Number

Key Findings:
1. Finding Name
   Location: Reference
   Details: Description
   Impact: Effect

2. Finding Name
   Location: Reference
   Details: Description
   Impact: Effect

Risk Assessment:
- Risk level
- Risk factors
- Risk controls

Requirements:
1. Required item
2. Required item

Next Steps:
1. Action item
2. Action item"""),
            ("human", """Analyze this contract:
ID: {contract_id}
Type: {contract_type}

Content:
{content}

Find key risks and requirements.""")
        ])
        logger.info("Analysis template ready")
        
        # Setup output parser
        self.parser = StrOutputParser()

    async def analyze_contract(self, contract: Contract) -> str:
        """Analyze contract content."""
        logger.info(f"Analyzing contract: {contract.contract_id}")
        
        try:
            # Split content
            chunks = self.splitter.split_text(contract.content)
            # Configure logging to suppress FAISS GPU warning
            logging.getLogger('faiss').setLevel(logging.ERROR)
            
            logger.info(f"Content split into {len(chunks)} chunks")
            
            try:
                # Try to create vector store
                vectorstore = FAISS.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    metadatas=[{"chunk": i} for i in range(len(chunks))]
                )
                logger.info("Vector store ready")
                
                # Get top relevant chunks
                query = f"key clauses in {contract.type.value}"
                relevant = vectorstore.similarity_search(query, k=3)
                relevant_text = "\n\n".join(chunk.page_content for chunk in relevant)
                logger.info("Found relevant sections")
                
            except Exception as e:
                logger.warning(f"Vector search unavailable: {str(e)}")
                # Use first few chunks as fallback
                relevant_text = "\n\n".join(chunks[:3])
                logger.info("Using first sections")
            
            # Get initial analysis
            messages = self.template.format_messages(
                contract_id=contract.contract_id,
                contract_type=contract.type.value,
                content=contract.content[:2000]  # Initial part
            )
            
            # Update with relevant parts
            messages[1].content += f"\n\nAdditional sections:\n{relevant_text}"
            
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
    logger.info("Starting contract analysis demo...")
    
    try:
        # Create analyzer
        analyzer = ContractAnalyzer()
        
        # Example contract
        contract = Contract(
            contract_id="CONT-2025-001",
            type=ContractType.LOAN,
            content="""LOAN AGREEMENT

1. DEFINITIONS
1.1 "Lender" means Acme Bank Ltd.
1.2 "Borrower" means the customer.
1.3 "Principal" means initial loan amount.

2. LOAN TERMS
2.1 Principal amount: $500,000
2.2 Term: 60 months
2.3 Interest: Variable rate LIBOR + 2.5%
2.4 Payment: Monthly installments

3. CONDITIONS
3.1 Purpose: Business expansion
3.2 Collateral: Business assets
3.3 Early repayment: 2% penalty
3.4 Default: Immediate repayment

4. REPRESENTATIONS
4.1 Borrower warrants all information true
4.2 No bankruptcy or litigation pending
4.3 Maintains required insurance

5. COVENANTS
5.1 Monthly financial statements
5.2 Maintain asset ratios
5.3 No additional debt
5.4 Asset sales restricted

6. EVENTS OF DEFAULT
6.1 Payment failure
6.2 Covenant breach
6.3 Material adverse change
6.4 Change of control

7. MISCELLANEOUS
7.1 Governing law
7.2 Amendments in writing
7.3 Notices
7.4 Assignment restricted"""
        )
        
        print("\nAnalyzing Contract")
        print("=================")
        print(f"Contract: {contract.contract_id}")
        print(f"Type: {contract.type.value}\n")
        
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
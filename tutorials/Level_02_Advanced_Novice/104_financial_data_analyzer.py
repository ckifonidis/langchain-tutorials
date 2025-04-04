#!/usr/bin/env python3
"""
LangChain Financial Data Analyzer (104) (LangChain v3)

This example demonstrates a financial data analysis system using three key concepts:
1. Text Splitters: Process large financial datasets
2. Retrievers: Semantic search in financial data
3. Vector Stores: Store and retrieve financial embeddings

It provides comprehensive financial data analysis for data science teams in banking.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialDocument(BaseModel):
    """Schema for financial documents."""
    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    content: str = Field(description="Document content")
    category: str = Field(description="Document category")
    metadata: Dict = Field(description="Document metadata")

class AnalysisResult(BaseModel):
    """Schema for analysis results."""
    query: str = Field(description="Search query")
    relevant_sections: List[str] = Field(description="Relevant document sections")
    financial_metrics: Dict = Field(description="Extracted financial metrics")
    insights: List[str] = Field(description="Key insights")

class FinancialDataAnalyzer:
    def __init__(self):
        # Initialize Azure OpenAI embeddings
        deployment = os.getenv("AZURE_DEPLOYMENT")
        if not deployment:
            raise ValueError("AZURE_DEPLOYMENT environment variable must be set")
            
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=deployment,
            model=os.getenv("AZURE_MODEL_NAME", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ",", " "],
            length_function=len
        )
        
        # Initialize document store
        self.doc_store = InMemoryStore()
        
        # Initialize vector store with persist_directory
        persist_dir = "./financial_data_store"
        os.makedirs(persist_dir, exist_ok=True)
        
        self.vector_store = Chroma(
            collection_name="financial_data",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        
        # Initialize retriever with default parameters
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.doc_store,
            child_splitter=self.text_splitter,
            search_kwargs={"k": 2}  # Reduced to avoid empty results warning
        )

    def extract_financial_metric(self, text: str) -> List[Tuple[str, str]]:
        """Extract financial metrics with proper context."""
        metrics = []
        
        # Pattern for financial values (numbers with optional $ and M/B/K suffixes)
        pattern = r'([^:\n]+):\s*(\$?\d+\.?\d*[MBK%]?)'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            metric_name = match.group(1).strip()
            value = match.group(2).strip()
            metrics.append((metric_name, value))
            
        return metrics

    async def process_document(self, document: FinancialDocument) -> bool:
        """Process and index a financial document."""
        try:
            # Split and index the document
            splits = self.text_splitter.create_documents(
                texts=[document.content],
                metadatas=[{
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "category": document.category,
                    **document.metadata
                }]
            )
            
            # Add directly to vector store
            self.vector_store.add_documents(splits)
            
            # Store full document in document store
            self.doc_store.mset([(doc.page_content, doc) for doc in splits])
            return True
            
        except Exception as e:
            print(f"Error processing document {document.doc_id}: {str(e)}")
            return False

    async def search_financial_data(self, query: str) -> AnalysisResult:
        """Search financial data and analyze results."""
        try:
            # Use similarity search directly
            docs = self.vector_store.similarity_search(query)
            
            # Extract financial metrics
            metrics = {}
            for doc in docs:
                extracted = self.extract_financial_metric(doc.page_content)
                metrics.update({k: v for k, v in extracted})
            
            # Generate insights
            insights = [
                f"Found {len(docs)} relevant sections",
                f"Extracted {len(metrics)} financial metrics",
                "Analysis based on most recent data"
            ]
            
            return AnalysisResult(
                query=query,
                relevant_sections=[doc.page_content for doc in docs],
                financial_metrics=metrics,
                insights=insights
            )
            
        except Exception as e:
            print(f"Error searching data: {str(e)}")
            return AnalysisResult(
                query=query,
                relevant_sections=[],
                financial_metrics={},
                insights=[f"Error during analysis: {str(e)}"]
            )

async def demonstrate_financial_analyzer():
    print("\nFinancial Data Analyzer Demo")
    print("===========================\n")

    try:
        analyzer = FinancialDataAnalyzer()
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        return

    # Example documents
    documents = [
        FinancialDocument(
            doc_id="fin_001",
            title="Q2 2025 Financial Report",
            content="""Quarterly financial performance summary:
            Revenue: $125.3M
            Operating Expenses: $82.7M
            Net Profit: $42.6M
            Customer Acquisition Cost: $250
            Customer Lifetime Value: $2,800
            Churn Rate: 2.3%""",
            category="Financial Reports",
            metadata={"department": "Finance", "quarter": "Q2", "year": "2025"}
        ),
        FinancialDocument(
            doc_id="fin_002",
            title="Risk Analysis Report",
            content="""Market risk assessment:
            Value at Risk (VaR): $15.2M
            Risk-weighted Assets: $890.5M
            Capital Adequacy Ratio: 15.8%
            Liquidity Coverage Ratio: 125%
            Non-performing Loans: 1.2%""",
            category="Risk Analysis",
            metadata={"department": "Risk", "period": "2025-H1"}
        )
    ]

    # Process documents
    for doc in documents:
        print(f"Processing Document: {doc.title}")
        print(f"Category: {doc.category}")
        print(f"Department: {doc.metadata['department']}\n")
        
        success = await analyzer.process_document(doc)
        if success:
            print("Document processed successfully")
        else:
            print("Error processing document")
        print("-" * 50 + "\n")

    # Example searches
    queries = [
        "What is our current revenue and profitability?",
        "What are our key risk metrics?"
    ]

    for query in queries:
        print(f"Query: {query}\n")
        result = await analyzer.search_financial_data(query)
        
        print("Analysis Results:")
        print("\nRelevant Sections:")
        for section in result.relevant_sections:
            print(f"- {section}")
        
        print("\nFinancial Metrics:")
        for metric, value in result.financial_metrics.items():
            print(f"- {metric}: {value}")
        
        print("\nInsights:")
        for insight in result.insights:
            print(f"- {insight}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_financial_analyzer())
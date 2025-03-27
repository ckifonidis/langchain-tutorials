#!/usr/bin/env python3
"""
LangChain Financial Document Analyzer (LangChain v3)

This example demonstrates how to build a financial document analysis system using
text splitting and retrieval capabilities. The system processes financial reports,
splits them intelligently, and enables semantic search for specific financial data.

Key concepts demonstrated:
1. Text Splitters: Intelligent document splitting for financial reports
2. Retrieval: Semantic search in financial documents
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://ai-agent-swarm-1.openai.azure.com/"
AZURE_API_KEY = "979b84bde7c04d8784208309bcdac5d0"
AZURE_API_VERSION = "2024-02-15-preview"

# Model Deployments
AZURE_CHAT_DEPLOYMENT = "gpt-4"  # For chat completions
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small-3"  # For embeddings

class FinancialData(BaseModel):
    """Schema for financial data extraction."""
    metric_name: str = Field(description="Financial metric name")
    value: float = Field(description="Metric value")
    period: str = Field(description="Reporting period")
    category: str = Field(description="Data category")

class SearchResult(BaseModel):
    """Schema for search results."""
    context: str = Field(description="Found context")
    relevance: float = Field(description="Relevance score")
    data: Optional[FinancialData] = Field(description="Extracted data")

class FinancialDocumentAnalyzer:
    """Financial document analysis system."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ","],
            keep_separator=True
        )
        
        # Initialize embeddings with correct deployment
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY
        )
        
        # Initialize LLM with chat deployment
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        
        self.vectorstore = None
    
    def process_document(self, text: str) -> None:
        """Process and index financial document."""
        try:
            print("Splitting document...")
            splits = self.text_splitter.split_text(text)
            
            print(f"Created {len(splits)} document chunks")
            
            # Create documents with metadata
            documents = []
            for i, split in enumerate(splits):
                doc = Document(
                    page_content=split,
                    metadata={
                        "chunk_id": i,
                        "source": "financial_report",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            print("Creating embeddings and vector store...")
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
            else:
                self.vectorstore.add_documents(documents)
            
            print("Document processing complete")
                
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    def search_financial_data(
        self,
        query: str,
        k: int = 3
    ) -> List[SearchResult]:
        """Search for specific financial information."""
        try:
            if not self.vectorstore:
                raise ValueError("No documents processed yet")
            
            print(f"Searching for: {query}")
            # Perform similarity search
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=k
            )
            
            print(f"Found {len(docs_and_scores)} relevant chunks")
            
            # Extract financial data from contexts
            results = []
            for doc, score in docs_and_scores:
                print(f"Processing chunk with score: {score}")
                # Create prompt for data extraction
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a financial data extraction specialist. Extract financial metrics from the given text and return them in JSON format."),
                    ("human", """Extract financial metrics from this text and return a valid JSON object:

Text to analyze:
{text}

The response must be a single JSON object with these fields:
- metric_name: name of the financial metric
- value: numeric value (remove any currency symbols)
- period: time period
- category: metric category

Example response:
{{"metric_name": "Total Revenue", "value": 2500000000, "period": "Q2 2024", "category": "Revenue"}}""".format(text=doc.page_content))
                ])
                
                # Extract data
                try:
                    messages = prompt.format_messages()
                    response = self.llm.invoke(messages)
                    
                    # Clean and parse the response
                    content = response.content.strip()
                    print(f"Raw response: {content}")
                    
                    # Remove any markdown formatting
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].strip()
                    
                    # Parse JSON
                    data = FinancialData(**json.loads(content))
                    print("Successfully extracted data")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}\nContent: {content}")
                    data = None
                except Exception as e:
                    print(f"Data extraction error: {str(e)}")
                    data = None
                
                # Create search result
                result = SearchResult(
                    context=doc.page_content,
                    relevance=float(score),
                    data=data
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching data: {str(e)}")
            return []

def demonstrate_document_analysis():
    """Demonstrate the document analysis system."""
    print("\nInitializing Financial Document Analyzer...\n")
    
    # Sample financial report text
    report = """
    Q2 2024 Financial Results
    
    Revenue Performance:
    Total revenue reached $2.5 billion in Q2 2024, representing a 15% increase
    year-over-year. Subscription revenue was $1.8 billion, up 18% from Q2 2023.
    
    Profitability Metrics:
    Gross margin improved to 72% in Q2 2024, compared to 70% in Q2 2023.
    Operating income was $500 million, with an operating margin of 20%.
    
    Cash Flow and Balance Sheet:
    Operating cash flow was $600 million in Q2 2024. Free cash flow reached
    $450 million. Cash and investments totaled $4.2 billion at quarter end.
    
    Key Performance Indicators:
    - Customer acquisition cost: $1,200
    - Monthly recurring revenue: $150 million
    - Customer lifetime value: $85,000
    - Churn rate: 2.5%
    """
    
    try:
        # Create analyzer
        analyzer = FinancialDocumentAnalyzer()
        
        # Process document
        print("Processing financial report...")
        analyzer.process_document(report)
        
        # Example queries
        queries = [
            "What was the total revenue?",
            "What are the profitability metrics?",
            "What is the cash position?"
        ]
        
        # Search for each query
        for query in queries:
            print(f"\nSearching: {query}")
            results = analyzer.search_financial_data(query)
            
            print("\nResults:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Relevance Score: {result.relevance:.2f}")
                print(f"Context: {result.context}")
                if result.data:
                    print("Extracted Data:")
                    print(json.dumps(
                        result.data.dict(),
                        indent=2
                    ))
            print("\n" + "="*50)
            
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Financial Document Analyzer...")
    demonstrate_document_analysis()

if __name__ == "__main__":
    main()
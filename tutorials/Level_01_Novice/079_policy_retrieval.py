#!/usr/bin/env python3
"""
Bank Policy Document Retrieval System (LangChain v3)

This example demonstrates a banking policy and procedure retrieval system using
key_methods for structured access and retrieval for finding relevant documents.
It provides efficient document search capabilities for banking compliance.

Key concepts demonstrated:
1. key_methods: Structured API interface for document access
2. retrieval: Intelligent document search and relevance ranking
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration - Use exact names from .env
AZURE_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_MODEL = os.getenv("AZURE_MODEL_NAME")

# Validate using correct environment variable names
missing_vars = []
for var_name, var_value in {
    "AZURE_EMBEDDING_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_API_KEY": AZURE_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION,
    "AZURE_DEPLOYMENT": AZURE_DEPLOYMENT,
    "AZURE_MODEL_NAME": AZURE_MODEL
}.items():
    if not var_value:
        missing_vars.append(var_name)

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

print("\nChecking Azure OpenAI configuration...")
print(f"Using deployment: {AZURE_DEPLOYMENT}")
print(f"Using model: {AZURE_MODEL}")

class PolicyDocument(BaseModel):
    """Bank policy document schema."""
    id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    category: str = Field(description="Policy category")
    content: str = Field(description="Document content")
    department: str = Field(description="Department")
    version: str = Field(description="Version number")
    last_updated: datetime = Field(description="Last update time")

class SearchQuery(BaseModel):
    """Search query schema."""
    query: str = Field(description="Search text")
    category: Optional[str] = Field(default=None, description="Category filter")
    department: Optional[str] = Field(default=None, description="Department filter")
    max_results: int = Field(default=5, description="Maximum results")

class SearchResult(BaseModel):
    """Search result schema."""
    document_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    score: float = Field(description="Relevance score")
    snippet: str = Field(description="Content snippet")
    metadata: Dict[str, Any] = Field(description="Metadata")

class PolicyRetriever:
    """Policy document retriever."""
    
    def __init__(self, documents: List[PolicyDocument]):
        """Initialize retrievers."""
        try:
            print("\nInitializing embeddings...")
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_DEPLOYMENT,
                openai_api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                model=AZURE_MODEL
            )
            
            # Store original documents and metadata
            print("Preparing documents...")
            self.documents = {doc.id: doc for doc in documents}
            self.texts = [doc.content for doc in documents]
            self.metadatas = [{
                "id": doc.id,
                "title": doc.title,
                "category": doc.category,
                "department": doc.department,
                "version": doc.version
            } for doc in documents]
            
            # Create vector store
            print("Creating vector store...")
            self.vectorstore = FAISS.from_texts(
                texts=self.texts,
                embedding=self.embeddings,
                metadatas=self.metadatas
            )
            
            # Create keyword retriever with metadata
            print("Creating keyword retriever...")
            self.keyword_retriever = BM25Retriever.from_texts(
                self.texts,
                metadatas=self.metadatas
            )
            
            print("Initialization complete")
            
        except Exception as e:
            print(f"\nError during initialization: {str(e)}")
            raise
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search documents."""
        try:
            print(f"\nProcessing search: {query.query}")
            
            # Get semantic results
            vector_docs = self.vectorstore.similarity_search_with_score(
                query.query,
                k=query.max_results * 2
            )
            print(f"Found {len(vector_docs)} semantic matches")
            
            # Get keyword results
            keyword_docs = self.keyword_retriever.get_relevant_documents(
                query.query
            )[:query.max_results * 2]
            print(f"Found {len(keyword_docs)} keyword matches")
            
            # Combine and deduplicate results
            seen_ids = set()
            results = []
            
            # Process semantic results
            for doc, score in vector_docs:
                if len(results) >= query.max_results:
                    break
                    
                doc_id = doc.metadata["id"]
                if doc_id not in self.documents or doc_id in seen_ids:
                    continue
                
                # Apply filters
                if query.category and doc.metadata["category"] != query.category:
                    continue
                if query.department and doc.metadata["department"] != query.department:
                    continue
                
                results.append(SearchResult(
                    document_id=doc_id,
                    title=doc.metadata["title"],
                    score=float(score),
                    snippet=doc.page_content[:200] + "...",
                    metadata=doc.metadata
                ))
                seen_ids.add(doc_id)
            
            # Add keyword results
            for doc in keyword_docs:
                if len(results) >= query.max_results:
                    break
                    
                # Get metadata
                doc_id = doc.metadata["id"]
                if doc_id not in self.documents or doc_id in seen_ids:
                    continue
                
                metadata = doc.metadata
                
                # Apply filters
                if query.category and metadata["category"] != query.category:
                    continue
                if query.department and metadata["department"] != query.department:
                    continue
                
                results.append(SearchResult(
                    document_id=doc_id,
                    title=metadata["title"],
                    score=0.5,  # Default score for keyword matches
                    snippet=doc.page_content[:200] + "...",
                    metadata=metadata
                ))
                seen_ids.add(doc_id)
            
            print(f"Returning {len(results)} filtered results")
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            raise  # Re-raise the error for better debugging

def create_sample_documents() -> List[PolicyDocument]:
    """Create sample policy documents."""
    return [
        PolicyDocument(
            id="POL-001",
            title="Customer Due Diligence Policy",
            category="Compliance",
            content="""Detailed procedures for customer verification including:
            1. KYC requirements and documentation
            2. Identity verification process
            3. Risk assessment guidelines
            4. Due diligence levels and triggers
            5. Ongoing monitoring requirements""",
            department="Risk",
            version="1.2",
            last_updated=datetime.now()
        ),
        PolicyDocument(
            id="POL-002",
            title="Transaction Monitoring Procedures",
            category="Operations",
            content="""Guidelines for monitoring transactions including:
            1. Suspicious activity detection
            2. Alert thresholds and triggers
            3. Investigation procedures
            4. Reporting requirements
            5. Documentation standards""",
            department="Operations",
            version="2.1",
            last_updated=datetime.now()
        ),
        PolicyDocument(
            id="POL-003",
            title="Credit Risk Assessment",
            category="Risk",
            content="""Methods and procedures for credit risk including:
            1. Credit scoring models
            2. Risk factor analysis
            3. Approval thresholds
            4. Documentation requirements
            5. Review schedules""",
            department="Credit",
            version="1.5",
            last_updated=datetime.now()
        )
    ]

def demonstrate_search():
    """Demonstrate policy search."""
    print("\nBank Policy Search Demo")
    print("=====================")
    
    try:
        # Create retriever
        print("\nInitializing search system...")
        documents = create_sample_documents()
        retriever = PolicyRetriever(documents)
        
        # Example searches
        queries = [
            SearchQuery(
                query="customer verification procedures",
                category="Compliance",
                max_results=2
            ),
            SearchQuery(
                query="transaction monitoring",
                department="Operations",
                max_results=2
            ),
            SearchQuery(
                query="credit assessment methods",
                category="Risk",
                max_results=2
            )
        ]
        
        # Run searches
        for query in queries:
            print(f"\nSearching: {query.query}")
            print(f"Filters: category='{query.category}', department='{query.department}'")
            print("-" * 50)
            
            try:
                results = retriever.search(query)
                
                if results:
                    print(f"\nFound {len(results)} matching documents:\n")
                    for result in results:
                        print(f"Document: {result.title} ({result.document_id})")
                        print(f"Score: {result.score:.2f}")
                        print(f"Category: {result.metadata['category']}")
                        print(f"Department: {result.metadata['department']}")
                        print(f"Snippet: {result.snippet}")
                        print("-" * 50)
                else:
                    print("\nNo matching documents found")
                    print("-" * 50)
            except Exception as e:
                print(f"\nSearch failed: {str(e)}")
                print("-" * 50)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    demonstrate_search()
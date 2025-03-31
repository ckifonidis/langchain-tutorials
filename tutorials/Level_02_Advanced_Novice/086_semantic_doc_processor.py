#!/usr/bin/env python3
"""
LangChain Semantic Document Processor (LangChain v3)

This example demonstrates a document processing system using three key concepts:
1. document_loaders: Handle various document formats
2. text_splitters: Chunk documents properly
3. retrievers: Semantic document search

It provides efficient document processing and semantic search capabilities.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPDFLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Load environment variables
load_dotenv(".env")

class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"

class ChunkingStrategy(str, Enum):
    """Text splitting strategies."""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"

class SearchResult(BaseModel):
    """Document search result."""
    content: str = Field(description="Matched content")
    score: float = Field(description="Relevance score")
    metadata: Dict = Field(description="Document metadata")
    source: str = Field(description="Source document")

class DocumentProcessor:
    """Document processing with semantic search."""
    
    def __init__(self):
        """Initialize processor with embeddings and text splitters."""
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small-3"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT", "https://ai-agent-swarm-1.openai.azure.com/"),
            api_key=os.getenv("AZURE_API_KEY", "979b84bde7c04d8784208309bcdac5d0")
        )
        
        # Initialize text splitters
        self.splitters = {
            ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            ChunkingStrategy.CHARACTER: CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            ChunkingStrategy.TOKEN: TokenTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
        }
        
        # Initialize document loaders
        self.loaders = {
            DocumentType.TEXT: TextLoader,
            DocumentType.CSV: CSVLoader,
            DocumentType.JSON: JSONLoader,
            DocumentType.PDF: UnstructuredPDFLoader
        }
        
        # Initialize storage
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
    
    def load_document(self, file_path: str, doc_type: DocumentType) -> List[Document]:
        """Load document using appropriate loader."""
        try:
            loader_class = self.loaders.get(doc_type)
            if not loader_class:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            loader = loader_class(file_path)
            docs = loader.load()
            
            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            
            return docs
            
        except Exception as e:
            raise ValueError(f"Error loading document: {str(e)}")
    
    def process_documents(
        self,
        docs: List[Document],
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ) -> List[Document]:
        """Process documents using specified chunking strategy."""
        try:
            # Get text splitter
            splitter = self.splitters.get(strategy)
            if not splitter:
                raise ValueError(f"Unsupported chunking strategy: {strategy}")
            
            # Split documents
            chunks = splitter.split_documents(docs)
            
            # Store documents
            self.documents.extend(chunks)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            
            return chunks
            
        except Exception as e:
            raise ValueError(f"Error processing documents: {str(e)}")
    
    def search_documents(self, query: str) -> List[SearchResult]:
        """Search processed documents."""
        try:
            if not self.documents:
                raise ValueError("No documents processed yet")
            
            # Get results from both retrievers
            semantic_results = self.vector_store.similarity_search_with_score(
                query,
                k=3
            )
            keyword_results = self.bm25_retriever.get_relevant_documents(query)[:3]
            
            # Combine and score results
            search_results = []
            
            # Add semantic results
            for doc, score in semantic_results:
                search_results.append(
                    SearchResult(
                        content=doc.page_content,
                        score=0.7 * (1.0 - score),  # Convert distance to similarity
                        metadata=doc.metadata,
                        source=doc.metadata.get("source", "unknown")
                    )
                )
            
            # Add keyword results
            for i, doc in enumerate(keyword_results):
                # Assign decreasing scores based on position
                score = 0.3 * (1.0 - (i / len(keyword_results)))
                search_results.append(
                    SearchResult(
                        content=doc.page_content,
                        score=score,
                        metadata=doc.metadata,
                        source=doc.metadata.get("source", "unknown")
                    )
                )
            
            # Sort by score and return top 5
            search_results.sort(key=lambda x: x.score, reverse=True)
            return search_results[:5]
            
        except Exception as e:
            raise ValueError(f"Error searching documents: {str(e)}")

def demonstrate_processor():
    """Demonstrate the document processor."""
    print("\nSemantic Document Processor Demo")
    print("===============================\n")
    
    # Create processor
    processor = DocumentProcessor()
    
    # Sample documents
    documents = {
        "report.txt": """
        Q1 2025 Performance Report
        
        Key Highlights:
        1. Revenue increased by 25%
        2. Customer satisfaction at 92%
        3. New product launch successful
        """,
        
        "policy.txt": """
        Corporate Security Policy
        
        1. Data Protection
        All sensitive data must be encrypted.
        
        2. Access Control
        Use multi-factor authentication.
        
        3. Incident Response
        Report security incidents immediately.
        """,
        
        "email.txt": """
        Subject: Project Update
        
        Team,
        
        The AI integration project is on track:
        - Phase 1 complete
        - Testing successful
        - Deployment next week
        
        Great work everyone!
        """
    }
    
    try:
        # Create temporary files
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write documents
            file_paths = []
            for name, content in documents.items():
                path = os.path.join(temp_dir, name)
                with open(path, "w") as f:
                    f.write(content)
                file_paths.append((path, name))
            
            # Process documents
            print("Processing Documents:")
            print("-" * 40)
            
            all_docs = []
            for path, name in file_paths:
                print(f"\nLoading: {name}")
                docs = processor.load_document(path, DocumentType.TEXT)
                chunks = processor.process_documents(docs)
                print(f"Created {len(chunks)} chunks")
                all_docs.extend(chunks)
            
            print(f"\nTotal Documents: {len(all_docs)}")
            
            # Search examples
            queries = [
                "performance metrics",
                "security requirements",
                "project status"
            ]
            
            for query in queries:
                print(f"\nSearch: {query}")
                print("-" * 40)
                
                results = processor.search_documents(query)
                
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}")
                    print(f"Score: {result.score:.2f}")
                    print(f"Source: {result.source}")
                    print(f"Content: {result.content[:200]}...")
                
                print("-" * 40)
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    demonstrate_processor()
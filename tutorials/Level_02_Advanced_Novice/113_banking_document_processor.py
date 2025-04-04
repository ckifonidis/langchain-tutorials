#!/usr/bin/env python3
"""
LangChain Banking Document Processor (113) (LangChain v3)

This example demonstrates a banking document processing system using three key concepts:
1. Document Loaders: Multi-format document handling
2. Text Splitters: Smart document chunking
3. Vector Stores: Efficient retrieval

It provides document processing support for various banking departments.
"""

import os
import sys
import json
from enum import Enum
os.environ["FAISS_CPU_ONLY"] = "1"  # Force CPU-only FAISS

import logging
import warnings
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import httpx
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure FAISS and logging
logging.getLogger("faiss").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*GpuIndexIVFFlat.*')
warnings.filterwarnings('ignore', message='.*Failed to load GPU Faiss.*')
warnings.filterwarnings('ignore', category=UserWarning, module='faiss')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*AVX2.*')
warnings.filterwarnings('ignore', message='.*success.*')

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentType(str, Enum):
    """Banking document types."""
    API_SPEC = "api_specification"
    CONTRACT = "contract"
    POLICY = "policy"
    MARKETING = "marketing"
    FINANCIAL = "financial"
    AUDIT = "audit"

class DepartmentType(str, Enum):
    """Banking department types."""
    DEVELOPMENT = "development"
    LEGAL = "legal"
    HR = "hr"
    MARKETING = "marketing"
    FINANCE = "finance"
    RISK = "risk"

class ProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    separators: List[str] = Field(default=["\n\n", "\n", " ", ""], description="Split separators")

class ProcessingMetrics(BaseModel):
    """Document processing metrics."""
    total_docs: int = Field(description="Total documents processed")
    total_chunks: int = Field(description="Total chunks created")
    avg_chunk_size: float = Field(description="Average chunk size")
    embedding_dim: int = Field(description="Embedding dimensions")
    index_size: int = Field(description="Vector store size")

class BankingDocument(BaseModel):
    """Banking document metadata."""
    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    department: DepartmentType = Field(description="Department owner")
    doc_type: DocumentType = Field(description="Document type")
    filepath: str = Field(description="Document file path")
    created_at: str = Field(description="Creation timestamp")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class BankingDocumentProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the document processor."""
        self.config = config or ProcessingConfig()
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=self.config.separators
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.metrics = ProcessingMetrics(
            total_docs=0,
            total_chunks=0,
            avg_chunk_size=0.0,
            embedding_dim=1536,
            index_size=0
        )

    def load_document(self, doc_info: BankingDocument) -> List[Document]:
        """Load document based on type."""
        documents = []
        try:
            ext = os.path.splitext(doc_info.filepath)[1].lower()
            
            if ext in ['.txt', '.md']:
                try:
                    with open(doc_info.filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents = [Document(page_content=text)]
                except Exception as e:
                    logger.error(f"Error reading file {doc_info.filepath}")
                    raise
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "doc_id": doc_info.doc_id,
                    "title": doc_info.title,
                    "department": doc_info.department.value,
                    "doc_type": doc_info.doc_type.value,
                    "created_at": doc_info.created_at,
                    **doc_info.metadata
                })
            
            return documents
            
        except Exception as e:
            raise

    def process_documents(self, documents: List[BankingDocument]) -> ProcessingMetrics:
        """Process multiple documents."""
        try:
            print("Processing documents...")
            all_docs = []
            total_chars = 0
            
            # Load and split documents
            for doc_info in documents:
                # Load document
                docs = self.load_document(doc_info)
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(docs)
                all_docs.extend(chunks)
                
                # Update metrics
                total_chars += sum(len(chunk.page_content) for chunk in chunks)
                self.metrics.total_chunks += len(chunks)
            
            self.metrics.total_docs = len(documents)
            self.metrics.avg_chunk_size = (
                total_chars / self.metrics.total_chunks 
                if self.metrics.total_chunks > 0 else 0
            )
            
            # Create or update vector store
            if not self.vectorstore:
                self.vectorstore = FAISS.from_documents(
                    documents=all_docs,
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(all_docs)
            
            self.metrics.index_size = len(self.vectorstore.docstore._dict)
            return self.metrics
            
        except Exception as e:
            raise

    def search_documents(self, query: str, 
                        department: Optional[DepartmentType] = None,
                        doc_type: Optional[DocumentType] = None,
                        k: int = 5) -> List[Document]:
        """Search processed documents."""
        try:
            if not self.vectorstore:
                raise ValueError("No documents have been processed yet")
            
            # Create metadata filter
            filter_dict = {}
            if department:
                filter_dict["department"] = department.value
            if doc_type:
                filter_dict["doc_type"] = doc_type.value
                
            # Search with metadata filtering
            print("\nSearch Results:")
            docs = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict if filter_dict else None
            )
            
            return docs
            
        except Exception as e:
            raise

async def demonstrate_workflow_assistant():
    print("\nBanking Document Processor Demo")
    print("==============================\n")

    try:
        # Initialize processor with example documents
        documents = [
            BankingDocument(
                doc_id="API-2025-001",
                title="Payment API Documentation",
                department=DepartmentType.DEVELOPMENT,
                doc_type=DocumentType.API_SPEC,
                filepath="./docs/api_spec.md",
                created_at=datetime.now().isoformat(),
                metadata={"version": "2.0"}
            ),
            BankingDocument(
                doc_id="POL-2025-001",
                title="Data Privacy Policy",
                department=DepartmentType.LEGAL,
                doc_type=DocumentType.POLICY,
                filepath="./docs/policy.txt",
                created_at=datetime.now().isoformat(),
                metadata={"region": "EU"}
            ),
            BankingDocument(
                doc_id="FIN-2025-001",
                title="Q1 Financial Report",
                department=DepartmentType.FINANCE,
                doc_type=DocumentType.FINANCIAL,
                filepath="./docs/q1_report.txt",
                created_at=datetime.now().isoformat(),
                metadata={"quarter": "Q1"}
            )
        ]

        # Process documents
        processor = BankingDocumentProcessor()
        metrics = processor.process_documents(documents)
        
        print("\nProcessing Metrics:")
        print(f"Total Documents: {metrics.total_docs}")
        print(f"Total Chunks: {metrics.total_chunks}")
        print(f"Average Chunk Size: {metrics.avg_chunk_size:.2f}")
        print(f"Vector Store Size: {metrics.index_size}")
        
        # Search documents
        results = processor.search_documents(
            query="payment api authentication",
            department=DepartmentType.DEVELOPMENT,
            k=2
        )
        
        for doc in results:
            print(f"\nDocument: {doc.metadata.get('title')}")
            print(f"Content: {doc.page_content[:200]}...")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
    
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_workflow_assistant())
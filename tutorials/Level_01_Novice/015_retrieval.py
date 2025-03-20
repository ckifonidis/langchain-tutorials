"""
LangChain Retrieval Example

This example demonstrates how to implement retrieval capabilities in LangChain,
showing basic similarity search functionality. Compatible with LangChain v0.3
and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.

Important: Make sure to install the FAISS package:
    For CPU: pip install faiss-cpu
    For GPU: pip install faiss-gpu
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Check if required environment variables are available
required_vars = [
    "AZURE_EMBEDDING_ENDPOINT",
    "AZURE_API_KEY",
    "AZURE_DEPLOYMENT"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                     "Please add them to your .env file.")

class SearchResult(BaseModel):
    """Schema for search results."""
    query: str = Field(description="The search query used")
    document: str = Field(description="The retrieved document content")
    relevance_score: float = Field(description="Relevance score of the document")
    metadata: Dict[str, Any] = Field(description="Document metadata")

def get_embeddings_client() -> AzureOpenAI:
    """
    Create an Azure OpenAI client for embeddings.
    
    Returns:
        AzureOpenAI: Configured client for embeddings
    """
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY")
    )
    return client

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings vectors
    """
    client = get_embeddings_client()
    response = client.embeddings.create(
        input=texts,
        model=os.getenv("AZURE_DEPLOYMENT")
    )
    return [item.embedding for item in response.data]

def create_document_store() -> FAISS:
    """
    Create and populate a vector store with sample documents.
    
    Returns:
        FAISS: A vector store containing the sample documents
    """
    # Create sample documents
    documents = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"category": "programming", "difficulty": "beginner"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"category": "ai", "difficulty": "intermediate"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks in animal brains.",
            metadata={"category": "ai", "difficulty": "advanced"}
        ),
        Document(
            page_content="Data structures are ways of organizing and storing data for efficient access and modification.",
            metadata={"category": "programming", "difficulty": "intermediate"}
        ),
        Document(
            page_content="Deep learning is part of machine learning based on artificial neural networks.",
            metadata={"category": "ai", "difficulty": "advanced"}
        )
    ]
    
    # Extract texts from documents and compute their embeddings
    texts = [doc.page_content for doc in documents]
    embeddings = embed_documents(texts)
    
    try:
        # Create FAISS index by providing the embedding function and the precomputed text embeddings as tuples
        index = FAISS.from_embeddings(
            embedding=embed_documents,                    # Embedding function for future queries
            text_embeddings=list(zip(texts, embeddings)),  # List of (text, embedding) tuples
            metadatas=[doc.metadata for doc in documents]
        )
        return index
        
    except Exception as e:
        print(f"Error creating document store: {str(e)}")
        raise

def perform_basic_retrieval(vectorstore: FAISS, query: str, k: int = 2) -> List[SearchResult]:
    """
    Perform basic similarity search retrieval.
    
    Args:
        vectorstore: The vector store to search in
        query: The search query
        k: Number of results to retrieve
        
    Returns:
        List of SearchResult objects
    """
    try:
        # Generate query embedding
        query_embedding = embed_documents([query])[0]
        
        # Perform similarity search
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            query_embedding, k=k
        )
        
        # Convert results to SearchResult objects
        results = []
        for doc, score in docs_and_scores:
            result = SearchResult(
                query=query,
                document=doc.page_content,
                relevance_score=float(score),
                metadata=doc.metadata
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error performing retrieval: {str(e)}")
        raise

def demonstrate_retrieval():
    """Demonstrate different retrieval capabilities."""
    try:
        print("\nDemonstrating LangChain Retrieval...\n")
        
        # Create vector store
        print("Creating document store...")
        vectorstore = create_document_store()
        print("Document store created with sample documents.")
        
        # Example 1: Basic Retrieval
        print("\nExample 1: Basic Similarity Search")
        print("-" * 50)
        
        query = "What is machine learning?"
        results = perform_basic_retrieval(vectorstore, query)
        
        print(f"Query: {query}")
        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result.document}")
            print(f"   Score: {result.relevance_score:.4f}")
            print(f"   Category: {result.metadata['category']}")
            print(f"   Difficulty: {result.metadata['difficulty']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_retrieval()

if __name__ == "__main__":
    main()
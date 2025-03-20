"""
LangChain Text Splitters Example

This example demonstrates how to use different text splitters in LangChain to
effectively divide text into manageable chunks. Compatible with LangChain v0.3
and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class TextChunk(BaseModel):
    """Schema for text chunks."""
    content: str = Field(description="The content of the text chunk")
    chunk_index: int = Field(description="Index of the chunk in sequence")
    metadata: Dict[str, Any] = Field(description="Metadata about the chunk")

def create_sample_text() -> str:
    """Create a sample text for demonstration."""
    return """
    Machine Learning and Artificial Intelligence

    Machine learning is a subset of artificial intelligence (AI) that focuses on developing computer systems that can learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze patterns in data.

    Types of Machine Learning:
    1. Supervised Learning: The algorithm learns from labeled data and makes predictions.
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
    3. Reinforcement Learning: The algorithm learns through trial and error with rewards/penalties.

    Applications include:
    - Image and speech recognition
    - Natural language processing
    - Recommendation systems
    - Automated decision-making

    Deep Learning, a subset of machine learning, uses neural networks with multiple layers (deep neural networks) to process complex patterns. These networks are inspired by the human brain's structure.

    Key Concepts:
    - Training Data: Used to teach the algorithm
    - Features: Input variables for the model
    - Labels: Expected outputs for supervised learning
    - Model: The mathematical representation learned from data
    - Inference: Using the trained model to make predictions

    Best Practices:
    1. Data preprocessing and cleaning
    2. Feature selection and engineering
    3. Model validation and testing
    4. Regular retraining with new data
    5. Monitoring model performance
    """

def split_text_recursive(text: str, chunk_size: int = 100, chunk_overlap: int = 20) -> List[TextChunk]:
    """
    Split text using RecursiveCharacterTextSplitter.
    
    Args:
        text: The text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of TextChunk objects
    """
    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Split text into documents
    documents = splitter.create_documents([text])
    
    # Convert to TextChunk objects
    chunks = []
    for i, doc in enumerate(documents):
        chunk = TextChunk(
            content=doc.page_content,
            chunk_index=i,
            metadata={"length": len(doc.page_content), "start_char": i * (chunk_size - chunk_overlap)}
        )
        chunks.append(chunk)
    
    return chunks

def split_text_character(text: str, chunk_size: int = 100, separator: str = "\n") -> List[TextChunk]:
    """
    Split text using CharacterTextSplitter.
    
    Args:
        text: The text to split
        chunk_size: Size of each chunk
        separator: Character to split on
    
    Returns:
        List of TextChunk objects
    """
    # Create splitter
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        separator=separator
    )
    
    # Split text into documents
    documents = splitter.create_documents([text])
    
    # Convert to TextChunk objects
    chunks = []
    for i, doc in enumerate(documents):
        chunk = TextChunk(
            content=doc.page_content,
            chunk_index=i,
            metadata={"length": len(doc.page_content), "separator": separator}
        )
        chunks.append(chunk)
    
    return chunks

def split_text_token(text: str, chunk_size: int = 50) -> List[TextChunk]:
    """
    Split text using TokenTextSplitter.
    
    Args:
        text: The text to split
        chunk_size: Number of tokens per chunk
    
    Returns:
        List of TextChunk objects
    """
    # Create splitter with chunk_overlap set to 0 (or a value < chunk_size)
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    
    # Split text into documents
    documents = splitter.create_documents([text])
    
    # Convert to TextChunk objects
    chunks = []
    for i, doc in enumerate(documents):
        chunk = TextChunk(
            content=doc.page_content,
            chunk_index=i,
            metadata={"length": len(doc.page_content), "tokens": chunk_size}
        )
        chunks.append(chunk)
    
    return chunks

def demonstrate_text_splitting():
    """Demonstrate different text splitting methods."""
    try:
        print("\nDemonstrating LangChain Text Splitters...\n")
        
        # Get sample text
        text = create_sample_text()
        print("Original Text Length:", len(text))
        print("=" * 50)
        
        # Example 1: Recursive Character Splitting
        print("\nExample 1: Recursive Character Splitting")
        print("-" * 50)
        recursive_chunks = split_text_recursive(text, chunk_size=200, chunk_overlap=50)
        
        print(f"Number of chunks: {len(recursive_chunks)}")
        for i, chunk in enumerate(recursive_chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Length: {chunk.metadata['length']}")
            print("Content:", chunk.content[:100], "...")
        print("=" * 50)
        
        # Example 2: Simple Character Splitting
        print("\nExample 2: Simple Character Splitting")
        print("-" * 50)
        char_chunks = split_text_character(text, chunk_size=200, separator="\n")
        
        print(f"Number of chunks: {len(char_chunks)}")
        for i, chunk in enumerate(char_chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Length: {chunk.metadata['length']}")
            print("Content:", chunk.content[:100], "...")
        print("=" * 50)
        
        # Example 3: Token-based Splitting
        print("\nExample 3: Token-based Splitting")
        print("-" * 50)
        token_chunks = split_text_token(text, chunk_size=50)
        
        print(f"Number of chunks: {len(token_chunks)}")
        for i, chunk in enumerate(token_chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Length: {chunk.metadata['length']}")
            print("Content:", chunk.content[:100], "...")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_text_splitting()

if __name__ == "__main__":
    main()
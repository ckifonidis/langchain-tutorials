"""
LangChain Document Loaders Example

This example demonstrates how to use document loaders in LangChain to load
and process different types of documents. Shows how to handle various file
formats and extract structured information. Compatible with LangChain v0.3
and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class DocumentInfo(BaseModel):
    """Schema for document information."""
    filename: str = Field(description="Name of the document file")
    doc_type: str = Field(description="Type of document (text, markdown, csv, json)")
    content_length: int = Field(description="Length of the document content")
    chunk_count: int = Field(description="Number of chunks after splitting")
    metadata: Dict[str, Any] = Field(description="Document metadata")

def load_text_document(file_path: str) -> List[Document]:
    """
    Load a text document.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Document objects
    """
    loader = TextLoader(file_path)
    return loader.load()


def load_csv_document(file_path: str) -> List[Document]:
    """
    Load a CSV document.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Document objects
    """
    loader = CSVLoader(file_path)
    return loader.load()

def load_json_document(file_path: str, jq_schema: str = ".[]") -> List[Document]:
    """
    Load a JSON document.
    
    Args:
        file_path: Path to the JSON file
        jq_schema: JQ schema for extracting data
        
    Returns:
        List of Document objects
    """
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,
        text_content=False
    )
    return loader.load()

def process_document(file_path: str, doc_type: str) -> DocumentInfo:
    """
    Process a document and return information about it.
    
    Args:
        file_path: Path to the document
        doc_type: Type of document (text, markdown, csv, json)
        
    Returns:
        DocumentInfo object with document details
    """
    # Load the document based on type
    if doc_type == "text":
        docs = load_text_document(file_path)
    elif doc_type == "csv":
        docs = load_csv_document(file_path)
    elif doc_type == "json":
        docs = load_json_document(file_path)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split documents
    splits = text_splitter.split_documents(docs)
    
    # Calculate total content length
    total_length = sum(len(doc.page_content) for doc in docs)
    
    # Collect metadata
    metadata = {}
    for doc in docs:
        metadata.update(doc.metadata)
    
    return DocumentInfo(
        filename=os.path.basename(file_path),
        doc_type=doc_type,
        content_length=total_length,
        chunk_count=len(splits),
        metadata=metadata
    )

def demonstrate_document_loading():
    """Demonstrate different document loading capabilities."""
    try:
        print("\nDemonstrating LangChain Document Loaders...\n")
        
        # Create example documents in various formats
        os.makedirs("examples", exist_ok=True)
        
        # Example 1: Text Document
        print("Example 1: Text Document")
        print("-" * 50)
        
        text_content = """
        This is an example text document.
        It contains multiple lines of text.
        This will be used to demonstrate document loading.
        """
        
        with open("examples/sample.txt", "w") as f:
            f.write(text_content)
        
        info = process_document("examples/sample.txt", "text")
        print(f"Filename: {info.filename}")
        print(f"Type: {info.doc_type}")
        print(f"Content Length: {info.content_length} characters")
        print(f"Chunks: {info.chunk_count}")
        print(f"Metadata: {info.metadata}")
        print("=" * 50)
        
        # Example 2: CSV Document
        print("\nExample 2: CSV Document")
        print("-" * 50)
        
        
        csv_content = """name,age,city
John Doe,30,New York
Jane Smith,25,London
Bob Johnson,35,Paris
"""
        
        with open("examples/sample.csv", "w") as f:
            f.write(csv_content)
        
        info = process_document("examples/sample.csv", "csv")
        print(f"Filename: {info.filename}")
        print(f"Type: {info.doc_type}")
        print(f"Content Length: {info.content_length} characters")
        print(f"Chunks: {info.chunk_count}")
        print(f"Metadata: {info.metadata}")
        print("=" * 50)
        
        # Clean up example files
        try:
            # Clean up all files in examples directory
            for file in os.listdir("examples"):
                os.remove(os.path.join("examples", file))
            os.rmdir("examples")
        except Exception as e:
            print(f"\nWarning: Cleanup failed: {str(e)}")
            print("You may need to manually remove the 'examples' directory")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_document_loading()

if __name__ == "__main__":
    main()
"""
LangChain Runnable Interface Example

This example demonstrates how to use the Runnable interface in LangChain,
showing how to create, compose, and chain operations using runnables.
Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Check if required environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                    "Please add them to your .env file.")

class TextProcessor(Runnable):
    """Example runnable that processes text input."""
    
    def invoke(self, input: str, config: RunnableConfig | None = None) -> str:
        """Process the input text."""
        # Simulate text processing
        return input.strip().title()

class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis results."""
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    key_points: List[str] = Field(description="Key points from the text")

def create_sentiment_chain():
    """
    Create a chain for sentiment analysis using the Runnable interface.
    
    Returns:
        A runnable chain that performs sentiment analysis
    """
    # Create the language model
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create a prompt template
    prompt = PromptTemplate.from_template("""
    Analyze the sentiment of the following text and provide key points.
    Return the response in the following format:
    Sentiment: (positive/negative/neutral)
    Confidence: (0.0-1.0)
    Key Points:
    - Point 1
    - Point 2
    - etc.

    Text: {input}
    """)
    
    # Create a chain of operations
    chain = (
        prompt 
        | model 
        | StrOutputParser()
    )
    
    return chain

class DocumentProcessor(Runnable):
    """Example of a configurable runnable for document processing."""
    
    def __init__(self, max_length: int = 100):
        """Initialize with configuration."""
        self.max_length = max_length
    
    def invoke(self, input: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        """Process a document with metadata."""
        # Extract and validate input
        text = input.get("text", "")
        metadata = input.get("metadata", {})
        
        # Process the text
        processed_text = text[:self.max_length] if len(text) > self.max_length else text
        
        # Return processed result with metadata
        return {
            "processed_text": processed_text,
            "original_length": len(text),
            "truncated": len(text) > self.max_length,
            "metadata": metadata
        }

def demonstrate_runnables():
    """Demonstrate various ways to use the Runnable interface."""
    try:
        print("\nDemonstrating LangChain Runnable Interface...\n")
        
        # Example 1: Simple Runnable
        print("Example 1: Simple Text Processing")
        print("-" * 50)
        processor = TextProcessor()
        result = processor.invoke("hello, world!")
        print(f"Input: 'hello, world!'")
        print(f"Output: '{result}'")
        print("=" * 50)
        
        # Example 2: Runnable Chain for Sentiment Analysis
        print("\nExample 2: Sentiment Analysis Chain")
        print("-" * 50)
        chain = create_sentiment_chain()
        text = """The new restaurant exceeded all expectations! 
        The food was amazing, service was impeccable, and the 
        atmosphere was perfect for a special evening."""
        
        print("Input text:")
        print(text)
        print("\nAnalysis:")
        result = chain.invoke({"input": text})
        print(result)
        print("=" * 50)
        
        # Example 3: Configurable Runnable with Metadata
        print("\nExample 3: Document Processing with Metadata")
        print("-" * 50)
        processor = DocumentProcessor(max_length=50)
        document = {
            "text": "This is a long document that might need to be truncated based on configuration settings.",
            "metadata": {
                "author": "John Doe",
                "date": "2024-03-18"
            }
        }
        
        print("Input document:")
        print(document)
        print("\nProcessed result:")
        result = processor.invoke(document)
        print(f"Processed text: {result['processed_text']}")
        print(f"Original length: {result['original_length']}")
        print(f"Was truncated: {result['truncated']}")
        print(f"Metadata: {result['metadata']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_runnables()

if __name__ == "__main__":
    main()
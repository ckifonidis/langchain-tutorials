"""
LangChain Testing Example

This example demonstrates how to implement proper testing for LangChain components
and applications. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import pytest
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please add them to your .env file."
    )

class QueryResult(BaseModel):
    """Schema for query results."""
    query: str = Field(description="The input query")
    response: str = Field(description="The model's response")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_qa_chain():
    """Create a simple question-answering chain."""
    model = create_chat_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant providing clear, concise answers."),
        ("human", "{question}")
    ])
    
    return prompt | model | StrOutputParser()

class FactChecker:
    """Simple fact checker for testing responses."""
    
    def __init__(self):
        """Initialize with known facts."""
        self.facts = {
            "capital_france": "Paris",
            "earth_satellite": "Moon",
            "largest_planet": "Jupiter"
        }
    
    def verify_fact(self, category: str, statement: str) -> bool:
        """
        Verify if a statement matches known facts.
        
        Args:
            category: Fact category to check
            statement: Statement to verify
            
        Returns:
            bool: True if statement matches facts
        """
        known_fact = self.facts.get(category, "").lower()
        return known_fact in statement.lower()

def demonstrate_testing():
    """Demonstrate different testing approaches."""
    try:
        print("\nDemonstrating LangChain Testing...\n")
        
        # Example 1: Unit Testing Components
        print("Example 1: Unit Testing Components")
        print("-" * 50)
        
        # Test prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}")
        ])
        
        # Format prompt
        formatted = prompt.format_messages(question="What is Python?")
        print("\nTesting Prompt Formatting:")
        print(f"Number of messages: {len(formatted)}")
        print(f"System message: {formatted[0].content}")
        print(f"Human message: {formatted[1].content}")
        
        # Test model configuration
        model = create_chat_model()
        print("\nTesting Model Configuration:")
        print(f"Model type: {type(model).__name__}")
        print(f"Temperature: {model.temperature}")
        print("=" * 50)
        
        # Example 2: Integration Testing
        print("\nExample 2: Integration Testing")
        print("-" * 50)
        
        # Create and test chain
        chain = create_qa_chain()
        
        # Test questions
        test_questions = [
            "What is the capital of France?",
            "What is Earth's natural satellite?",
            "What is the largest planet in our solar system?"
        ]
        
        # Create fact checker
        checker = FactChecker()
        
        print("\nTesting Chain Responses:")
        for question in test_questions:
            response = chain.invoke({"question": question})
            print(f"\nQuestion: {question}")
            print(f"Response: {response}")
            
            # Verify response
            if "france" in question.lower():
                is_correct = checker.verify_fact("capital_france", response)
            elif "satellite" in question.lower():
                is_correct = checker.verify_fact("earth_satellite", response)
            elif "largest planet" in question.lower():
                is_correct = checker.verify_fact("largest_planet", response)
            else:
                is_correct = None
            
            if is_correct is not None:
                print(f"Fact Check: {'✓' if is_correct else '✗'}")
        print("=" * 50)
        
        # Example 3: Error Handling Tests
        print("\nExample 3: Error Handling Tests")
        print("-" * 50)
        
        print("\nTesting Error Scenarios:")
        
        # Test missing variable
        try:
            invalid_prompt = ChatPromptTemplate.from_messages([
                ("human", "{nonexistent}")
            ])
            invalid_prompt.format_messages()
        except Exception as e:
            print(f"Expected error caught: {type(e).__name__}")
        
        # Test invalid input
        try:
            response = chain.invoke({})
        except Exception as e:
            print(f"Expected error caught: {type(e).__name__}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_testing()

if __name__ == "__main__":
    main()
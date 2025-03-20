"""
LangChain Example Selectors Example

This example demonstrates how to use example selectors in LangChain to dynamically
choose the most relevant examples for few-shot prompting. Compatible with 
LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.example_selector import (
    SemanticSimilarityExampleSelector,
    LengthBasedExampleSelector
)
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_EMBEDDING_ENDPOINT",
    "AZURE_API_KEY",
    "AZURE_DEPLOYMENT"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please add them to your .env file."
    )

class Example(BaseModel):
    """Schema for examples."""
    question: str = Field(description="The input question")
    answer: str = Field(description="The example answer")
    difficulty: str = Field(description="Difficulty level of the question")
    category: str = Field(description="Category of the question")

def create_example_pool() -> List[Dict[str, str]]:
    """Create a pool of example question-answer pairs."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "difficulty": "beginner",
            "category": "programming"
        },
        {
            "question": "Explain object-oriented programming.",
            "answer": "Object-oriented programming is a programming paradigm based on objects containing data and code.",
            "difficulty": "intermediate",
            "category": "programming"
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables systems to learn from data and improve from experience.",
            "difficulty": "intermediate",
            "category": "ai"
        },
        {
            "question": "How does a neural network work?",
            "answer": "A neural network processes data through interconnected nodes in layers, mimicking human brain functions.",
            "difficulty": "advanced",
            "category": "ai"
        },
        {
            "question": "What is version control?",
            "answer": "Version control is a system that records changes to files over time, enabling collaboration and history tracking.",
            "difficulty": "beginner",
            "category": "development"
        }
    ]

def create_semantic_selector(examples: List[Dict[str, str]], k: int = 2) -> SemanticSimilarityExampleSelector:
    """
    Create a semantic similarity-based example selector.
    
    Args:
        examples: List of example dictionaries.
        k: Number of examples to select.
        
    Returns:
        SemanticSimilarityExampleSelector: Configured selector.
    """
    # Initialize embeddings with correct configuration.
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        deployment=os.getenv("AZURE_DEPLOYMENT")
    )
    
    # Create example selector using FAISS as the vector store.
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        FAISS,
        k=k,
        input_keys=["question"]
    )
    
    return selector

def create_length_based_selector(examples: List[Dict[str, str]], max_length: int = 1000) -> LengthBasedExampleSelector:
    """
    Create a length-based example selector.
    
    Args:
        examples: List of example dictionaries.
        max_length: Maximum combined length of examples.
        
    Returns:
        LengthBasedExampleSelector: Configured selector.
    """
    # Instead of a ChatPromptTemplate instance, supply a dict with a "template" field.
    example_prompt = {"template": "Q: {question}\nA: {answer}"}
    return LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=max_length,
        length_function=len,
        input_keys=["question", "answer"]
    )

def demonstrate_example_selection():
    """Demonstrate different example selection methods."""
    try:
        print("\nDemonstrating LangChain Example Selectors...\n")
        
        # Initialize model.
        model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Get example pool.
        examples = create_example_pool()
        
        # Example 1: Semantic Similarity Selection.
        print("Example 1: Semantic Similarity Selection")
        print("-" * 50)
        
        semantic_selector = create_semantic_selector(examples)
        
        # Test questions for semantic selection.
        test_questions = [
            "How do I start learning Python?",
            "What are deep neural networks?",
            "How do I use Git?"
        ]
        
        for question in test_questions:
            selected_examples = semantic_selector.select_examples({"question": question})
            
            print(f"\nInput Question: {question}")
            print("\nSelected Examples:")
            for i, example in enumerate(selected_examples, 1):
                print(f"\n{i}. Q: {example['question']}")
                print(f"   A: {example['answer']}")
        print("=" * 50)
        
        # Example 2: Length-Based Selection.
        print("\nExample 2: Length-Based Selection")
        print("-" * 50)
        
        length_selector = create_length_based_selector(examples)
        
        # Test with different max lengths.
        max_lengths = [100, 200, 300]
        
        for max_length in max_lengths:
            length_selector.max_length = max_length
            selected_examples = length_selector.select_examples({})
            
            print(f"\nMax Length: {max_length}")
            print(f"Number of examples selected: {len(selected_examples)}")
            print("\nSelected Examples:")
            for i, example in enumerate(selected_examples, 1):
                print(f"\n{i}. Q: {example['question']}")
                print(f"   A: {example['answer']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_example_selection()

if __name__ == "__main__":
    main()
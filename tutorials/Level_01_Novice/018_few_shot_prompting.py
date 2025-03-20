"""
LangChain Few-Shot Prompting Example

This example demonstrates how to use few-shot prompting in LangChain to improve
model responses by providing example patterns. Compatible with LangChain v0.3
and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

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

class ExampleResponse(BaseModel):
    """Schema for example responses."""
    question: str = Field(description="The input question")
    answer: str = Field(description="The model's answer")
    explanation: str = Field(description="Explanation of how the answer follows the pattern")

def create_few_shot_prompt(examples: List[Dict[str, str]]) -> FewShotChatMessagePromptTemplate:
    """
    Create a few-shot prompt template from examples.
    
    Args:
        examples: List of example dictionaries with 'question' and 'answer' keys
    
    Returns:
        FewShotChatMessagePromptTemplate: The configured prompt template
    """
    # Define example prompt format
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("assistant", "{answer}")
    ])
    
    # Create few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    return few_shot_prompt

def create_model_and_examples():
    """
    Create the language model and example cases.
    
    Returns:
        Tuple[AzureChatOpenAI, List[Dict[str, str]]]: Model and examples
    """
    # Initialize the model
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Define example cases
    examples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris. This city serves as both the country's political and cultural center."
        },
        {
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Tokyo. This metropolitan city is the country's political, economic, and cultural hub."
        },
        {
            "question": "What is the capital of Australia?",
            "answer": "The capital of Australia is Canberra. This planned city was specifically designed to serve as the nation's capital."
        }
    ]
    
    return model, examples

def demonstrate_few_shot_prompting():
    """Demonstrate few-shot prompting capabilities."""
    try:
        print("\nDemonstrating LangChain Few-Shot Prompting...\n")
        
        # Create model and get examples
        model, examples = create_model_and_examples()
        
        # Create few-shot prompt
        few_shot_prompt = create_few_shot_prompt(examples)
        
        # Create final prompt template
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that provides clear and informative answers about capital cities."),
            few_shot_prompt,
            ("human", "{question}")
        ])
        
        # Example 1: Basic Few-Shot Pattern
        print("Example 1: Basic Few-Shot Pattern")
        print("-" * 50)
        
        # Test questions
        test_questions = [
            "What is the capital of Italy?",
            "What is the capital of Canada?",
            "What is the capital of Brazil?"
        ]
        
        for question in test_questions:
            # Format the prompt
            formatted_prompt = final_prompt.format_messages(question=question)
            
            # Get response
            response = model.invoke(formatted_prompt)
            
            print(f"\nQuestion: {question}")
            print(f"Response: {response.content}")
            print("-" * 50)
        
        # Example 2: Few-Shot with Specific Format
        print("\nExample 2: Few-Shot with Format Guidance")
        print("-" * 50)
        
        # Add format guidance to system message
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that provides answers about capital cities.
            Follow this format:
            1. Name of the capital
            2. Brief description
            3. One interesting fact"""),
            few_shot_prompt,
            ("human", "{question}")
        ])
        
        format_questions = [
            "What is the capital of Spain?",
            "What is the capital of Germany?"
        ]
        
        for question in format_questions:
            # Format the prompt
            formatted_prompt = format_prompt.format_messages(question=question)
            
            # Get response
            response = model.invoke(formatted_prompt)
            
            print(f"\nQuestion: {question}")
            print(f"Response: {response.content}")
            print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_few_shot_prompting()

if __name__ == "__main__":
    main()
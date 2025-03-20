"""
LangChain Async Programming Example

This example demonstrates how to use async programming in LangChain for better
performance and concurrency. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import time
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    question: str = Field(description="The original question")
    answer: str = Field(description="The model's response")
    time_taken: float = Field(description="Time taken to process in seconds")

def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        streaming=False  # Important for async comparison
    )

async def process_question_async(
    chain, question: str, sleep_time: int = 0
) -> QueryResult:
    """
    Process a single question asynchronously.
    
    Args:
        chain: The LangChain chain to use
        question: Question to process
        sleep_time: Optional delay to simulate other processing
        
    Returns:
        QueryResult: The processed result
    """
    start_time = time.time()
    
    # Simulate other processing if needed
    if sleep_time > 0:
        await asyncio.sleep(sleep_time)
    
    # Process the question
    response = await chain.ainvoke({"question": question})
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    return QueryResult(
        question=question,
        answer=response,
        time_taken=time_taken
    )

def process_questions_sync(chain, questions: List[str]) -> List[QueryResult]:
    """Process questions synchronously."""
    results = []
    
    # Process each question sequentially
    for question in questions:
        start_time = time.time()
        response = chain.invoke({"question": question})
        time_taken = time.time() - start_time
        
        results.append(
            QueryResult(
                question=question,
                answer=response,
                time_taken=time_taken
            )
        )
    
    return results

async def process_questions_async(chain, questions: List[str]) -> List[QueryResult]:
    """Process questions asynchronously."""
    # Create tasks for all questions
    tasks = [
        process_question_async(chain, question)
        for question in questions
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    return results

async def demonstrate_async_capabilities():
    """Demonstrate async vs sync processing capabilities."""
    try:
        print("\nDemonstrating LangChain Async Programming...\n")
        
        # Initialize model and chain
        model = create_chat_model()
        
        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Answer this question: {question}")
        ])
        
        # Create the chain
        chain = prompt | model | StrOutputParser()
        
        # Example 1: Basic Sync vs Async Comparison
        print("Example 1: Basic Sync vs Async Comparison")
        print("-" * 50)
        
        # Test questions
        questions = [
            "What is the capital of France?",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical formula for water?"
        ]
        
        # Synchronous processing
        print("\nProcessing synchronously...")
        start_time = time.time()
        sync_results = process_questions_sync(chain, questions)
        sync_total_time = time.time() - start_time
        
        print(f"Total sync time: {sync_total_time:.2f} seconds")
        for result in sync_results:
            print(f"\nQuestion: {result.question}")
            print(f"Answer: {result.answer}")
            print(f"Time taken: {result.time_taken:.2f} seconds")
        print("=" * 50)
        
        # Asynchronous processing
        print("\nProcessing asynchronously...")
        start_time = time.time()
        async_results = await process_questions_async(chain, questions)
        async_total_time = time.time() - start_time
        
        print(f"Total async time: {async_total_time:.2f} seconds")
        for result in async_results:
            print(f"\nQuestion: {result.question}")
            print(f"Answer: {result.answer}")
            print(f"Time taken: {result.time_taken:.2f} seconds")
        print("=" * 50)
        
        # Example 2: Mixed Processing Times
        print("\nExample 2: Mixed Processing Times")
        print("-" * 50)
        
        # Process questions with varying delays
        tasks = [
            process_question_async(chain, "What is Python?", sleep_time=1),
            process_question_async(chain, "What is async programming?", sleep_time=2),
            process_question_async(chain, "What are coroutines?", sleep_time=0)
        ]
        
        print("\nProcessing with varying delays...")
        start_time = time.time()
        mixed_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"Total time: {total_time:.2f} seconds")
        for result in mixed_results:
            print(f"\nQuestion: {result.question}")
            print(f"Answer: {result.answer}")
            print(f"Time taken: {result.time_taken:.2f} seconds")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    asyncio.run(demonstrate_async_capabilities())

if __name__ == "__main__":
    main()
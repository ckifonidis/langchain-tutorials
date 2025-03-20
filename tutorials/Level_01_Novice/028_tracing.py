"""
LangChain Tracing Example

This example demonstrates how to use tracing in LangChain to monitor and debug
application flows. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.callbacks import FileCallbackHandler
from langchain_core.tracers.context import tracing_v2_enabled  # New tracing context manager
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Check required environment variables for the LLM
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

class QueryTrace(BaseModel):
    """Schema for tracing query execution."""
    query: str = Field(description="The input query")
    timestamp: datetime = Field(description="When the query was processed")
    duration: float = Field(description="Processing time in seconds")
    response: str = Field(description="Model response")

def setup_tracers(log_file: str = "trace.log") -> List[Any]:
    """
    Set up tracing handlers.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        List[Any]: Console and file handlers.
    """
    # Create handlers directory if it doesn't exist
    os.makedirs("handlers", exist_ok=True)
    
    console_handler = ConsoleCallbackHandler()
    file_handler = FileCallbackHandler(f"handlers/{log_file}")
    
    return [console_handler, file_handler]

def create_chat_model(handlers: List[Any]) -> AzureChatOpenAI:
    """
    Create an Azure ChatOpenAI instance with tracing.
    
    Args:
        handlers: List of callback handlers.
        
    Returns:
        AzureChatOpenAI: Configured model instance.
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        callbacks=handlers,
        temperature=0
    )

def demonstrate_tracing():
    """Demonstrate different tracing capabilities."""
    try:
        print("\nDemonstrating LangChain Tracing...\n")
        
        # Set up tracers
        handlers = setup_tracers()
        
        # Initialize model with tracers
        model = create_chat_model(handlers)
        
        # Example 1: Basic Query Tracing
        print("Example 1: Basic Query Tracing")
        print("-" * 50)
        
        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "What is the capital of {country}?")
        ])
        
        # Create chain: prompt | model | string output parser
        chain = prompt | model | StrOutputParser()
        
        # Process queries with tracing
        countries = ["France", "Japan", "Brazil"]
        traces = []
        
        for country in countries:
            print(f"\nProcessing query for {country}...")
            
            # Use the new tracing context manager.
            with tracing_v2_enabled(project_name="TracingExample") as session:
                start_time = datetime.now()
                response = chain.invoke({"country": country})
                duration = (datetime.now() - start_time).total_seconds()
                
                # Record trace
                trace = QueryTrace(
                    query=f"What is the capital of {country}?",
                    timestamp=start_time,
                    duration=duration,
                    response=response
                )
                traces.append(trace)
                
                print(f"Response: {response}")
                print(f"Duration: {duration:.2f} seconds")
                # Removed printing session.session_id as it's no longer available.
                print("Tracing session started.")
        print("=" * 50)
        
        # Example 2: Error Tracing
        print("\nExample 2: Error Tracing")
        print("-" * 50)
        
        # Create an invalid prompt to trigger an error
        error_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Process this: {nonexistent_variable}")
        ])
        
        error_chain = error_prompt | model | StrOutputParser()
        
        print("\nAttempting to process invalid prompt...")
        try:
            with tracing_v2_enabled(project_name="TracingExample") as session:
                error_chain.invoke({})
        except Exception as e:
            print(f"Caught error: {str(e)}")
            print("Error tracked in tracing session.")
        print("=" * 50)
        
        # Example 3: Trace Analysis
        print("\nExample 3: Trace Analysis")
        print("-" * 50)
        
        total_queries = len(traces)
        total_duration = sum(t.duration for t in traces)
        avg_duration = total_duration / total_queries if total_queries else 0
        
        print("\nTrace Analysis:")
        print(f"Total Queries: {total_queries}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Average Duration: {avg_duration:.2f} seconds")
        
        print("\nDetailed Traces:")
        for trace in traces:
            print(f"\nQuery: {trace.query}")
            print(f"Timestamp: {trace.timestamp}")
            print(f"Duration: {trace.duration:.2f} seconds")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_tracing()

if __name__ == "__main__":
    main()

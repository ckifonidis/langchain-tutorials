"""
LangChain Callbacks Example

This example demonstrates how to use callbacks in LangChain to monitor and modify
execution flow. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    CallbackManager,
    BaseCallbackHandler
)
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

class TimingCallback(BaseCallbackHandler):
    """Callback handler that tracks timing information."""
    
    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.start_time = None
        self.timing_logs = []
    
    def on_llm_start(self, *args, **kwargs) -> None:
        """Record start time of LLM operation."""
        self.start_time = datetime.now()
        self.timing_logs.append(f"LLM operation started at {self.start_time}")
    
    def on_llm_end(self, *args, **kwargs) -> None:
        """Record end time and duration of LLM operation."""
        if self.start_time:
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.timing_logs.append(
                f"LLM operation completed at {end_time}. "
                f"Duration: {duration.total_seconds():.2f} seconds"
            )

class LoggingCallback(BaseCallbackHandler):
    """Callback handler that logs all events."""
    
    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.logs = []
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log when LLM starts processing."""
        self.logs.append(f"Starting LLM with prompts: {prompts}")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log when LLM completes processing."""
        self.logs.append(f"LLM completed with response: {response}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log any LLM errors."""
        self.logs.append(f"LLM encountered error: {str(error)}")

def create_chat_model(callbacks: List[BaseCallbackHandler]) -> AzureChatOpenAI:
    """
    Create an Azure ChatOpenAI instance with callbacks.
    
    Args:
        callbacks: List of callback handlers to use
        
    Returns:
        AzureChatOpenAI: Configured model instance
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        callback_manager=CallbackManager(callbacks),
        temperature=0
    )

def demonstrate_callbacks():
    """Demonstrate different callback capabilities."""
    try:
        print("\nDemonstrating LangChain Callbacks...\n")
        
        # Example 1: Basic Timing and Logging
        print("Example 1: Basic Timing and Logging")
        print("-" * 50)
        
        # Create callbacks
        timing_callback = TimingCallback()
        logging_callback = LoggingCallback()
        
        # Initialize model with callbacks
        model = create_chat_model([timing_callback, logging_callback])
        
        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "What is the capital of {country}?")
        ])
        
        # Create chain
        chain = prompt | model | StrOutputParser()
        
        # Process some questions
        countries = ["France", "Japan", "Brazil"]
        for country in countries:
            print(f"\nProcessing question for {country}...")
            response = chain.invoke({"country": country})
            print(f"Response: {response}")
        
        # Display timing logs
        print("\nTiming Information:")
        for log in timing_callback.timing_logs:
            print(log)
        
        # Display operation logs
        print("\nOperation Logs:")
        for log in logging_callback.logs:
            print(log)
        print("=" * 50)
        
        # Example 2: Error Handling
        print("\nExample 2: Error Handling")
        print("-" * 50)
        
        # Create an invalid prompt to trigger an error
        error_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Process this: {nonexistent_variable}")
        ])
        
        error_chain = error_prompt | model | StrOutputParser()
        
        print("\nAttempting to process invalid prompt...")
        try:
            error_chain.invoke({})
        except Exception as e:
            print(f"Caught error: {str(e)}")
            print("\nError was logged in callback logs:")
            error_logs = [log for log in logging_callback.logs if "error" in log.lower()]
            for log in error_logs:
                print(log)
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_callbacks()

if __name__ == "__main__":
    main()
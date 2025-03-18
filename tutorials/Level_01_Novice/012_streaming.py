"""
LangChain Streaming Example

This example demonstrates how to implement streaming in LangChain,
showing different ways to handle streaming responses from language models.
Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import Dict, Any
from queue import Queue
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
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

class QueueCallback(BaseCallbackHandler):
    """Custom callback handler that puts tokens into a queue."""
    
    def __init__(self, queue: Queue):
        """Initialize with a queue."""
        self.queue = queue
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Put new tokens into the queue."""
        self.queue.put(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Signal the end of the stream."""
        self.queue.put(None)
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Handle errors by putting them in the queue."""
        self.queue.put(error)

class StreamProcessor:
    """Handles streaming text generation and processing."""
    
    def __init__(self):
        """Initialize the stream processor."""
        self.chat_model = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            model_kwargs={"stream": True},
            streaming=True,
            temperature=0.7,
        )
    
    def stream_story(self, topic: str) -> Queue:
        """
        Stream a short story about the given topic.
        
        Args:
            topic: The topic to write about
            
        Returns:
            Queue containing the streamed tokens
        """
        queue = Queue()
        handler = QueueCallback(queue)
        
        try:
            messages = [
                SystemMessage(content="You are a creative storyteller."),
                HumanMessage(content=f"Write a short story about {topic} in about 100 words.")
            ]
            
            # Use generate instead of invoke for streaming
            response = self.chat_model.generate(
                [messages],
                callbacks=[handler]
            )
            
        except Exception as e:
            print(f"Error in stream_story: {str(e)}")
            queue.put(e)
        
        return queue
    
    def stream_with_processing(self, prompt: str, processor: callable) -> Queue:
        """
        Stream text with custom token processing.
        
        Args:
            prompt: The prompt to send to the model
            processor: Function to process each token
            
        Returns:
            Queue containing processed tokens
        """
        queue = Queue()
        
        class ProcessingCallback(BaseCallbackHandler):
            def on_llm_new_token(self_cb, token: str, **kwargs) -> None:
                processed = processor(token)
                queue.put(processed)
            
            def on_llm_end(self_cb, response: LLMResult, **kwargs) -> None:
                queue.put(None)
            
            def on_llm_error(self_cb, error: Exception, **kwargs) -> None:
                queue.put(error)
        
        try:
            messages = [HumanMessage(content=prompt)]
            
            # Use generate instead of invoke for streaming
            response = self.chat_model.generate(
                [messages],
                callbacks=[ProcessingCallback()]
            )
            
        except Exception as e:
            print(f"Error in stream_with_processing: {str(e)}")
            queue.put(e)
        
        return queue

def process_stream(queue: Queue) -> None:
    """Process tokens from a stream queue."""
    while True:
        token = queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            print(f"\nError: {str(token)}")
            break
        print(token, end="", flush=True)

def demonstrate_streaming():
    """Demonstrate different streaming capabilities."""
    try:
        print("\nDemonstrating LangChain Streaming...\n")
        
        processor = StreamProcessor()
        
        # Example 1: Basic Streaming
        print("Example 1: Basic Story Streaming")
        print("-" * 50)
        print("Generating a story about space exploration...")
        print("\nStory:")
        
        queue = processor.stream_story("space exploration")
        process_stream(queue)
        print("\n" + "=" * 50)
        
        # Example 2: Streaming with Processing
        print("\nExample 2: Streaming with Token Processing")
        print("-" * 50)
        print("Generating text with uppercase processing...")
        print()
        
        def uppercase_processor(token: str) -> str:
            return token.upper()
        
        queue = processor.stream_with_processing(
            "Write a short sentence about programming.",
            uppercase_processor
        )
        process_stream(queue)
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_streaming()

if __name__ == "__main__":
    main()
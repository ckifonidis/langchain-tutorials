"""
LangChain Streaming Example

This example demonstrates how to use streaming capabilities in LangChain,
showing real-time response handling and custom callback implementations.
"""

import os
import sys
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Load environment variables from the .env file
load_dotenv()

# Check if required Azure OpenAI environment variables are available
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

class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    def __init__(self, prefix: str = ""):
        """Initialize with optional prefix for output."""
        self.prefix = prefix
        self.tokens_seen = 0
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print when LLM starts processing."""
        print(f"\n{self.prefix}LLM processing started...")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print new token as it arrives."""
        self.tokens_seen += 1
        print(token, end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Print summary when LLM processing ends."""
        print(f"\n\n{self.prefix}LLM processing finished. "
              f"Total tokens received: {self.tokens_seen}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Print error if LLM processing fails."""
        print(f"\n{self.prefix}Error during LLM processing: {str(error)}")

def init_streaming_chat_model(callbacks: List[BaseCallbackHandler]) -> AzureChatOpenAI:
    """Initialize the Azure OpenAI chat model with streaming."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        streaming=True,
        callbacks=callbacks
    )

def demonstrate_basic_streaming():
    """Demonstrate basic streaming with a simple callback handler."""
    # Initialize chat model with streaming callback
    callback = StreamingStdOutCallbackHandler(prefix="Basic Stream: ")
    chat_model = init_streaming_chat_model([callback])
    
    try:
        # Create messages for a long response that benefits from streaming
        system_msg = SystemMessage(content="""
            You are a detailed story narrator. When asked about a topic,
            provide a thoughtful, multi-paragraph response with rich details.
        """)
        
        human_msg = HumanMessage(content=
            "Tell me an interesting story about the invention of the telescope."
        )
        
        # Stream the response
        chat_model.invoke([system_msg, human_msg])
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def simulate_processing_stream():
    """Demonstrate processing tokens as they arrive."""
    # Initialize chat model with streaming
    tokens_received = []
    
    class TokenProcessor(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            tokens_received.append(token)
            # Simulate processing each token
            sys.stdout.write(f"\rProcessing tokens: {len(tokens_received)}")
            sys.stdout.flush()
            time.sleep(0.1)  # Simulate processing time
    
    chat_model = init_streaming_chat_model([TokenProcessor()])
    
    try:
        # Create messages for response
        system_msg = SystemMessage(content="""
            You are a technical expert. Provide a detailed explanation
            of a complex topic in clear, step-by-step terms.
        """)
        
        human_msg = HumanMessage(content=
            "Explain how a digital camera converts light into a digital image."
        )
        
        # Process the streaming response
        chat_model.invoke([system_msg, human_msg])
        
        # Show final statistics
        print(f"\n\nTotal tokens processed: {len(tokens_received)}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def main():
    print("\nDemonstrating LangChain Streaming Capabilities...")
    
    print("\n1. Basic Streaming Example:")
    demonstrate_basic_streaming()
    
    print("\n2. Token Processing Example:")
    simulate_processing_stream()

if __name__ == "__main__":
    main()

# Expected Output:
# Basic Stream: LLM processing started...
# [Streaming response about telescope invention...]
# Basic Stream: LLM processing finished. Total tokens received: X
#
# Token Processing Example:
# Processing tokens: Y
# Total tokens processed: Y
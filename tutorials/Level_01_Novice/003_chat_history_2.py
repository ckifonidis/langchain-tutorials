"""
LangChain Chat History Example

This example demonstrates how to manage chat history in LangChain,
showing how to maintain context across multiple interactions.
"""

import os
from dotenv import load_dotenv

# Patch Pydantic's SecretStr to add the __get_pydantic_json_schema__ method for compatibility with Pydantic v2
import pydantic
from pydantic import SecretStr
if not hasattr(SecretStr, '__get_pydantic_json_schema__'):
    @classmethod
    def _get_pydantic_json_schema(cls, schema: dict, model: type) -> dict:
        # Simply return the schema without modifications.
        return schema
    SecretStr.__get_pydantic_json_schema__ = _get_pydantic_json_schema

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import AzureChatOpenAI
from typing import List

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

class SimpleChatHistory(BaseChatMessageHistory):
    """Simple implementation of chat history management."""
    
    def __init__(self):
        self.messages: List = []
    
    def add_message(self, message):
        """Add a message to the history."""
        self.messages.append(message)
    
    def clear(self):
        """Clear chat history."""
        self.messages = []

def init_chat_model():
    """Initialize the Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

def demonstrate_chat_history():
    """Demonstrate chat history management with multiple interactions."""
    # Initialize chat model and history
    chat_model = init_chat_model()
    chat_history = SimpleChatHistory()
    
    # Set up system message
    system_msg = SystemMessage(content="""
        You are a helpful assistant that remembers previous interactions.
        Keep your responses concise and refer to previous context when relevant.
    """)
    chat_history.add_message(system_msg)
    
    try:
        # First interaction
        human_msg1 = HumanMessage(content="What are the three primary colors?")
        chat_history.add_message(human_msg1)
        
        response1 = chat_model.invoke(chat_history.messages)
        chat_history.add_message(response1)
        print("\nFirst Response:", response1.content)
        
        # Second interaction - using context
        human_msg2 = HumanMessage(content="What colors do you get when you mix them?")
        chat_history.add_message(human_msg2)
        
        response2 = chat_model.invoke(chat_history.messages)
        chat_history.add_message(response2)
        print("\nSecond Response:", response2.content)
        
        # Third interaction - referring to all previous context
        human_msg3 = HumanMessage(content="Which of these mixed colors is your favorite and why?")
        chat_history.add_message(human_msg3)
        
        response3 = chat_model.invoke(chat_history.messages)
        chat_history.add_message(response3)
        print("\nThird Response:", response3.content)
        
        # Display message count
        print(f"\nTotal messages in history: {len(chat_history.messages)}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    
    # Clear history after conversation
    chat_history.clear()
    print("\nChat history cleared.")

def main():
    print("\nDemonstrating LangChain Chat History Management...")
    demonstrate_chat_history()

if __name__ == "__main__":
    main()


# Expected Output:
# First Response: The three primary colors are red, blue, and yellow.
# Second Response: When mixing the primary colors, you get purple (red + blue),
# green (blue + yellow), and orange (red + yellow).
# Third Response: Of these secondary colors, purple is my favorite because it
# combines the warmth of red with the depth of blue, creating a rich and
# sophisticated color.
# Total messages in history: 7
# Chat history cleared.

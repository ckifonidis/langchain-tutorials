"""
LangChain Chat History Example

This example demonstrates how to manage chat history in LangChain,
showing how to maintain context across multiple interactions.
"""

import os
from dotenv import load_dotenv

# Patch Pydantic's SecretStr to add the __get_pydantic_json_schema__ method for compatibility with Pydantic v2
import pydantic
from pydantic import SecretStr, BaseModel, Field, field_validator
if not hasattr(SecretStr, '__get_pydantic_json_schema__'):
    @classmethod
    def _get_pydantic_json_schema(cls, schema: dict, model: type) -> dict:
        # Simply return the schema without modifications.
        return schema
    SecretStr.__get_pydantic_json_schema__ = _get_pydantic_json_schema

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import AzureChatOpenAI
from typing import List, Optional
from datetime import datetime

class SensitiveContent(BaseModel):
    """Model for handling sensitive information."""
    value: SecretStr
    access_level: str = Field(..., pattern='^(public|private|restricted)$')

class Message(BaseModel):
    """Base message model with validation."""
    content: str = Field(..., min_length=1, description="The message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    sensitive_data: Optional[SensitiveContent] = None
    
    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()

    def has_sensitive_data(self) -> bool:
        return self.sensitive_data is not None

class ChatMessage(Message):
    """Enhanced chat message model with metadata."""
    message_type: str = Field(..., pattern='^(system|human|ai|sensitive|debug)$')
    metadata: Optional[dict] = Field(default_factory=dict)
    priority: Optional[int] = Field(None, ge=1, le=5)
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v > 3:
            print(f"Warning: High priority message ({v}) detected!")
        return v

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
    """Enhanced implementation of chat history management with Pydantic validation."""
    
    def __init__(self):
        self.messages: List = []  # LangChain messages
        self.enhanced_messages: List[ChatMessage] = []  # Our enhanced message storage
        self.max_messages: int = 100  # Prevent unbounded growth
    
    def add_message(self, message):
        """Add a message to the history with validation."""
        # Store original LangChain message
        if len(self.messages) >= self.max_messages:
            self.messages.pop(0)
        self.messages.append(message)
        
        # Store enhanced version
        chat_message = ChatMessage(
            content=message.content,
            message_type=message.__class__.__name__.lower().replace('message', ''),
            metadata={
                'timestamp': datetime.now().isoformat(),
                'type': message.__class__.__name__
            }
        )
        
        if len(self.enhanced_messages) >= self.max_messages:
            self.enhanced_messages.pop(0)
        self.enhanced_messages.append(chat_message)
    
    def clear(self):
        """Clear chat history."""
        self.messages = []
        self.enhanced_messages = []
        
    def get_recent_messages(self, count: int = 5) -> List[ChatMessage]:
        """Get the most recent enhanced messages."""
        return self.enhanced_messages[-count:]
    
    def get_message_count(self) -> int:
        """Get the current number of messages."""
        return len(self.messages)  # Both lists will have same length

def init_chat_model():
    """Initialize the Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

def demonstrate_chat_history():
    """Demonstrate enhanced chat history management with multiple interactions."""
    # Initialize chat model and history
    chat_model = init_chat_model()
    chat_history = SimpleChatHistory()
    
    # Set up system message
    system_msg = SystemMessage(content="""
        You are a helpful assistant that remembers previous interactions.
        Keep your responses concise and refer to previous context when relevant.
    """)
    chat_history.add_message(system_msg)
    print(f"\nSystem message added. Current message count: {chat_history.get_message_count()}")
    
    try:
        # Testing with valid content
        human_msg1 = HumanMessage(content="What are the three primary colors?")
        chat_history.add_message(human_msg1)
        
        # Test empty content validation
        try:
            empty_msg = HumanMessage(content="   ")
            chat_history.add_message(empty_msg)
        except ValueError as e:
            print("\nValidation caught empty message:", str(e))
        
        # Test sensitive data handling
        sensitive_msg = HumanMessage(content="Here's my API key: sk_test_123")
        sensitive_content = SensitiveContent(
            value=SecretStr("sk_test_123"),
            access_level="restricted"
        )
        # Store enhanced message with sensitive data
        chat_message = ChatMessage(
            content="[REDACTED SENSITIVE CONTENT]",
            message_type="sensitive",
            priority=4,
            sensitive_data=sensitive_content,
            metadata={"original_length": len(sensitive_msg.content)}
        )
        chat_history.enhanced_messages.append(chat_message)
        print("\nSensitive message handling:")
        print(f"- Raw SecretStr representation: {chat_message.sensitive_data.value}")  # Shows as '**********'
        print(f"- Retrieved secret value: {chat_message.sensitive_data.value.get_secret_value()}")  # Shows actual value
        print(f"- Access level: {chat_message.sensitive_data.access_level}")
        print(f"- Priority level: {chat_message.priority}")
        
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
        # Display enhanced message info
        print("\nMessage Statistics:")
        print(f"Total messages in history: {chat_history.get_message_count()}")
        recent_messages = chat_history.get_recent_messages(3)
        print("\nMost recent messages:")
        for msg in recent_messages:
            print(f"- Type: {msg.message_type}, Time: {msg.timestamp}, Content length: {len(msg.content)}")
        
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
# Demonstrating LangChain Chat History Management...

# System message added. Current message count: 1

# Validation caught empty message: 1 validation error for ChatMessage
# content
#   Value error, Message content cannot be empty [type=value_error, input_value='   ', input_type=str]
#     For further information visit https://errors.pydantic.dev/2.11/v/value_error
# Warning: High priority message (4) detected!

# Sensitive message handling:
# - Raw SecretStr representation: **********
# - Retrieved secret value: sk_test_123
# - Access level: restricted
# - Priority level: 4

# First Response: The three primary colors are red, blue, and yellow. These colors cannot be created by mixing other colors and are the basis for creating other hues.

# Second Response: When you mix the primary colors, you get the following secondary colors:

# - Mixing red and blue gives you purple.
# - Mixing blue and yellow gives you green.
# - Mixing red and yellow gives you orange.

# Third Response: As an AI, I don't have personal preferences or feelings, but I can tell you that many people find purple appealing due to its associations with creativity and luxury.

# Message Statistics:
# Total messages in history: 8

# Most recent messages:
# - Type: ai, Time: 2025-04-04 19:34:35.389767, Content length: 197
# - Type: human, Time: 2025-04-04 19:34:35.389841, Content length: 53
# - Type: ai, Time: 2025-04-04 19:34:36.057112, Content length: 166

# Chat history cleared.

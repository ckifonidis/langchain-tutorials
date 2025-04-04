"""
LangChain Basic Messages Example

This example demonstrates the usage of different message types in LangChain,
showing how to create and use various message roles with a chat model.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

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

def init_chat_model():
    """Initialize the Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model="gpt-4o",
        temperature=0
    )

def demonstrate_message_types():
    """Demonstrate different types of messages and their usage."""
    # Initialize chat model
    chat_model = init_chat_model()
    
    # Create messages of different types
    system_message = SystemMessage(content="""
        You are a helpful assistant that provides clear, concise responses.
        Always format your responses in a single paragraph.
    """)
    
    human_message = HumanMessage(content=
        "What are the three primary colors? List them in a single sentence."
    )
    
    try:
        # First interaction with system and human messages
        response = chat_model.invoke([system_message, human_message])
        print("\nFirst Response:", response.content)
        
        # Create a follow-up interaction using the AI's response
        ai_message = AIMessage(content=response.content)
        follow_up = HumanMessage(content=
            "Now tell me what colors you get when you mix these primary colors."
        )
        
        # Second interaction including previous context
        messages = [system_message, human_message, ai_message, follow_up]
        response = chat_model.invoke(messages)
        print("\nSecond Response:", response.content)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def main():
    print("\nDemonstrating LangChain Message Types...")
    demonstrate_message_types()

if __name__ == "__main__":
    main()

# Expected Output:
# First Response: The three primary colors are red, blue, and yellow.
# Second Response: When you mix the primary colors, red and blue make purple, 
# blue and yellow make green, and red and yellow make orange.
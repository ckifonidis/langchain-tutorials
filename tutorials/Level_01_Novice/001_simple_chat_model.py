"""
LangChain Basic Chat Model Example

This example demonstrates the fundamental usage of chat models in LangChain,
focusing on Azure OpenAI integration with alternative provider configurations.
"""

import os
from dotenv import load_dotenv

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

from langchain_openai import AzureChatOpenAI

def init_chat_model():
    """
    Initialize the chat model with Azure OpenAI.
    
    Alternative provider configurations are shown in comments below.
    """
    # Primary implementation using Azure OpenAI
    model = AzureChatOpenAI(
         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
         openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
         model="gpt-4o",
         temperature=0
    )

    # Alternative implementations (commented):
    
    # For Groq:
    # from langchain.chat_models import init_chat_model
    # model = init_chat_model("llama3-8b-8192", model_provider="groq")
    
    # For OpenAI:
    # from langchain.chat_models import init_chat_model
    # model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    # For Anthropic:
    # from langchain.chat_models import init_chat_model
    # model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

    return model

def main():
    # Initialize the chat model
    chat_model = init_chat_model()
    
    # Simple prompt to test the model
    prompt = "What is the capital of France? Give me a one-sentence answer."
    
    try:
        # Invoke the model and get the response
        response = chat_model.invoke(prompt)
        print("\nResponse:", response.content)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        
if __name__ == "__main__":
    main()

# Expected Output:
# Response: Paris is the capital of France.

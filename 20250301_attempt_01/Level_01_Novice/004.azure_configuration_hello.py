import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# Check if all required Azure OpenAI environment variables are available
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

# Initialize the chat model with the Azure OpenAI provider
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Invoke the model and print the output.
result = model.invoke("Hello, world!")
print(result)

# Expected Output (example):
# Hello, world!
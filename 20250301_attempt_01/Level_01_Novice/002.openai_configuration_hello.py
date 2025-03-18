import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# Check if the OPENAI_API_KEY is available from the .env file.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it to your .env file.")

from langchain.chat_models import init_chat_model

# Initialize the chat model with the OpenAI provider.
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Invoke the model and print the output.
result = model.invoke("Hello, world!")
print(result)

# Expected Output (example):
# Hello, world!


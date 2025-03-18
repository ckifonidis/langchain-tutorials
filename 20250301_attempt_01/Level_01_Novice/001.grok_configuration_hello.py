import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# Ensure the GROQ_API_KEY is available from the .env file.
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it to your .env file.")

from langchain.chat_models import init_chat_model

# Initialize the chat model with the Groq provider.
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Invoke the model and print the output.
result = model.invoke("Hello, world!")
print(result)


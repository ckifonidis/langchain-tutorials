#!/usr/bin/env python3
"""
LangChain Chatbot with Memory (096) (LangChain v3)

This example demonstrates a conversational AI system using three key concepts:
1. Chat Models: Enable natural language interaction
2. Memory: Maintain conversation context
3. Retrieval: Access relevant information

It provides a responsive and context-aware chatbot for customer support in banking applications.
"""

import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define message model
class Message(BaseModel):
    sender: str = Field(description="Sender of the message")
    content: str = Field(description="Content of the message")

# Define chatbot
class Chatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_DEPLOYMENT"),
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.chat_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.5
        )
        self.vectorstore = self.create_document_store()

    def add_message_to_memory(self, message: Message) -> None:
        self.memory.add_message(message)

    def create_document_store(self) -> FAISS:
        # Create sample documents
        documents = [
            Document(
                page_content="Your account balance is $5,000.",
                metadata={"category": "banking", "type": "balance"}
            ),
            Document(
                page_content="Loan applications require proof of income and credit history.",
                metadata={"category": "banking", "type": "loan"}
            ),
            Document(
                page_content="Savings accounts offer interest rates up to 2.5%.",
                metadata={"category": "banking", "type": "savings"}
            )
        ]
        
        # Extract texts from documents and compute their embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_documents(texts)
        
        # Create FAISS index
        index = FAISS.from_embeddings(
            embedding=self.embeddings,
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=[doc.metadata for doc in documents]
        )
        return index

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def retrieve_information(self, query: str) -> str:
        query_embedding = self.embeddings.embed_query(query)
        docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(query_embedding, k=1)
        return docs_and_scores[0][0].page_content

    async def generate_response(self, user_input: str) -> str:
        # Retrieve relevant information
        retrieved_info = self.retrieve_information(user_input)
        
        # Create prompt with memory and retrieved information
        messages = [
            SystemMessage(content="You are a helpful banking assistant."),
            HumanMessage(content=user_input),
            SystemMessage(content=retrieved_info)
        ]
        
        # Generate response
        response = await self.chat_model.ainvoke(messages)
        return response.content

async def demonstrate_chatbot():
    print("\nChatbot with Memory Demo")
    print("========================\n")
    chatbot = Chatbot()

    # Simulate conversation
    user_inputs = [
        "What is my account balance?",
        "Can you help me with a loan application?",
        "What are the interest rates for savings accounts?",
        "What is my account balance?",  # Repeated question to demonstrate memory effect
        "Tell me more about loan applications."
    ]

    for user_input in user_inputs:
        print(f"User: {user_input}")
        response = await chatbot.generate_response(user_input)
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_chatbot())
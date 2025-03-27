#!/usr/bin/env python3
"""
LangChain Streaming Support Agent (LangChain v3)

This example demonstrates an intelligent banking support agent using three key concepts:
1. retrieval: Access relevant banking policies and FAQs
2. few_shot_prompting: Learn from example responses for consistency
3. streaming: Provide real-time response updates to customers

It provides responsive, knowledge-based customer support for banking applications.
"""

import os
from typing import Dict, List, Any, Generator, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models.fake import FakeListChatModel
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Load environment variables
load_dotenv()

class QueryType(str, Enum):
    """Support query categories."""
    ACCOUNT = "account"
    TRANSACTION = "transaction"
    CARD = "card"
    LOAN = "loan"
    OTHER = "other"

class SupportResponse(BaseModel):
    """Structured support response."""
    type: QueryType = Field(description="Type of query")
    response: str = Field(description="Main response")
    references: List[str] = Field(description="Policy references")
    next_steps: Optional[str] = Field(description="Recommended actions")

# Example banking support responses
EXAMPLE_RESPONSES = [
    {
        "query": "How do I check my account balance?",
        "type": QueryType.ACCOUNT,
        "response": "You can check your balance through our mobile app, online banking, or ATM. For security, please verify your identity first.",
        "references": ["Account Access Policy", "Security Guidelines"],
        "next_steps": "Download our mobile app or visit online banking"
    },
    {
        "query": "My card was declined",
        "type": QueryType.CARD,
        "response": "I'll help you troubleshoot the card decline. First, let's verify your identity and check recent transactions.",
        "references": ["Card Security Policy", "Transaction Guidelines"],
        "next_steps": "Review recent transactions and contact issuer if needed"
    }
]

# Sample banking policies and FAQs
BANKING_POLICIES = [
    "Account Balance Inquiries: Customers must verify identity before receiving balance information.",
    "Card Declines: Immediate investigation required for security. Verify recent transactions.",
    "Transaction Disputes: Must be reported within 60 days. Provide transaction details.",
    "Loan Applications: Credit check required. Income verification needed."
]

def create_support_agent():
    """Create a streaming support agent with retrieval and few-shot learning."""
    
    # Initialize mock LLM with realistic responses
    llm = FakeListChatModel(
        responses=[
            "Based on our account access policy, I'll help you check your balance. First, please verify your identity for security purposes. Once verified, you can access your balance through our mobile app, online banking portal, or any ATM. Would you like me to guide you through the verification process?",
            "I understand your card was declined while traveling. This is often due to our security measures protecting your account. Based on our card security policy, I'll help you resolve this. First, we need to verify recent transactions and update your travel status. Would you like me to connect you with our urgent support team?",
            "Regarding mortgage rates, let me assist you. According to our loan policy, we offer competitive rates that depend on factors like loan term and credit score. Currently, our rates range from 3.5% to 5.5%. You'll need to provide income verification and complete a credit check. Would you like to start a preliminary application?"
        ],
        streaming=True
    )
    
    # Set up retriever with documents
    retriever = BM25Retriever.from_texts(
        texts=BANKING_POLICIES,
        preprocess_func=lambda x: x.lower()  # Simple lowercase preprocessing
    )
    
    # Create example prompt template
    example_prompt = PromptTemplate(
        input_variables=["query", "type", "response", "references", "next_steps"],
        template="""
Query: {query}
Type: {type}
Response: {response}
References: {references}
Next Steps: {next_steps}
"""
    )
    
    # Create example selector
    example_selector = LengthBasedExampleSelector(
        examples=EXAMPLE_RESPONSES,
        example_prompt=example_prompt,
        max_length=1000
    )
    
    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="You are a helpful banking support agent. Respond to queries using these examples:",
        suffix="Query: {query}\nRelevant Policies: {context}\nResponse:",
        input_variables=["query", "context"]
    )
    
    # Create retrieval chain
    retrieval_chain = RunnablePassthrough.assign(
        context = lambda x: "\n".join(doc.page_content for doc in retriever.invoke(x["query"]))
    )
    
    # Create streaming chain
    def process_query(inputs: Dict[str, Any]) -> Generator:
        # Create and stream chain
        chain = (
            retrieval_chain
            | few_shot_prompt
            | llm
        )
        
        # Stream response
        for chunk in chain.stream(inputs):
            yield chunk
    
    return process_query

def demonstrate_support():
    """Demonstrate the streaming support agent."""
    print("\nStreaming Support Agent Demo")
    print("==========================\n")
    
    # Initialize agent
    agent = create_support_agent()
    
    # Test queries
    queries = [
        "How do I check my account balance?",
        "My card was declined while traveling",
        "I need information about mortgage rates"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nCustomer: {query}")
        print("Agent: ", end="", flush=True)
        
        # Stream response
        try:
            for chunk in agent({"query": query}):
                print(chunk.content if hasattr(chunk, 'content') else chunk, end="", flush=True)
            print("\n" + "-" * 40)
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            print("-" * 40)

if __name__ == "__main__":
    demonstrate_support()
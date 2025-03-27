#!/usr/bin/env python3
"""
LangChain Customer Support Agent (LangChain v3)

This example demonstrates a customer support agent for banking using three key concepts:
1. memory: Tracks conversation history for context-aware responses
2. chat_models: Handles natural language interaction with customers
3. callbacks: Monitors conversations and logs important events

It provides context-aware customer support for banking queries while maintaining 
proper logging and monitoring.
"""

import os
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_models.fake import FakeListChatModel

# Load environment variables
load_dotenv()

class QueryType(str, Enum):
    """Support query categories."""
    ACCOUNT = "account"
    TRANSACTION = "transaction"
    CARD = "card"
    LOAN = "loan"
    OTHER = "other"

class CustomerIntent(BaseModel):
    """Parsed customer intent."""
    topic: QueryType = Field(description="Main topic of customer query")
    urgency: str = Field(description="Query urgency (low/medium/high)")
    needs_escalation: bool = Field(description="Whether query needs supervisor")
    summary: str = Field(description="Brief summary of customer's issue")

class SimpleMemory:
    """Simple conversation memory implementation."""
    
    def __init__(self):
        """Initialize empty memory."""
        self.chat_history: List[BaseMessage] = []
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load chat history."""
        return {"chat_history": self.chat_history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save new context."""
        self.chat_history.extend([
            HumanMessage(content=inputs["input"]),
            AIMessage(content=outputs["output"])
        ])
    
    def clear(self) -> None:
        """Clear memory."""
        self.chat_history = []

class SupportCallbackHandler(BaseCallbackHandler):
    """Monitors and logs support conversations."""
    
    def __init__(self):
        """Initialize conversation metrics."""
        self.start_time = None
        self.messages = []
        self.needs_supervisor = False
    
    def on_llm_start(self, *args, **kwargs):
        """Log conversation start."""
        if not self.start_time:
            self.start_time = datetime.now()
            print("\nConversation started at:", self.start_time)
    
    def on_llm_end(self, response, *args, **kwargs):
        """Monitor conversation for important events."""
        if hasattr(response, 'generations'):
            message = response.generations[0][0].text
            self.messages.append(message)
            
            # Check for supervisor needs
            if "supervisor" in message.lower():
                self.needs_supervisor = True
                print("\n[ALERT] Supervisor assistance may be needed!")
            
            # Log metrics
            duration = datetime.now() - self.start_time
            print(f"\nConversation metrics:")
            print(f"- Duration: {duration}")
            print(f"- Messages: {len(self.messages)}")
            if self.needs_supervisor:
                print("- Supervisor flag raised!")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """Log any errors."""
        print(f"\n[ERROR] Chat model error: {str(error)}")

def create_support_agent():
    """Create a customer support agent with memory and monitoring."""
    
    # Initialize components
    handler = SupportCallbackHandler()
    
    # Initialize mock LLM with callbacks
    llm = FakeListChatModel(
        responses=[
            "I'll help you check your balance. First, I need to verify your identity for security purposes.",
            "I understand you can't see recent transactions. Let me check the status for you.",
            "I'll investigate the deposit. Could you provide the deposit amount and date?",
            "I understand your card was declined. This could be due to our security measures for international transactions. Let me connect you with a supervisor.",
            "For foreign transactions, you should notify us before traveling. I'll escalate this to our urgent support team.",
            "I understand this is urgent. A supervisor will help resolve this immediately.",
            "I can help you with loan information. We offer various mortgage options.",
            "Our current mortgage rates range from 3.5% to 5.5% depending on the term.",
            "For a mortgage application, you'll need: income verification, bank statements, and employment history."
        ],
        callbacks=[handler]
    )
    
    # Initialize memory
    memory = SimpleMemory()
    
    # Create chat prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful banking customer support agent.
Always be polite and professional. If you cannot help, escalate to a supervisor.
Never share sensitive information. Verify identity before discussing accounts."""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}")
    ])
    
    # Create support chain
    support_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
        )
        | prompt 
        | llm
        | (lambda x: {"response": x.content})
    )
    
    # Create function to process inputs and update memory
    def process_input(user_input: Dict[str, Any]) -> Dict[str, Any]:
        # Run chain
        result = support_chain.invoke(user_input)
        
        # Update memory
        memory.save_context(
            {"input": user_input["input"]},
            {"output": result["response"]}
        )
        
        return result
    
    return process_input

def demonstrate_support():
    """Demonstrate the customer support agent."""
    print("\nBank Customer Support Demo")
    print("========================\n")
    
    # Create agent
    agent = create_support_agent()
    
    # Test conversations
    conversations = [
        [
            "Hi, I need help with my account balance",
            "I can't see my recent transactions",
            "I made a deposit yesterday but it's not showing"
        ],
        [
            "Hello, my credit card was declined",
            "I'm traveling abroad",
            "I need this fixed urgently"
        ],
        [
            "I want to apply for a loan",
            "I need information about mortgage rates",
            "What documents do I need?"
        ]
    ]
    
    # Process each conversation
    for i, messages in enumerate(conversations, 1):
        print(f"\nConversation {i}")
        print("-" * 40)
        
        for message in messages:
            print(f"\nCustomer: {message}")
            try:
                result = agent({"input": message})
                print(f"Agent: {result['response']}")
            except Exception as e:
                print(f"Error: {str(e)}")
            
        print("-" * 40)

if __name__ == "__main__":
    demonstrate_support()
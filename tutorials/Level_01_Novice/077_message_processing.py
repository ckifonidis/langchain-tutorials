#!/usr/bin/env python3
"""
Structured Message Processing System (LangChain v3)

This example demonstrates systematic message processing using key_methods and messages.
It provides a robust message handling system for banking/fintech applications.

Key concepts demonstrated:
1. key_methods: Utilizing core LangChain methods for message processing
2. messages: Structured handling of different message types and formats
"""

import os
import json
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

# Load environment variables
load_dotenv()

class MessageCategory(str, Enum):
    """Categories of banking messages."""
    TRANSACTION = "transaction_notification"
    ALERT = "security_alert"
    SERVICE = "service_update"
    SUPPORT = "customer_support"

class MessagePriority(str, Enum):
    """Message priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ProcessedMessage(BaseModel):
    """Structure for processed messages."""
    category: MessageCategory
    priority: MessagePriority
    timestamp: datetime
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_action: bool = False
    action_details: Optional[str] = None

class MessageProcessor:
    """Banking message processing system."""
    
    def __init__(self):
        """Initialize the message processor."""
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Initialize message history
        self.message_history: List[Union[HumanMessage, AIMessage, SystemMessage]] = [
            SystemMessage(content="""You are an expert Banking Message Processing Assistant.

IMPORTANT: Always respond with a valid JSON object containing:
- category: Exactly one of [transaction_notification, security_alert, service_update, customer_support]
- priority: Exactly one of [high, medium, low]
- requires_action: Boolean value (true/false)
- metadata: Object with relevant extracted information

Process incoming messages and categorize them appropriately using only the specified categories.
Identify urgency, required actions, and relevant metadata.""")
        ]
        
        # Create processing chain
        self.process_chain = (
            ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            | self.llm
            | StrOutputParser()
        )

    def _analyze_content(self, content: str) -> Dict:
        """Analyze message content for categorization."""
        print(f"\nAnalyzing message: {content}")
        
        # Add message to history with clear categorization rules
        self.message_history.append(HumanMessage(content=f"""Analyze this banking message: {content}

Follow these categorization rules:
1. Security alerts (suspicious activity, unauthorized access): 
   - category: security_alert
   - priority: high
2. Transaction notifications (deposits, transfers):
   - category: transaction_notification
   - priority: medium
3. Service updates (maintenance, system changes):
   - category: service_update
   - priority: low
4. Support requests (customer inquiries, callbacks):
   - category: customer_support
   - priority: medium"""))
        
        # Get AI analysis
        result = self.process_chain.invoke({
            "history": self.message_history,
            "input": """Return ONLY a valid JSON object with no additional text:
{
    "category": "transaction_notification|security_alert|service_update|customer_support",
    "priority": "high|medium|low",
    "requires_action": true|false,
    "action_details": "action description if needed",
    "metadata": {
        "type": "message type",
        "details": "relevant information"
    }
}"""
        })
        
        # Add AI response to history and debug output
        self.message_history.append(AIMessage(content=result))
        print("\nAI Response:", result[:200])
        
        try:
            # Try to extract JSON from response
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Validate category and priority
                if analysis["category"] not in [e.value for e in MessageCategory]:
                    analysis["category"] = "service_update"
                if analysis["priority"] not in [e.value for e in MessagePriority]:
                    analysis["priority"] = "low"
                
                return analysis
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"\nError processing response: {str(e)}")
            print(f"Raw response: {result[:200]}")
            return {
                "category": "service_update",
                "priority": "low",
                "requires_action": False,
                "action_details": None,
                "metadata": {
                    "error": "Failed to process response",
                    "details": str(e)
                }
            }

    def process_message(self, content: str, timestamp: Optional[datetime] = None) -> ProcessedMessage:
        """Process an incoming message."""
        # Analyze content
        analysis = self._analyze_content(content)
        
        # Create processed message
        return ProcessedMessage(
            timestamp=timestamp or datetime.now(),
            category=analysis["category"],
            priority=analysis["priority"],
            content=content,
            metadata=analysis.get("metadata", {}),
            requires_action=analysis.get("requires_action", False),
            action_details=analysis.get("action_details")
        )

def demonstrate_processing():
    """Demonstrate the message processing system."""
    processor = MessageProcessor()
    
    # Example messages
    messages = [
        "Suspicious login attempt detected from IP 192.168.1.100 at 2:30 AM",
        "Your account ending in 4321 has received a deposit of $5,000.00",
        "Scheduled maintenance: Online banking will be unavailable on Sunday 2 AM - 4 AM",
        "Customer requested callback regarding recent wire transfer"
    ]
    
    print("\nProcessing Banking Messages")
    print("=========================\n")
    
    for msg in messages:
        print(f"\nOriginal Message: {msg}")
        result = processor.process_message(msg)
        print("\nProcessed Result:")
        print(f"Category: {result.category}")
        print(f"Priority: {result.priority}")
        print(f"Timestamp: {result.timestamp}")
        if result.requires_action:
            print("Action Required!")
            print(f"Action Details: {result.action_details}")
        else:
            print("No action required")
        print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
        print("-" * 50)

if __name__ == "__main__":
    demonstrate_processing()

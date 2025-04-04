#!/usr/bin/env python3
"""
Chatbot Optimizer (136) (LangChain v3)

This example demonstrates conversation analysis using:
1. Message History: Context tracking
2. Few Shot Learning: Response patterns
3. Prompt Templates: Format control

It helps UX teams optimize banking chatbot interactions.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class IntentType(str, Enum):
    """Chat intent types."""
    BALANCE = "check_balance"
    TRANSFER = "money_transfer"
    PAYMENT = "bill_payment"
    SUPPORT = "customer_support"
    PRODUCT = "product_info"
    ACCOUNT = "account_service"

class MoodType(str, Enum):
    """User mood types."""
    HAPPY = "satisfied"
    NEUTRAL = "neutral"
    UPSET = "dissatisfied"
    ANGRY = "very_dissatisfied"

class Conversation(BaseModel):
    """Conversation details."""
    session_id: str = Field(description="Session ID")
    intent: IntentType = Field(description="User intent")
    messages: List[Dict] = Field(description="Message history")
    metrics: Dict[str, float] = Field(description="Chat metrics")
    metadata: Dict = Field(default_factory=dict)

class ChatOptimizer:
    """Chatbot optimization system."""

    def __init__(self):
        """Initialize optimizer."""
        logger.info("Starting chat optimizer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup memory
        self.memory = ChatMessageHistory()
        logger.info("Chat memory ready")
        
        # Example response patterns
        self.examples = [
            {
                "input": "What's my account balance?",
                "output": """Balance Check Pattern:
- Direct response
- Account details
- Clear amount
- Next steps"""
            },
            {
                "input": "I need to transfer money",
                "output": """Transfer Pattern:
- Security checks
- Amount confirmation
- Clear process
- Status updates"""
            }
        ]
        logger.info("Examples ready")
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a conversation optimization expert.
Analyze chat patterns and suggest improvements.

Format your response exactly like this:

CHAT ANALYSIS
------------
Session: ID
Intent: Type
Mood: Level

Pattern Review:
- Flow structure
- Response style
- Key elements

Effectiveness:
1. Strong Points
   - Detail
   - Reason
   - Impact

2. Weak Points
   - Detail
   - Reason
   - Impact

Optimization Plan:
1. Change Name
   Current: State
   Target: Goal
   Steps: Implementation

2. Change Name
   Current: State
   Target: Goal
   Steps: Implementation

Metrics to Track:
- Metric name
- Metric name

Next Review: YYYY-MM-DD"""),
            ("human", """Analyze this conversation:

Session: {session_id}
Intent: {intent}
Messages:
{messages}

Metrics:
{metrics}

Consider these patterns:
{examples}

Provide optimization recommendations.""")
        ])
        logger.info("Analysis template ready")
        
        # Setup output parser
        self.parser = StrOutputParser()

    def add_message(self, message: Dict) -> None:
        """Add message to history."""
        if message["role"] == "user":
            self.memory.add_user_message(message["content"])
        else:
            self.memory.add_ai_message(message["content"])
        logger.debug("Message added")

    async def analyze_conversation(self, chat: Conversation) -> str:
        """Analyze chat patterns."""
        logger.info(f"Analyzing session: {chat.session_id}")
        
        try:
            # Format messages
            messages = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in chat.messages
            )
            
            # Format metrics
            metrics = "\n".join(
                f"{k}: {v:.2f}"
                for k, v in chat.metrics.items()
            )
            
            # Format example patterns
            examples = "\n".join(
                f"Pattern {i+1}:\n{ex['output']}"
                for i, ex in enumerate(self.examples)
            )
            
            # Add to history
            for message in chat.messages:
                self.add_message(message)
            
            # Format request
            messages = self.template.format_messages(
                session_id=chat.session_id,
                intent=chat.intent.value,
                messages=messages,
                metrics=metrics,
                examples=examples
            )
            logger.debug("Request formatted")
            
            # Get analysis
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            logger.info("Analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting optimization demo...")
    
    try:
        # Create optimizer
        optimizer = ChatOptimizer()
        
        # Example conversation
        conversation = Conversation(
            session_id="CHAT-2025-001",
            intent=IntentType.TRANSFER,
            messages=[
                {"role": "user", "content": "I want to send money to my friend"},
                {"role": "assistant", "content": "I'll help you with the transfer. Which account would you like to send from?"},
                {"role": "user", "content": "My checking account"},
                {"role": "assistant", "content": "How much would you like to send?"},
                {"role": "user", "content": "$500"},
                {"role": "assistant", "content": "Could you provide your friend's account details for the transfer?"}
            ],
            metrics={
                "duration": 45.5,
                "turns": 6.0,
                "satisfaction": 0.85,
                "completion": 1.0,
                "clarity": 0.95,
                "efficiency": 0.80
            }
        )
        
        print("\nAnalyzing Conversation")
        print("====================")
        print(f"Session: {conversation.session_id}")
        print(f"Intent: {conversation.intent.value}\n")
        
        print("Chat History:")
        for msg in conversation.messages:
            role = msg["role"].title()
            content = msg["content"]
            print(f"{role}: {content}")
        
        print("\nChat Metrics:")
        for name, value in conversation.metrics.items():
            if name in ["satisfaction", "completion", "clarity", "efficiency"]:
                print(f"{name}: {value:.1%}")
            elif name == "duration":
                print(f"{name}: {value:.1f}s")
            else:
                print(f"{name}: {value:.1f}")
        
        try:
            # Get analysis
            result = await optimizer.analyze_conversation(conversation)
            print("\nAnalysis Results:")
            print("================")
            print(result)
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
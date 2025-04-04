#!/usr/bin/env python3
"""
Chat Summarizer (117) (LangChain v3)

This example demonstrates chat summarization using:
1. Message handling
2. Streaming output
3. Clear summaries

It helps teams understand key points from chat conversations.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MessageType(str, Enum):
    """Message types."""
    QUESTION = "question"
    ANSWER = "answer"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    CONCLUSION = "conclusion"

class ChatMessage(BaseModel):
    """Chat message details."""
    speaker: str = Field(description="Speaker name")
    type: MessageType = Field(description="Message type")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="Message time")

class ChatThread(BaseModel):
    """Chat conversation thread."""
    title: str = Field(description="Thread title")
    participants: List[str] = Field(description="Chat participants")
    messages: List[ChatMessage] = Field(description="Chat messages")
    metadata: Dict = Field(default_factory=dict)

class ChatSummarizer:
    """Chat summarization system."""

    def __init__(self):
        """Initialize summarizer."""
        logger.info("Starting chat summarizer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        logger.info("Chat model ready")
        
        # Setup summary prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a chat summarizer. Create clear summaries of conversations.
For each summary include:

1. Overview
- Main topics discussed
- Key participants
- Time span covered

2. Key Points
- Important questions
- Main answers
- Critical decisions
- Action items

3. Follow-ups
- Open questions
- Next steps
- Required actions

Use clear sections and bullet points."""),
            ("human", """Summarize this chat:

Title: {title}
Participants: {participants}

Messages:
{messages}

Provide a complete summary.""")
        ])
        logger.info("Prompt template ready")

    def format_messages(self, thread: ChatThread) -> str:
        """Format chat messages for summary."""
        messages = []
        for msg in thread.messages:
            messages.append(f"[{msg.timestamp}] {msg.speaker} ({msg.type}):\n{msg.content}\n")
        return "\n".join(messages)

    async def summarize(self, thread: ChatThread) -> None:
        """Summarize a chat thread."""
        logger.info(f"Summarizing: {thread.title}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                title=thread.title,
                participants=", ".join(thread.participants),
                messages=self.format_messages(thread)
            )
            logger.info("Request formatted")
            
            # Stream summary
            print("\nGenerating Summary...")
            print("==================\n")
            await self.llm.ainvoke(messages)
            print("\n")
            
        except Exception as e:
            logger.error(f"Summary failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting demo...")
    
    try:
        # Create summarizer
        summarizer = ChatSummarizer()
        
        # Example chat
        chat = ChatThread(
            title="API Design Discussion",
            participants=["Alice (Tech Lead)", "Bob (Developer)", "Carol (PM)"],
            messages=[
                ChatMessage(
                    speaker="Carol",
                    type=MessageType.QUESTION,
                    content="What authentication approach should we use for the new API?",
                    timestamp="14:02"
                ),
                ChatMessage(
                    speaker="Bob",
                    type=MessageType.ANSWER,
                    content="I recommend OAuth 2.0 with JWT tokens. It's secure and widely supported.",
                    timestamp="14:03"
                ),
                ChatMessage(
                    speaker="Alice",
                    type=MessageType.FOLLOW_UP,
                    content="Good suggestion. What about refresh token handling?",
                    timestamp="14:04"
                ),
                ChatMessage(
                    speaker="Bob",
                    type=MessageType.ANSWER,
                    content="We can use sliding refresh tokens with a 7-day expiry.",
                    timestamp="14:05"
                ),
                ChatMessage(
                    speaker="Carol",
                    type=MessageType.CLARIFICATION,
                    content="Will this work with our existing user database?",
                    timestamp="14:06"
                ),
                ChatMessage(
                    speaker="Alice",
                    type=MessageType.ANSWER,
                    content="Yes, we already have the required user fields. We'll need to add a tokens table.",
                    timestamp="14:07"
                ),
                ChatMessage(
                    speaker="Bob",
                    type=MessageType.FOLLOW_UP,
                    content="Should we implement rate limiting too?",
                    timestamp="14:08"
                ),
                ChatMessage(
                    speaker="Alice",
                    type=MessageType.ANSWER,
                    content="Yes, let's use token bucket algorithm. Start with 100 requests per minute.",
                    timestamp="14:09"
                ),
                ChatMessage(
                    speaker="Carol",
                    type=MessageType.CONCLUSION,
                    content="Sounds good! Can you create a technical spec with these details?",
                    timestamp="14:10"
                ),
                ChatMessage(
                    speaker="Alice",
                    type=MessageType.ANSWER,
                    content="I'll draft it today and share for review tomorrow morning.",
                    timestamp="14:11"
                )
            ]
        )
        
        print("\nProcessing Chat Thread")
        print("===================")
        print(f"Title: {chat.title}")
        print(f"Participants: {', '.join(chat.participants)}")
        print(f"Messages: {len(chat.messages)}\n")
        
        try:
            # Get summary
            await summarizer.summarize(chat)
            
        except Exception as e:
            print(f"\nSummary failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
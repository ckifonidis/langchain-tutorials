"""
LangChain Chat Session Analyzer Example

This example demonstrates how to combine memory management and streaming capabilities
to create a system that can analyze chat sessions in real-time, tracking metrics
and generating insights.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class MessageMetrics(BaseModel):
    """Schema for message-level metrics."""
    message_count: int = Field(description="Number of messages in conversation")
    avg_length: float = Field(description="Average message length")
    response_time: float = Field(description="Average response time in seconds")
    sentiment_score: float = Field(description="Overall sentiment (-1 to 1)")
    engagement_level: str = Field(description="User engagement level")

class TopicAnalysis(BaseModel):
    """Schema for topic analysis."""
    main_topics: List[str] = Field(description="Main conversation topics")
    topic_shifts: int = Field(description="Number of topic changes")
    unresolved_topics: List[str] = Field(description="Topics needing follow-up")
    key_terms: Dict[str, int] = Field(description="Frequently used terms and counts")

class SessionAnalysis(BaseModel):
    """Schema for comprehensive session analysis."""
    session_id: str = Field(description="Unique session identifier")
    start_time: datetime = Field(description="Session start time")
    duration: float = Field(description="Session duration in minutes")
    metrics: MessageMetrics = Field(description="Session metrics")
    topics: TopicAnalysis = Field(description="Topic analysis")
    engagement_factors: List[str] = Field(description="Engagement indicators")
    improvement_suggestions: List[str] = Field(description="Areas for improvement")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "session_id": "CHAT001",
                "start_time": "2025-03-19T15:30:00",
                "duration": 15.5,
                "metrics": {
                    "message_count": 25,
                    "avg_length": 42.5,
                    "response_time": 2.3,
                    "sentiment_score": 0.65,
                    "engagement_level": "High"
                },
                "topics": {
                    "main_topics": [
                        "Product Features",
                        "Technical Support",
                        "Account Settings"
                    ],
                    "topic_shifts": 3,
                    "unresolved_topics": [
                        "Advanced Features Tutorial"
                    ],
                    "key_terms": {
                        "login": 5,
                        "settings": 4,
                        "help": 3
                    }
                },
                "engagement_factors": [
                    "Quick responses",
                    "Detailed questions",
                    "Follow-up queries"
                ],
                "improvement_suggestions": [
                    "Provide tutorial links proactively",
                    "Follow up on advanced features"
                ]
            }]
        }
    }

def create_chat_model() -> AzureChatOpenAI:
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

async def analyze_chat_stream(
    session_id: str,
    memory: ConversationSummaryMemory,
    parser: PydanticOutputParser,
    update_interval: float = 60.0
) -> AsyncIterator[SessionAnalysis]:
    """
    Generate stream of chat session analyses.
    
    Args:
        session_id: Session identifier
        memory: Conversation memory
        parser: Output parser for analysis
        update_interval: Time between updates in seconds
        
    Yields:
        SessionAnalysis: Updated session analysis
    """
    start_time = datetime.now()
    message_times = []
    
    while True:
        # Get conversation history
        history = memory.load_memory_variables({})
        messages = history.get("history", "").split("\n")
        
        # Calculate basic metrics
        message_count = len(messages)
        avg_length = sum(len(msg) for msg in messages) / max(message_count, 1)
        
        # Calculate response times
        if len(message_times) > 1:
            response_times = [
                (t2 - t1).total_seconds()
                for t1, t2 in zip(message_times[:-1], message_times[1:])
            ]
            avg_response_time = sum(response_times) / len(response_times)
        else:
            avg_response_time = 0
        
        # Create metrics
        metrics = MessageMetrics(
            message_count=message_count,
            avg_length=avg_length,
            response_time=avg_response_time,
            sentiment_score=0.5,  # Placeholder - implement sentiment analysis
            engagement_level="High" if avg_response_time < 5 else "Medium"
        )
        
        # Analyze topics (simplified)
        topics = TopicAnalysis(
            main_topics=["General Discussion"],  # Placeholder
            topic_shifts=0,
            unresolved_topics=[],
            key_terms={"chat": message_count}
        )
        
        # Generate analysis
        analysis = SessionAnalysis(
            session_id=session_id,
            start_time=start_time,
            duration=(datetime.now() - start_time).total_seconds() / 60,
            metrics=metrics,
            topics=topics,
            engagement_factors=[
                "Regular responses" if avg_response_time < 10 else "Delayed responses"
            ],
            improvement_suggestions=[
                "Maintain current engagement" if metrics.engagement_level == "High"
                else "Improve response time"
            ]
        )
        
        yield analysis
        await asyncio.sleep(update_interval)

async def demonstrate_session_analysis():
    """Demonstrate chat session analysis capabilities."""
    try:
        print("\nDemonstrating Chat Session Analysis...\n")
        
        # Initialize components
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=SessionAnalysis)
        memory = ConversationSummaryMemory(llm=chat_model, memory_key="history")
        
        # Example 1: New Chat Session
        print("Example 1: Analyzing New Chat Session")
        print("-" * 50)
        
        # Simulate some chat messages
        messages = [
            ("user", "Hi, I need help with the product settings."),
            ("assistant", "I'll be happy to help. What specific settings are you looking to adjust?"),
            ("user", "I can't find the advanced features section."),
            ("assistant", "The advanced features can be found under Settings > Advanced. Would you like me to guide you through them?")
        ]
        
        for role, content in messages:
            memory.save_context(
                {"input" if role == "user" else "output": content},
                {"input" if role == "assistant" else "output": "Chat message"}
            )
        
        # Monitor session
        count = 0
        async for analysis in analyze_chat_stream("CHAT001", memory, parser):
            print(f"\nAnalysis Update {count + 1}:")
            print(f"Session ID: {analysis.session_id}")
            print(f"Duration: {analysis.duration:.1f} minutes")
            
            print("\nMetrics:")
            print(f"Messages: {analysis.metrics.message_count}")
            print(f"Avg Length: {analysis.metrics.avg_length:.1f}")
            print(f"Response Time: {analysis.metrics.response_time:.1f}s")
            print(f"Engagement: {analysis.metrics.engagement_level}")
            
            print("\nEngagement Factors:")
            for factor in analysis.engagement_factors:
                print(f"- {factor}")
            
            count += 1
            if count >= 3:  # Show 3 updates
                break
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Chat Session Analyzer...")
    asyncio.run(demonstrate_session_analysis())

if __name__ == "__main__":
    main()
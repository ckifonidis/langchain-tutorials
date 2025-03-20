"""
LangChain Conversation History Analyzer Example

This example demonstrates how to combine structured output parsing with conversation
analysis capabilities to create a system that can analyze dialogue patterns and
generate meaningful insights.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

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

class DialogueTurn(BaseModel):
    """Schema for a single dialogue turn."""
    speaker: str = Field(description="Speaker identifier")
    message: str = Field(description="Message content")
    timestamp: str = Field(description="Message timestamp")
    sentiment: str = Field(description="Message sentiment (Positive/Neutral/Negative)")
    intent: str = Field(description="Speaker's intent or purpose")

class ConversationMetrics(BaseModel):
    """Schema for conversation-level metrics."""
    turn_count: int = Field(description="Number of dialogue turns")
    avg_message_length: float = Field(description="Average message length")
    sentiment_distribution: Dict[str, int] = Field(description="Count of each sentiment")
    topic_shifts: int = Field(description="Number of topic changes")
    engagement_score: float = Field(description="Overall engagement score (0-100)")

class ConversationAnalysis(BaseModel):
    """Schema for comprehensive conversation analysis."""
    conversation_id: str = Field(description="Unique conversation identifier")
    participants: List[str] = Field(description="List of participants")
    dialogue: List[DialogueTurn] = Field(description="Dialogue turns")
    metrics: ConversationMetrics = Field(description="Conversation metrics")
    main_topics: List[str] = Field(description="Main conversation topics")
    key_insights: List[str] = Field(description="Key conversation insights")
    suggestions: List[str] = Field(description="Improvement suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def analyze_conversation(messages: List[Dict], llm: AzureChatOpenAI) -> ConversationAnalysis:
    """
    Analyze conversation history and generate insights.
    
    Args:
        messages: List of conversation messages
        llm: Language model for analysis
        
    Returns:
        ConversationAnalysis: Comprehensive conversation analysis
    """
    # Format conversation for analysis
    conversation_text = "\n".join(
        f"{msg['speaker']}: {msg['message']}" for msg in messages
    )
    
    # System message with analysis instructions
    system_message = SystemMessage(content="""You are a conversation analyst. 
    Analyze the provided conversation and return a JSON object containing dialogue analysis, 
    metrics, topics, insights, and suggestions.""")
    
    # Human message with conversation and format requirements
    human_message = HumanMessage(content=f"""Analyze this conversation:

{conversation_text}

Provide your analysis in this exact JSON format:
{{
    "dialogue": [
        {{
            "speaker": "string",
            "message": "string",
            "timestamp": "string",
            "sentiment": "string",
            "intent": "string"
        }}
    ],
    "metrics": {{
        "turn_count": 0,
        "avg_message_length": 0,
        "sentiment_distribution": {{
            "Positive": 0,
            "Neutral": 0,
            "Negative": 0
        }},
        "topic_shifts": 0,
        "engagement_score": 0
    }},
    "main_topics": [],
    "key_insights": [],
    "suggestions": []
}}""")
    
    # Get analysis from LLM
    response = llm.invoke([system_message, human_message])
    content = response.content.strip()
    
    try:
        # Try to parse the JSON response
        analysis_data = json.loads(content)
    except json.JSONDecodeError:
        # Provide default analysis if JSON parsing fails
        analysis_data = {
            "dialogue": [
                {
                    "speaker": msg["speaker"],
                    "message": msg["message"],
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": "Neutral",
                    "intent": "General communication"
                }
                for msg in messages
            ],
            "metrics": {
                "turn_count": len(messages),
                "avg_message_length": sum(len(msg["message"]) for msg in messages) / max(len(messages), 1),
                "sentiment_distribution": {"Positive": 0, "Neutral": len(messages), "Negative": 0},
                "topic_shifts": 0,
                "engagement_score": 50.0
            },
            "main_topics": ["General Discussion"],
            "key_insights": ["Basic conversation analysis"],
            "suggestions": ["Consider more detailed analysis"]
        }
    
    return ConversationAnalysis(
        conversation_id=f"CONV{hash(conversation_text) % 1000:03d}",
        participants=list(set(msg["speaker"] for msg in messages)),
        dialogue=analysis_data.get("dialogue", []),
        metrics=ConversationMetrics(**analysis_data.get("metrics", {})),
        main_topics=analysis_data.get("main_topics", []),
        key_insights=analysis_data.get("key_insights", []),
        suggestions=analysis_data.get("suggestions", [])
    )

def demonstrate_conversation_analysis():
    """Demonstrate conversation analysis capabilities."""
    try:
        print("\nDemonstrating Conversation Analysis...\n")
        
        # Initialize components
        llm = create_chat_model()
        
        # Example: Technical Support Conversation
        print("Example: Technical Support Conversation")
        print("-" * 50)
        
        # Simulate conversation
        conversation = [
            {"speaker": "User", "message": "Hi, I need help with my account settings. I can't find where to change my password."},
            {"speaker": "Assistant", "message": "I'll be happy to help you with that. The password settings can be found in the Security section of your Account Settings."},
            {"speaker": "User", "message": "Thanks, but I don't see the Security section. Where exactly is it located?"},
            {"speaker": "Assistant", "message": "Let me guide you step by step: 1. Click on your profile icon in the top right corner. 2. Select 'Account Settings' from the dropdown menu. 3. Look for 'Security' in the left sidebar menu."},
            {"speaker": "User", "message": "Perfect, I found it! How do I make sure my new password is secure enough?"},
            {"speaker": "Assistant", "message": "Good question! A secure password should: 1. Be at least 12 characters long 2. Include uppercase and lowercase letters 3. Have numbers and special characters 4. Avoid personal information. The system will show a password strength indicator as you type."}
        ]
        
        # Analyze conversation
        analysis = analyze_conversation(conversation, llm)
        
        print("\nConversation Analysis:")
        print(f"ID: {analysis.conversation_id}")
        print(f"Participants: {', '.join(analysis.participants)}")
        
        print("\nMetrics:")
        print(f"Turns: {analysis.metrics.turn_count}")
        print(f"Average Message Length: {analysis.metrics.avg_message_length:.1f}")
        print(f"Topic Shifts: {analysis.metrics.topic_shifts}")
        print(f"Engagement Score: {analysis.metrics.engagement_score:.1f}")
        
        print("\nSentiment Distribution:")
        for sentiment, count in analysis.metrics.sentiment_distribution.items():
            print(f"{sentiment}: {count}")
        
        print("\nMain Topics:")
        for topic in analysis.main_topics:
            print(f"- {topic}")
        
        print("\nKey Insights:")
        for insight in analysis.key_insights:
            print(f"- {insight}")
        
        print("\nSuggestions:")
        for suggestion in analysis.suggestions:
            print(f"- {suggestion}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Conversation History Analyzer...")
    demonstrate_conversation_analysis()

if __name__ == "__main__":
    main()

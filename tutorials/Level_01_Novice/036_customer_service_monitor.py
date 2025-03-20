"""
LangChain Customer Service Quality Monitor Example

This example demonstrates how to combine memory management and evaluation capabilities
to create a system that monitors customer service interactions and evaluates response
quality while maintaining conversation context.

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

class ResponseMetrics(BaseModel):
    """Schema for response quality metrics."""
    clarity: int = Field(description="Clarity score (1-10)")
    relevance: int = Field(description="Relevance score (1-10)")
    completeness: int = Field(description="Completeness score (1-10)")
    professionalism: int = Field(description="Professionalism score (1-10)")
    empathy: int = Field(description="Empathy score (1-10)")
    resolution_rate: int = Field(description="Issue resolution score (1-10)")
    average_score: float = Field(description="Average of all scores")

class InteractionEvaluation(BaseModel):
    """Schema for comprehensive interaction evaluation."""
    interaction_id: str = Field(description="Unique interaction identifier")
    customer_query: str = Field(description="Original customer query")
    service_response: str = Field(description="Service representative's response")
    context_summary: str = Field(description="Summary of conversation context")
    metrics: ResponseMetrics = Field(description="Response quality metrics")
    strengths: List[str] = Field(description="Response strengths")
    areas_for_improvement: List[str] = Field(description="Areas needing improvement")
    suggested_improvements: List[str] = Field(description="Specific improvement suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "interaction_id": "CS001",
                "customer_query": "I'm having trouble with my account login",
                "service_response": "I understand how frustrating login issues can be. Let's get this resolved for you. Could you please tell me what specific error message you're seeing?",
                "context_summary": "First interaction with customer regarding login issues",
                "metrics": {
                    "clarity": 9,
                    "relevance": 8,
                    "completeness": 7,
                    "professionalism": 9,
                    "empathy": 9,
                    "resolution_rate": 7,
                    "average_score": 8.2
                },
                "strengths": [
                    "Shows empathy towards customer's frustration",
                    "Professional tone throughout",
                    "Clear request for specific information"
                ],
                "areas_for_improvement": [
                    "Could provide initial troubleshooting steps",
                    "Could mention common login issues"
                ],
                "suggested_improvements": [
                    "Include basic troubleshooting steps in initial response",
                    "Add link to password reset page proactively",
                    "Mention system status if relevant"
                ]
            }]
        }
    }

def create_chat_model():
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def evaluate_service_interaction(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    memory: ConversationSummaryMemory,
    interaction_data: Dict
) -> InteractionEvaluation:
    """
    Evaluate a customer service interaction.
    
    Args:
        chat_model: The chat model to use
        parser: The output parser for structured evaluation
        memory: Conversation memory for context
        interaction_data: Current interaction data
        
    Returns:
        InteractionEvaluation: Structured evaluation of the interaction
    """
    # Get format instructions for output structure
    format_instructions = parser.get_format_instructions()
    
    # Load conversation history
    history = memory.load_memory_variables({}).get("history", "")
    
    # Construct system message with context
    system_text = (
        "You are a customer service quality analyst. Evaluate the provided interaction "
        "considering the conversation history and context. Assess response quality, "
        "identify strengths and areas for improvement. Score metrics on a scale of 1-10.\n\n"
        f"Previous conversation context:\n{history}\n\n"
        "Respond with a JSON object that exactly follows this schema (no additional text):\n\n"
        f"{format_instructions}\n"
    )
    
    # Create messages
    system_msg = SystemMessage(content=system_text)
    human_msg = HumanMessage(
        content=f"Evaluate this customer service interaction: {json.dumps(interaction_data)}"
    )
    
    # Get evaluation
    response = chat_model.invoke([system_msg, human_msg])
    evaluation = parser.parse(response.content)
    
    # Update memory with current interaction
    memory.save_context(
        {"input": interaction_data["customer_query"]},
        {"output": interaction_data["service_response"]}
    )
    
    return evaluation

def demonstrate_service_monitoring():
    """Demonstrate customer service monitoring capabilities."""
    try:
        print("\nDemonstrating Customer Service Quality Monitoring...\n")
        
        # Initialize components
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=InteractionEvaluation)
        memory = ConversationSummaryMemory(llm=chat_model, memory_key="history")
        
        # Example 1: Initial Customer Query
        print("Example 1: Login Issue Interaction")
        print("-" * 50)
        
        interaction1 = {
            "interaction_id": "CS001",
            "customer_query": "I can't log into my account. It keeps saying 'invalid password' even though I'm sure it's correct.",
            "service_response": "I understand how frustrating login issues can be. Let me help you resolve this. Could you tell me if you're seeing any specific error message? Also, have you tried resetting your password using the 'Forgot Password' link?",
            "context_summary": "First interaction - customer reporting login issues"
        }
        
        evaluation1 = evaluate_service_interaction(chat_model, parser, memory, interaction1)
        
        print("\nInteraction Evaluation:")
        print(f"ID: {evaluation1.interaction_id}")
        print("\nMetrics:")
        print(f"Clarity: {evaluation1.metrics.clarity}/10")
        print(f"Relevance: {evaluation1.metrics.relevance}/10")
        print(f"Completeness: {evaluation1.metrics.completeness}/10")
        print(f"Professionalism: {evaluation1.metrics.professionalism}/10")
        print(f"Empathy: {evaluation1.metrics.empathy}/10")
        print(f"Resolution Rate: {evaluation1.metrics.resolution_rate}/10")
        print(f"Average Score: {evaluation1.metrics.average_score:.1f}")
        
        print("\nStrengths:")
        for strength in evaluation1.strengths:
            print(f"- {strength}")
        
        print("\nAreas for Improvement:")
        for area in evaluation1.areas_for_improvement:
            print(f"- {area}")
        
        # Example 2: Follow-up Interaction
        print("\nExample 2: Password Reset Follow-up")
        print("-" * 50)
        
        interaction2 = {
            "interaction_id": "CS002",
            "customer_query": "I tried resetting my password but haven't received the reset email yet.",
            "service_response": "Thank you for trying the password reset. Let me check the status for you. First, could you verify if you're using the same email address associated with your account? Also, please check your spam/junk folder for the reset email. If you still don't see it, I can resend the reset link.",
            "context_summary": "Follow-up to login issue - password reset email not received"
        }
        
        evaluation2 = evaluate_service_interaction(chat_model, parser, memory, interaction2)
        
        print("\nInteraction Evaluation:")
        print(f"ID: {evaluation2.interaction_id}")
        print("\nMetrics:")
        print(f"Clarity: {evaluation2.metrics.clarity}/10")
        print(f"Relevance: {evaluation2.metrics.relevance}/10")
        print(f"Completeness: {evaluation2.metrics.completeness}/10")
        print(f"Professionalism: {evaluation2.metrics.professionalism}/10")
        print(f"Empathy: {evaluation2.metrics.empathy}/10")
        print(f"Resolution Rate: {evaluation2.metrics.resolution_rate}/10")
        print(f"Average Score: {evaluation2.metrics.average_score:.1f}")
        
        print("\nStrengths:")
        for strength in evaluation2.strengths:
            print(f"- {strength}")
        
        print("\nAreas for Improvement:")
        for area in evaluation2.areas_for_improvement:
            print(f"- {area}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Customer Service Quality Monitor...")
    demonstrate_service_monitoring()

if __name__ == "__main__":
    main()
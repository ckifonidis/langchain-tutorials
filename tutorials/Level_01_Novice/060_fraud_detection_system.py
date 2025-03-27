#!/usr/bin/env python3
"""
LangChain Multi-Agent Fraud Detection System (LangChain v3)

This example demonstrates a sophisticated fraud detection system using multiple
coordinated agents with memory capabilities. The system analyzes banking
transactions and user behavior patterns to detect potential fraud while
maintaining context across interactions.

Key concepts demonstrated:
1. Multi-Agent System with Orchestration
2. Memory Management for Context Retention
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

class TransactionData(BaseModel):
    """Schema for transaction data."""
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    merchant: str = Field(description="Merchant name")
    timestamp: str = Field(description="Transaction timestamp")
    location: str = Field(description="Transaction location")
    category: str = Field(description="Transaction category")
    user_id: str = Field(description="User identifier")

class UserProfile(BaseModel):
    """Schema for user profile."""
    user_id: str = Field(description="User identifier")
    risk_level: str = Field(description="User risk level")
    usual_locations: List[str] = Field(description="Common transaction locations")
    usual_merchants: List[str] = Field(description="Frequent merchants")
    average_transaction: float = Field(description="Average transaction amount")

class FraudDetectionTools:
    """Collection of fraud detection tools."""
    
    @staticmethod
    def get_user_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile and transaction patterns."""
        # Simulated user profile data
        return {
            "user_id": user_id,
            "risk_level": "LOW",
            "usual_locations": ["New York, USA", "Boston, USA"],
            "usual_merchants": ["Amazon", "Walmart", "Target"],
            "average_transaction": 150.00
        }
    
    @staticmethod
    def get_transaction_history(user_id: str) -> List[Dict[str, Any]]:
        """Get recent transaction history."""
        # Simulated transaction history
        return [
            {
                "transaction_id": "T123",
                "amount": 50.00,
                "merchant": "Amazon",
                "timestamp": "2025-03-20T10:00:00",
                "location": "New York, USA",
                "category": "Shopping"
            },
            {
                "transaction_id": "T124",
                "amount": 1500.00,
                "merchant": "Unknown Store",
                "timestamp": "2025-03-20T10:30:00",
                "location": "Lagos, Nigeria",
                "category": "Electronics"
            }
        ]
    
    @staticmethod
    def analyze_location_pattern(
        location: str,
        user_locations: List[str]
    ) -> Dict[str, Any]:
        """Analyze location against user patterns."""
        return {
            "is_usual": location in user_locations,
            "risk_level": "HIGH" if location not in user_locations else "LOW",
            "distance_from_usual": "Far" if location not in user_locations else "Near"
        }

class StreamingCallback(BaseCallbackHandler):
    """Handle streaming updates."""
    def __init__(self):
        self.updates = []
        self.current_agent = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Handle start of LLM operations."""
        agent_name = self.current_agent or "System"
        print(f"\n{agent_name} Processing...")
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Process streaming tokens."""
        print(token, end="", flush=True)
        self.updates.append(token)
    
    def on_llm_end(self, response: Any, **kwargs):
        """Handle completion of LLM operations."""
        agent_name = self.current_agent or "System"
        print(f"\n{agent_name} Complete.")

def create_pattern_analysis_agent() -> AgentExecutor:
    """Create agent for analyzing transaction patterns."""
    prompt = PromptTemplate(
        input_variables=["transaction", "user_profile", "agent_scratchpad"],
        template="""You are an expert in analyzing transaction patterns.
        
        Transaction: {transaction}
        User Profile: {user_profile}
        
        Analyze the transaction pattern and identify anomalies.
        
        {agent_scratchpad}
        
        Return your analysis in this exact format:
        {{"response": {{
            "analysis": "detailed explanation",
            "risk_factors": ["list", "of", "risks"],
            "confidence": confidence_score,
            "recommendation": "action to take"
        }}}}
        """
    )
    
    tools = [
        Tool(
            name="analyze_location",
            func=FraudDetectionTools.analyze_location_pattern,
            description="Analyze location patterns"
        )
    ]
    
    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_functions_agent(
            llm=create_chat_model(streaming=True),
            prompt=prompt,
            tools=tools
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def create_risk_assessment_agent() -> AgentExecutor:
    """Create agent for risk assessment."""
    prompt = PromptTemplate(
        input_variables=["transaction", "pattern_analysis", "agent_scratchpad"],
        template="""You are an expert in fraud risk assessment.
        
        Transaction: {transaction}
        Pattern Analysis: {pattern_analysis}
        
        Assess the fraud risk level.
        
        {agent_scratchpad}
        
        Return your assessment in this exact format:
        {{"response": {{
            "risk_level": "HIGH|MEDIUM|LOW",
            "factors": ["risk", "factors"],
            "confidence": confidence_score,
            "action": "recommended action"
        }}}}
        """
    )
    
    tools = [
        Tool(
            name="get_transaction_history",
            func=FraudDetectionTools.get_transaction_history,
            description="Get transaction history"
        )
    ]
    
    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_functions_agent(
            llm=create_chat_model(streaming=True),
            prompt=prompt,
            tools=tools
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def create_chat_model(streaming: bool = False) -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        streaming=streaming
    )

def create_orchestrator_agent() -> RunnableLambda:
    """Create the main orchestrator agent."""
    def clean_json_response(response: str) -> str:
        """Clean JSON response by removing markdown markers."""
        import re
        response = re.sub(r'```json\s*|\s*```', '', response)
        return response.strip()
    
    def orchestrate(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate fraud detection process."""
        try:
            # Parse transaction
            transaction = TransactionData(**inputs["transaction"])
            
            # Get user profile
            user_profile = UserProfile(
                **FraudDetectionTools.get_user_profile(transaction.user_id)
            )
            
            # Initialize memory
            memory = ConversationBufferMemory(
                chat_memory=RedisChatMessageHistory(
                    url="redis://localhost:6379/0",
                    session_id=f"user_{transaction.user_id}"
                )
            )
            
            # Initialize agents
            pattern_agent = create_pattern_analysis_agent()
            risk_agent = create_risk_assessment_agent()
            
            # Analyze patterns
            pattern_result = pattern_agent.invoke({
                "transaction": transaction.json(),
                "user_profile": user_profile.json(),
                "agent_scratchpad": ""
            })
            
            # Store analysis in memory
            memory.save_context(
                {"input": "Pattern Analysis"},
                {"output": pattern_result["output"]}
            )
            
            # Assess risk
            risk_result = risk_agent.invoke({
                "transaction": transaction.json(),
                "pattern_analysis": pattern_result["output"],
                "agent_scratchpad": ""
            })
            
            # Store assessment in memory
            memory.save_context(
                {"input": "Risk Assessment"},
                {"output": risk_result["output"]}
            )
            
            # Get memory contents
            memory_contents = memory.load_memory_variables({})
            
            # Clean and parse responses
            pattern_response = json.loads(
                clean_json_response(pattern_result["output"])
            )
            risk_response = json.loads(
                clean_json_response(risk_result["output"])
            )
            
            return {
                "status": "success",
                "pattern_analysis": pattern_response["response"],
                "risk_assessment": risk_response["response"],
                "memory_contents": memory_contents,
                "metadata": {
                    "transaction_id": transaction.transaction_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    return RunnableLambda(orchestrate)

def demonstrate_fraud_detection():
    """Demonstrate the fraud detection system."""
    print("\nInitializing Fraud Detection System...\n")
    
    # Example transactions
    transactions = [
        {
            "transaction": {
                "transaction_id": "T123",
                "amount": 50.00,
                "merchant": "Amazon",
                "timestamp": "2025-03-20T10:00:00",
                "location": "New York, USA",
                "category": "Shopping",
                "user_id": "U789"
            }
        },
        {
            "transaction": {
                "transaction_id": "T124",
                "amount": 1500.00,
                "merchant": "Unknown Store",
                "timestamp": "2025-03-20T10:30:00",
                "location": "Lagos, Nigeria",
                "category": "Electronics",
                "user_id": "U789"
            }
        }
    ]
    
    # Create orchestrator
    orchestrator = create_orchestrator_agent()
    
    # Process transactions
    for tx in transactions:
        print(f"\nAnalyzing Transaction: {tx['transaction']['transaction_id']}")
        result = orchestrator.invoke(tx)
        
        # Display results
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))
        print("\n" + "="*50)

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Fraud Detection System...")
    demonstrate_fraud_detection()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
LangChain Multi-Agent Banking Service Example (LangChain v3)

This example demonstrates a sophisticated banking customer service system using
multiple specialized agents coordinated by an orchestrator agent. The system
provides real-time responses through streaming while handling complex banking
queries across different domains.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

class CustomerQuery(BaseModel):
    """Schema for customer service queries."""
    query_text: str = Field(description="Customer's original query")
    context: Dict[str, Any] = Field(description="Additional context")
    priority: str = Field(description="Query priority (HIGH|MEDIUM|LOW)")
    category: str = Field(description="Query category")
    customer_id: str = Field(description="Customer identifier")

class ServiceResponse(BaseModel):
    """Schema for service responses."""
    answer: str = Field(description="Response to customer")
    sources: List[str] = Field(description="Information sources")
    confidence: float = Field(description="Confidence score")
    follow_up: Optional[str] = Field(description="Suggested follow-up")

class StreamingCallback(BaseCallbackHandler):
    """Custom callback handler for streaming responses."""
    
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

class BankingTools:
    """Collection of banking-related tools."""
    
    @staticmethod
    def get_account_info(account_id: str) -> Dict[str, Any]:
        """Get account information."""
        # Simulated account data
        return {
            "account_id": account_id,
            "type": "checking",
            "balance": 5000.00,
            "status": "active"
        }
    
    @staticmethod
    def get_transaction_history(account_id: str) -> List[Dict[str, Any]]:
        """Get transaction history."""
        # Simulated transaction data
        return [
            {
                "date": "2025-03-20",
                "type": "deposit",
                "amount": 1000.00
            },
            {
                "date": "2025-03-19",
                "type": "withdrawal",
                "amount": 50.00
            }
        ]
    
    @staticmethod
    def check_loan_eligibility(customer_id: str) -> Dict[str, Any]:
        """Check loan eligibility."""
        # Simulated eligibility check
        return {
            "eligible": True,
            "max_amount": 25000.00,
            "rate": 5.5,
            "term_months": 60
        }

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

def create_accounts_agent() -> AgentExecutor:
    """Create an agent specializing in account operations."""
    template = """You are an expert in banking accounts.
    
    Query: {query}
    Account Info: {account_info}
    
    Provide accurate information about accounts and balances.
    
    {agent_scratchpad}
    
    Return your response in this exact format:
    {{"response": {{"answer": "detailed explanation", "sources": ["sources"], "confidence": 0.95, "follow_up": "suggestion"}}}}
    """
    
    prompt = PromptTemplate(
        input_variables=["query", "account_info", "agent_scratchpad"],
        template=template
    )
    
    tools = [
        Tool(
            name="get_account_info",
            func=BankingTools.get_account_info,
            description="Get account information"
        ),
        Tool(
            name="get_transaction_history",
            func=BankingTools.get_transaction_history,
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

def create_loans_agent() -> AgentExecutor:
    """Create an agent specializing in loans."""
    template = """You are an expert in banking loans.
    
    Query: {query}
    Customer Info: {customer_info}
    
    Provide accurate loan information and eligibility.
    
    {agent_scratchpad}
    
    Return your response in this exact format:
    {{"response": {{"answer": "detailed explanation", "sources": ["sources"], "confidence": 0.95, "follow_up": "suggestion"}}}}
    """
    
    prompt = PromptTemplate(
        input_variables=["query", "customer_info", "agent_scratchpad"],
        template=template
    )
    
    tools = [
        Tool(
            name="check_loan_eligibility",
            func=BankingTools.check_loan_eligibility,
            description="Check loan eligibility"
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

def clean_json_response(response: str) -> str:
    """Clean JSON response by removing markdown markers."""
    # Remove markdown code block markers
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'\s*```', '', response)
    return response

def create_orchestrator_agent() -> RunnableLambda:
    """Create the main orchestrator agent."""
    def orchestrate(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate query handling across specialized agents."""
        try:
            # Parse query
            query = CustomerQuery(**inputs)
            
            # Initialize agents
            accounts_agent = create_accounts_agent()
            loans_agent = create_loans_agent()
            
            # Route to appropriate agent
            if "account" in query.category.lower():
                result = accounts_agent.invoke({
                    "query": query.query_text,
                    "account_info": BankingTools.get_account_info(
                        query.context.get("account_id", "")
                    ),
                    "agent_scratchpad": ""
                })
            elif "loan" in query.category.lower():
                result = loans_agent.invoke({
                    "query": query.query_text,
                    "customer_info": {
                        "id": query.customer_id,
                        "context": query.context
                    },
                    "agent_scratchpad": ""
                })
            else:
                raise ValueError(f"Unknown query category: {query.category}")
            
            # Clean and parse the response
            cleaned_output = clean_json_response(result["output"])
            print(f"\nCleaned output: {cleaned_output}")  # Debug info
            parsed_response = json.loads(cleaned_output)
            
            return {
                "status": "success",
                "response": parsed_response["response"],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "category": query.category,
                    "priority": query.priority
                }
            }
            
        except json.JSONDecodeError as e:
            print(f"\nJSON Decode Error: {str(e)}")
            print(f"Problematic content: {result.get('output', 'No output')}")
            return {
                "status": "error",
                "error": "Failed to parse response",
                "details": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            print(f"\nGeneral Error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    return RunnableLambda(orchestrate)

def demonstrate_banking_service():
    """Demonstrate the banking service capabilities."""
    print("\nInitializing Multi-Agent Banking Service...\n")
    
    # Example queries
    queries = [
        {
            "query_text": "What's my current balance and recent transactions?",
            "context": {"account_id": "12345"},
            "priority": "MEDIUM",
            "category": "account",
            "customer_id": "C789"
        },
        {
            "query_text": "Am I eligible for a home loan? What's the maximum amount?",
            "context": {"income": 75000, "credit_score": 720},
            "priority": "HIGH",
            "category": "loan",
            "customer_id": "C789"
        }
    ]
    
    # Create orchestrator
    orchestrator = create_orchestrator_agent()
    
    # Process queries
    for query in queries:
        print(f"\nProcessing Query: {query['query_text']}")
        result = orchestrator.invoke(query)
        
        # Display results
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        print("\n" + "="*50)

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Multi-Agent Banking Service...")
    demonstrate_banking_service()

if __name__ == "__main__":
    main()
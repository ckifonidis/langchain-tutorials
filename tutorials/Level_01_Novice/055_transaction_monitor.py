#!/usr/bin/env python3
"""
LangChain Transaction Monitor Example (LangChain v3)

This example demonstrates how to combine memory and streaming capabilities to create
a sophisticated transaction monitoring system that can process real-time financial
data while maintaining context.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any, Generator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

class Transaction(BaseModel):
    """Schema for financial transactions."""
    id: str = Field(description="Transaction identifier")
    timestamp: datetime = Field(description="Transaction time")
    amount: float = Field(description="Transaction amount")
    type: str = Field(description="Transaction type")
    description: str = Field(description="Transaction description")
    merchant: str = Field(description="Merchant name")
    category: str = Field(description="Transaction category")
    risk_score: float = Field(description="Risk assessment score")

class Alert(BaseModel):
    """Schema for transaction alerts."""
    transaction_id: str = Field(description="Related transaction ID")
    severity: str = Field(description="Alert severity level")
    reason: str = Field(description="Alert trigger reason")
    details: str = Field(description="Detailed explanation")
    timestamp: datetime = Field(default_factory=datetime.now)

class MonitorCallback(BaseCallbackHandler):
    """Custom callback handler for streaming transaction analysis."""
    
    def __init__(self):
        self.analysis = []
        self.current_transaction = None
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Handle start of LLM generation."""
        print("\nAnalyzing transaction...\n")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process streaming tokens."""
        print(token, end="", flush=True)
        self.analysis.append(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle completion of LLM generation."""
        print("\nAnalysis complete.\n")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle LLM errors."""
        print(f"\nError during analysis: {str(error)}\n")

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,  # Low temperature for consistent analysis
        streaming=True  # Enable streaming for real-time analysis
    )

def create_transaction_analyzer() -> RunnablePassthrough:
    """Create a chain for transaction analysis."""
    template = """You are an expert financial transaction monitor. Analyze this transaction:

Transaction Details:
{transaction}

Previous Context:
{history}

Analyze for:
1. Transaction patterns
2. Risk indicators
3. Suspicious activity
4. Compliance issues

Return your analysis as a JSON object with this structure:

{{
    "risk_level": "<LOW|MEDIUM|HIGH>",
    "analysis": {{
        "patterns": ["<pattern>", ...],
        "indicators": ["<indicator>", ...],
        "suspicious": ["<activity>", ...],
        "compliance": ["<issue>", ...]
    }},
    "alerts": [
        {{
            "severity": "<level>",
            "reason": "<description>",
            "details": "<explanation>"
        }}
    ],
    "recommendations": ["<action>", ...]
}}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["transaction", "history"]
    )
    
    return prompt | create_chat_model()

def analyze_transaction(tx: Transaction, memory: ConversationBufferMemory) -> Generator[str, None, None]:
    """Analyze a transaction and yield results."""
    try:
        # Get conversation history
        history = memory.load_memory_variables({})
        
        # Create analyzer chain
        analyzer = create_transaction_analyzer()
        
        # Analyze transaction with callback
        callback = MonitorCallback()
        result = analyzer.invoke(
            {
                "transaction": tx.model_dump_json(indent=2),
                "history": history.get("history", "No previous history.")
            },
            config={"callbacks": [callback]}
        )
        
        # Update memory
        memory.save_context(
            {"input": f"Transaction: {tx.id}"},
            {"output": result.content}
        )
        
        yield result.content
        
    except Exception as e:
        yield f"Error analyzing transaction: {str(e)}"

def demonstrate_transaction_monitor():
    """Demonstrate the Transaction Monitor capabilities."""
    try:
        print("\nInitializing Transaction Monitor...\n")
        
        # Initialize memory
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # Example transactions
        transactions = [
            Transaction(
                id="TX001",
                timestamp=datetime.now(),
                amount=1500.00,
                type="purchase",
                description="Electronics Store Purchase",
                merchant="TechHub",
                category="Electronics",
                risk_score=0.2
            ),
            Transaction(
                id="TX002",
                timestamp=datetime.now(),
                amount=5000.00,
                type="transfer",
                description="International Wire Transfer",
                merchant="SWIFT",
                category="Wire Transfer",
                risk_score=0.7
            ),
            Transaction(
                id="TX003",
                timestamp=datetime.now(),
                amount=2500.00,
                type="withdrawal",
                description="ATM Withdrawal",
                merchant="ATM",
                category="Cash",
                risk_score=0.4
            )
        ]
        
        # Process transactions
        for tx in transactions:
            print(f"Processing Transaction {tx.id}...")
            print(f"Amount: ${tx.amount:.2f}")
            print(f"Type: {tx.type}")
            print(f"Initial Risk Score: {tx.risk_score:.2f}")
            
            for result in analyze_transaction(tx, memory):
                print(result)
                
            print("\n" + "="*50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Transaction Monitor...")
    demonstrate_transaction_monitor()

if __name__ == "__main__":
    main()
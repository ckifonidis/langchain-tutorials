#!/usr/bin/env python3
"""
LangChain v3 Multi-Agent Fraud Investigation System
"""

import os
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent

# Load environment variables
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

class TransactionType(str, Enum):
    TRANSFER = "transfer"
    PAYMENT = "payment"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"

class TransactionData(BaseModel):
    transaction_id: str = Field(description="Unique transaction identifier")
    type: TransactionType = Field(description="Type of transaction")
    amount: float = Field(description="Transaction amount")
    currency: str = Field(description="Currency code")
    source: str = Field(description="Source account/entity")
    destination: str = Field(description="Destination account/entity")
    timestamp: datetime = Field(description="Transaction timestamp")
    description: Optional[str] = Field(description="Transaction description")

class FraudInvestigator:
    def __init__(self):
        print("\nValidating Azure OpenAI deployments...")
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        self.llm.invoke("Test connection")
        print("✓ Chat model deployment validated")
        print("✓ System initialized")
        self.coordinator = self._create_coordinator()

    def _analyze_patterns(self, data: Dict) -> Dict:
        amount = float(data.get('amount', 0))
        is_international = 'FOREIGN' in str(data.get('destination', '')).upper()
        findings = []
        if amount > 50000:
            findings.append("High value transaction")
        if is_international:
            findings.append("International transfer")
        return {"patterns": findings}

    def _assess_risk(self, data: Dict) -> Dict:
        patterns = data.get('patterns', [])
        is_high_value = "High value" in patterns
        is_international = "International" in patterns
        
        risk_level = "low"
        if is_high_value and is_international:
            risk_level = "high"
        elif is_high_value or is_international:
            risk_level = "medium"
            
        return {
            "level": risk_level,
            "findings": patterns,
            "review_required": risk_level == "high"
        }

    def _create_coordinator(self) -> AgentExecutor:
        tools = [
            Tool(
                name="analyze_patterns",
                description="Analyze transaction patterns",
                func=lambda x: json.dumps(self._analyze_patterns(x)),
                return_direct=False
            ),
            Tool(
                name="assess_risk",
                description="Evaluate risk level",
                func=lambda x: json.dumps(self._assess_risk(x)),
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Fraud Investigator. Analyze transactions by:
1. Using analyze_patterns to check for suspicious patterns
2. Using assess_risk to evaluate findings
3. Return a JSON report only

Required JSON Format:
{{
    "findings": ["list findings"],
    "risk_level": "high/medium/low",
    "requires_review": true/false
}}"""),
            ("human", "Transaction to analyze:\n{input}"),
            ("human", "Steps taken: {agent_scratchpad}")
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=3,
            handle_parsing_errors=True
        )

        return executor

    def investigate_transaction(self, transaction: TransactionData) -> Dict:
        try:
            if transaction.amount <= 0:
                raise ValueError("Invalid amount")

            investigation = {
                "transaction": transaction.model_dump(),
                "status": "in_progress",
                "timestamp": datetime.now()
            }

            input_text = f"""
Type: {transaction.type}
Amount: {transaction.amount} {transaction.currency}
Source: {transaction.source}
Destination: {transaction.destination}
Description: {transaction.description}
"""
            result = self.coordinator.invoke({"input": input_text.strip()})

            if result.get("output"):
                try:
                    output = json.loads(result["output"]) if isinstance(result["output"], str) else result["output"]
                    investigation.update({
                        "status": "completed",
                        "findings": output.get("findings", []),
                        "risk_level": output.get("risk_level", "unknown"),
                        "requires_review": output.get("requires_review", False),
                        "completion_time": datetime.now()
                    })
                except Exception as e:
                    investigation.update({
                        "status": "error",
                        "error": f"Failed to parse results: {str(e)}",
                        "raw_output": result["output"],
                        "completion_time": datetime.now()
                    })
            else:
                investigation.update({
                    "status": "failed",
                    "error": "No output produced",
                    "completion_time": datetime.now()
                })

            return investigation

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction.transaction_id,
                "timestamp": datetime.now()
            }

def demonstrate_investigation():
    print("\nFraud Investigation System Demo")
    print("===============================")
    
    transactions = [
        TransactionData(
            transaction_id="INV001",
            type=TransactionType.TRANSFER,
            amount=75000.00,
            currency="USD",
            source="account_123",
            destination="FOREIGN_ACC_456",
            timestamp=datetime.now(),
            description="International wire transfer"
        ),
        TransactionData(
            transaction_id="INV002",
            type=TransactionType.PAYMENT,
            amount=12000.00,
            currency="EUR",
            source="account_789",
            destination="vendor_555",
            timestamp=datetime.now(),
            description="Multiple vendor payments"
        )
    ]
    
    try:
        print("\nInitializing Fraud Investigation System...")
        investigator = FraudInvestigator()
        print("\nSystem initialized successfully")
        
        for idx, txn in enumerate(transactions, 1):
            print(f"\nInvestigating Transaction {idx}/{len(transactions)}")
            print("=" * 40)
            print(f"Transaction ID: {txn.transaction_id}")
            print(f"Type: {txn.type}")
            print(f"Amount: {txn.amount} {txn.currency}")
            print("-" * 40)
            
            result = investigator.investigate_transaction(txn)
            
            print("\nInvestigation Results:")
            print(json.dumps(result, indent=2, cls=DateTimeEncoder))
            print("=" * 40)
            
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")

if __name__ == "__main__":
    demonstrate_investigation()
#!/usr/bin/env python3
"""
Fraud Detector (137) (LangChain v3)

This example demonstrates fraud detection using:
1. Tool Calling: External system integration
2. Agents: Autonomous investigation
3. Message History: Context tracking

It helps risk teams detect banking fraud patterns.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import Tool
from langchain_core.agents import AgentFinish, AgentAction
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AlertLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_risk"

class FraudType(str, Enum):
    """Fraud categories."""
    ACCOUNT = "account_takeover"
    PAYMENT = "payment_fraud"
    IDENTITY = "identity_theft"
    SYNTHETIC = "synthetic_fraud"
    INTERNAL = "insider_threat"
    COLLUSION = "collusion_fraud"

class Transaction(BaseModel):
    """Transaction details."""
    transaction_id: str = Field(description="Transaction ID")
    type: str = Field(description="Transaction type")
    amount: float = Field(description="Amount")
    timestamp: str = Field(description="Time of transaction")
    details: Dict = Field(description="Transaction details")
    metadata: Dict = Field(default_factory=dict)

class Alert(BaseModel):
    """Alert details."""
    alert_id: str = Field(description="Alert ID")
    level: AlertLevel = Field(description="Alert level")
    type: FraudType = Field(description="Fraud type")
    source: str = Field(description="Alert source")
    details: Dict = Field(description="Alert details")
    metadata: Dict = Field(default_factory=dict)

class FraudDetector:
    """Fraud detection system."""

    def __init__(self):
        """Initialize detector."""
        logger.info("Starting fraud detector...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup tools
        self.tools = [
            Tool(
                name="check_transaction_history",
                description="Look up previous transactions for patterns",
                func=self.check_history
            ),
            Tool(
                name="verify_identity",
                description="Verify user identity and authentication",
                func=self.verify_identity
            ),
            Tool(
                name="analyze_behavior",
                description="Analyze user behavior patterns",
                func=self.analyze_behavior
            )
        ]
        
        # Convert tools to OpenAI functions
        functions = [convert_to_openai_function(t) for t in self.tools]
        
        # Setup agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fraud detection expert.
You have access to tools for investigation.

Review alerts and transactions for fraud.
Follow this process:
1. Review alert details
2. Check transaction history
3. Verify identity if needed
4. Analyze behavior patterns
5. Make recommendations

Format your final response like this:

FRAUD ANALYSIS
-------------
Alert: ID
Level: SEVERITY
Type: CATEGORY

Findings:
1. Finding Name
   Evidence: Details
   Impact: Description
   Risk: Level

2. Finding Name
   Evidence: Details
   Impact: Description
   Risk: Level

Required Actions:
1. Action step
2. Action step

Next Steps:
1. Required check
2. Required check"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Setup memory
        self.messages = []
        self.window_size = 5
        logger.info("Memory initialized")
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Setup agent executor
        self.agent_chain = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
        logger.info("Agent ready")

    def check_history(self, transaction_id: str) -> str:
        """Check transaction history."""
        logger.info(f"Checking history for: {transaction_id}")
        
        # Simulate history check
        history = {
            "last_24h": 5,
            "similar_pattern": 2,
            "risk_score": 0.75
        }
        
        return f"Found {history['last_24h']} related transactions, {history['similar_pattern']} match pattern, risk score: {history['risk_score']}"

    def verify_identity(self, transaction_id: str) -> str:
        """Verify user identity."""
        logger.info(f"Verifying identity for: {transaction_id}")
        
        # Simulate verification
        factors = {
            "device_match": True,
            "location_match": False,
            "behavior_match": True
        }
        
        issues = [k for k, v in factors.items() if not v]
        return f"Identity check complete. Issues found: {', '.join(issues)}"

    def analyze_behavior(self, transaction_id: str) -> str:
        """Analyze user behavior."""
        logger.info(f"Analyzing behavior for: {transaction_id}")
        
        # Simulate analysis
        patterns = {
            "typical_amount": False,
            "normal_time": True,
            "usual_location": False,
            "normal_merchant": True
        }
        
        anomalies = [k for k, v in patterns.items() if not v]
        return f"Behavior analysis complete. Anomalies: {', '.join(anomalies)}"

    async def investigate_alert(self, alert: Alert, transaction: Transaction) -> str:
        """Investigate fraud alert."""
        logger.info(f"Investigating alert: {alert.alert_id}")
        
        try:
            # Format input with available tools listed in message
            tools_list = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tools
            ])
            
            input_text = f"""Available tools:
{tools_list}

Investigate this alert:
Alert ID: {alert.alert_id}
Level: {alert.level.value}
Type: {alert.type.value}
Source: {alert.source}

Related Transaction:
ID: {transaction.transaction_id}
Type: {transaction.type}
Amount: ${transaction.amount:,.2f}
Time: {transaction.timestamp}

Analyze for potential fraud."""
            
            # Update memory
            self.messages.append({"role": "user", "content": input_text})
            if len(self.messages) > self.window_size:
                self.messages.pop(0)
            
            # Run investigation
            result = await self.agent_chain.ainvoke({"input": input_text})
            logger.info("Investigation complete")
            
            return result["output"]
            
        except Exception as e:
            logger.error(f"Investigation failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting fraud detection demo...")
    
    try:
        # Create detector
        detector = FraudDetector()
        
        # Example alert
        alert = Alert(
            alert_id="ALERT-2025-001",
            level=AlertLevel.HIGH,
            type=FraudType.ACCOUNT,
            source="behavior_monitor",
            details={
                "trigger": "unusual_access",
                "attempts": 3,
                "window": "10m",
                "ip_changes": True,
                "device_changes": True
            }
        )
        
        # Related transaction
        transaction = Transaction(
            transaction_id="TXN-2025-001",
            type="large_transfer",
            amount=25000.00,
            timestamp="2025-04-03T19:30:00Z",
            details={
                "source": "checking_account",
                "destination": "external_account",
                "routing": "foreign_bank",
                "method": "wire_transfer"
            }
        )
        
        print("\nInvestigating Alert")
        print("==================")
        print(f"Alert: {alert.alert_id}")
        print(f"Level: {alert.level.value}")
        print(f"Type: {alert.type.value}")
        print(f"Source: {alert.source}\n")
        
        print("Related Transaction:")
        print(f"ID: {transaction.transaction_id}")
        print(f"Type: {transaction.type}")
        print(f"Amount: ${transaction.amount:,.2f}")
        print(f"Time: {transaction.timestamp}\n")
        
        try:
            # Get analysis
            result = await detector.investigate_alert(alert, transaction)
            print("\nInvestigation Results:")
            print("=====================")
            print(result)
            
        except Exception as e:
            print(f"\nInvestigation failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
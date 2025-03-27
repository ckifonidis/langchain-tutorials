#!/usr/bin/env python3
"""
Transaction Rule Engine (LangChain v3)

This example demonstrates a banking transaction rule engine using LangChain's LCEL for
composing rule chains and tool_calling for executing transaction actions. It provides
flexible rule-based transaction processing for banking operations.

Key concepts demonstrated:
1. lcel: Chain composition for rule evaluation and processing
2. tool_calling: Dynamic action execution based on rule outcomes
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from decimal import Decimal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables
load_dotenv()

class TransactionType(str, Enum):
    """Supported transaction types."""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"

class RuleAction(str, Enum):
    """Possible rule actions."""
    APPROVE = "approve"
    DENY = "deny"
    FLAG = "flag"
    REVIEW = "review"
    NOTIFY = "notify"

class Transaction(BaseModel):
    """Banking transaction model."""
    id: str = Field(description="Transaction ID")
    type: TransactionType = Field(description="Transaction type")
    amount: Decimal = Field(description="Transaction amount")
    currency: str = Field(description="Currency code")
    source_account: str = Field(description="Source account")
    destination_account: str = Field(description="Destination account")
    timestamp: datetime = Field(description="Transaction time")
    description: Optional[str] = Field(default=None, description="Optional description")

class RuleResult(BaseModel):
    """Rule evaluation result."""
    action: RuleAction = Field(description="Action to take")
    reason: str = Field(description="Reason for the action")
    risk_level: str = Field(description="Risk level assessment")
    required_actions: List[str] = Field(default_factory=list, description="Required follow-up actions")

@tool("approve_transaction")
def approve_transaction(transaction_id: str, notes: str) -> Dict[str, Any]:
    """Approve a transaction with notes."""
    return {
        "status": "approved",
        "transaction_id": transaction_id,
        "timestamp": datetime.now().isoformat(),
        "notes": notes
    }

@tool("flag_transaction")
def flag_transaction(transaction_id: str, risk_level: str, reason: str) -> Dict[str, Any]:
    """Flag a transaction for review."""
    return {
        "status": "flagged",
        "transaction_id": transaction_id,
        "risk_level": risk_level,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }

@tool("deny_transaction")
def deny_transaction(transaction_id: str, reason: str) -> Dict[str, Any]:
    """Deny a transaction with reason."""
    return {
        "status": "denied",
        "transaction_id": transaction_id,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }

@tool("notify_compliance")
def notify_compliance(transaction_id: str, details: str) -> Dict[str, Any]:
    """Send notification to compliance team."""
    return {
        "status": "notified",
        "transaction_id": transaction_id,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }

def evaluate_transaction(transaction: Transaction) -> RuleResult:
    """Evaluate transaction using fixed rules."""
    amount = float(transaction.amount)
    is_foreign = "FOREIGN_" in transaction.destination_account
    
    # Base evaluation by amount
    if amount < 10000:
        result = RuleResult(
            action=RuleAction.APPROVE,
            reason=f"Amount ${amount:,.2f} USD is within standard limit",
            risk_level="low",
            required_actions=[]
        )
    elif amount <= 50000:
        result = RuleResult(
            action=RuleAction.FLAG,
            reason=f"Amount ${amount:,.2f} USD requires review",
            risk_level="medium",
            required_actions=["notify_compliance"]
        )
    else:
        result = RuleResult(
            action=RuleAction.DENY,
            reason=f"Amount ${amount:,.2f} USD exceeds maximum limit",
            risk_level="high",
            required_actions=["notify_compliance"]
        )
    
    # Adjust for foreign destination
    if is_foreign:
        # Increase risk level
        if result.risk_level == "low":
            result.risk_level = "medium"
        elif result.risk_level == "medium":
            result.risk_level = "high"
        
        # Update reason and actions
        result.reason += " and foreign destination detected"
        if "notify_compliance" not in result.required_actions:
            result.required_actions.append("notify_compliance")
    
    return result

def create_rule_chain():
    """Create the rule evaluation chain."""
    # Evaluation chain
    eval_chain = RunnableLambda(evaluate_transaction)

    # Action execution chain
    def execute_actions(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs["result"]
        transaction = inputs["transaction"]
        
        responses = []
        
        # Execute primary action using tool.invoke()
        if result.action == RuleAction.APPROVE:
            responses.append(
                approve_transaction.invoke({
                    "transaction_id": transaction.id,
                    "notes": result.reason
                })
            )
        elif result.action == RuleAction.DENY:
            responses.append(
                deny_transaction.invoke({
                    "transaction_id": transaction.id,
                    "reason": result.reason
                })
            )
        elif result.action == RuleAction.FLAG:
            responses.append(
                flag_transaction.invoke({
                    "transaction_id": transaction.id,
                    "risk_level": result.risk_level,
                    "reason": result.reason
                })
            )
        
        # Execute required actions
        if "notify_compliance" in result.required_actions:
            responses.append(
                notify_compliance.invoke({
                    "transaction_id": transaction.id,
                    "details": result.reason
                })
            )
        
        return {
            "transaction_id": transaction.id,
            "primary_action": result.action,
            "reason": result.reason,
            "risk_level": result.risk_level,
            "action_results": responses
        }

    # Combine evaluation and execution
    chain = (
        RunnablePassthrough()
        | {
            "result": eval_chain,
            "transaction": RunnablePassthrough()
        }
        | RunnableLambda(execute_actions)
    )

    return chain

def demonstrate_rules():
    """Demonstrate the rule engine."""
    print("\nTransaction Rule Engine Demo")
    print("=========================\n")
    
    # Create test transactions
    transactions = [
        Transaction(
            id="TRX001",
            type=TransactionType.TRANSFER,
            amount=Decimal("5000.00"),
            currency="USD",
            source_account="ACCT123",
            destination_account="ACCT456",
            timestamp=datetime.now(),
            description="Regular transfer"
        ),
        Transaction(
            id="TRX002",
            type=TransactionType.TRANSFER,
            amount=Decimal("75000.00"),
            currency="USD",
            source_account="ACCT789",
            destination_account="FOREIGN_ACCT",
            timestamp=datetime.now(),
            description="International wire transfer"
        ),
        Transaction(
            id="TRX003",
            type=TransactionType.PAYMENT,
            amount=Decimal("12500.00"),
            currency="USD",
            source_account="ACCT999",
            destination_account="MERCHANT888",
            timestamp=datetime.now(),
            description="Vendor payment"
        )
    ]
    
    # Initialize rule chain
    print("Initializing rule chain...")
    chain = create_rule_chain()
    print("Rule chain initialized\n")
    
    # Process transactions
    for tx in transactions:
        print(f"\nProcessing Transaction: {tx.id}")
        print(f"Type: {tx.type.value}")
        print(f"Amount: ${float(tx.amount):,.2f} {tx.currency}")
        print(f"From: {tx.source_account}")
        print(f"To: {tx.destination_account}")
        print("-" * 40)
        
        try:
            result = chain.invoke(tx)
            
            print("\nRule Evaluation Result:")
            print(f"Action: {result['primary_action'].upper()}")
            print(f"Reason: {result['reason']}")
            print(f"Risk Level: {result['risk_level'].upper()}")
            
            print("\nExecuted Actions:")
            for action in result["action_results"]:
                details = action.get("reason") or action.get("details") or action.get("notes", "No details")
                print(f"- {action['status'].title()}: {details}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing transaction: {str(e)}")
            print("-" * 40)

if __name__ == "__main__":
    demonstrate_rules()
#!/usr/bin/env python3
"""
LangChain Async Agent Callbacks (095) (LangChain v3)

This example demonstrates real-time financial transaction processing using three key concepts:
1. Agents: Manage transaction workflows
2. Async: Handle concurrent transaction processing
3. Callbacks: Trigger actions upon transaction completion

It provides efficient and scalable transaction management for banking applications.
"""

import asyncio
import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define transaction status
class TransactionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Define transaction model
class Transaction(BaseModel):
    transaction_id: str = Field(description="Unique transaction identifier")
    amount: float = Field(description="Transaction amount")
    status: TransactionStatus = Field(description="Current transaction status")

# Define transaction agent
class TransactionAgent:
    def __init__(self, agent_id: str, callback_manager: CallbackManager, llm: AzureChatOpenAI):
        self.agent_id = agent_id
        self.callback_manager = callback_manager
        self.llm = llm

    async def process_transaction(self, transaction: Transaction) -> None:
        try:
            transaction.status = TransactionStatus.PROCESSING
            # Simulate processing with AI decision making
            messages = [
                SystemMessage(content="You are a financial expert."),
                HumanMessage(content=f"Process transaction {transaction.transaction_id} with amount {transaction.amount}.")
            ]
            decision = await self.llm.ainvoke(messages)
            print(f"AI Decision for {transaction.transaction_id}: {decision.content}")
            transaction.status = TransactionStatus.COMPLETED
            self.callback_manager.trigger_callback("transaction_completed", transaction)
        except Exception as e:
            transaction.status = TransactionStatus.FAILED
            print(f"Error processing transaction {transaction.transaction_id}: {str(e)}")

# Define callback manager
class TransactionCallbackManager(CallbackManager):
    def __init__(self, handlers: List[BaseCallbackHandler]):
        super().__init__(handlers=handlers)
        self.callbacks: Dict[str, List[Any]] = {}

    def register_callback(self, event: str, callback: Any) -> None:
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def trigger_callback(self, event: str, data: Any) -> None:
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(data)

# Define async transaction processor
class AsyncTransactionProcessor:
    def __init__(self, num_agents: int = 3):
        self.callback_manager = TransactionCallbackManager(handlers=[])
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.2
        )
        self.agents: List[TransactionAgent] = [
            TransactionAgent(agent_id=f"agent_{i}", callback_manager=self.callback_manager, llm=self.llm)
            for i in range(num_agents)
        ]

    async def process_transactions(self, transactions: List[Transaction]) -> None:
        tasks = [agent.process_transaction(transaction) for agent, transaction in zip(self.agents, transactions)]
        await asyncio.gather(*tasks)

    def register_completion_callback(self, callback: Any) -> None:
        self.callback_manager.register_callback("transaction_completed", callback)

async def demonstrate_async_processing():
    print("\nAsync Agent Callbacks Demo")
    print("==========================\n")
    processor = AsyncTransactionProcessor(num_agents=3)

    # Register a callback to handle completed transactions
    processor.register_completion_callback(lambda t: print(f"Transaction {t.transaction_id} completed with status {t.status}"))

    transactions = [
        Transaction(transaction_id="txn_001", amount=100.0, status=TransactionStatus.PENDING),
        Transaction(transaction_id="txn_002", amount=200.0, status=TransactionStatus.PENDING),
        Transaction(transaction_id="txn_003", amount=300.0, status=TransactionStatus.PENDING)
    ]

    print("Processing transactions...")
    await processor.process_transactions(transactions)
    print("\nAsync processing demonstration completed")

if __name__ == "__main__":
    asyncio.run(demonstrate_async_processing())
#!/usr/bin/env python3
"""
LangChain Financial Advisor (097) (LangChain v3)

This example demonstrates a financial advisory system using three key concepts:
1. Agents: Manage advisory workflows
2. Evaluation: Assess financial strategies
3. Retrieval: Access relevant financial data

It provides personalized financial advice and strategy evaluation for banking applications.
"""

import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define financial strategy model
class FinancialStrategy(BaseModel):
    strategy_id: str = Field(description="Unique strategy identifier")
    description: str = Field(description="Strategy description")
    risk_level: str = Field(description="Risk level of the strategy")
    expected_return: float = Field(description="Expected return percentage")

# Define financial advisor agent
class FinancialAdvisorAgent:
    def __init__(self, agent_id: str, llm: AzureChatOpenAI, eval_chain: QAEvalChain, vectorstore: FAISS, embeddings: AzureOpenAIEmbeddings):
        self.agent_id = agent_id
        self.llm = llm
        self.eval_chain = eval_chain
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    async def provide_advice(self, strategy: FinancialStrategy) -> str:
        # Retrieve relevant financial data
        query_embedding = self.embeddings.embed_query(strategy.description)
        docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(query_embedding, k=2)
        
        # Evaluate strategy
        examples = [{"query": strategy.description, "answer": "Expected financial outcome"}]
        predictions = [{"result": "Predicted financial outcome"}]
        evaluation_results = self.eval_chain.evaluate(examples, predictions)
        
        # Generate advice
        messages = [
            SystemMessage(content="You are a financial advisor."),
            HumanMessage(content=f"Evaluate the following strategy: {strategy.description}"),
            SystemMessage(content=f"Financial Data: {docs_and_scores}"),
            SystemMessage(content=f"Evaluation: {evaluation_results}")
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content

def create_document_store(embeddings: AzureOpenAIEmbeddings) -> FAISS:
    """
    Create and populate a vector store with sample financial documents.
    
    Returns:
        FAISS: A vector store containing the sample documents
    """
    # Create sample documents
    documents = [
        Document(
            page_content="Investing in tech stocks can yield high returns but comes with significant risks.",
            metadata={"category": "investment", "risk": "high"}
        ),
        Document(
            page_content="Diversifying with bonds provides stable returns and lowers overall portfolio risk.",
            metadata={"category": "investment", "risk": "low"}
        )
    ]
    
    # Extract texts from documents and compute their embeddings
    texts = [doc.page_content for doc in documents]
    embeddings_vectors = embeddings.embed_documents(texts)
    
    # Create FAISS index
    index = FAISS.from_embeddings(
        embedding=embeddings,
        text_embeddings=list(zip(texts, embeddings_vectors)),
        metadatas=[doc.metadata for doc in documents]
    )
    return index

async def demonstrate_financial_advisor():
    print("\nFinancial Advisor Demo")
    print("======================\n")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.5
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        model=os.getenv("AZURE_DEPLOYMENT")
    )
    eval_chain = QAEvalChain.from_llm(llm=llm)
    vectorstore = create_document_store(embeddings)

    advisor = FinancialAdvisorAgent(agent_id="advisor_1", llm=llm, eval_chain=eval_chain, vectorstore=vectorstore, embeddings=embeddings)

    strategies = [
        FinancialStrategy(strategy_id="strat_001", description="Invest in tech stocks", risk_level="High", expected_return=15.0),
        FinancialStrategy(strategy_id="strat_002", description="Diversify with bonds", risk_level="Low", expected_return=5.0)
    ]

    for strategy in strategies:
        print(f"Strategy: {strategy.description}")
        advice = await advisor.provide_advice(strategy)
        print(f"Advice: {advice}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_financial_advisor())
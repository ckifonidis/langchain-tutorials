#!/usr/bin/env python3
"""
LangChain Risk Assessment (098) (LangChain v3)

This example demonstrates a risk assessment system using three key concepts:
1. Evaluation: Assess risk levels
2. Retrieval: Access relevant risk data
3. Structured Output: Provide clear risk reports

It provides comprehensive risk analysis and reporting for financial applications.
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

# Define risk assessment model
class RiskAssessment(BaseModel):
    assessment_id: str = Field(description="Unique assessment identifier")
    description: str = Field(description="Assessment description")
    risk_level: str = Field(description="Risk level of the assessment")
    impact_score: float = Field(description="Impact score of the risk")

# Define risk assessment agent
class RiskAssessmentAgent:
    def __init__(self, agent_id: str, llm: AzureChatOpenAI, eval_chain: QAEvalChain, vectorstore: FAISS, embeddings: AzureOpenAIEmbeddings):
        self.agent_id = agent_id
        self.llm = llm
        self.eval_chain = eval_chain
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    async def assess_risk(self, assessment: RiskAssessment) -> Dict[str, str]:
        # Retrieve relevant risk data
        query_embedding = self.embeddings.embed_query(assessment.description)
        docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(query_embedding, k=2)
        
        # Evaluate risk
        examples = [{"query": assessment.description, "answer": "Expected risk outcome"}]
        predictions = [{"result": "Predicted risk outcome"}]
        evaluation_results = self.eval_chain.evaluate(examples, predictions)
        
        # Generate risk report
        messages = [
            SystemMessage(content="You are a risk assessment agent."),
            HumanMessage(content=f"Assess the following risk: {assessment.description}"),
            SystemMessage(content=f"Risk Data: {docs_and_scores}"),
            SystemMessage(content=f"Evaluation: {evaluation_results}")
        ]
        
        response = await self.llm.ainvoke(messages)
        return {
            "content": response.content,
            "risk_level": assessment.risk_level,
            "impact_score": str(assessment.impact_score)
        }

def create_document_store(embeddings: AzureOpenAIEmbeddings) -> FAISS:
    """
    Create and populate a vector store with sample risk documents.
    
    Returns:
        FAISS: A vector store containing the sample documents
    """
    # Create sample documents
    documents = [
        Document(
            page_content="Investing in volatile markets can lead to high risks and potential losses.",
            metadata={"category": "investment", "risk": "high"}
        ),
        Document(
            page_content="Stable markets provide lower risks and more predictable returns.",
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

async def demonstrate_risk_assessment():
    print("\nRisk Assessment Demo")
    print("====================\n")
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

    agent = RiskAssessmentAgent(agent_id="agent_1", llm=llm, eval_chain=eval_chain, vectorstore=vectorstore, embeddings=embeddings)

    assessments = [
        RiskAssessment(assessment_id="assess_001", description="Investing in volatile markets", risk_level="High", impact_score=8.5),
        RiskAssessment(assessment_id="assess_002", description="Investing in stable markets", risk_level="Low", impact_score=3.0)
    ]

    for assessment in assessments:
        print(f"Assessment: {assessment.description}")
        report = await agent.assess_risk(assessment)
        print(f"Risk Report: {report['content']}\nRisk Level: {report['risk_level']}\nImpact Score: {report['impact_score']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_risk_assessment())
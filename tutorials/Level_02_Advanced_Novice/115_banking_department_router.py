#!/usr/bin/env python3
"""
LangChain Banking Department Router (115) (LangChain v3)

This example demonstrates routing internal banking queries to appropriate departments using:
1. Chat Models: Smart query understanding
2. Output Parsers: Structured routing responses
3. Vector Store: Department matching with FAISS CPU

It helps efficiently route queries to the right banking departments.
"""

import os
import json
from enum import Enum
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Department(str, Enum):
    """Banking departments."""
    DEVELOPMENT = "development"
    DATA_SCIENCE = "data_science"
    LEGAL = "legal"
    HR = "hr"
    MARKETING = "marketing"
    RISK = "risk"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"

class Priority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Query(BaseModel):
    """Internal query structure."""
    id: str = Field(description="Query identifier")
    content: str = Field(description="Query content")
    sender: str = Field(description="Sender name")
    priority: Priority = Field(description="Query priority")
    metadata: Dict = Field(default_factory=dict)

class Response(BaseModel):
    """Routing response structure."""
    id: str = Field(description="Response identifier")
    query_id: str = Field(description="Original query ID")
    department: Department = Field(description="Assigned department")
    priority: Priority = Field(description="Assigned priority")
    analysis: str = Field(description="Routing explanation")
    assigned_to: List[str] = Field(description="Team members")
    next_steps: List[str] = Field(description="Required actions")

DEPARTMENTS = {
    Department.DEVELOPMENT: "Technical development team handling APIs, systems, and integrations",
    Department.DATA_SCIENCE: "Data analysis team managing ML models and analytics",
    Department.LEGAL: "Legal team handling compliance and contracts",
    Department.HR: "Human resources team managing personnel and training",
    Department.MARKETING: "Marketing team handling campaigns and communications",
    Department.RISK: "Risk management team handling assessments and security",
    Department.COMPLIANCE: "Compliance team ensuring regulatory adherence",
    Department.OPERATIONS: "Operations team managing daily processes and support"
}

class Router:
    def __init__(self):
        """Initialize the router."""
        print("Initializing Banking Department Router...")
        
        # Setup LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        print("Chat model initialized")
        
        # Setup embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-3",
            model="text-embedding-3-small",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        print("Embeddings model initialized")
        
        # Setup vector store
        descriptions = list(DEPARTMENTS.values())
        departments = list(DEPARTMENTS.keys())
        self.vectorstore = FAISS.from_texts(
            texts=descriptions,
            embedding=self.embeddings,
            metadatas=[{"dept": d.value} for d in departments]
        )
        print("Vector store ready")
        
        # Setup prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a banking department router that responds only in JSON format."),
            ("human", """Analyze this request: {query}

Return response as JSON:
{
  "department": "department_name",
  "priority": "priority_level",
  "assigned_to": ["name1"],
  "analysis": "explanation",
  "next_steps": ["step1"]
}""")
        ])
        print("Router ready")

    async def route(self, query: Query) -> Response:
        """Route query to appropriate department."""
        # Find closest department
        results = self.vectorstore.similarity_search_with_score(
            query.content, k=1
        )
        dept = results[0][0].metadata["dept"]
        print(f"Matched department: {dept}")
        
        # Get routing recommendation
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are a banking department router. 
Respond ONLY with a JSON object containing routing details."""),
            HumanMessage(content=f"""Query ID: {query.id}
From: {query.sender}
Content: {query.content}
Priority: {query.priority.value}
Department: {dept}

Respond with ONLY a JSON object like this:
{{
  "department": "{dept}",
  "priority": "low/medium/high/urgent",
  "assigned_to": ["name"],
  "analysis": "explanation",
  "next_steps": ["step"]
}}""")
        ])
        
        print(f"Raw response:\n{response.content}")
        
        # Parse JSON response
        json_str = response.content[
            response.content.find('{'):
            response.content.rfind('}')+1
        ]
        data = json.loads(json_str)
        
        # Create response
        return Response(
            id=f"RES-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            query_id=query.id,
            department=Department(data["department"]),
            priority=Priority(data["priority"]),
            analysis=data["analysis"],
            assigned_to=data["assigned_to"],
            next_steps=data["next_steps"]
        )

async def main():
    """Demonstrate the router."""
    router = Router()
    
    queries = [
        Query(
            id="DEV-001",
            content="Need help implementing OAuth2 in our payment API",
            sender="Sarah Chen",
            priority=Priority.HIGH
        ),
        Query(
            id="DS-001", 
            content="Need a fraud detection model for transactions",
            sender="Alex Patel",
            priority=Priority.URGENT
        ),
        Query(
            id="HR-001",
            content="Planning compliance training for new regulations",
            sender="Jordan Lee",
            priority=Priority.MEDIUM
        )
    ]
    
    print("\nProcessing Queries:")
    print("==================")
    
    for query in queries:
        print(f"\nQuery: {query.id}")
        print(f"From: {query.sender}")
        print(f"Content: {query.content}")
        
        response = await router.route(query)
        
        print(f"\nRouted to: {response.department.value}")
        print(f"Priority: {response.priority.value}")
        print(f"Assigned: {', '.join(response.assigned_to)}")
        print(f"Analysis: {response.analysis}")
        print("\nNext Steps:")
        for step in response.next_steps:
            print(f"- {step}")
        
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

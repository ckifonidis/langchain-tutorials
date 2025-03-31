#!/usr/bin/env python3
"""
LangChain Semantic Banking Assistant (LangChain v3)

This example demonstrates a semantic banking assistant using three key concepts:
1. chat_models: Natural language interaction with customers
2. embedding_models: Semantic understanding of queries
3. vector_stores: Efficient knowledge retrieval

It provides intelligent banking assistance with semantic search capabilities.
"""

import os
import base64
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# Load environment variables
load_dotenv(".env")

class QueryCategory(str, Enum):
    """Banking query categories."""
    ACCOUNT = "account"
    PRODUCTS = "products"
    LOANS = "loans"
    INVESTMENTS = "investments"
    GENERAL = "general"

class BankingQuery(BaseModel):
    """Banking query with metadata."""
    text: str = Field(description="Query text")
    category: QueryCategory = Field(description="Query category")
    timestamp: str = Field(description="Query timestamp")
    context: Dict = Field(description="Additional context")

class AssistantResponse(BaseModel):
    """Assistant response with metadata."""
    text: str = Field(description="Response text")
    sources: List[str] = Field(description="Information sources")
    confidence: float = Field(description="Response confidence")
    suggestions: List[str] = Field(description="Follow-up suggestions")

class SemanticBankingAssistant:
    """Banking assistant with semantic search capabilities."""
    
    def __init__(self):
        """Initialize the banking assistant."""
        # Initialize chat model
        self.chat_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.3
        )
        
        # Initialize embeddings model with specific deployment
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_DEPLOYMENT", "text-embedding-3-small-3"),
            model=os.getenv("AZURE_MODEL_NAME", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Initialize knowledge base with banking information
        self.knowledge_base = self._create_knowledge_base()
        
        # System prompts for different query categories
        self.prompts = {
            QueryCategory.ACCOUNT: """You are a helpful banking assistant. Provide accurate information 
            about account services, balances, and transactions. Always maintain privacy and security. 
            Never share specific account details.""",
            
            QueryCategory.PRODUCTS: """You are a helpful banking assistant. Explain banking products, 
            their features, benefits, and requirements. Provide factual information to help customers 
            make informed decisions.""",
            
            QueryCategory.LOANS: """You are a helpful banking assistant specializing in loans. 
            Explain loan types, terms, requirements, and processes. Provide general guidance without 
            making specific promises or approvals.""",
            
            QueryCategory.INVESTMENTS: """You are a helpful banking assistant specializing in investments. 
            Explain investment options, risks, and strategies. Always include risk disclaimers. Never 
            provide specific investment advice.""",
            
            QueryCategory.GENERAL: """You are a helpful banking assistant. Provide general banking 
            information and guidance. Maintain professionalism and accuracy in all responses."""
        }
    
    def _create_knowledge_base(self) -> FAISS:
        """Create and populate the vector store."""
        # Sample banking knowledge
        documents = [
            Document(
                page_content="Checking accounts offer daily banking services with debit cards, checks, and online banking.",
                metadata={"category": "account", "type": "product_info"}
            ),
            Document(
                page_content="Savings accounts earn interest on deposits. Interest rates vary based on account type and balance.",
                metadata={"category": "account", "type": "product_info"}
            ),
            Document(
                page_content="Personal loans can be used for debt consolidation, home improvement, or major purchases.",
                metadata={"category": "loans", "type": "product_info"}
            ),
            Document(
                page_content="Investment accounts include mutual funds, stocks, bonds, and retirement planning options.",
                metadata={"category": "investments", "type": "product_info"}
            ),
            Document(
                page_content="Online banking features include bill pay, transfers, mobile check deposit, and account alerts.",
                metadata={"category": "general", "type": "service_info"}
            )
        ]
        
        # Create vector store
        return FAISS.from_documents(documents, self.embeddings)
    
    def _categorize_query(self, query: str) -> QueryCategory:
        """Determine query category based on content."""
        # Create category embeddings for comparison
        category_docs = {
            QueryCategory.ACCOUNT: "account balance transactions banking statements deposits withdrawals",
            QueryCategory.PRODUCTS: "banking products services features benefits cards accounts",
            QueryCategory.LOANS: "loans borrowing mortgage credit financing rates terms",
            QueryCategory.INVESTMENTS: "investments stocks bonds funds retirement planning wealth",
            QueryCategory.GENERAL: "general banking information help support assistance guidance"
        }
        
        # Get query embedding
        query_emb = self.embeddings.embed_query(query)
        
        # Get category embeddings and compare
        scores = {
            cat: np.dot(query_emb, self.embeddings.embed_query(text))
            for cat, text in category_docs.items()
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def get_response(self, query: BankingQuery) -> AssistantResponse:
        """Generate response using semantic search and chat model."""
        try:
            # Get relevant documents
            docs = self.knowledge_base.similarity_search(
                query.text,
                k=2,
                fetch_k=4
            )
            
            # Prepare context
            context = "\n".join(doc.page_content for doc in docs)
            
            # Get category-specific prompt
            system_prompt = self.prompts.get(
                query.category,
                self.prompts[QueryCategory.GENERAL]
            )
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Context: {context}
                
                Query: {query.text}
                
                Provide a helpful response with relevant information. Include 2-3 follow-up suggestions.
                """)
            ]
            
            response = await self.chat_model.ainvoke(messages)
            
            # Extract suggestions (assuming they're on new lines starting with -)
            suggestions = [
                line.strip("- ") for line in response.content.split("\n")
                if line.strip().startswith("-")
            ] or ["Check account balance", "Review transaction history", "Contact support"]
            
            return AssistantResponse(
                text=response.content,
                sources=[doc.metadata.get("type", "general") for doc in docs],
                confidence=0.85,  # Could be calculated based on relevance scores
                suggestions=suggestions[:3]  # Limit to 3 suggestions
            )
            
        except Exception as e:
            return AssistantResponse(
                text=f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support.",
                sources=["error_handler"],
                confidence=0.0,
                suggestions=[
                    "Try rephrasing your question",
                    "Contact customer support",
                    "Visit our help center"
                ]
            )

async def demonstrate_system():
    """Demonstrate the semantic banking assistant."""
    print("\nSemantic Banking Assistant Demo")
    print("===============================\n")
    
    # Create assistant
    assistant = SemanticBankingAssistant()
    
    # Test queries
    queries = [
        BankingQuery(
            text="What types of checking accounts do you offer?",
            category=QueryCategory.PRODUCTS,
            timestamp=datetime.now().isoformat(),
            context={"user_type": "new_customer"}
        ),
        BankingQuery(
            text="How do I start investing for retirement?",
            category=QueryCategory.INVESTMENTS,
            timestamp=datetime.now().isoformat(),
            context={"user_type": "existing_customer"}
        )
    ]
    
    try:
        # Process queries
        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query.text}")
            print(f"Category: {query.category}")
            print("-" * 40)
            
            response = await assistant.get_response(query)
            
            print("\nResponse:")
            print(response.text)
            print("\nSources:", ", ".join(response.sources))
            print(f"Confidence: {response.confidence:.2f}")
            print("\nSuggested Follow-ups:")
            for suggestion in response.suggestions:
                print(f"- {suggestion}")
            print("-" * 40 + "\n")
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_system())
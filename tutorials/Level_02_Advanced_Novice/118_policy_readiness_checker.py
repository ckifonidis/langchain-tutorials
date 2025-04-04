#!/usr/bin/env python3
"""
LangChain Policy Readiness Checker (117) (LangChain v3)

This example demonstrates a policy readiness assessment system using:
1. Chat Models: Policy analysis
2. Document Processing: Content extraction  
3. Chain of Thought: Validation process

It helps teams evaluate and improve their policy documents.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Department(str, Enum):
    """Organization departments."""
    SECURITY = "security"
    LEGAL = "legal"
    HR = "human_resources"
    OPERATIONS = "operations"
    FINANCE = "finance"

class PolicyType(str, Enum):
    """Types of policies."""
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"

class PolicyDocument(BaseModel):
    """Policy document details."""
    title: str = Field(description="Policy title")
    department: Department = Field(description="Owner department")
    type: PolicyType = Field(description="Policy type")
    content: str = Field(description="Policy content")
    metadata: Dict = Field(default_factory=dict)

class ReadinessChecker:
    """Policy readiness assessment system."""

    def __init__(self):
        """Initialize checker."""
        logger.info("Initializing readiness checker...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Create analysis chain
        analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a policy analyst. Given a policy document:
1. Assess its readiness and structure
2. Identify gaps and weaknesses
3. Suggest concrete improvements
4. Estimate completion percentage

Your response should be a clear report with:
- Overall Assessment 
- Key Findings
- Required Updates
- Readiness Score (0-100%)"""),
            ("human", """Please analyze this policy:

Title: {title}
Department: {department}
Type: {type}

Content:
{content}

Provide a detailed analysis.""")
        ])
        
        self.chain = (
            {"title": RunnablePassthrough(), 
             "department": RunnablePassthrough(),
             "type": RunnablePassthrough(),
             "content": RunnablePassthrough()} 
            | analyzer_prompt 
            | self.llm 
            | StrOutputParser()
        )
        logger.info("Analysis chain ready")

    async def check_policy(self, policy: PolicyDocument) -> str:
        """Analyze policy readiness."""
        logger.info(f"Analyzing policy: {policy.title}")
        
        try:
            # Run analysis
            result = await self.chain.ainvoke({
                "title": policy.title,
                "department": policy.department.value,
                "type": policy.type.value,
                "content": policy.content.strip()
            })
            logger.info("Analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting policy readiness demo...")
    
    try:
        # Create checker
        checker = ReadinessChecker()
        
        # Example policy
        policy = PolicyDocument(
            title="Data Access Control Policy",
            department=Department.SECURITY,
            type=PolicyType.SECURITY,
            content="""1. Purpose
This policy establishes data access control requirements.

2. Access Management
- Role-based access control required
- Regular access reviews
- Account deactivation process

3. Authentication
- Multi-factor authentication required
- Password complexity rules
- Session timeout controls

4. Data Classification
- Confidential data handling
- Data labeling standards
- Storage requirements

5. Compliance
- Access logging enabled
- Regular audits required
- Incident reporting"""
        )
        
        print("\nChecking Policy Readiness")
        print("========================")
        print(f"Title: {policy.title}")
        print(f"Department: {policy.department.value}")
        print(f"Type: {policy.type.value}\n")
        
        try:
            # Get analysis
            result = await checker.check_policy(policy)
            print("\nReadiness Analysis:")
            print("------------------")
            print(result)
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
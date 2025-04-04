#!/usr/bin/env python3
"""
Simple Policy Reviewer (117) (LangChain v3)

This example demonstrates policy review using:
1. Clear prompts
2. Simple chains
3. Clean output

It helps teams review and improve their policy documents.
"""

import os
import logging
from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
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
    FINANCE = "finance"

class PolicyType(str, Enum):
    """Types of policies."""
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"

class PolicyDocument(BaseModel):
    """Policy document details."""
    title: str = Field(description="Policy title")
    department: Department = Field(description="Department")
    type: PolicyType = Field(description="Policy type")
    content: str = Field(description="Content")
    metadata: Dict = Field(default_factory=dict)

class PolicyReviewer:
    """Simple policy review system."""

    def __init__(self):
        """Initialize reviewer."""
        logger.info("Starting policy reviewer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a policy reviewer. Review policies and provide clear feedback.
Include in your review:

1. Structure Review
- Is the policy well organized?
- Are sections clearly defined?
- Is formatting consistent?

2. Content Analysis
- Is content clear and specific?
- Are requirements actionable?
- Are responsibilities defined?

3. Required Updates
- What sections need work?
- What details are missing?
- What should be added?

4. Overall Rating
- Score policy from 0-100
- Explain the score
- List top priorities"""),
            ("human", """Review this policy:

Title: {title}
Department: {department}
Type: {type}

Content:
{content}

Provide a complete review.""")
        ])
        logger.info("Prompt template ready")

    async def review_policy(self, policy: PolicyDocument) -> str:
        """Review a policy document."""
        logger.info(f"Reviewing: {policy.title}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                title=policy.title,
                department=policy.department.value,
                type=policy.type.value,
                content=policy.content.strip()
            )
            logger.info("Request formatted")
            
            # Get review
            response = await self.llm.ainvoke(messages)
            logger.info("Review complete")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Review failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting demo...")
    
    try:
        # Create reviewer
        reviewer = PolicyReviewer()
        
        # Example policy
        policy = PolicyDocument(
            title="Remote Access Security Policy",
            department=Department.SECURITY,
            type=PolicyType.SECURITY,
            content="""1. Overview
This policy defines security requirements for remote access.

2. Access Requirements
- VPN required for remote access
- Multi-factor authentication mandatory
- Encrypted connections only

3. Device Security
- Up-to-date antivirus required
- System patches must be current
- Local firewall enabled

4. User Responsibilities
- Keep credentials secure
- Report suspicious activity
- Lock screen when away

5. Monitoring
- Access logs reviewed
- Usage monitored
- Violations reported"""
        )
        
        print("\nReviewing Policy")
        print("===============")
        print(f"Title: {policy.title}")
        print(f"Department: {policy.department.value}")
        print(f"Type: {policy.type.value}\n")
        
        try:
            # Get review
            review = await reviewer.review_policy(policy)
            print("\nPolicy Review:")
            print("-------------")
            print(review)
            
        except Exception as e:
            print(f"\nReview failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
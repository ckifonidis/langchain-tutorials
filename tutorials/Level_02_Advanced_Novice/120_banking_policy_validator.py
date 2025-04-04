#!/usr/bin/env python3
"""
Banking Policy Validator (120) (LangChain v3)

This example demonstrates a banking policy validator using:
1. Chat models
2. Streaming output
3. Clear feedback

It helps bank teams validate their policy documents.
"""

import os
import logging
from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Department(str, Enum):
    """Bank departments."""
    SECURITY = "security"
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    HR = "human_resources"

class PolicyType(str, Enum):
    """Types of policies."""
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"

class PolicyDocument(BaseModel):
    """Policy document details."""
    title: str = Field(description="Policy title")
    department: Department = Field(description="Department")
    type: PolicyType = Field(description="Policy type")
    content: str = Field(description="Policy content")
    metadata: Dict = Field(default_factory=dict)

class PolicyValidator:
    """Banking policy validation system."""

    def __init__(self):
        """Initialize validator."""
        logger.debug("Starting validator...")
        
        # Setup chat model with streaming
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        logger.debug("Chat model ready")
        
        # Setup validation prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking policy validator. Review policies and provide structured feedback.
Include in your analysis:

1. Policy Overview
- Document structure
- Content clarity
- Key sections

2. Compliance Check
- Required elements
- Missing components
- Regulatory alignment

3. Detailed Analysis
- Section by section review
- Specific issues found
- Required updates

4. Recommendations
- Critical updates needed
- Suggested improvements
- Priority order

5. Final Assessment
- Overall score (0-100)
- Pass/Fail status
- Next review date

Format with clear sections and bullet points."""),
            ("human", """Review this policy:

Title: {title}
Department: {department}
Type: {type}

Content:
{content}

Provide a complete validation report.""")
        ])
        logger.debug("Prompt template ready")

    async def validate(self, policy: PolicyDocument) -> None:
        """Validate a policy document."""
        logger.debug(f"Processing policy: {policy.title}")
        
        try:
            # Format request
            messages = self.prompt.format_messages(
                title=policy.title,
                department=policy.department.value,
                type=policy.type.value,
                content=policy.content
            )
            logger.debug("Request formatted")
            
            # Stream validation report
            print("\nValidation Report")
            print("================\n")
            await self.llm.ainvoke(messages)
            print("\n")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting demo")
    
    try:
        # Create validator
        validator = PolicyValidator()
        
        # Example policy
        policy = PolicyDocument(
            title="API Security Policy",
            department=Department.SECURITY,
            type=PolicyType.SECURITY,
            content="""1. Purpose and Scope
This policy establishes security requirements for all APIs.

2. Authentication Requirements
- OAuth 2.0 authentication required
- JWT tokens must be used
- Refresh tokens handled securely
- Regular key rotation mandatory

3. Authorization Controls
- Role-based access control (RBAC)
- Least privilege principle enforced
- Regular access reviews required
- Clear role definitions needed

4. Data Protection
- TLS 1.3 encryption required
- Data classification enforced
- PII handling procedures
- Encryption at rest mandatory

5. Security Monitoring
- Real-time threat detection
- Comprehensive audit logging
- Incident response procedures
- Regular security testing

6. Compliance
- Regulatory requirements met
- Regular compliance reviews
- Documentation maintained
- Training requirements"""
        )
        
        print("\nValidating Security Policy")
        print("========================")
        print(f"Title: {policy.title}")
        print(f"Department: {policy.department.value}")
        print(f"Type: {policy.type.value}\n")
        
        try:
            # Get validation
            await validator.validate(policy)
            
        except Exception as e:
            print(f"\nValidation failed: {str(e)}")
        
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
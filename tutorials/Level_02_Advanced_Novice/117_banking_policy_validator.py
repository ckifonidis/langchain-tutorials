#!/usr/bin/env python3
"""
LangChain Banking Policy Validator (117) (LangChain v3)

This example demonstrates a banking policy validation system using:
1. Chat Models: Policy validation
2. Function Calling: Structured output
3. Debug Logging: Error tracking

It helps bank departments ensure their policies meet regulatory requirements.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Department(str, Enum):
    """Bank departments."""
    SECURITY = "security"
    DATA_SCIENCE = "data_science"
    LEGAL = "legal"
    HR = "human_resources"
    MARKETING = "marketing"

class PolicyType(str, Enum):
    """Types of banking policies."""
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"

class ValidationIssue(BaseModel):
    """Policy validation issue."""
    section: str = Field(description="Section name")
    issue: str = Field(description="Issue description")
    severity: str = Field(description="low/medium/high/critical")
    recommendation: str = Field(description="Fix recommendation")
    reference: str = Field(description="Reference ID")

class ValidationResult(BaseModel):
    """Policy validation results."""
    valid: bool = Field(description="Overall validity")
    score: int = Field(description="Score from 0-100")
    issues: List[ValidationIssue] = Field(description="Found issues")
    summary: str = Field(description="Brief summary")
    next_steps: List[str] = Field(description="Required actions")
    review_date: str = Field(description="YYYY-MM-DD")

    @field_validator('score')
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v

class PolicyDocument(BaseModel):
    """Bank policy document."""
    title: str = Field(description="Policy title")
    department: Department = Field(description="Department")
    type: PolicyType = Field(description="Policy type")
    content: str = Field(description="Policy content")
    metadata: Dict = Field(default_factory=dict)

SYSTEM_PROMPT = """You are a banking policy validator. Analyze policies and return the validation results.
Follow these rules:
1. Return ONLY a JSON object
2. No text before/after JSON
3. Use this format:
{
  "valid": true/false,
  "score": number 0-100,
  "issues": [{"section":"name", "issue":"desc", "severity":"level", "recommendation":"fix", "reference":"id"}],
  "summary": "brief text",
  "next_steps": ["action 1", "action 2"],
  "review_date": "YYYY-MM-DD"
}"""

HUMAN_PROMPT = """Review this policy:

TITLE: {title}
TYPE: {type} policy
DEPT: {department}

CONTENT:
{content}

Return validation as JSON."""

class PolicyValidator:
    """Banking policy validator."""

    def __init__(self):
        """Initialize validator."""
        logger.debug("Starting validator...")
        
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.debug("Chat model ready")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT)
        ])
        logger.debug("Prompt template ready")

    async def validate(self, policy: PolicyDocument) -> ValidationResult:
        """Validate a policy document."""
        logger.debug(f"Processing policy: {policy.title}")
        
        try:
            # Prepare request
            messages = self.prompt.format_messages(
                title=policy.title,
                type=policy.type.value,
                department=policy.department.value,
                content=policy.content
            )
            logger.debug("Request formatted")
            
            # Get validation
            response = await self.llm.ainvoke(messages)
            logger.debug(f"Got response:\n{response.content}")
            
            # Parse result
            try:
                data = json.loads(response.content)
                result = ValidationResult(**data)
                logger.info(f"✓ Score: {result.score}/100")
                return result
                
            except json.JSONDecodeError as e:
                logger.error("❌ Invalid JSON response")
                logger.error(f"Error: {str(e)}")
                logger.error(f"Response:\n{response.content}")
                raise ValueError("Bad response format")
                
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting demo")
    
    try:
        # Create validator
        validator = PolicyValidator()
        
        # Test policy
        policy = PolicyDocument(
            title="API Security Policy",
            department=Department.SECURITY,
            type=PolicyType.SECURITY,
            content="""1. Authentication
- OAuth 2.0 & JWT required
- Regular key rotation
- Secure token storage

2. Authorization
- Role-based access control
- Least privilege principle
- Access reviews required

3. Data Protection
- TLS 1.3 required
- At-rest encryption
- Key management policy

4. Monitoring
- Real-time detection
- Audit logging
- Incident response"""
        )
        
        print("\nValidating Security Policy")
        print("========================")
        print(f"Title: {policy.title}")
        print(f"Department: {policy.department.value}")
        print(f"Type: {policy.type.value}\n")
        
        try:
            # Get validation
            result = await validator.validate(policy)
            
            # Show results
            print("\nValidation Results:")
            print(f"Valid: {'✓' if result.valid else '✗'}")
            print(f"Score: {result.score}/100")
            print(f"Summary: {result.summary}")
            
            if result.issues:
                print("\nIssues Found:")
                for i, issue in enumerate(result.issues, 1):
                    print(f"\n{i}. Section: {issue.section}")
                    print(f"   Severity: {issue.severity}")
                    print(f"   Issue: {issue.issue}")
                    print(f"   Fix: {issue.recommendation}")
                    print(f"   Ref: {issue.reference}")
            else:
                print("\nNo issues found")
            
            print("\nNext Steps:")
            for i, step in enumerate(result.next_steps, 1):
                print(f"{i}. {step}")
            
            print(f"\nNext Review: {result.review_date}")
            
        except Exception as e:
            print(f"\nValidation failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
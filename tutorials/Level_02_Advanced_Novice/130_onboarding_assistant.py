#!/usr/bin/env python3
"""
Employee Onboarding Assistant (130) (LangChain v3)

This example demonstrates HR onboarding automation using:
1. Message History: Contextual conversations
2. Few Shot Learning: Example-based responses
3. Chain Templates: Structured workflows

It helps HR teams automate employee onboarding in banking.
"""

import os
import logging
from enum import Enum
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Department(str, Enum):
    """Banking departments."""
    RETAIL = "retail_banking"
    CORPORATE = "corporate_banking"
    WEALTH = "wealth_management"
    RISK = "risk_management"
    COMPLIANCE = "compliance"
    IT = "information_technology"

class Role(str, Enum):
    """Employee roles."""
    ANALYST = "analyst"
    MANAGER = "manager"
    SPECIALIST = "specialist"
    ADVISOR = "advisor"
    DEVELOPER = "developer"
    ADMIN = "administrator"

class Employee(BaseModel):
    """Employee details."""
    employee_id: str = Field(description="Employee ID")
    name: str = Field(description="Full name")
    department: Department = Field(description="Department")
    role: Role = Field(description="Job role")
    start_date: str = Field(description="Start date")
    metadata: Dict = Field(default_factory=dict)

class OnboardingAssistant:
    """Employee onboarding automation system."""

    def __init__(self):
        """Initialize assistant."""
        logger.info("Starting onboarding assistant...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Example conversations
        self.examples = [
            [
                HumanMessage(content="What systems do I need access to?"),
                AIMessage(content="""Based on your role, you need:

1. Core Systems:
   - Email and calendar
   - Employee portal
   - Department tools

2. Banking Systems:
   - Customer database
   - Transaction system
   - Risk controls

Please submit access requests through the IT portal.""")
            ],
            [
                HumanMessage(content="Where do I find company policies?"),
                AIMessage(content="""You can find policies in these locations:

1. Employee Portal:
   - HR policies
   - Code of conduct
   - Department guidelines

2. Compliance Hub:
   - Banking regulations
   - Security policies
   - Data protection

Access both through your employee dashboard.""")
            ]
        ]
        
        # Setup conversation template
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", """You are an HR onboarding assistant.
Help new employees with questions and procedures.
Use clear, structured responses with steps and links.

Base your responses on these examples:
{examples}

Consider the employee's department and role."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Setup onboarding template
        self.plan_template = """Create a personalized onboarding plan.
Consider the department, role, and requirements.

Format tasks like this:

ONBOARDING PLAN
--------------
Employee: {name}
Department: {department}
Role: {role}

Required Tasks:
1. Task name
   Due: Timeline
   Details: Description

2. Task name
   Due: Timeline
   Details: Description

Create a complete plan with specific tasks."""
        
        # Setup progress template
        self.progress_template = """Review onboarding progress.
Check task completion and identify issues.

Employee: {name}
Department: {department}
Role: {role}

Current Progress:
{progress}

Provide a structured review with:
- Completion percentage
- Finished tasks
- Pending items
- Next actions"""
        
        logger.info("Assistant ready")

    async def get_response(self, question: str, history: List = None) -> str:
        """Get conversational response."""
        logger.info("Processing question")
        
        try:
            # Format with examples and history
            messages = self.chat_template.format_messages(
                examples=self.examples,
                history=history or [],
                input=question
            )
            
            # Get response
            response = await self.llm.ainvoke(messages)
            result = StrOutputParser().parse(response.content)
            logger.info("Response ready")
            return result
            
        except Exception as e:
            logger.error(f"Response failed: {str(e)}")
            raise

    async def create_plan(self, employee: Employee) -> str:
        """Create onboarding plan."""
        logger.info(f"Creating plan for: {employee.employee_id}")
        
        try:
            # Format plan request
            content = self.plan_template.format(
                name=employee.name,
                department=employee.department.value,
                role=employee.role.value
            )
            
            # Get plan
            messages = [HumanMessage(content=content)]
            response = await self.llm.ainvoke(messages)
            result = StrOutputParser().parse(response.content)
            logger.info("Plan created")
            return result
            
        except Exception as e:
            logger.error(f"Plan failed: {str(e)}")
            raise

    async def check_progress(self, employee: Employee, progress: str) -> str:
        """Review onboarding progress."""
        logger.info(f"Checking progress for: {employee.employee_id}")
        
        try:
            # Format progress request
            content = self.progress_template.format(
                name=employee.name,
                department=employee.department.value,
                role=employee.role.value,
                progress=progress
            )
            
            # Get review
            messages = [HumanMessage(content=content)]
            response = await self.llm.ainvoke(messages)
            result = StrOutputParser().parse(response.content)
            logger.info("Progress reviewed")
            return result
            
        except Exception as e:
            logger.error(f"Review failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting onboarding demo...")
    
    try:
        # Create assistant
        assistant = OnboardingAssistant()
        
        # Example employee
        employee = Employee(
            employee_id="EMP001",
            name="John Smith",
            department=Department.CORPORATE,
            role=Role.ANALYST,
            start_date="2025-04-15"
        )
        
        print("\nCreating Onboarding Plan")
        print("=======================")
        print(f"Employee: {employee.name}")
        print(f"Department: {employee.department.value}")
        print(f"Role: {employee.role.value}")
        print(f"Start Date: {employee.start_date}\n")
        
        try:
            # Create plan
            plan = await assistant.create_plan(employee)
            print("\nOnboarding Plan:")
            print("===============")
            print(plan)
            
            # Example progress
            progress = """Tasks Completed:
1. IT Setup (2025-04-15)
2. Policy Review (2025-04-16)
3. System Access (2025-04-16)

Tasks Pending:
1. Department Training (Due: 2025-04-18)
2. Compliance Course (Due: 2025-04-19)
3. Team Meeting (Due: 2025-04-22)"""
            
            # Check progress
            result = await assistant.check_progress(employee, progress)
            print("\nProgress Review:")
            print("===============")
            print(result)
            
            # Example question
            print("\nQuestion Handling:")
            print("=================")
            question = "What training do I need to complete?"
            print(f"Q: {question}")
            
            response = await assistant.get_response(question)
            print("\nA:")
            print(response)
            
        except Exception as e:
            print(f"\nProcessing failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
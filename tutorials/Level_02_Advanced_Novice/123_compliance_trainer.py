#!/usr/bin/env python3
"""
Compliance Training Assistant (123) (LangChain v3)

This example demonstrates banking compliance training using:
1. Learning Analysis: Skill assessment
2. Content Adaptation: Personalized learning
3. Progress Tracking: Skill development

It helps HR teams deliver personalized compliance training.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
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
    """Bank departments."""
    RETAIL = "retail_banking"
    CORPORATE = "corporate_banking"
    INVESTMENT = "investment_banking"
    OPERATIONS = "operations"
    TECHNOLOGY = "technology"
    RISK = "risk_management"

class ComplianceArea(str, Enum):
    """Compliance areas."""
    AML = "anti_money_laundering"
    KYC = "know_your_customer"
    GDPR = "data_protection"
    TRADING = "trading_compliance"
    SECURITY = "information_security"
    ETHICS = "business_ethics"

class SkillLevel(str, Enum):
    """Skill levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class TrainingModule(BaseModel):
    """Training module details."""
    module_id: str = Field(description="Module ID")
    title: str = Field(description="Module title")
    area: ComplianceArea = Field(description="Compliance area")
    level: SkillLevel = Field(description="Skill level")
    content: str = Field(description="Training content")
    duration: int = Field(description="Minutes required")

class EmployeeProfile(BaseModel):
    """Employee training profile."""
    employee_id: str = Field(description="Employee ID")
    department: Department = Field(description="Department")
    role: str = Field(description="Job role")
    areas: List[ComplianceArea] = Field(description="Required areas")
    completed: List[str] = Field(description="Completed modules")
    skills: Dict[str, SkillLevel] = Field(description="Current skills")
    metadata: Dict = Field(default_factory=dict)

class ComplianceTrainer:
    """Compliance training assistant system."""

    def __init__(self):
        """Initialize trainer."""
        logger.info("Starting compliance trainer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Create training chain
        trainer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a compliance training assistant.
Analyze employee profiles and training modules to create personalized learning plans.

Your analysis should include:

1. Skill Assessment
- Current knowledge
- Required skills
- Skill gaps
- Learning goals

2. Training Plan
- Module selection
- Learning path
- Time requirements
- Priority order

3. Progress Tracking
- Completion targets
- Key milestones
- Success metrics
- Review points

4. Learning Support
- Focus areas
- Practice needs
- Review topics
- Additional resources

Format with clear sections and actionable steps."""),
            ("human", """Create a training plan:

Module:
ID: {module_id}
Title: {title}
Area: {area}
Level: {level}
Duration: {duration} minutes

Employee:
ID: {employee_id}
Department: {department}
Role: {role}
Required Areas: {required_areas}
Current Skills: {current_skills}

Provide a personalized learning plan.""")
        ])
        
        self.chain = (
            {"module_id": RunnablePassthrough(), 
             "title": RunnablePassthrough(),
             "area": RunnablePassthrough(),
             "level": RunnablePassthrough(),
             "duration": RunnablePassthrough(),
             "employee_id": RunnablePassthrough(),
             "department": RunnablePassthrough(),
             "role": RunnablePassthrough(),
             "required_areas": RunnablePassthrough(),
             "current_skills": RunnablePassthrough()} 
            | trainer_prompt 
            | self.llm 
            | StrOutputParser()
        )
        logger.info("Training chain ready")

    async def create_training_plan(self, module: TrainingModule, employee: EmployeeProfile) -> str:
        """Create personalized training plan."""
        logger.info(f"Creating plan for {employee.employee_id} - Module: {module.module_id}")
        
        try:
            # Run analysis
            result = await self.chain.ainvoke({
                "module_id": module.module_id,
                "title": module.title,
                "area": module.area.value,
                "level": module.level.value,
                "duration": module.duration,
                "employee_id": employee.employee_id,
                "department": employee.department.value,
                "role": employee.role,
                "required_areas": ", ".join(area.value for area in employee.areas),
                "current_skills": ", ".join(f"{k}: {v.value}" for k, v in employee.skills.items())
            })
            logger.info("Training plan created")
            return result
            
        except Exception as e:
            logger.error(f"Plan creation failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting compliance training demo...")
    
    try:
        # Create trainer
        trainer = ComplianceTrainer()
        
        # Example module
        module = TrainingModule(
            module_id="AML-201",
            title="Advanced Transaction Monitoring",
            area=ComplianceArea.AML,
            level=SkillLevel.ADVANCED,
            content="""Course covers:
1. Transaction Pattern Analysis
- Complex transaction flows
- Alert investigation techniques
- Risk scoring methodologies
- Documentation requirements

2. Case Management
- Alert triage process
- Investigation workflows
- Evidence collection
- Report generation

3. Regulatory Requirements
- Reporting thresholds
- Filing timelines
- Documentation standards
- Quality assurance

4. Advanced Detection
- Machine learning models
- Behavior analytics
- Network analysis
- Risk indicators""",
            duration=120
        )
        
        # Example employee
        employee = EmployeeProfile(
            employee_id="EMP001",
            department=Department.RISK,
            role="Transaction Monitoring Analyst",
            areas=[
                ComplianceArea.AML,
                ComplianceArea.KYC,
                ComplianceArea.SECURITY
            ],
            completed=[
                "AML-101",
                "AML-102",
                "KYC-201"
            ],
            skills={
                "aml_basics": SkillLevel.INTERMEDIATE,
                "kyc_procedures": SkillLevel.INTERMEDIATE,
                "transaction_monitoring": SkillLevel.INTERMEDIATE,
                "risk_assessment": SkillLevel.INTERMEDIATE
            }
        )
        
        print("\nCreating Training Plan")
        print("====================")
        print(f"Module: {module.title}")
        print(f"Employee: {employee.employee_id}")
        print(f"Department: {employee.department.value}")
        print(f"Role: {employee.role}\n")
        
        try:
            # Get plan
            result = await trainer.create_training_plan(module, employee)
            print("\nPersonalized Training Plan:")
            print("=========================")
            print(result)
            
        except Exception as e:
            print(f"\nPlan creation failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
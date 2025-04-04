#!/usr/bin/env python3
"""
LangChain Banking Task Prioritizer (116) (LangChain v3)

This example demonstrates a banking task prioritization system using:
1. Chat Models: Task analysis and impact assessment
2. Structured Output: Priority determination
3. Pydantic: Task validation and scoring

It helps bank departments prioritize tasks based on impact, urgency, and resources.
"""

import os
import json
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Definitions
class Department(str, Enum):
    """Bank departments."""
    DEVELOPMENT = "development"
    DATA_SCIENCE = "data_science"
    LEGAL = "legal"
    HR = "human_resources"
    MARKETING = "marketing"
    RISK = "risk"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    FINANCE = "finance"
    SECURITY = "security"

class Impact(str, Enum):
    """Task impact levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    """Types of banking tasks."""
    DEVELOPMENT = "development"
    ANALYSIS = "analysis"
    COMPLIANCE = "compliance"
    SUPPORT = "support"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    TRAINING = "training"
    REVIEW = "review"

class BankingTask(BaseModel):
    """Banking task details."""
    task_id: str = Field(description="Task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    department: Department = Field(description="Responsible department")
    type: TaskType = Field(description="Task type")
    deadline: Optional[str] = Field(description="Task deadline", default=None)
    resources: List[str] = Field(description="Required resources")
    stakeholders: List[str] = Field(description="Task stakeholders")
    metadata: Dict = Field(default_factory=dict)

class TaskAnalysis(BaseModel):
    """Task analysis results."""
    impact: Impact = Field(description="Task impact level")
    urgency_score: int = Field(description="Urgency score (1-10)")
    complexity: int = Field(description="Complexity score (1-10)")
    resource_demand: int = Field(description="Resource demand (1-10)")
    dependencies: List[str] = Field(description="Task dependencies")
    risks: List[str] = Field(description="Potential risks")

    @field_validator('urgency_score', 'complexity', 'resource_demand')
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("Score must be between 1 and 10")
        return v

class TaskPriority(BaseModel):
    """Task prioritization results."""
    task_id: str = Field(description="Original task ID")
    priority_score: int = Field(description="Priority score (1-100)")
    impact_level: Impact = Field(description="Assessed impact")
    recommendations: List[str] = Field(description="Action recommendations")
    required_resources: List[str] = Field(description="Required resources")
    timeline: str = Field(description="Suggested timeline")
    escalation_level: str = Field(description="Escalation requirements")

    @field_validator('priority_score')
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("Priority score must be between 1 and 100")
        return v

# Department weights for priority calculation
DEPARTMENT_WEIGHTS = {
    Department.SECURITY: 1.5,     # High priority for security tasks
    Department.COMPLIANCE: 1.4,   # High priority for compliance
    Department.RISK: 1.3,         # Important risk management
    Department.FINANCE: 1.2,      # Financial tasks
    Department.OPERATIONS: 1.1,   # Core operations
    Department.DEVELOPMENT: 1.0,  # Standard development
    Department.DATA_SCIENCE: 1.0, # Analytics tasks
    Department.LEGAL: 1.0,        # Legal work
    Department.HR: 0.9,           # HR tasks
    Department.MARKETING: 0.8     # Marketing activities
}

class BankingTaskPrioritizer:
    def __init__(self):
        """Initialize the task prioritizer."""
        print("Initializing Banking Task Prioritizer...")
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Setup analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a banking task analyzer. Return only a JSON object with no extra text."),
            ("human", """Analyze this task: {task}

Return a JSON with:
"impact": "low/medium/high/critical",
"urgency_score": number 1-10,
"complexity": number 1-10,
"resource_demand": number 1-10,
"dependencies": ["dep1"],
"risks": ["risk1"]""")
        ])
        
        # Setup priority prompt
        self.priority_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a banking task prioritizer. Return only a JSON object with no extra text."),
            ("human", """Details:
Task: {task}
Analysis: {analysis}
Weight: {weight}

Return a JSON with:
"priority_score": number 1-100,
"impact_level": "low/medium/high/critical",
"recommendations": ["rec1"],
"required_resources": ["res1"],
"timeline": "duration",
"escalation_level": "level\"""")
        ])
        
        print("Router initialized")

    def extract_json(self, text: str) -> dict:
        """Extract and parse JSON from text."""
        # Clean text
        text = text.strip()
        
        # Find JSON boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in:\n{text}")
            
        # Extract JSON
        json_str = text[start:end + 1]
        print(f"Found JSON:\n{json_str}")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {str(e)}")
            print(f"Raw text:\n{text}")
            raise

    async def prioritize_task(self, task: BankingTask) -> TaskPriority:
        """Analyze and prioritize a banking task."""
        print(f"\nProcessing Task: {task.task_id}")
        print(f"Title: {task.title}")
        print(f"Department: {task.department.value}")
        
        try:
            # Get analysis
            print("\nGetting task analysis...")
            messages = self.analysis_prompt.format_messages(
                task=json.dumps(task.model_dump())
            )
            response = await self.llm.ainvoke(messages)
            
            # Parse analysis
            analysis_data = self.extract_json(response.content)
            analysis = TaskAnalysis(**analysis_data)
            
            print(f"Analysis complete:")
            print(f"Impact: {analysis.impact}")
            print(f"Urgency: {analysis.urgency_score}/10")
            print(f"Complexity: {analysis.complexity}/10\n")
            
            # Get priority
            print("Calculating priority...")
            weight = DEPARTMENT_WEIGHTS.get(task.department, 1.0)
            print(f"Department weight: {weight}")
            
            messages = self.priority_prompt.format_messages(
                task=json.dumps(task.model_dump()),
                analysis=json.dumps(analysis.model_dump()),
                weight=weight
            )
            response = await self.llm.ainvoke(messages)
            
            # Parse priority
            priority_data = self.extract_json(response.content)
            priority_data["task_id"] = task.task_id
            priority = TaskPriority(**priority_data)
            
            print(f"Priority set:")
            print(f"Score: {priority.priority_score}/100")
            print(f"Impact: {priority.impact_level}\n")
            
            return priority
            
        except Exception as e:
            print(f"Error processing task {task.task_id}: {str(e)}")
            raise

async def main():
    """Demo the task prioritizer."""
    prioritizer = BankingTaskPrioritizer()
    
    # Example tasks
    tasks = [
        BankingTask(
            task_id="SEC-001",
            title="OAuth2 Security Update",
            description="Implement OAuth2 security update for payment API system",
            department=Department.SECURITY,
            type=TaskType.SECURITY,
            deadline="2025-04-15",
            resources=["Security Team", "API Documentation", "Test Environment"],
            stakeholders=["Security Team", "Development Team", "Compliance"]
        ),
        BankingTask(
            task_id="DS-001",
            title="Fraud Detection Model",
            description="Develop ML model for real-time transaction fraud detection",
            department=Department.DATA_SCIENCE,
            type=TaskType.DEVELOPMENT,
            deadline="2025-05-01",
            resources=["Data Science Team", "Transaction Data", "ML Platform"],
            stakeholders=["Risk Team", "Data Science Team", "Operations"]
        ),
        BankingTask(
            task_id="HR-001",
            title="Compliance Training",
            description="Develop training program for new banking regulations",
            department=Department.HR,
            type=TaskType.COMPLIANCE,
            deadline="2025-04-30",
            resources=["Training Team", "Compliance Docs", "LMS"],
            stakeholders=["HR Team", "Compliance Team", "All Departments"]
        )
    ]
    
    print("\nBanking Task Prioritizer Demo")
    print("============================\n")
    
    for task in tasks:
        try:
            priority = await prioritizer.prioritize_task(task)
            
            print("Recommendations:")
            for i, rec in enumerate(priority.recommendations, 1):
                print(f"{i}. {rec}")
            
            print("\nRequired Resources:")
            for res in priority.required_resources:
                print(f"- {res}")
            
            print(f"\nTimeline: {priority.timeline}")
            print(f"Escalation: {priority.escalation_level}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nFailed to process task {task.task_id}:")
            print(f"Error: {str(e)}")
            print("-" * 50)
            continue
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
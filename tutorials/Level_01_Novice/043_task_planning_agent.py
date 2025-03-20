"""
LangChain Task Planning Agent Example

This example demonstrates how to combine agents and chains capabilities to create
a system that can autonomously plan and break down complex tasks while handling
dependencies and generating structured execution plans.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import SequentialChain
from langchain.chains.llm import LLMChain

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class TaskStep(BaseModel):
    """Schema for individual task steps."""
    step_id: str = Field(description="Unique step identifier")
    name: str = Field(description="Step name")
    description: str = Field(description="Detailed step description")
    duration: float = Field(description="Estimated duration in hours")
    dependencies: List[str] = Field(description="Required prior steps")
    resources: List[str] = Field(description="Required resources")

class TaskPlan(BaseModel):
    """Schema for task execution plan."""
    task_id: str = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    steps: List[TaskStep] = Field(description="Ordered task steps")
    total_duration: float = Field(description="Total estimated duration")
    critical_path: List[str] = Field(description="Critical path steps")
    risks: List[str] = Field(description="Identified risks")
    assumptions: List[str] = Field(description="Planning assumptions")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "task_id": "PRJ001",
                "title": "Website Redesign",
                "description": "Modernize company website",
                "steps": [{
                    "step_id": "S1",
                    "name": "Requirements Analysis",
                    "description": "Gather and document requirements",
                    "duration": 4.0,
                    "dependencies": [],
                    "resources": ["Project Manager", "Business Analyst"]
                }],
                "total_duration": 4.0,
                "critical_path": ["S1"],
                "risks": ["Incomplete requirements"],
                "assumptions": ["Stakeholders available"],
                "timestamp": "2024-04-01T12:00:00"
            }]
        }
    }

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def plan_task(task_description: str, llm: AzureChatOpenAI, parser: PydanticOutputParser) -> TaskPlan:
    """
    Generate a comprehensive task plan as a structured JSON object.
    
    The LLM is instructed to output only a JSON object that exactly follows the schema.
    
    Args:
        task_description: The task to be planned.
        llm: The language model for planning.
        parser: The output parser for the TaskPlan schema.
    
    Returns:
        TaskPlan: The structured task plan.
    """
    # Escape curly braces in the schema by doubling them.
    structured_template = """
You are a task planning expert. Generate a comprehensive task plan as a JSON object that follows exactly this schema:
{{
  "task_id": "<string>",
  "title": "<string>",
  "description": "<string>",
  "steps": [
      {{
          "step_id": "<string>",
          "name": "<string>",
          "description": "<string>",
          "duration": <number>,
          "dependencies": ["<string>", ...],
          "resources": ["<string>", ...]
      }}
  ],
  "total_duration": <number>,
  "critical_path": ["<string>", ...],
  "risks": ["<string>", ...],
  "assumptions": ["<string>", ...],
  "timestamp": "<ISO 8601 datetime>"
}}

Task Description: {input_task}

Respond with only the JSON.
"""
    prompt_text = structured_template.format(input_task=task_description)
    message = HumanMessage(content=prompt_text)
    response = llm.invoke([message])
    return parser.parse(response.content)

def demonstrate_task_planning():
    """Demonstrate task planning capabilities."""
    try:
        print("\nDemonstrating Task Planning System...\n")
        
        llm = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=TaskPlan)
        
        # Example 1: Software Project Planning
        print("Example 1: Software Project Planning")
        print("-" * 50)
        
        software_task = (
            "Develop a new customer portal with user authentication, profile management, "
            "and order history features. The portal should integrate with existing backend systems "
            "and follow company security guidelines."
        )
        
        plan1 = plan_task(software_task, llm, parser)
        
        print("\nTask Plan:")
        print(f"ID: {plan1.task_id}")
        print(f"Title: {plan1.title}")
        print("\nSteps:")
        for step in plan1.steps:
            print(f"\n{step.step_id}: {step.name}")
            print(f"Description: {step.description}")
            print(f"Duration: {step.duration} hours")
            if step.dependencies:
                print(f"Dependencies: {', '.join(step.dependencies)}")
            if step.resources:
                print(f"Resources: {', '.join(step.resources)}")
        
        print("\nRisks:")
        for risk in plan1.risks:
            print(f"- {risk}")
        print("\nAssumptions:")
        for assumption in plan1.assumptions:
            print(f"- {assumption}")
        
        # Example 2: Marketing Campaign Planning
        print("\nExample 2: Marketing Campaign Planning")
        print("-" * 50)
        
        marketing_task = (
            "Plan and execute a multi-channel marketing campaign for product launch. "
            "Include social media, email marketing, and content creation. The campaign duration is 3 months "
            "with a focus on brand awareness and lead generation."
        )
        
        plan2 = plan_task(marketing_task, llm, parser)
        
        print("\nTask Plan:")
        print(f"ID: {plan2.task_id}")
        print(f"Title: {plan2.title}")
        print("\nSteps:")
        for step in plan2.steps:
            print(f"\n{step.step_id}: {step.name}")
            print(f"Description: {step.description}")
            print(f"Duration: {step.duration} hours")
            if step.dependencies:
                print(f"Dependencies: {', '.join(step.dependencies)}")
            if step.resources:
                print(f"Resources: {', '.join(step.resources)}")
        
        print("\nRisks:")
        for risk in plan2.risks:
            print(f"- {risk}")
        print("\nAssumptions:")
        for assumption in plan2.assumptions:
            print(f"- {assumption}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Task Planning Agent...")
    demonstrate_task_planning()

if __name__ == "__main__":
    main()

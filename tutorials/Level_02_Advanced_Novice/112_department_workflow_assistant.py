#!/usr/bin/env python3
"""
LangChain Department Workflow Assistant (112) (LangChain v3)

This example demonstrates a banking department workflow system using three key concepts:
1. Agents: Department-specific task handling
2. Memory: Context retention across interactions
3. Tools: Department-specific integrations

It provides workflow automation support for various banking departments.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DepartmentType(str, Enum):
    """Banking department types."""
    DEVELOPMENT = "development"
    DATA_SCIENCE = "data_science"
    LEGAL = "legal"
    HR = "hr"
    MARKETING = "marketing"
    FINANCE = "finance"
    RISK = "risk"

class WorkflowStatus(str, Enum):
    """Workflow status types."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

class WorkflowStep(BaseModel):
    """Schema for workflow steps."""
    name: str = Field(description="Step name")
    status: WorkflowStatus = Field(description="Current status")
    assignee: str = Field(description="Step assignee")
    due_date: str = Field(description="Due date")
    notes: List[str] = Field(description="Step notes")

class WorkflowRequest(BaseModel):
    """Schema for workflow requests."""
    workflow_id: str = Field(description="Workflow identifier")
    department: DepartmentType = Field(description="Department type")
    title: str = Field(description="Workflow title")
    description: str = Field(description="Workflow description")
    priority: int = Field(description="Priority level (1-5)")
    steps: List[WorkflowStep] = Field(description="Workflow steps")

class DepartmentWorkflowAssistant:
    def __init__(self):
        logger.info("Initializing Department Workflow Assistant...")
        try:
            # Initialize Azure OpenAI
            logger.info("Setting up Azure OpenAI...")
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            
            # Setup components
            logger.info("Setting up tools...")
            self.setup_tools()
            
            # Setup memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Setup agent
            logger.info("Setting up agent...")
            self.setup_agent()
            logger.info("Assistant initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}")
            raise

    def setup_tools(self):
        """Initialize department-specific tools."""
        self.tools = [
            # Development tools
            StructuredTool.from_function(
                func=self.check_deployment_status,
                name="check_deployment_status",
                description="Check deployment status"
            ),
            StructuredTool.from_function(
                func=self.validate_model,
                name="validate_model",
                description="Validate ML model"
            ),
            StructuredTool.from_function(
                func=self.analyze_document,
                name="analyze_document",
                description="Analyze document contents"
            )
        ]

    def setup_agent(self):
        """Initialize the workflow agent."""
        # Agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a banking workflow assistant that helps manage tasks."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    async def process_workflow(self, request: WorkflowRequest) -> Dict:
        """Process a department workflow request."""
        logger.info(f"Processing workflow: {request.workflow_id}")
        try:
            # Create workflow context
            workflow_context = (
                f"Process workflow request for {request.department.value} department:\n"
                f"Title: {request.title}\n"
                f"Description: {request.description}\n"
                f"Priority: {request.priority}\n"
                f"Steps: {len(request.steps)} steps"
            )
            
            # Process with agent
            logger.debug("Invoking agent...")
            result = await self.agent_executor.ainvoke({"input": workflow_context})
            logger.info("Workflow processing complete")
            
            return {
                "workflow_id": request.workflow_id,
                "department": request.department.value,
                "status": "processed",
                "actions": result.get("output", "No actions specified"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing workflow: {str(e)}")
            raise

    # Tool implementations
    def check_deployment_status(self, deployment_id: str) -> str:
        """Check deployment status."""
        return (
            f"Deployment {deployment_id} status:\n"
            "- Security patches verified\n"
            "- API endpoints updated\n"
            "- Health checks passing"
        )

    def validate_model(self, model_id: str) -> str:
        """Validate ML model."""
        return f"Model {model_id} validation passed"

    def analyze_document(self, document_id: str) -> str:
        """Analyze document."""
        return f"Document {document_id} analyzed"

async def demonstrate_workflow_assistant():
    logger.info("\nDepartment Workflow Assistant Demo")
    logger.info("==================================\n")

    try:
        logger.info("Initializing assistant...")
        assistant = DepartmentWorkflowAssistant()

        # Example workflow
        logger.info("Setting up example workflow...")
        workflow = WorkflowRequest(
            workflow_id="DEV-2025-001",
            department=DepartmentType.DEVELOPMENT,
            title="API Security Update",
            description="Deploy critical security patches to API endpoints",
            priority=1,
            steps=[
                WorkflowStep(
                    name="Code Review",
                    status=WorkflowStatus.COMPLETED,
                    assignee="tech.lead",
                    due_date="2025-04-01",
                    notes=["Security patches verified"]
                )
            ]
        )

        # Process workflow
        logger.info("Processing workflow...")
        result = await assistant.process_workflow(workflow)
        
        logger.info("\nResults:")
        logger.info(f"Status: {result['status']}")
        logger.info("\nActions:")
        logger.info(result['actions'])
        
    except Exception as e:
        logger.error(f"\nError during demonstration: {str(e)}", exc_info=True)
    
    logger.info("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_workflow_assistant())
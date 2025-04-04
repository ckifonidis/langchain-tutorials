#!/usr/bin/env python3
"""
LangChain Banking Customer Service Assistant (114) (LangChain v3)

This example demonstrates a banking customer service system using three key concepts:
1. Chat Models: Natural conversation handling
2. Output Parsers: Direct JSON formatting
3. Prompt Engineering: Clear instructions

It provides comprehensive customer service support across banking departments.
"""

import os
import sys
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DepartmentType(str, Enum):
    """Banking department types."""
    SUPPORT = "support"
    PRODUCT = "product"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    SALES = "sales"
    SECURITY = "security"

class ResponseType(str, Enum):
    """Response types."""
    GENERAL = "GENERAL"
    TECHNICAL = "TECHNICAL"
    PRODUCT = "PRODUCT"
    SECURITY = "SECURITY"

class CustomerQuery(BaseModel):
    """Customer query structure."""
    query_id: str = Field(description="Query identifier")
    content: str = Field(description="Query content")
    department: DepartmentType = Field(description="Target department")
    priority: int = Field(description="Priority level (1-5)")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class ServiceResponse(BaseModel):
    """Service response structure."""
    response_id: str = Field(description="Response identifier")
    query_id: str = Field(description="Original query ID")
    content: str = Field(description="Response content")
    department: DepartmentType = Field(description="Responding department")
    response_type: ResponseType = Field(description="Response category")
    follow_up: Optional[List[str]] = Field(default_factory=list)
    resources: Optional[List[str]] = Field(default_factory=list)

PROMPT_TEMPLATE = """You are a helpful banking assistant. Respond to the customer query below.

Return your response as a valid JSON object with these exact fields:
{{
    "content": "your detailed response here",
    "response_type": "one of: GENERAL, TECHNICAL, PRODUCT, SECURITY",
    "follow_up": ["action 1", "action 2"],
    "resources": ["resource 1", "resource 2"]
}}

Customer Query: {input}

Assistant Response:"""

class CustomerServiceAssistant:
    def __init__(self):
        """Initialize the customer service assistant."""
        logger.info("Initializing Banking Customer Service Assistant...")
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        self.setup_prompt()
        logger.info("Assistant ready")

    def setup_prompt(self):
        """Setup prompt template."""
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["input"]
        )

    async def process_query(self, query: CustomerQuery) -> ServiceResponse:
        """Process a customer query."""
        try:
            # Format query and get response
            formatted_prompt = self.prompt.format(input=query.content.strip())
            response = await self.llm.ainvoke(
                [HumanMessage(content=formatted_prompt)]
            )
            response_text = response.content.strip()
            
            # Find and extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}')
            
            if start >= 0 and end > start:
                json_str = response_text[start:end + 1].strip()
                response_data = json.loads(json_str)
                
                # Add metadata
                response_data.update({
                    "response_id": f"RES-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "query_id": query.query_id,
                    "department": query.department.value
                })
                
                return ServiceResponse(**response_data)
            else:
                raise ValueError("No JSON found in response")
            
        except Exception as e:
            logger.error("Processing failed: %s", str(e))
            raise

async def demonstrate_customer_service():
    print("\nBanking Customer Service Assistant Demo")
    print("=====================================\n")

    try:
        # Initialize assistant
        assistant = CustomerServiceAssistant()

        # Example queries
        queries = [
            CustomerQuery(
                query_id="TECH-2025-001",
                content="How do I enable international transactions on my card?",
                department=DepartmentType.TECHNICAL,
                priority=2,
                metadata={"channel": "mobile"}
            ),
            CustomerQuery(
                query_id="SEC-2025-001",
                content="I noticed a suspicious transaction on my account",
                department=DepartmentType.SECURITY,
                priority=1,
                metadata={"urgent": True}
            ),
            CustomerQuery(
                query_id="PROD-2025-001",
                content="What are the benefits of your premium checking account?",
                department=DepartmentType.PRODUCT,
                priority=3,
                metadata={"prospect": True}
            )
        ]

        # Process queries
        for query in queries:
            print(f"\nProcessing Query: {query.query_id}")
            print(f"Content: {query.content}")
            print(f"Department: {query.department.value}")
            
            response = await assistant.process_query(query)
            
            print("\nResponse:")
            print(f"Content: {response.content}")
            
            if response.follow_up:
                print("\nFollow-up Actions:")
                for action in response.follow_up:
                    print(f"- {action}")
                    
            if response.resources:
                print("\nHelpful Resources:")
                for resource in response.resources:
                    print(f"- {resource}")
                    
            print("\n" + "-" * 50)
        
    except Exception as e:
        print("\n❌ Error during demonstration:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
    
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_customer_service())

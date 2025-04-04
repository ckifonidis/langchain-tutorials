#!/usr/bin/env python3
"""
API Monitor (133) (LangChain v3)

This example demonstrates API documentation using:
1. Few Shot Learning: Example-based analysis
2. Template Chaining: Multi-step processing
3. Async Callbacks: Event handling

It helps development teams monitor banking APIs.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ApiCategory(str, Enum):
    """API categories."""
    ACCOUNTS = "account_services"
    PAYMENTS = "payment_services"
    CARDS = "card_services"
    LOANS = "loan_services"
    AUTH = "authentication"
    USERS = "user_services"

class ApiStatus(str, Enum):
    """API status levels."""
    OK = "operational"
    WARN = "degraded"
    ERROR = "failing"
    DOWN = "offline"

class ApiEndpoint(BaseModel):
    """API endpoint details."""
    endpoint_id: str = Field(description="Endpoint ID")
    category: ApiCategory = Field(description="API category")
    method: str = Field(description="HTTP method")
    path: str = Field(description="API path")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    metadata: Dict = Field(default_factory=dict)

class MonitorCallback(AsyncCallbackHandler):
    """API monitoring callback."""
    
    async def on_llm_start(self, *args, **kwargs):
        logger.info("Starting API analysis")
        
    async def on_llm_end(self, *args, **kwargs):
        logger.info("Analysis complete")
        
    async def on_llm_error(self, error: Exception, *args, **kwargs):
        logger.error(f"Analysis error: {str(error)}")

class ApiMonitor:
    """API monitoring system."""

    def __init__(self):
        """Initialize monitor."""
        logger.info("Starting API monitor...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            callbacks=[MonitorCallback()]
        )
        logger.info("Chat model ready")
        
        # Example analyses
        self.examples = [
            [
                HumanMessage(content="High latency on /accounts/balance"),
                AIMessage(content="""API ANALYSIS
-----------
Status: WARN
Service: Account Services
Issue: Response Time

Problem:
- 95th percentile latency above SLA
- Average response time increasing
- Database query slowdown

Impact:
- User experience degraded
- Mobile app affected
- Higher error rates

Actions:
1. Optimize DB queries
2. Scale up resources
3. Update cache config

Priority: HIGH
Timeline: 24 hours""")
            ],
            [
                HumanMessage(content="Authentication service errors"),
                AIMessage(content="""API ANALYSIS
-----------
Status: ERROR
Service: Authentication
Issue: Failed Requests

Problem:
- Token validation failures
- High error rates
- Session timeouts

Impact:
- Login issues
- Access denied
- Security alerts

Actions:
1. Check token service
2. Review error logs
3. Monitor sessions

Priority: CRITICAL
Timeline: 2 hours""")
            ]
        ]
        
        # Setup analysis chain
        self.check = ChatPromptTemplate.from_messages([
            ("system", """You are an API monitoring expert.
Review endpoints and identify issues.

Base your analysis on these examples:
{examples}

Consider:
- Response times
- Error rates
- Resource usage
- User impact"""),
            ("human", """Analyze this endpoint:

ID: {endpoint_id}
Category: {category}
Path: {method} {path}

Metrics:
{metrics}

Provide a complete analysis.""")
        ])
        
        # Setup recommendation chain
        self.suggest = ChatPromptTemplate.from_messages([
            ("system", """Generate API optimization recommendations.

Format your response like this:

OPTIMIZATION PLAN
---------------
Status: Current state
Priority: HIGH/MEDIUM/LOW

Issues Found:
1. Issue Name
   Impact: Description
   Fix: Solution
   Time: Timeline

2. Issue Name
   Impact: Description
   Fix: Solution
   Time: Timeline

Required Actions:
1. Action step
2. Action step

Next Check: YYYY-MM-DD"""),
            ("human", """Plan improvements for:
ID: {endpoint_id}
Analysis: {analysis}

Provide specific recommendations.""")
        ])
        
        # Setup output parser
        self.parser = StrOutputParser()
        logger.info("Monitor ready")

    async def analyze_endpoint(self, endpoint: ApiEndpoint) -> str:
        """Analyze API endpoint."""
        logger.info(f"Analyzing endpoint: {endpoint.endpoint_id}")
        
        try:
            # Format metrics
            metrics = "\n".join(
                f"{k}: {v:.2f}" 
                for k, v in endpoint.metrics.items()
            )
            
            # Check endpoint
            messages = self.check.format_messages(
                examples=self.examples,
                endpoint_id=endpoint.endpoint_id,
                category=endpoint.category.value,
                method=endpoint.method,
                path=endpoint.path,
                metrics=metrics
            )
            logger.debug("Analysis formatted")
            
            # Get analysis
            response = await self.llm.ainvoke(messages)
            analysis = self.parser.parse(response.content)
            logger.debug("First step complete")
            
            # Get recommendations
            messages = self.suggest.format_messages(
                endpoint_id=endpoint.endpoint_id,
                analysis=analysis
            )
            
            # Get and combine results
            response = await self.llm.ainvoke(messages)
            recommendations = self.parser.parse(response.content)
            logger.info("Analysis complete")
            
            return f"Analysis Results:\n{analysis}\n\nRecommendations:\n{recommendations}"
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting monitoring demo...")
    
    try:
        # Create monitor
        monitor = ApiMonitor()
        
        # Example endpoint
        endpoint = ApiEndpoint(
            endpoint_id="API-2025-001",
            category=ApiCategory.ACCOUNTS,
            method="GET",
            path="/v1/accounts/{id}/balance",
            metrics={
                "requests": 15000,
                "errors": 450,
                "latency_p95": 2.5,
                "latency_p50": 0.8,
                "success_rate": 97.0,
                "cpu_usage": 65.5,
                "memory_usage": 78.2,
                "cache_hits": 85.5
            }
        )
        
        print("\nAnalyzing Endpoint")
        print("=================")
        print(f"Endpoint: {endpoint.endpoint_id}")
        print(f"Category: {endpoint.category.value}")
        print(f"Path: {endpoint.method} {endpoint.path}\n")
        
        print("Performance Metrics:")
        for name, value in endpoint.metrics.items():
            if name.startswith("latency"):
                print(f"{name}: {value:.2f}s")
            elif name.endswith("rate"):
                print(f"{name}: {value:.1f}%")
            elif name.endswith("usage"):
                print(f"{name}: {value:.1f}%")
            else:
                print(f"{name}: {value:,.0f}")
        
        try:
            # Get analysis
            result = await monitor.analyze_endpoint(endpoint)
            print("\nAnalysis Results:")
            print("================")
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
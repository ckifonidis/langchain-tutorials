#!/usr/bin/env python3
"""
Multi-Agent Financial Data Analysis System

This example demonstrates a collaborative financial data analysis system using
retrieval, tool_calling, and tracing for coordinated data processing.
"""

import os
import json
import csv
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")

def debug_print(prefix: str, obj: Any) -> None:
    """Helper to print debug information."""
    try:
        truncated = str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
        print(f"\n{prefix}: {truncated}")
    except Exception as e:
        print(f"Error printing debug info: {str(e)}")

class AnalysisType(str, Enum):
    """Types of financial analysis."""
    TREND = "trend_analysis"
    RISK = "risk_assessment"
    FORECAST = "forecasting"
    ANOMALY = "anomaly_detection"
    SUMMARY = "data_summary"

class AnalysisRequest(BaseModel):
    """Analysis request specification."""
    analysis_type: AnalysisType
    data_source: str = Field(description="CSV file path")
    parameters: Dict = Field(description="Analysis parameters")
    user_query: Optional[str] = Field(description="User's specific question")

class AnalysisResult(BaseModel):
    """Analysis result details."""
    type: AnalysisType
    findings: List[str]
    metrics: Dict
    recommendations: List[str]
    confidence: float

class MultiAgentAnalyzer:
    """Collaborative financial data analysis system."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        print("\nInitializing Multi-Agent Analysis System...")
        
        # Initialize tracing
        self.callbacks = [ConsoleCallbackHandler()]
        if os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true":
            tracer = LangChainTracer()
            self.callbacks.append(tracer)
        
        # Initialize AI models
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT,
                openai_api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                temperature=0,
                callbacks=self.callbacks
            )
            
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
                openai_api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY
            )
            print("✓ AI models initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI models: {str(e)}")

        # Initialize vector store
        try:
            self.vectorstore = FAISS.from_texts(
                ["Financial analysis system initialized"],
                self.embeddings
            )
            print("✓ Vector store initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

        # Initialize agents in dependency order
        self.data_processor = self._create_data_processor()
        self.analyst = self._create_analyst()
        self.risk_assessor = self._create_risk_assessor()
        self.report_generator = self._create_report_generator()
        self.coordinator = self._create_coordinator()
        print("✓ Specialist agents initialized")
        
        print("✓ System ready\n")

    def _load_csv_data(self, file_path: str) -> Dict:
        """Load and preprocess CSV data."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            debug_print("Loaded CSV rows", rows)

            # Add to vector store
            texts = [json.dumps(row) for row in rows]
            self.vectorstore.add_texts(texts)
            
            return {
                "data": {
                    "rows": rows,
                    "summary": [f"Loaded {len(rows)} records"]
                },
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

    def _create_analyst(self) -> AgentExecutor:
        """Create the financial analyst agent."""
        tools = [
            Tool(
                name="calculate_metrics",
                description="Calculate financial metrics",
                func=self._calculate_metrics,
                return_direct=False
            ),
            Tool(
                name="search_data",
                description="Search through loaded data",
                func=lambda q: [{"content": d.page_content} for d in self.vectorstore.similarity_search(q)],
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Financial Analyst.
Analyze data and identify patterns.
Return analysis in format:
{
    "patterns": ["identified patterns"],
    "metrics": {"metric": "value"},
    "insights": ["key findings"]
}"""),
            ("human", "{input}"),
            ("human", "Analysis steps: {agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            ),
            tools=tools,
            max_iterations=3,
            handle_parsing_errors=True
        )

    @tool
    def _calculate_metrics(self, input_data: Any) -> Dict:
        """Calculate financial metrics."""
        try:
            debug_print("Raw metrics input", input_data)
            
            # Parse input if needed
            if isinstance(input_data, str):
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError:
                    return {
                        "error": "Invalid JSON input",
                        "status": "failed"
                    }
            else:
                data = input_data

            if not data:
                return {
                    "error": "Empty input data",
                    "status": "failed"
                }

            # Extract rows from nested structure
            rows = []
            if isinstance(data, dict):
                if "data" in data:
                    data_content = data["data"]
                    if isinstance(data_content, str):
                        data_content = json.loads(data_content)
                    if isinstance(data_content, dict):
                        rows = data_content.get("rows", [])
                    else:
                        rows = [data_content]
                else:
                    rows = data.get("rows", [data])
            elif isinstance(data, list):
                rows = data
            else:
                return {
                    "error": f"Invalid data structure: {type(data)}",
                    "status": "failed"
                }

            debug_print("Processing rows", rows)

            # Calculate metrics
            amounts = []
            types = {}
            
            for row in rows:
                try:
                    amount = float(row.get("amount", "0").replace(",", ""))
                    amounts.append(amount)
                    
                    trans_type = row.get("type", "unknown")
                    types[trans_type] = types.get(trans_type, 0) + 1
                except (ValueError, TypeError):
                    debug_print("Warning: Could not process row", row)
                    continue

            if not amounts:
                return {
                    "error": "No valid amounts found in data",
                    "status": "failed"
                }

            result = {
                "metrics": {
                    "total": sum(amounts),
                    "average": sum(amounts) / len(amounts),
                    "count": len(amounts),
                    "min": min(amounts),
                    "max": max(amounts),
                    "types": types
                },
                "status": "success"
            }

            debug_print("Calculated metrics", result)
            return result

        except Exception as e:
            debug_print("Error in calculate_metrics", str(e))
            return {
                "error": str(e),
                "status": "failed"
            }

    def _create_coordinator(self) -> AgentExecutor:
        """Create the coordinator agent."""
        tools = [
            Tool(
                name="process_data",
                description="Load and process financial data from CSV",
                func=lambda x: self.data_processor.invoke({"input": json.dumps({"action": "load", "source": x})}),
                return_direct=False
            ),
            Tool(
                name="analyze_data",
                description="Perform financial analysis on processed data",
                func=lambda x: self.analyst.invoke({"input": json.dumps({"data": x})}),
                return_direct=False
            ),
            Tool(
                name="assess_risk",
                description="Evaluate financial risks from analysis",
                func=lambda x: self.risk_assessor.invoke({"input": json.dumps({"metrics": x})}),
                return_direct=False
            ),
            Tool(
                name="generate_report",
                description="Create analysis report with findings",
                func=lambda x: self.report_generator.invoke({"input": json.dumps({"results": x})}),
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Lead Financial Analysis Coordinator.
Your role is to coordinate the analysis process by:
1. Processing data first
2. Analyzing the processed data
3. Assessing risks if needed
4. Generating final reports

Execute these steps in sequence using the available tools.

Return structured analysis results as:
{
    "findings": ["detailed findings"],
    "metrics": {"metric_name": "value"},
    "recommendations": ["action items"],
    "confidence": 0.95
}"""),
            ("human", "{input}"),
            ("human", "Previous steps: {agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            ),
            tools=tools,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def _create_data_processor(self) -> AgentExecutor:
        """Create the data processing agent."""
        tools = [
            Tool(
                name="load_csv",
                description="Load CSV data file",
                func=self._load_csv_data,
                return_direct=False
            ),
            Tool(
                name="search_data",
                description="Search through loaded data",
                func=lambda q: [{"content": d.page_content} for d in self.vectorstore.similarity_search(q)],
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Data Processing Specialist.
Process financial data and return in structured format:
{
    "data": {
        "rows": [processed records],
        "summary": ["key points"]
    },
    "status": "success"
}"""),
            ("human", "{input}"),
            ("human", "Steps taken: {agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            ),
            tools=tools,
            max_iterations=3,
            handle_parsing_errors=True
        )

    def _create_risk_assessor(self) -> AgentExecutor:
        """Create the risk assessment agent."""
        tools = [
            Tool(
                name="assess_metrics",
                description="Evaluate risk metrics",
                func=self._assess_risk_metrics,
                return_direct=False
            ),
            Tool(
                name="search_context",
                description="Get historical risk data",
                func=lambda q: [{"content": d.page_content} for d in self.vectorstore.similarity_search(q)],
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Risk Assessment Specialist.
Evaluate risks and provide recommendations.
Return assessment in format:
{
    "risk_level": "high/medium/low",
    "factors": ["risk factors"],
    "mitigations": ["recommended actions"]
}"""),
            ("human", "{input}"),
            ("human", "Assessment steps: {agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            ),
            tools=tools,
            max_iterations=3,
            handle_parsing_errors=True
        )

    def _create_report_generator(self) -> AgentExecutor:
        """Create the report generation agent."""
        tools = [
            Tool(
                name="format_findings",
                description="Format analysis findings",
                func=self._format_findings,
                return_direct=False
            ),
            Tool(
                name="search_context",
                description="Get relevant context",
                func=lambda q: [{"content": d.page_content} for d in self.vectorstore.similarity_search(q)],
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Report Generation Specialist.
Create clear and actionable reports.
Return report in format:
{
    "summary": ["key points"],
    "details": ["full findings"],
    "actions": ["next steps"]
}"""),
            ("human", "{input}"),
            ("human", "Report steps: {agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            ),
            tools=tools,
            max_iterations=3,
            handle_parsing_errors=True
        )

    def analyze_data(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform comprehensive financial analysis."""
        try:
            with tracing_v2_enabled():
                print(f"\nStarting {request.analysis_type} analysis...")
                
                # Prepare input
                input_data = {
                    "type": request.analysis_type,
                    "source": request.data_source,
                    "parameters": request.parameters,
                    "query": request.user_query
                }

                debug_print("Analysis input", input_data)

                # Run analysis
                result = self.coordinator.invoke({
                    "input": json.dumps(input_data)
                })

                debug_print("Analysis result", result)

                # Process results
                if isinstance(result, dict) and result.get("output"):
                    try:
                        output = json.loads(result["output"]) if isinstance(result["output"], str) else result["output"]
                        return AnalysisResult(
                            type=request.analysis_type,
                            findings=output.get("findings", []),
                            metrics=output.get("metrics", {}),
                            recommendations=output.get("recommendations", []),
                            confidence=output.get("confidence", 0.0)
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to parse analysis output: {str(e)}")
                else:
                    raise ValueError("No analysis output produced")

        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise

def demonstrate_analysis():
    """Demonstrate the multi-agent analysis system."""
    print("\nMulti-Agent Financial Analysis Demo")
    print("===================================")
    
    request = AnalysisRequest(
        analysis_type=AnalysisType.TREND, 
        data_source="test_financial_data.csv", 
        parameters={"time_period": "1Y", "metrics": ["daily_volume", "transaction_patterns", "growth_rate"]},
        user_query="Analyze growth trends and identify potential risks")
    
    print("Request created, initializing analyzer...")
    
    try:
        analyzer = MultiAgentAnalyzer()
        result = analyzer.analyze_data(request)
        print("\nFinancial Analysis Results:")
        print(json.dumps(result.dict(), indent=2))
    except Exception as e:
        print(f"\nDemo failed: {str(e)}")

if __name__ == "__main__":
    demonstrate_analysis()
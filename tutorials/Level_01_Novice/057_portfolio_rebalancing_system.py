#!/usr/bin/env python3
"""
LangChain Portfolio Rebalancing System Example (LangChain v3)

This example demonstrates how to create a sophisticated portfolio rebalancing system
using multiple coordinated agents with streaming capabilities. The system shows how
to orchestrate different specialized agents for market analysis, risk assessment,
and trade execution, all providing real-time updates.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import re
import json
from typing import List, Dict, Any, Generator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

class MarketData(BaseModel):
    """Schema for market data analysis."""
    symbol: str = Field(description="Asset symbol")
    current_price: float = Field(description="Current market price")
    trend: str = Field(description="Market trend analysis")
    volatility: float = Field(description="Volatility measure")
    volume: int = Field(description="Trading volume")
    timestamp: datetime = Field(default_factory=datetime.now)

class RiskMetrics(BaseModel):
    """Schema for risk assessment."""
    asset_risk: float = Field(description="Individual asset risk score")
    portfolio_impact: float = Field(description="Impact on portfolio risk")
    correlation: float = Field(description="Correlation with portfolio")
    var_ratio: float = Field(description="Value at Risk ratio")
    recommendation: str = Field(description="Risk-based recommendation")

class TradeDetails(BaseModel):
    """Schema for trade execution."""
    symbol: str = Field(description="Asset symbol")
    action: str = Field(description="Buy or sell")
    quantity: int = Field(description="Number of units")
    price: float = Field(description="Target price")
    urgency: str = Field(description="Trade urgency level")

class PortfolioUpdate(BaseModel):
    """Schema for portfolio updates."""
    asset_changes: List[Dict[str, Any]] = Field(description="Asset allocation changes")
    risk_profile: str = Field(description="Updated risk profile")
    rebalance_status: str = Field(description="Rebalancing status")
    execution_summary: List[str] = Field(description="Execution details")
    timestamp: datetime = Field(default_factory=datetime.now)

class StreamingCallback(BaseCallbackHandler):
    """Custom callback handler for streaming updates."""
    
    def __init__(self):
        self.updates = []
        self.current_agent = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Handle start of LLM operations."""
        agent_name = self.current_agent or "System"
        print(f"\n{agent_name} Analysis Started...")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process streaming tokens."""
        print(token, end="", flush=True)
        self.updates.append(token)
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Handle completion of LLM operations."""
        agent_name = self.current_agent or "System"
        print(f"\n{agent_name} Analysis Complete.")

def clean_json_str(json_str: str) -> str:
    """Clean up JSON string by removing formatting artifacts."""
    # Remove markdown code block markers
    json_str = re.sub(r'```json\s*|\s*```', '', json_str)
    # Remove escaped quotes
    json_str = json_str.replace('\\"', '"')
    # Remove redundant newlines and spaces
    json_str = re.sub(r'\s+', ' ', json_str).strip()
    return json_str

def format_json_output(data: Any) -> str:
    """Format JSON data for clean display."""
    if isinstance(data, str):
        try:
            data = json.loads(clean_json_str(data))
        except:
            return data
    return json.dumps(data, indent=2)

def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON string with error handling."""
    try:
        return json.loads(clean_json_str(json_str))
    except json.JSONDecodeError as e:
        print(f"\nWarning: JSON parsing error - {str(e)}")
        print("Original string:", json_str)
        return {}

def create_chat_model(streaming: bool = False) -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,  # Low temperature for consistent analysis
        streaming=streaming  # Enable streaming for real-time updates
    )

def create_market_analysis_agent() -> AgentExecutor:
    """Create an agent for market analysis."""
    prompt = PromptTemplate(
        template="""You are an expert market analyst specializing in real-time market data analysis.

Current Portfolio Context:
{portfolio}

Analyze the following market conditions:
{market_data}

{agent_scratchpad}

Your task is to analyze the market data and provide recommendations.

Tools available:
- get_market_data: Get real-time market data for a symbol

Return a clean, well-formatted JSON response with this structure:
{{{{
    "analysis": {{{{
        "market_conditions": "<summary>",
        "opportunities": ["<opportunity>"],
        "risks": ["<risk>"],
        "recommendations": ["<action>"]
    }}}}
}}}}""",
        input_variables=["portfolio", "market_data", "agent_scratchpad"]
    )
    
    tools = [
        Tool(
            name="get_market_data",
            func=lambda x: f"Market data simulation for {x}: Price trending up, volume stable",
            description="Get real-time market data for a symbol"
        )
    ]
    
    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_functions_agent(
            llm=create_chat_model(streaming=True),
            prompt=prompt,
            tools=tools
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def create_risk_assessment_agent() -> AgentExecutor:
    """Create an agent for risk assessment."""
    prompt = PromptTemplate(
        template="""You are an expert risk analyst specializing in portfolio risk assessment.

Portfolio State:
{portfolio}

Market Analysis:
{market_analysis}

{agent_scratchpad}

Follow these steps exactly:
1. Call calculate_risk_metrics ONCE to get base metrics
2. Process the metrics result
3. Return a final assessment in the exact JSON format shown below

Return your response in this exact JSON structure:
{{{{
    "risk_assessment": {{{{
        "overall_risk": "HIGH|MODERATE|LOW",
        "risk_factors": [
            "Detailed risk factor description",
            "Another risk factor"
        ],
        "mitigation_strategies": [
            "Specific strategy to mitigate risks",
            "Another strategy"
        ],
        "recommendations": [
            "Actionable recommendation",
            "Another recommendation"
        ]
    }}}}
}}}}

Important:
- Use calculate_risk_metrics only ONCE
- Ensure response is in valid JSON format
- Include specific, actionable recommendations""",
        input_variables=["portfolio", "market_analysis", "agent_scratchpad"]
    )
    
    def calculate_risk_metrics(portfolio_data: str) -> str:
        """Calculate risk metrics with enhanced output."""
        metrics = {
            "risk_level": "MODERATE",
            "portfolio_metrics": {
                "concentration_risk": "Medium - Tech sector dominant",
                "volatility": "Within acceptable range",
                "correlation": "High inter-asset correlation"
            },
            "recommendations": [
                "Consider sector diversification",
                "Monitor tech sector exposure",
                "Review asset correlations"
            ]
        }
        return json.dumps(metrics)
    
    tools = [
        Tool(
            name="calculate_risk_metrics",
            func=calculate_risk_metrics,
            description="Calculate comprehensive risk metrics for the portfolio. Use this tool only once."
        )
    ]
    
    agent = create_openai_functions_agent(
        llm=create_chat_model(streaming=True),
        prompt=prompt,
        tools=tools
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2  # Strict limit to prevent loops
    )

def create_trading_agent() -> AgentExecutor:
    """Create an agent for trade execution."""
    prompt = PromptTemplate(
        template="""You are an expert trading agent specializing in efficient trade execution.

Trade Requirements:
{trade_requirements}

Market Conditions:
{market_conditions}

{agent_scratchpad}

Your task is to plan and simulate trade execution.

Tools available:
- execute_trade: Execute a trade order

Return a clean, well-formatted JSON response with this structure:
{{{{
    "execution_plan": {{{{
        "trades": [{{{{
            "symbol": "<symbol>",
            "action": "BUY|SELL",
            "quantity": <number>,
            "strategy": "<approach>"
        }}}}],
        "timing": "<execution timing>",
        "contingencies": ["<contingency>"]
    }}}}
}}}}""",
        input_variables=["trade_requirements", "market_conditions", "agent_scratchpad"]
    )
    
    tools = [
        Tool(
            name="execute_trade",
            func=lambda x: f"Trade simulation for {x}: Order executed successfully",
            description="Execute a trade order"
        )
    ]
    
    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_functions_agent(
            llm=create_chat_model(streaming=True),
            prompt=prompt,
            tools=tools
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def create_orchestrator_agent() -> RunnableLambda:
    """Create the main orchestrator agent."""
    def orchestrate(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the portfolio rebalancing process."""
        try:
            # Initialize agents
            market_agent = create_market_analysis_agent()
            risk_agent = create_risk_assessment_agent()
            trading_agent = create_trading_agent()
            
            # Step 1: Market Analysis
            print("\nPerforming Market Analysis...")
            market_result = market_agent.invoke({
                "input": "Analyze market conditions",
                "portfolio": json.dumps(inputs["portfolio"]),
                "market_data": json.dumps(inputs["market_data"])
            })
            market_analysis = safe_json_loads(market_result.get("output", "{}"))
            
            # Step 2: Risk Assessment
            print("\nPerforming Risk Assessment...")
            risk_result = risk_agent.invoke({
                "input": "Assess portfolio risks",
                "portfolio": json.dumps(inputs["portfolio"]),
                "market_analysis": json.dumps(market_analysis)
            })
            
            risk_assessment = {}
            if risk_result and "output" in risk_result:
                try:
                    risk_assessment = safe_json_loads(risk_result["output"])
                except Exception as e:
                    print(f"\nWarning: Risk assessment parsing error - {str(e)}")
                    risk_assessment = {
                        "risk_assessment": {
                            "overall_risk": "MODERATE",
                            "risk_factors": ["Unable to parse detailed risk factors"],
                            "mitigation_strategies": ["Review portfolio manually"],
                            "recommendations": ["Consult risk management team"]
                        }
                    }
            
            # Step 3: Trade Execution
            try:
                risk_level = risk_assessment["risk_assessment"]["overall_risk"]
            except:
                risk_level = "MODERATE"
                
            if risk_level != "HIGH":
                print("\nPlanning Trade Execution...")
                trade_result = trading_agent.invoke({
                    "input": "Plan trade execution",
                    "trade_requirements": json.dumps(inputs["requirements"]),
                    "market_conditions": json.dumps(market_analysis)
                })
                trade_plan = safe_json_loads(trade_result.get("output", "{}"))
                
                status = "EXECUTED"
                execution = trade_plan
            else:
                status = "HALTED"
                execution = {"reason": "Risk level too high"}
            
            return {
                "status": status,
                "market_analysis": market_analysis,
                "risk_assessment": risk_assessment,
                "execution": execution
            }
            
        except Exception as e:
            print(f"\nOrchestration error: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "market_analysis": {},
                "risk_assessment": {},
                "execution": {"reason": "Error during processing"}
            }
    
    return RunnableLambda(orchestrate)

def demonstrate_portfolio_rebalancing():
    """Demonstrate the Portfolio Rebalancing System capabilities."""
    try:
        print("\nInitializing Portfolio Rebalancing System...\n")
        
        # Example portfolio data
        portfolio = {
            "assets": [
                {"symbol": "AAPL", "allocation": 0.25, "current_value": 25000},
                {"symbol": "GOOGL", "allocation": 0.25, "current_value": 25000},
                {"symbol": "MSFT", "allocation": 0.25, "current_value": 25000},
                {"symbol": "AMZN", "allocation": 0.25, "current_value": 25000}
            ],
            "total_value": 100000,
            "risk_profile": "MODERATE",
            "rebalance_threshold": 0.05
        }
        
        # Example market data
        market_data = {
            "AAPL": {"price": 180.5, "change": 0.015},
            "GOOGL": {"price": 140.2, "change": -0.02},
            "MSFT": {"price": 375.8, "change": 0.03},
            "AMZN": {"price": 175.4, "change": 0.01}
        }
        
        # Example rebalancing requirements
        requirements = {
            "target_allocations": {
                "AAPL": 0.30,
                "GOOGL": 0.20,
                "MSFT": 0.30,
                "AMZN": 0.20
            },
            "max_deviation": 0.05,
            "trade_urgency": "NORMAL"
        }
        
        print("Current Portfolio:")
        print(format_json_output(portfolio))
        print("\nProcessing Rebalancing Request...")
        
        # Create and run orchestrator
        orchestrator = create_orchestrator_agent()
        result = orchestrator.invoke({
            "portfolio": portfolio,
            "market_data": market_data,
            "requirements": requirements
        })
        
        # Display results with clean formatting
        print("\nRebalancing Analysis Complete:")
        
        print("\nMarket Analysis:")
        print(format_json_output(result["market_analysis"]))
        
        print("\nRisk Assessment:")
        print(format_json_output(result["risk_assessment"]))
        
        print("\nExecution Results:")
        print(format_json_output(result["execution"]))
        
        print(f"\nFinal Status: {result['status']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Portfolio Rebalancing System...")
    demonstrate_portfolio_rebalancing()

if __name__ == "__main__":
    main()
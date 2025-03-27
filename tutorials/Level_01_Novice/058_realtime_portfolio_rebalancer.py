#!/usr/bin/env python3
"""
LangChain Real-Time Portfolio Rebalancer Example (LangChain v3)

This example enhances the Portfolio Rebalancing System by incorporating real-time
market data from multiple providers (yfinance, wallstreet, realtime-stock).
It demonstrates provider failover, data validation, and real-time market analysis
capabilities while maintaining the multi-agent architecture.

Requirements:
- yfinance: Python module for market data from Yahoo Finance
- wallstreet: Python module for real-time stock data from Google Finance
- realtime-stock: Python module for real-time stock data scraping

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import re
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import yfinance as yf
from wallstreet import Stock
from rtstock.stock import Stock as RTStock

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
    price: float = Field(description="Current market price")
    change: float = Field(description="Price change percentage")
    volume: int = Field(description="Trading volume")
    provider: str = Field(description="Data provider name")
    timestamp: str = Field(description="Data timestamp")

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
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MarketDataProvider:
    """Handles market data retrieval with failover between providers."""
    
    @staticmethod
    def get_yfinance_data(symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data from yfinance."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            
            if hist.empty:
                return None
                
            return {
                "price": float(hist['Close'].iloc[-1]),
                "change": float((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]),
                "volume": int(hist['Volume'].iloc[-1]),
                "provider": "yfinance"
            }
        except Exception as e:
            print(f"yfinance error for {symbol}: {str(e)}")
            return None

    @staticmethod
    def get_wallstreet_data(symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data from wallstreet."""
        try:
            stock = Stock(symbol)
            
            # Fetch basic data
            data = {
                "price": float(stock.price),
                "change": float(stock.change),
                "volume": int(stock.volume),
                "provider": "wallstreet",
                "exchange": stock.exchange,
                "last_trade": stock.last_trade
            }
            
            # Calculate percentage change
            data["change_percent"] = (data["change"] / (data["price"] - data["change"])) * 100
            
            # Get additional historical context if available
            try:
                import pandas as pd
                hist_data = stock.historical(days_back=5, frequency='d')
                if not hist_data.empty:
                    data.update({
                        "5d_high": float(hist_data['High'].max()),
                        "5d_low": float(hist_data['Low'].min()),
                        "5d_avg_volume": int(hist_data['Volume'].mean())
                    })
            except:
                pass  # Historical data is optional
            
            return data
            
        except AttributeError as e:
            print(f"wallstreet error - missing attribute for {symbol}: {str(e)}")
            return None
        except ValueError as e:
            print(f"wallstreet error - invalid value for {symbol}: {str(e)}")
            return None
        except Exception as e:
            print(f"wallstreet error for {symbol}: {str(e)}")
            return None

    @staticmethod
    def get_realtime_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data from realtime-stock."""
        try:
            stock = RTStock(symbol)
            latest = stock.get_latest_price()
            
            if not latest or 'LastTradePriceOnly' not in latest:
                return None
            
            # Get additional info for better analysis
            info = stock.get_info()
            
            current_price = float(latest['LastTradePriceOnly'])
            
            # Calculate change if possible
            try:
                prev_price = float(info.get('PreviousClose', current_price))
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
            except:
                change = 0.0
                change_percent = 0.0
            
            return {
                "price": current_price,
                "change": float(change),
                "change_percent": float(change_percent),
                "volume": int(info.get('Volume', 0)),
                "trade_time": latest.get('LastTradeTime', 'N/A'),
                "provider": "realtime-stock"
            }
        except Exception as e:
            print(f"realtime-stock error for {symbol}: {str(e)}")
            return None

    @classmethod
    def get_market_data(cls, symbol: str) -> Dict[str, Any]:
        """Get market data with provider failover."""
        data = None
        errors = []
        
        # Try each provider in sequence with delay between attempts
        for provider_method in [cls.get_yfinance_data, cls.get_wallstreet_data, cls.get_realtime_stock_data]:
            try:
                data = provider_method(symbol)
                if data:
                    # Add standard fields if missing
                    if "change_percent" not in data and "change" in data and "price" in data:
                        try:
                            base_price = data["price"] - data["change"]
                            data["change_percent"] = (data["change"] / base_price) * 100
                        except:
                            data["change_percent"] = 0.0
                    
                    break
                time.sleep(0.5)  # Delay between providers
            except Exception as e:
                errors.append(f"{provider_method.__name__} error: {str(e)}")
        
        if not data:
            error_msg = "; ".join(errors)
            raise ValueError(f"Unable to fetch market data for {symbol}: {error_msg}")
        
        return {
            **data,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }

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

def serialize_numpy(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def clean_json_str(json_str: str) -> str:
    """Clean up JSON string by removing formatting artifacts."""
    json_str = re.sub(r'```json\s*|\s*```', '', json_str)
    json_str = json_str.replace('\\"', '"')
    json_str = re.sub(r'\s+', ' ', json_str).strip()
    return json_str

def format_json_output(data: Any) -> str:
    """Format JSON data for clean display."""
    if isinstance(data, str):
        try:
            data = json.loads(clean_json_str(data))
        except:
            return data
    return json.dumps(data, indent=2, default=serialize_numpy)

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
        temperature=0,
        streaming=streaming
    )

def fetch_market_data(symbol: str) -> Dict[str, Any]:
    """Fetch real-time market data with error handling."""
    try:
        data = MarketDataProvider.get_market_data(symbol)
        # Convert numpy types to Python native types
        return json.loads(json.dumps(data, default=serialize_numpy))
    except Exception as e:
        print(f"Error fetching market data for {symbol}: {str(e)}")
        return {
            "price": 0.0,
            "change": 0.0,
            "volume": 0,
            "provider": "fallback",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "error": str(e)
        }

def create_market_analysis_agent() -> AgentExecutor:
    """Create an agent for market analysis."""
    prompt = PromptTemplate(
        template="""You are an expert market analyst specializing in real-time market data analysis.

Current Portfolio Context:
{portfolio}

Analyze the following market conditions:
{market_data}

{agent_scratchpad}

Your task is to analyze the real-time market data and provide recommendations.

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
            func=fetch_market_data,
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
        max_iterations=2
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

Your task is to plan and simulate trade execution using real-time market data.

Tools available:
- execute_trade: Execute a trade order
- get_market_data: Get real-time market data for a symbol

Return a clean, well-formatted JSON response with this structure:
{{{{
    "execution_plan": {{{{
        "trades": [{{{{
            "symbol": "<symbol>",
            "action": "BUY|SELL",
            "quantity": <number>,
            "strategy": "<approach>",
            "limit_price": <price>
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
        ),
        Tool(
            name="get_market_data",
            func=fetch_market_data,
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

def create_orchestrator_agent() -> RunnableLambda:
    """Create the main orchestrator agent."""
    def orchestrate(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the portfolio rebalancing process."""
        # Initialize response with default error structure
        response = {
            "status": "ERROR",
            "market_analysis": {},
            "risk_assessment": {},
            "execution": {"reason": "Error during processing"},
            "metadata": {
                "providers": {},
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            # Initialize agents
            market_agent = create_market_analysis_agent()
            risk_agent = create_risk_assessment_agent()
            trading_agent = create_trading_agent()
            
            # Fetch real-time market data for all assets
            portfolio_symbols = [
                asset["symbol"] for asset in inputs["portfolio"]["assets"]
            ]
            
            market_data = {}
            providers_used = {}
            
            for symbol in portfolio_symbols:
                try:
                    data = fetch_market_data(symbol)
                    market_data[symbol] = data
                    providers_used[symbol] = data.get("provider", "unknown")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"\nError fetching data for {symbol}: {str(e)}")
                    providers_used[symbol] = "failed"
            
            # Update metadata
            response["metadata"]["providers"] = providers_used
            
            if not market_data:
                raise ValueError("Failed to fetch market data for any symbol")
            
            # Convert data to JSON-serializable format
            market_data = json.loads(json.dumps(market_data, default=serialize_numpy))
            
            # Step 1: Market Analysis
            print("\nPerforming Market Analysis...")
            market_result = market_agent.invoke({
                "input": "Analyze current market conditions",
                "portfolio": json.dumps(inputs["portfolio"]),
                "market_data": json.dumps(market_data)
            })
            market_analysis = safe_json_loads(market_result.get("output", "{}"))
            response["market_analysis"] = market_analysis
            
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
            response["risk_assessment"] = risk_assessment
            
            # Step 3: Trade Execution
            try:
                risk_level = risk_assessment["risk_assessment"]["overall_risk"]
            except:
                risk_level = "MODERATE"
                
            if risk_level != "HIGH":
                print("\nPlanning Trade Execution...")
                trade_result = trading_agent.invoke({
                    "input": "Plan trade execution with current market data",
                    "trade_requirements": json.dumps(inputs["requirements"]),
                    "market_conditions": json.dumps({
                        "analysis": market_analysis,
                        "current_data": market_data
                    })
                })
                trade_plan = safe_json_loads(trade_result.get("output", "{}"))
                
                response.update({
                    "status": "EXECUTED",
                    "execution": trade_plan
                })
            else:
                response.update({
                    "status": "HALTED",
                    "execution": {"reason": "Risk level too high"}
                })
            
            return response
            
        except Exception as e:
            print(f"\nOrchestration error: {str(e)}")
            response["error"] = str(e)
            return response
    
    return RunnableLambda(orchestrate)

def demonstrate_portfolio_rebalancing():
    """Demonstrate the Portfolio Rebalancing System capabilities."""
    try:
        print("\nInitializing Real-Time Portfolio Rebalancing System...\n")
        
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
        print("\nProcessing Real-Time Rebalancing Request...")
        
        # Create and run orchestrator
        orchestrator = create_orchestrator_agent()
        result = orchestrator.invoke({
            "portfolio": portfolio,
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
        
        print("\nData Providers Used:")
        print(format_json_output(result.get("metadata", {}).get("providers", {})))
        
        print(f"\nFinal Status: {result['status']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Real-Time Portfolio Rebalancer...")
    demonstrate_portfolio_rebalancing()

if __name__ == "__main__":
    main()
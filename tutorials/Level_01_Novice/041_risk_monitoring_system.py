"""
LangChain Risk Monitoring System Example

This example demonstrates how to combine tool calling and streaming capabilities
to create a system that can calculate risk metrics and provide real-time risk
monitoring updates.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

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

class RiskMetrics(BaseModel):
    """Schema for risk metrics."""
    volatility: float = Field(description="Price volatility")
    value_at_risk: float = Field(description="Value at Risk (VaR)")
    sharpe_ratio: float = Field(description="Risk-adjusted return metric")
    correlation: float = Field(description="Correlation with market")
    beta: float = Field(description="Market sensitivity")
    max_drawdown: float = Field(description="Maximum historical loss")

class PositionData(BaseModel):
    """Schema for position information."""
    asset_id: str = Field(description="Asset identifier")
    quantity: float = Field(description="Position size")
    entry_price: float = Field(description="Average entry price")
    current_price: float = Field(description="Current market price")
    unrealized_pnl: float = Field(description="Unrealized profit/loss")
    market_value: float = Field(description="Current position value")

class RiskAssessment(BaseModel):
    """Schema for comprehensive risk assessment."""
    position: PositionData = Field(description="Position details")
    metrics: RiskMetrics = Field(description="Risk metrics")
    risk_level: str = Field(description="Overall risk level")
    risk_factors: List[str] = Field(description="Contributing risk factors")
    alerts: List[str] = Field(description="Risk alerts and warnings")
    recommendations: List[str] = Field(description="Risk management suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "position": {
                    "asset_id": "AAPL",
                    "quantity": 100,
                    "entry_price": 150.00,
                    "current_price": 155.00,
                    "unrealized_pnl": 500.00,
                    "market_value": 15500.00
                },
                "metrics": {
                    "volatility": 15.5,
                    "value_at_risk": 1250.00,
                    "sharpe_ratio": 1.8,
                    "correlation": 0.65,
                    "beta": 1.2,
                    "max_drawdown": 8.5
                },
                "risk_level": "Moderate",
                "risk_factors": [
                    "High volatility",
                    "Elevated VaR"
                ],
                "alerts": [
                    "Monitor volatility",
                    "Large position size"
                ],
                "recommendations": [
                    "Reduce position",
                    "Set stop-loss"
                ],
                "timestamp": "2025-03-19T01:06:36.100849"
            }]
        }
    }

def create_chat_model() -> AzureChatOpenAI:
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

async def monitor_risk_stream(
    asset_id: str,
    calculator: Any,  # Assume calculator is an instance that provides async risk calculation.
    update_interval: float = 1.0
) -> AsyncIterator[RiskAssessment]:
    """
    Generate a stream of risk assessments.
    
    Args:
        asset_id: Asset to monitor.
        calculator: Risk calculation tool (could be a function or async method).
        update_interval: Time between updates in seconds.
        
    Yields:
        RiskAssessment: Updated risk assessment.
    """
    base_price = 150.00
    while True:
        # Simulate price movement.
        price_change = (hash(datetime.now().isoformat()) % 100 - 50) / 100
        current_price = base_price * (1 + price_change)
        
        # Create position data.
        position_data = {
            "asset_id": asset_id,
            "quantity": 100,
            "entry_price": base_price,
            "current_price": current_price,
            "unrealized_pnl": (current_price - base_price) * 100,
            "market_value": current_price * 100
        }
        
        # Calculate risk metrics using the calculator (assumed synchronous here).
        metrics = calculator._run(position_data)
        
        # Build alerts, filtering out any None values.
        alerts = [alert for alert in [
            "Monitor volatility" if metrics["volatility"] > 15 else None,
            "Large position size" if position_data["market_value"] > 15000 else None
        ] if alert is not None]
        
        assessment = RiskAssessment(
            position=PositionData(**position_data),
            metrics=RiskMetrics(**metrics),
            risk_level="High" if metrics["volatility"] > 15 else "Moderate" if metrics["volatility"] > 10 else "Low",
            risk_factors=[
                "High volatility" if metrics["volatility"] > 15 else "Normal volatility",
                "Elevated VaR" if metrics["value_at_risk"] > 1000 else "Acceptable VaR"
            ],
            alerts=alerts,
            recommendations=[
                "Reduce position" if metrics["volatility"] > 15 else "Maintain position",
                "Set stop-loss" if metrics["max_drawdown"] > 10 else "Monitor closely"
            ]
        )
        
        yield assessment
        await asyncio.sleep(update_interval)

async def demonstrate_risk_monitoring():
    """Demonstrate risk monitoring capabilities."""
    try:
        print("\nDemonstrating Risk Monitoring System...\n")
        
        # Initialize a dummy risk calculator (using RiskCalculator from previous examples).
        # Here, we'll create a simple calculator instance.
        class RiskCalculator(BaseTool):
            name: str = "risk_calculator"
            description: str = "Calculate risk metrics for a given position"
            
            def _run(self, position_data: dict) -> dict:
                price = position_data.get("current_price", 0)
                entry = position_data.get("entry_price", 0)
                quantity = position_data.get("quantity", 0)
                volatility = abs((price - entry) / entry * 100)
                market_value = price * quantity
                var_95 = market_value * 0.05
                return {
                    "volatility": round(volatility, 2),
                    "value_at_risk": round(var_95, 2),
                    "sharpe_ratio": 1.8,
                    "correlation": 0.65,
                    "beta": 1.2,
                    "max_drawdown": 8.5
                }
            
            async def _arun(self, position_data: dict) -> dict:
                return self._run(position_data)
        
        calculator = RiskCalculator()
        
        print("Example 1: Single Position Monitoring")
        print("-" * 50)
        
        count = 0
        async for assessment in monitor_risk_stream("AAPL", calculator):
            print(f"\nUpdate {count + 1}:")
            print(f"Asset: {assessment.position.asset_id}")
            print(f"Price: ${assessment.position.current_price:.2f}")
            print(f"P&L: ${assessment.position.unrealized_pnl:.2f}")
            print("\nRisk Metrics:")
            print(f"Volatility: {assessment.metrics.volatility:.1f}%")
            print(f"VaR: ${assessment.metrics.value_at_risk:.2f}")
            print(f"Risk Level: {assessment.risk_level}")
            print("\nAlerts:")
            for alert in assessment.alerts:
                print(f"- {alert}")
            count += 1
            if count >= 5:  # Show 5 updates
                break
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Risk Monitoring System...")
    asyncio.run(demonstrate_risk_monitoring())

if __name__ == "__main__":
    main()

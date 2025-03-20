"""
LangChain Trading Strategy Evaluator Example

This example demonstrates how to combine evaluation and structured output parsing
to create a system that can assess trading strategies and provide detailed,
formatted performance analysis.

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

class TradeMetrics(BaseModel):
    """Schema for trade performance metrics."""
    win_rate: float = Field(description="Percentage of winning trades")
    profit_factor: float = Field(description="Ratio of gross profits to gross losses")
    max_drawdown: float = Field(description="Maximum peak to trough decline")
    avg_win: float = Field(description="Average profit on winning trades")
    avg_loss: float = Field(description="Average loss on losing trades")
    risk_reward_ratio: float = Field(description="Ratio of average win to average loss")

class StrategyAnalysis(BaseModel):
    """Schema for comprehensive strategy analysis."""
    strategy_name: str = Field(description="Name of the trading strategy")
    time_period: str = Field(description="Analysis time period")
    metrics: TradeMetrics = Field(description="Performance metrics")
    strengths: List[str] = Field(description="Strategy strengths")
    weaknesses: List[str] = Field(description="Strategy weaknesses")
    risk_assessment: str = Field(description="Risk level assessment")
    recommendations: List[str] = Field(description="Improvement recommendations")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "strategy_name": "Moving Average Crossover",
                "time_period": "Jan 2024 - Mar 2024",
                "metrics": {
                    "win_rate": 65.5,
                    "profit_factor": 1.8,
                    "max_drawdown": 12.5,
                    "avg_win": 250.0,
                    "avg_loss": 150.0,
                    "risk_reward_ratio": 1.67
                },
                "strengths": [
                    "Consistent performance in trending markets",
                    "Clear entry and exit signals",
                    "Good risk management"
                ],
                "weaknesses": [
                    "Underperforms in ranging markets",
                    "Multiple false signals in volatile conditions"
                ],
                "risk_assessment": "Medium",
                "recommendations": [
                    "Add trend filter for ranging markets",
                    "Implement volatility-based position sizing",
                    "Consider longer timeframe confirmation"
                ],
                "timestamp": "2024-03-31T12:00:00"
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

def evaluate_trading_strategy(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    strategy_data: Dict
) -> StrategyAnalysis:
    """
    Evaluate a trading strategy using the provided data.
    
    Args:
        chat_model: The chat model to use.
        parser: The output parser for structured analysis.
        strategy_data: Trading strategy performance data.
        
    Returns:
        StrategyAnalysis: Structured analysis of the strategy.
    """
    # Get the format instructions from the parser.
    format_instructions = parser.get_format_instructions()
    
    # Manually build the system message by concatenating the instructions.
    system_text = (
        "You are a trading strategy analyst. Evaluate the provided strategy data and provide a comprehensive analysis.\n"
        "Respond with a JSON object that exactly follows the schema below (do not include any extra text):\n\n"
        f"{format_instructions}\n"
    )
    
    system_msg = SystemMessage(content=system_text)
    human_msg = HumanMessage(content=f"Analyze this trading strategy: {json.dumps(strategy_data)}")
    
    # Use these messages as the prompt.
    messages = [system_msg, human_msg]
    response = chat_model.invoke(messages)
    return parser.parse(response.content)

def demonstrate_strategy_evaluation():
    """Demonstrate trading strategy evaluation capabilities."""
    try:
        print("\nDemonstrating Trading Strategy Evaluation...\n")
        
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=StrategyAnalysis)
        
        # Example 1: Moving Average Strategy
        print("Example 1: Moving Average Strategy Evaluation")
        print("-" * 50)
        
        ma_strategy_data = {
            "strategy_name": "Moving Average Crossover",
            "time_period": "Jan 2024 - Mar 2024",
            "trades": {
                "total_trades": 100,
                "winning_trades": 65,
                "losing_trades": 35,
                "total_profit": 16250,
                "total_loss": 9000,
                "largest_drawdown": 12.5
            },
            "market_conditions": [
                "Strong trends in January",
                "Increased volatility in February",
                "Range-bound in March"
            ]
        }
        
        ma_analysis = evaluate_trading_strategy(chat_model, parser, ma_strategy_data)
        
        print("\nStrategy Analysis:")
        print(f"Name: {ma_analysis.strategy_name}")
        print(f"Period: {ma_analysis.time_period}")
        print("\nPerformance Metrics:")
        print(f"Win Rate: {ma_analysis.metrics.win_rate:.1f}%")
        print(f"Profit Factor: {ma_analysis.metrics.profit_factor:.2f}")
        print(f"Max Drawdown: {ma_analysis.metrics.max_drawdown:.1f}%")
        print(f"Risk/Reward: {ma_analysis.metrics.risk_reward_ratio:.2f}")
        print("\nStrengths:")
        for strength in ma_analysis.strengths:
            print(f"- {strength}")
        print("\nWeaknesses:")
        for weakness in ma_analysis.weaknesses:
            print(f"- {weakness}")
        print(f"\nRisk Assessment: {ma_analysis.risk_assessment}")
        print("\nRecommendations:")
        for rec in ma_analysis.recommendations:
            print(f"- {rec}")
        
        # Example 2: Mean Reversion Strategy
        print("\nExample 2: Mean Reversion Strategy Evaluation")
        print("-" * 50)
        
        mr_strategy_data = {
            "strategy_name": "Mean Reversion RSI",
            "time_period": "Jan 2024 - Mar 2024",
            "trades": {
                "total_trades": 80,
                "winning_trades": 48,
                "losing_trades": 32,
                "total_profit": 12000,
                "total_loss": 8000,
                "largest_drawdown": 15.0
            },
            "market_conditions": [
                "High volatility periods",
                "Multiple market reversals",
                "Some trending periods"
            ]
        }
        
        mr_analysis = evaluate_trading_strategy(chat_model, parser, mr_strategy_data)
        
        print("\nStrategy Analysis:")
        print(f"Name: {mr_analysis.strategy_name}")
        print(f"Period: {mr_analysis.time_period}")
        print("\nPerformance Metrics:")
        print(f"Win Rate: {mr_analysis.metrics.win_rate:.1f}%")
        print(f"Profit Factor: {mr_analysis.metrics.profit_factor:.2f}")
        print(f"Max Drawdown: {mr_analysis.metrics.max_drawdown:.1f}%")
        print(f"Risk/Reward: {mr_analysis.metrics.risk_reward_ratio:.2f}")
        print("\nStrengths:")
        for strength in mr_analysis.strengths:
            print(f"- {strength}")
        print("\nWeaknesses:")
        for weakness in mr_analysis.weaknesses:
            print(f"- {weakness}")
        print(f"\nRisk Assessment: {mr_analysis.risk_assessment}")
        print("\nRecommendations:")
        for rec in mr_analysis.recommendations:
            print(f"- {rec}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Trading Strategy Evaluator...")
    demonstrate_strategy_evaluation()

if __name__ == "__main__":
    main()

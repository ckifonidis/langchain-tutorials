"""
LangChain Market Data Aggregator Example

This example demonstrates how to combine async programming and tool calling to create
a system that efficiently fetches and processes market data from multiple sources,
providing aggregated insights.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function

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

class MarketData(BaseModel):
    """Schema for market data."""
    symbol: str = Field(description="Asset symbol")
    price: float = Field(description="Current price")
    volume: int = Field(description="Trading volume")
    change_percent: float = Field(description="24h price change percentage")
    market_cap: float = Field(description="Market capitalization")
    timestamp: datetime = Field(default_factory=datetime.now)

class MarketMetrics(BaseModel):
    """Schema for calculated market metrics."""
    moving_average: float = Field(description="20-day moving average")
    volatility: float = Field(description="Historical volatility")
    rsi: float = Field(description="Relative Strength Index")
    volume_ma: float = Field(description="Volume moving average")
    support_level: float = Field(description="Calculated support level")
    resistance_level: float = Field(description="Calculated resistance level")

class MarketAnalysis(BaseModel):
    """Schema for market analysis results."""
    asset_data: MarketData = Field(description="Raw market data")
    metrics: MarketMetrics = Field(description="Calculated metrics")
    trends: List[str] = Field(description="Identified market trends")
    signals: List[str] = Field(description="Trading signals")
    risk_level: str = Field(description="Risk assessment")
    timestamp: datetime = Field(default_factory=datetime.now)

class DataFetcher(BaseTool):
    """Tool for fetching market data from different sources."""
    
    name: str = "market_data_fetcher"
    description: str = "Fetch market data for specified assets"
    
    async def _fetch_price_data(self, symbol: str) -> Dict[str, Any]:
        """Simulate fetching price data with delay."""
        await asyncio.sleep(1)  # Simulate API delay
        # Simulated market data (replace with actual API calls)
        data = {
            "BTC": {"price": 65000, "volume": 25000, "change": 2.5},
            "ETH": {"price": 3500, "volume": 15000, "change": 1.8},
            "AAPL": {"price": 175, "volume": 1000000, "change": 0.5},
            "GOOGL": {"price": 150, "volume": 750000, "change": -0.3}
        }
        return data.get(symbol, {"price": 0, "volume": 0, "change": 0})
    
    async def _fetch_market_cap(self, symbol: str) -> float:
        """Simulate fetching market cap data."""
        await asyncio.sleep(0.5)  # Simulate API delay
        # Simulated market cap data
        data = {
            "BTC": 1.2e12,
            "ETH": 4.2e11,
            "AAPL": 2.8e12,
            "GOOGL": 1.9e12
        }
        return data.get(symbol, 0)
    
    async def _arun(self, symbol: str) -> Dict[str, Any]:
        """
        Asynchronously fetch all market data.
        
        Args:
            symbol: Asset symbol to fetch data for
            
        Returns:
            dict: Collected market data
        """
        # Fetch data concurrently
        price_data, market_cap = await asyncio.gather(
            self._fetch_price_data(symbol),
            self._fetch_market_cap(symbol)
        )
        
        # Combine and format data
        return {
            "symbol": symbol,
            "price": price_data["price"],
            "volume": price_data["volume"],
            "change_percent": price_data["change"],
            "market_cap": market_cap,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        """Synchronous version that runs async code."""
        return asyncio.run(self._arun(symbol))

class MetricsCalculator(BaseTool):
    """Tool for calculating market metrics."""
    
    name: str = "metrics_calculator"
    description: str = "Calculate technical metrics from market data"
    
    def _run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate market metrics from data.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            dict: Calculated metrics
        """
        price = market_data.get("price", 0)
        
        # Simulate metric calculations (replace with actual calculations)
        return {
            "moving_average": price * 0.95,  # Simplified calculation
            "volatility": 15.5,  # Example value
            "rsi": 65.0,  # Example value
            "volume_ma": market_data.get("volume", 0) * 0.9,
            "support_level": price * 0.85,
            "resistance_level": price * 1.15
        }

async def analyze_market_data(
    symbols: List[str],
    data_fetcher: DataFetcher,
    metrics_calculator: MetricsCalculator
) -> List[MarketAnalysis]:
    """
    Analyze market data for multiple assets concurrently.
    
    Args:
        symbols: List of asset symbols to analyze
        data_fetcher: Tool for fetching market data
        metrics_calculator: Tool for calculating metrics
        
    Returns:
        List[MarketAnalysis]: Analysis results for each asset
    """
    async def analyze_symbol(symbol: str) -> MarketAnalysis:
        # Fetch market data
        market_data = await data_fetcher._arun(symbol)
        
        # Calculate metrics
        metrics = metrics_calculator._run(market_data)
        
        # Generate analysis
        return MarketAnalysis(
            asset_data=MarketData(**market_data),
            metrics=MarketMetrics(**metrics),
            trends=[
                "Upward trend" if market_data["change_percent"] > 0 else "Downward trend",
                "High volume" if market_data["volume"] > 500000 else "Normal volume"
            ],
            signals=[
                "Buy" if metrics["rsi"] < 30 else "Sell" if metrics["rsi"] > 70 else "Hold"
            ],
            risk_level="High" if metrics["volatility"] > 20 else "Medium" if metrics["volatility"] > 10 else "Low"
        )
    
    # Analyze all symbols concurrently
    return await asyncio.gather(*(analyze_symbol(symbol) for symbol in symbols))

async def demonstrate_market_analysis():
    """Demonstrate market data aggregation and analysis capabilities."""
    try:
        print("\nDemonstrating Market Data Analysis...\n")
        
        # Initialize tools
        data_fetcher = DataFetcher()
        metrics_calculator = MetricsCalculator()
        
        # Example 1: Analyze Crypto Assets
        print("Example 1: Cryptocurrency Market Analysis")
        print("-" * 50)
        
        crypto_symbols = ["BTC", "ETH"]
        crypto_analysis = await analyze_market_data(
            crypto_symbols, data_fetcher, metrics_calculator
        )
        
        for analysis in crypto_analysis:
            print(f"\nAnalysis for {analysis.asset_data.symbol}:")
            print(f"Price: ${analysis.asset_data.price:,.2f}")
            print(f"24h Change: {analysis.asset_data.change_percent:.1f}%")
            print(f"Volume: {analysis.asset_data.volume:,}")
            print(f"Market Cap: ${analysis.asset_data.market_cap:,.2f}")
            
            print("\nMetrics:")
            print(f"RSI: {analysis.metrics.rsi:.1f}")
            print(f"Volatility: {analysis.metrics.volatility:.1f}%")
            print(f"Support: ${analysis.metrics.support_level:,.2f}")
            print(f"Resistance: ${analysis.metrics.resistance_level:,.2f}")
            
            print("\nSignals:", ", ".join(analysis.signals))
            print("Risk Level:", analysis.risk_level)
        
        # Example 2: Analyze Stocks
        print("\nExample 2: Stock Market Analysis")
        print("-" * 50)
        
        stock_symbols = ["AAPL", "GOOGL"]
        stock_analysis = await analyze_market_data(
            stock_symbols, data_fetcher, metrics_calculator
        )
        
        for analysis in stock_analysis:
            print(f"\nAnalysis for {analysis.asset_data.symbol}:")
            print(f"Price: ${analysis.asset_data.price:.2f}")
            print(f"24h Change: {analysis.asset_data.change_percent:.1f}%")
            print(f"Volume: {analysis.asset_data.volume:,}")
            print(f"Market Cap: ${analysis.asset_data.market_cap:,.2f}")
            
            print("\nMetrics:")
            print(f"RSI: {analysis.metrics.rsi:.1f}")
            print(f"Volatility: {analysis.metrics.volatility:.1f}%")
            print(f"Support: ${analysis.metrics.support_level:.2f}")
            print(f"Resistance: ${analysis.metrics.resistance_level:.2f}")
            
            print("\nSignals:", ", ".join(analysis.signals))
            print("Risk Level:", analysis.risk_level)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Market Data Aggregator...")
    asyncio.run(demonstrate_market_analysis())

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Portfolio Optimizer (135) (LangChain v3)

This example demonstrates investment analysis using:
1. Streaming: Real-time analysis
2. Few Shot Learning: Strategy matching
3. Prompt Templates: Recommendation format

It helps investment teams optimize client portfolios.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AssetType(str, Enum):
    """Investment asset types."""
    STOCK = "stock"
    BOND = "bond"
    ETF = "etf"
    FUND = "mutual_fund"
    CASH = "cash_equivalent"
    CRYPTO = "cryptocurrency"

class RiskLevel(str, Enum):
    """Investment risk levels."""
    LOW = "conservative"
    MEDIUM = "moderate"
    HIGH = "aggressive"
    SPEC = "speculative"

class Portfolio(BaseModel):
    """Portfolio details."""
    portfolio_id: str = Field(description="Portfolio ID")
    client_name: str = Field(description="Client name")
    risk_level: RiskLevel = Field(description="Risk level")
    assets: Dict[str, float] = Field(description="Asset allocations")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    metadata: Dict = Field(default_factory=dict)

class PortfolioOptimizer:
    """Investment portfolio optimization system."""

    def __init__(self):
        """Initialize optimizer."""
        logger.info("Starting portfolio optimizer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
            streaming=True
        )
        logger.info("Chat model ready")
        
        # Example strategies
        self.examples = [
            [
                HumanMessage(content="High risk growth portfolio"),
                AIMessage(content="""PORTFOLIO STRATEGY
-----------------
Risk Level: AGGRESSIVE
Style: Growth Focus

Asset Mix:
- Stocks: 80%
- ETFs: 15%
- Cash: 5%

Recommendations:
1. Tech sector focus
2. Growth markets
3. Momentum plays

Risk Controls:
- Stop losses
- Position limits
- Regular rebalance""")
            ],
            [
                HumanMessage(content="Conservative income portfolio"),
                AIMessage(content="""PORTFOLIO STRATEGY
-----------------
Risk Level: CONSERVATIVE
Style: Income Focus

Asset Mix:
- Bonds: 60%
- Dividend Stocks: 30%
- Cash: 10%

Recommendations:
1. Quality bonds
2. Blue chips
3. High yield

Risk Controls:
- Duration limits
- Credit quality
- Sector limits""")
            ]
        ]
        
        # Setup analysis template
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are an investment portfolio analyst.
Review portfolios and suggest optimizations.

Base your analysis on these examples:
{examples}

Format your response exactly like this:

PORTFOLIO ANALYSIS
----------------
Client: Name
Risk Level: LEVEL
Style: Strategy type

Current Mix:
- Asset allocations
- Key metrics
- Performance data

Optimization Plan:
1. Change Name
   Current: State
   Target: Goal
   Reason: Explanation

2. Change Name
   Current: State
   Target: Goal
   Reason: Explanation

Implementation:
1. Action step
2. Action step

Risk Management:
- Control measure
- Control measure

Monitor:
- Metric to track
- Metric to track"""),
            ("human", """Analyze this portfolio:

Client: {client_name}
Risk Level: {risk_level}

Current Allocation:
{assets}

Performance Metrics:
{metrics}

Provide optimization recommendations.""")
        ])
        logger.info("Analysis template ready")
        
        # Setup output parser
        self.parser = StrOutputParser()

    async def analyze_portfolio(self, portfolio: Portfolio) -> AsyncGenerator[str, None]:
        """Analyze portfolio with streaming."""
        logger.info(f"Analyzing portfolio: {portfolio.portfolio_id}")
        
        try:
            # Format data
            assets = "\n".join(
                f"- {k}: {v:.1f}%" 
                for k, v in portfolio.assets.items()
            )
            
            metrics = "\n".join(
                f"- {k}: {v:.2f}" 
                for k, v in portfolio.metrics.items()
            )
            
            # Format request
            messages = self.template.format_messages(
                examples=self.examples,
                client_name=portfolio.client_name,
                risk_level=portfolio.risk_level.value,
                assets=assets,
                metrics=metrics
            )
            logger.debug("Request formatted")
            
            # Stream analysis
            async for chunk in self.llm.astream(messages):
                yield chunk.content
                
            logger.info("Analysis complete")
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting optimization demo...")
    
    try:
        # Create optimizer
        optimizer = PortfolioOptimizer()
        
        # Example portfolio
        portfolio = Portfolio(
            portfolio_id="PORT-2025-001",
            client_name="John Smith",
            risk_level=RiskLevel.HIGH,
            assets={
                "US_TECH": 35.0,
                "US_GROWTH": 25.0,
                "INTL_DEV": 15.0,
                "EMERG_MKT": 15.0,
                "CASH": 10.0
            },
            metrics={
                "returns_1y": 18.5,
                "volatility": 15.2,
                "sharpe": 1.05,
                "beta": 1.25,
                "alpha": 2.5,
                "max_drawdown": -12.5
            }
        )
        
        print("\nAnalyzing Portfolio")
        print("==================")
        print(f"Portfolio: {portfolio.portfolio_id}")
        print(f"Client: {portfolio.client_name}")
        print(f"Risk Level: {portfolio.risk_level.value}\n")
        
        print("Current Allocation:")
        for asset, weight in portfolio.assets.items():
            print(f"{asset}: {weight:.1f}%")
        
        print("\nPerformance Metrics:")
        for name, value in portfolio.metrics.items():
            if name in ["returns_1y", "alpha", "max_drawdown"]:
                print(f"{name}: {value:+.1f}%")
            else:
                print(f"{name}: {value:.2f}")
        
        try:
            print("\nAnalysis Results:")
            print("================")
            # Stream results
            async for chunk in optimizer.analyze_portfolio(portfolio):
                print(chunk, end="")
            print()
            
        except Exception as e:
            print(f"\nAnalysis failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
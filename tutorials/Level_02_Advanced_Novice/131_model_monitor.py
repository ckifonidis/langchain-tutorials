#!/usr/bin/env python3
"""
Model Monitor (131) (LangChain v3)

This example demonstrates AI model monitoring using:
1. Memory: Chat history tracking
2. Output Parsing: Structured analysis
3. Chat Models: Pattern detection

It helps data science teams monitor AI models in banking.
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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

class ModelType(str, Enum):
    """AI model types."""
    FRAUD = "fraud_detection"
    CREDIT = "credit_scoring"
    RISK = "risk_assessment"
    CHURN = "churn_prediction"
    AML = "anti_money_laundering"
    KYC = "kyc_verification"

class AlertLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricChange(BaseModel):
    """Metric change details."""
    metric: str = Field(description="Metric name")
    value: float = Field(description="Current value")
    change: float = Field(description="Change amount")
    level: AlertLevel = Field(description="Alert level")
    impact: str = Field(description="Business impact")

class ModelAnalysis(BaseModel):
    """Model analysis results."""
    model_id: str = Field(description="Model identifier")
    type: ModelType = Field(description="Model type")
    status: str = Field(description="Current status")
    metrics: List[MetricChange] = Field(description="Changed metrics")
    actions: List[str] = Field(description="Required actions")
    review_date: str = Field(description="Next review")

class ModelData(BaseModel):
    """Model performance data."""
    model_id: str = Field(description="Model ID")
    type: ModelType = Field(description="Model type")
    metrics: Dict[str, float] = Field(description="Current metrics")
    history: Dict[str, List[float]] = Field(description="Metric history")
    metadata: Dict = Field(default_factory=dict)

class ModelMonitor:
    """AI model monitoring system."""

    def __init__(self):
        """Initialize monitor."""
        logger.info("Starting model monitor...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        logger.info("Chat model ready")
        
        # Setup memory
        self.memory = ChatMessageHistory()
        logger.info("Chat memory ready")
        
        # Setup parser
        self.parser = StrOutputParser()
        
        # Setup template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an AI model monitoring assistant.
Review model metrics and identify significant changes.
Consider historical patterns and business impact.

Format your response exactly like this:

MODEL ANALYSIS
-------------
Status: Current state
Risk Level: HIGH/MEDIUM/LOW

Metric Changes:
1. Metric Name
   Value: Current value
   Change: +/- amount
   Impact: Description
   Action: Required step

2. Metric Name
   Value: Current value
   Change: +/- amount
   Impact: Description
   Action: Required step

Required Actions:
1. Action step
2. Action step

Next Review: YYYY-MM-DD

Focus on:
1. Metric changes
2. Alert levels
3. Business impact
4. Required actions"""),
            ("human", """Analyze this model:

Model: {model_id}
Type: {type}

Current Metrics:
{metrics}

Historical Data:
{history}

Previous Analysis:
{previous}

Provide a complete analysis.""")
        ])
        logger.info("Analysis prompt ready")

    def add_observation(self, observation: str) -> None:
        """Add observation to memory."""
        self.memory.add_user_message(observation)
        logger.debug("Added observation")

    def add_analysis(self, analysis: str) -> None:
        """Add analysis to memory."""
        self.memory.add_ai_message(analysis)
        logger.debug("Added analysis")

    def get_history(self) -> List[str]:
        """Get analysis history."""
        messages = self.memory.messages
        return [msg.content for msg in messages]

    async def analyze_model(self, model: ModelData) -> ModelAnalysis:
        """Analyze model performance."""
        logger.info(f"Analyzing model: {model.model_id}")
        
        try:
            # Get previous analysis
            history = self.get_history()
            previous = history[-1] if history else "No previous analysis"
            
            # Format metrics and history
            current = "\n".join(f"{k}: {v:.4f}" for k, v in model.metrics.items())
            historical = "\n".join(
                f"{k}: {', '.join(f'{x:.4f}' for x in v)}"
                for k, v in model.history.items()
            )
            
            # Format request
            messages = self.prompt.format_messages(
                model_id=model.model_id,
                type=model.type.value,
                metrics=current,
                history=historical,
                previous=previous
            )
            
            # Get and parse analysis
            response = await self.llm.ainvoke(messages)
            result = self.parser.parse(response.content)
            
            # Update memory
            self.add_analysis(response.content)
            logger.info("Analysis complete")
            
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting monitoring demo...")
    
    try:
        # Create monitor
        monitor = ModelMonitor()
        
        # Example model
        model = ModelData(
            model_id="FRAUD-2025-001",
            type=ModelType.FRAUD,
            metrics={
                "accuracy": 0.9856,
                "precision": 0.9823,
                "recall": 0.9789,
                "f1_score": 0.9806,
                "roc_auc": 0.9912,
                "false_positives": 0.0177
            },
            history={
                "accuracy": [0.9901, 0.9889, 0.9872, 0.9856],
                "precision": [0.9867, 0.9845, 0.9834, 0.9823],
                "recall": [0.9845, 0.9823, 0.9801, 0.9789],
                "f1_score": [0.9856, 0.9834, 0.9817, 0.9806],
                "roc_auc": [0.9934, 0.9923, 0.9918, 0.9912],
                "false_positives": [0.0133, 0.0155, 0.0166, 0.0177]
            }
        )
        
        print("\nAnalyzing Model")
        print("==============")
        print(f"Model: {model.model_id}")
        print(f"Type: {model.type.value}\n")
        
        print("Current Metrics:")
        for name, value in model.metrics.items():
            print(f"{name}: {value:.4f}")
        
        try:
            # Add observation
            monitor.add_observation(
                "Performance metrics showing gradual decline over last 4 periods"
            )
            
            # Get analysis
            result = await monitor.analyze_model(model)
            print("\nAnalysis Results:")
            print("================")
            print(result)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            print("\nAnalysis failed:", str(e))
        
        # End marker
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
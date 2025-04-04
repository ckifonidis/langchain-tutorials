#!/usr/bin/env python3
"""
LangChain Model Performance Monitor (108) (LangChain v3)

This example demonstrates a model performance monitoring system using three key concepts:
1. Async: Handle concurrent model evaluations
2. Vector Stores: Store and query performance metrics
3. Embedding Models: Analyze performance patterns

It provides comprehensive model monitoring support for data science teams in banking.
"""

import os
import json
from typing import List, Dict, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FakeEmbeddings(Embeddings):
    """Fake embeddings for demonstration."""
    def __init__(self, size: int = 1536):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate fake embeddings for documents."""
        return [self._get_embedding() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate fake embedding for query."""
        return self._get_embedding()

    def _get_embedding(self) -> List[float]:
        """Generate a fake embedding vector."""
        return list(np.random.normal(0, 0.1, self.size))

class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    model_id: str = Field(description="Model identifier")
    timestamp: str = Field(description="Evaluation timestamp")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    features: List[str] = Field(description="Key features used")
    sample_size: int = Field(description="Evaluation sample size")

class AlertConfig(BaseModel):
    """Schema for performance alerts."""
    metric: str = Field(description="Metric to monitor")
    threshold: float = Field(description="Alert threshold")
    condition: str = Field(description="Comparison condition")
    severity: str = Field(description="Alert severity")

class PerformanceAlert(BaseModel):
    """Schema for alert results."""
    model_id: str = Field(description="Model identifier")
    metric: str = Field(description="Triggered metric")
    current_value: float = Field(description="Current metric value")
    threshold: float = Field(description="Alert threshold")
    severity: str = Field(description="Alert severity")
    recommendations: List[str] = Field(description="Improvement suggestions")

class ModelPerformanceMonitor:
    def __init__(self, use_fake_embeddings: bool = True):
        try:
            # Initialize Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            
            # Initialize embeddings
            if use_fake_embeddings:
                print("Using fake embeddings for demonstration")
                self.embeddings = FakeEmbeddings()
            else:
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY")
                )
            
            # Initialize vector store
            self.setup_vector_store()
            
            # Default alert configurations
            self.alert_configs = [
                AlertConfig(
                    metric="accuracy",
                    threshold=0.95,
                    condition="below",
                    severity="high"
                ),
                AlertConfig(
                    metric="latency",
                    threshold=100,
                    condition="above",
                    severity="medium"
                ),
                AlertConfig(
                    metric="drift",
                    threshold=0.1,
                    condition="above",
                    severity="high"
                )
            ]
        except Exception as e:
            print(f"Error initializing monitor: {str(e)}")
            raise

    def setup_vector_store(self):
        """Initialize the vector store."""
        try:
            # Initialize empty vector store
            self.vector_store = FAISS.from_texts(
                texts=["Initial document"],
                embedding=self.embeddings,
                metadatas=[{"init": True}]
            )
        except Exception as e:
            print(f"Error setting up vector store: {str(e)}")
            raise

    async def store_metrics(self, metrics: ModelMetrics):
        """Store model metrics in vector store."""
        try:
            # Create metadata
            metadata = {
                "model_id": metrics.model_id,
                "timestamp": metrics.timestamp,
                "sample_size": metrics.sample_size,
                **metrics.metrics
            }
            
            # Create text representation
            text_content = f"""
            Model: {metrics.model_id}
            Time: {metrics.timestamp}
            Metrics: {json.dumps(metrics.metrics)}
            Features: {', '.join(metrics.features)}
            Sample Size: {metrics.sample_size}
            """
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=[text_content],
                metadatas=[metadata]
            )
            
        except Exception as e:
            print(f"Error storing metrics: {str(e)}")
            raise

    async def check_alerts(self, metrics: ModelMetrics) -> List[PerformanceAlert]:
        """Check for performance alerts."""
        alerts = []
        
        try:
            for config in self.alert_configs:
                if config.metric not in metrics.metrics:
                    continue
                
                current_value = metrics.metrics[config.metric]
                triggered = False
                
                if config.condition == "above" and current_value > config.threshold:
                    triggered = True
                elif config.condition == "below" and current_value < config.threshold:
                    triggered = True
                
                if triggered:
                    # Find similar past incidents
                    similar_docs = self.vector_store.similarity_search(
                        f"Model performance issues with {config.metric}",
                        k=3
                    )
                    
                    # Generate recommendations
                    context = "\n".join(doc.page_content for doc in similar_docs)
                    prompt = f"""
                    Analyze this model performance issue and provide recommendations:
                    
                    Current Issue:
                    - Model: {metrics.model_id}
                    - Metric: {config.metric}
                    - Current Value: {current_value}
                    - Threshold: {config.threshold}
                    
                    Similar Past Issues:
                    {context}
                    
                    Provide specific recommendations to address this performance issue.
                    Focus on banking model requirements and compliance.
                    """
                    
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert ML model performance analyst."),
                        HumanMessage(content=prompt)
                    ])
                    
                    # Create alert
                    alert = PerformanceAlert(
                        model_id=metrics.model_id,
                        metric=config.metric,
                        current_value=current_value,
                        threshold=config.threshold,
                        severity=config.severity,
                        recommendations=[r.strip() for r in response.content.split("\n") if r.strip()]
                    )
                    alerts.append(alert)
        
        except Exception as e:
            print(f"Error checking alerts: {str(e)}")
            raise
        
        return alerts

    async def analyze_performance(self, model_id: str) -> Dict:
        """Analyze model performance trends."""
        try:
            # Query vector store for model history
            results = self.vector_store.similarity_search(
                f"Performance history for model {model_id}",
                k=10
            )
            
            # Extract metrics safely
            metrics_history = []
            for doc in results:
                if not doc.metadata.get("init") and doc.metadata.get("model_id") == model_id:
                    metrics_history.append(doc.metadata.copy())
            
            # Calculate trends
            if metrics_history:
                df = pd.DataFrame(metrics_history)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")
                    
                    trends = {}
                    for metric in df.columns:
                        if metric not in ["model_id", "timestamp", "sample_size"]:
                            try:
                                values = df[metric].astype(float).values
                                if len(values) > 1:  # Multiple points
                                    slope = (values[-1] - values[0]) / (len(values) - 1)
                                    trends[metric] = {
                                        "direction": "improving" if slope > 0 else "degrading",
                                        "change_rate": float(abs(slope)),
                                        "current_value": float(values[-1]),
                                        "previous_value": float(values[0])
                                    }
                                else:  # Single point
                                    trends[metric] = {
                                        "current_value": float(values[0]),
                                        "trend": "insufficient_data"
                                    }
                            except Exception as metric_error:
                                print(f"Error processing metric {metric}: {str(metric_error)}")
                                continue
                    
                    return {
                        "model_id": model_id,
                        "metrics_count": len(metrics_history),
                        "date_range": {
                            "start": df["timestamp"].min().isoformat(),
                            "end": df["timestamp"].max().isoformat()
                        },
                        "trends": trends
                    }
            
            # Return empty result if no data
            return {
                "model_id": model_id,
                "metrics_count": 0,
                "trends": {}
            }
            
        except Exception as e:
            print(f"Error analyzing performance: {str(e)}")
            return {
                "model_id": model_id,
                "metrics_count": 0,
                "error": str(e),
                "trends": {}
            }

async def demonstrate_performance_monitor():
    print("\nModel Performance Monitor Demo")
    print("==============================\n")

    try:
        # Initialize monitor with fake embeddings
        monitor = ModelPerformanceMonitor(use_fake_embeddings=True)

        # Example metrics
        metrics = ModelMetrics(
            model_id="credit_risk_v2",
            timestamp=datetime.now().isoformat(),
            metrics={
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.95,
                "f1": 0.93,
                "latency": 120,
                "drift": 0.15
            },
            features=["credit_score", "income", "debt_ratio", "payment_history"],
            sample_size=1000
        )

        print("Storing metrics...")
        await monitor.store_metrics(metrics)

        print("\nChecking alerts...")
        alerts = await monitor.check_alerts(metrics)
        for alert in alerts:
            print(f"\nAlert for {alert.model_id}:")
            print(f"Metric: {alert.metric}")
            print(f"Current Value: {alert.current_value}")
            print(f"Threshold: {alert.threshold}")
            print(f"Severity: {alert.severity}")
            print("\nRecommendations:")
            for rec in alert.recommendations:
                print(f"- {rec}")

        print("\nAnalyzing performance trends...")
        analysis = await monitor.analyze_performance(metrics.model_id)
        print("\nPerformance Analysis:")
        print(json.dumps(analysis, indent=2))
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_performance_monitor())
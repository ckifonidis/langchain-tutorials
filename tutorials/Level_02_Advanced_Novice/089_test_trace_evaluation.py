#!/usr/bin/env python3
"""
LangChain Test-Trace-Evaluate System (LangChain v3)

This example demonstrates a comprehensive testing and evaluation system using three key concepts:
1. testing: Systematic test coverage and validation
2. tracing: Performance and behavior monitoring
3. evaluation: Quality assessment and metrics

It provides robust quality assurance for banking/fintech applications.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.evaluation import load_evaluator

# Load environment variables
load_dotenv(".env")

class TestCase(BaseModel):
    """Test case definition."""
    input: str = Field(description="Test input")
    expected: str = Field(description="Expected output")
    description: str = Field(description="Test description")
    category: str = Field(description="Test category")

class TestResult(BaseModel):
    """Test execution result."""
    passed: bool = Field(description="Whether test passed")
    actual: str = Field(description="Actual output")
    error: Optional[str] = Field(description="Error if any", default=None)
    metrics: Dict[str, float] = Field(description="Performance metrics")
    trace_id: str = Field(description="Trace identifier")

class PerformanceMetric(BaseModel):
    """Performance metric."""
    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    threshold: float = Field(description="Acceptable threshold")
    passed: bool = Field(description="Whether threshold was met")

class CustomTracer(BaseCallbackHandler):
    """Custom tracing handler."""
    
    def __init__(self):
        """Initialize tracer."""
        super().__init__()
        self.metrics = {}
        self.current_trace = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Record start time."""
        self.metrics["start_time"] = datetime.now()
    
    def on_llm_end(self, response, **kwargs):
        """Calculate duration and record metrics."""
        end_time = datetime.now()
        duration = (end_time - self.metrics["start_time"]).total_seconds()
        self.metrics["duration"] = duration
        self.metrics["tokens"] = len(str(response).split())
        
class QualityEvaluator:
    """System quality evaluator."""
    
    def __init__(self):
        """Initialize evaluator."""
        # Initialize LLM with callbacks
        self.console_tracer = ConsoleCallbackHandler()
        self.custom_tracer = CustomTracer()
        
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Initialize evaluators
        self.correctness_evaluator = load_evaluator(
            "string_distance",
            distance_function="levenshtein"
        )

        # Define evaluation keywords
        self.security_keywords = [
            "login", "access", "bank", "help", "online", "balance", 
            "mobile", "app", "direct", "secure", "portal"
        ]
        
        self.financial_keywords = [
            "interest", "compound", "principal", "rate", "invest", 
            "growth", "calculation", "formula", "year", "percent"
        ]
    
    def _evaluate_criteria(self, text: str, keywords: List[str]) -> int:
        """Count keyword matches in text."""
        return sum(1 for keyword in keywords if keyword.lower() in text.lower())
    
    async def evaluate_response(
        self,
        question: str,
        response: str,
        expected: str
    ) -> Dict[str, Any]:
        """Evaluate response quality."""
        try:
            # Choose keywords based on question content
            keywords = (
                self.security_keywords 
                if "balance" in question.lower() 
                else self.financial_keywords
            )
            
            # Evaluate string similarity
            correctness_eval = self.correctness_evaluator.evaluate_strings(
                prediction=response,
                reference=expected
            ) if response and expected else {"score": 0}
            
            # Count keyword matches
            matches = self._evaluate_criteria(response, keywords)
            
            return {
                "correctness": correctness_eval,
                "criteria": {"matches": matches}
            }
            
        except Exception as e:
            return {
                "correctness": {"score": 0},
                "criteria": {"matches": 0}
            }
    
    async def run_tests(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Run test suite with tracing and evaluation."""
        results = []
        
        # Configure LLM with callbacks
        self.custom_tracer.metrics = {}  # Reset metrics for each test
        self.custom_tracer.metrics["start_time"] = datetime.now()
        
        llm_with_cb = self.llm.with_config(
            configurable={
                "callbacks": [self.custom_tracer, self.console_tracer]
            }
        )
        
        for test in test_cases:
            try:
                # Execute test with tracers
                response = await llm_with_cb.ainvoke(
                    input=[
                        SystemMessage(content="You are a helpful banking assistant."),
                        HumanMessage(content=test.input),
                    ]
                )
                
                # Get response content
                actual = response.content
                
                # Calculate duration
                end_time = datetime.now()
                self.custom_tracer.metrics["duration"] = (end_time - self.custom_tracer.metrics["start_time"]).total_seconds()
                self.custom_tracer.metrics["tokens"] = len(actual.split())
                
                # Get trace ID (timestamp)
                trace_id = datetime.now().isoformat()
                
                # Evaluate quality
                eval_results = await self.evaluate_response(
                    question=test.input,
                    response=actual,
                    expected=test.expected
                )
                
                # Calculate metrics
                metrics = {
                    "duration": self.custom_tracer.metrics.get("duration", 0),
                    "tokens": self.custom_tracer.metrics.get("tokens", 0),
                    "string_similarity": 1 - float(eval_results["correctness"].get("score", 1)),
                    "criteria_matches": float(eval_results["criteria"].get("matches", 0))
                }
                
                # Create result
                result = TestResult(
                    passed=metrics["string_similarity"] > 0.7,
                    actual=actual,
                    metrics=metrics,
                    trace_id=trace_id
                )
                
                results.append(result)
            
            except Exception as e:
                results.append(
                    TestResult(
                        passed=False,
                        actual="",
                        error=str(e),
                        metrics={},
                        trace_id=str(datetime.now().isoformat())
                    )
                )
        
        return results

async def demonstrate_system():
    """Demonstrate the test-trace-evaluate system."""
    print("\nTest-Trace-Evaluate System Demo")
    print("===============================\n")
    
    # Create system
    system = QualityEvaluator()
    
    # Define test cases
    test_cases = [
        TestCase(
            input="What's the current balance in my checking account?",
            expected="I apologize, but I cannot access your actual account balance. For security reasons, you would need to log into your online banking portal or contact your bank directly to get your current checking account balance.",
            description="Test security handling for sensitive data request",
            category="security"
        ),
        TestCase(
            input="Explain how compound interest works.",
            expected="Compound interest is when you earn interest on both your initial deposit (principal) and previously accumulated interest. This creates exponential growth over time. For example, if you invest $1000 at 5% annual compound interest, you'll earn interest not only on the $1000, but also on any interest you've already earned.",
            description="Test financial concept explanation",
            category="education"
        )
    ]
    
    try:
        # Run tests
        print("Running Tests:")
        print("-" * 40)
        
        results = await system.run_tests(test_cases)
        
        # Display results
        for i, (test, result) in enumerate(zip(test_cases, results), 1):
            print(f"\nTest {i}: {test.description}")
            print(f"Category: {test.category}")
            print(f"Passed: {result.passed}")
            if result.error:
                print(f"Error: {result.error}")
            else:
                print("\nMetrics:")
                for name, value in result.metrics.items():
                    print(f"- {name}: {value:.3f}")
            print("-" * 40)
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_system())
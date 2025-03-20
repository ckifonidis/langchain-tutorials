"""
LangChain Async Task Monitor Example

This example demonstrates how to combine async programming with tracing capabilities
to create a sophisticated task monitoring system that can execute multiple operations
concurrently while providing detailed execution insights.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager
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

class TaskMetrics(BaseModel):
    """Schema for task execution metrics."""
    task_id: str = Field(description="Unique task identifier")
    start_time: datetime = Field(description="Task start timestamp")
    end_time: datetime = Field(description="Task completion timestamp")
    duration: float = Field(description="Execution duration in seconds")
    success: bool = Field(description="Task completion status")
    error: str = Field(description="Error message if failed", default="")
    tokens_used: int = Field(description="Number of tokens consumed")
    cost: float = Field(description="Estimated cost of execution")

class BatchMetrics(BaseModel):
    """Schema for batch execution metrics."""
    batch_id: str = Field(description="Unique batch identifier")
    tasks: List[TaskMetrics] = Field(description="Individual task metrics")
    total_duration: float = Field(description="Total batch duration")
    total_tokens: int = Field(description="Total tokens consumed")
    total_cost: float = Field(description="Total execution cost")
    success_rate: float = Field(description="Percentage of successful tasks")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model(callbacks: List[Any] = None) -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    callback_manager = CallbackManager(handlers=callbacks) if callbacks else None
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        callbacks=callbacks
    )

async def execute_task(
    task_id: str,
    prompt: str,
    llm: AzureChatOpenAI
) -> TaskMetrics:
    """Execute a single task with monitoring."""
    start_time = datetime.now()
    success = True
    error_msg = ""
    tokens = 0
    cost = 0.0
    
    try:
        with get_openai_callback() as cb:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=prompt)
            ]
            response = await llm.ainvoke(messages)
            tokens = cb.total_tokens
            cost = cb.total_cost
            
    except Exception as e:
        success = False
        error_msg = str(e)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return TaskMetrics(
        task_id=task_id,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        success=success,
        error=error_msg,
        tokens_used=tokens,
        cost=cost
    )

async def execute_batch(
    prompts: List[str],
    batch_size: int = 3
) -> BatchMetrics:
    """Execute a batch of tasks concurrently."""
    # Initialize components
    console_handler = ConsoleCallbackHandler()
    llm = create_chat_model(callbacks=[console_handler])
    
    batch_id = f"BATCH{datetime.now().strftime('%Y%m%d%H%M%S')}"
    start_time = time.time()
    
    # Create task batches
    tasks = []
    for i, prompt in enumerate(prompts):
        task_id = f"TASK{i+1:03d}"
        task = execute_task(task_id, prompt, llm)
        tasks.append(task)
    
    # Execute tasks in batches
    all_metrics = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        metrics = await asyncio.gather(*batch)
        all_metrics.extend(metrics)
    
    # Calculate batch statistics
    end_time = time.time()
    total_duration = end_time - start_time
    total_tokens = sum(m.tokens_used for m in all_metrics)
    total_cost = sum(m.cost for m in all_metrics)
    successful_tasks = sum(1 for m in all_metrics if m.success)
    success_rate = successful_tasks / len(all_metrics) if all_metrics else 0
    
    return BatchMetrics(
        batch_id=batch_id,
        tasks=all_metrics,
        total_duration=total_duration,
        total_tokens=total_tokens,
        total_cost=total_cost,
        success_rate=success_rate
    )

async def demonstrate_async_monitoring():
    """Demonstrate the Async Task Monitor capabilities."""
    try:
        print("\nInitializing Async Task Monitor...\n")
        
        # Example prompts
        prompts = [
            "Explain the concept of quantum entanglement.",
            "Describe the process of photosynthesis.",
            "Explain how blockchain technology works.",
            "Describe the theory of relativity.",
            "Explain the concept of artificial neural networks.",
            "Describe the process of climate change.",
            "Explain how DNA replication works.",
            "Describe the principles of machine learning."
        ]
        
        # Execute batch
        print("Executing batch of tasks...\n")
        metrics = await execute_batch(prompts)
        
        # Display results
        print("\nBatch Execution Results:")
        print(f"Batch ID: {metrics.batch_id}")
        print(f"Total Duration: {metrics.total_duration:.2f} seconds")
        print(f"Total Tokens: {metrics.total_tokens}")
        print(f"Total Cost: ${metrics.total_cost:.4f}")
        print(f"Success Rate: {metrics.success_rate * 100:.1f}%")
        
        print("\nIndividual Task Metrics:")
        for task in metrics.tasks:
            print(f"\nTask ID: {task.task_id}")
            print(f"Duration: {task.duration:.2f} seconds")
            print(f"Tokens: {task.tokens_used}")
            print(f"Cost: ${task.cost:.4f}")
            print(f"Status: {'Success' if task.success else 'Failed'}")
            if task.error:
                print(f"Error: {task.error}")
            print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Async Task Monitor...")
    asyncio.run(demonstrate_async_monitoring())

if __name__ == "__main__":
    main()
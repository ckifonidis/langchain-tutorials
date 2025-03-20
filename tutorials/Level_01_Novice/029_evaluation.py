"""
LangChain Evaluation Example

This example demonstrates how to evaluate model outputs and chain performance in
LangChain. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please add them to your .env file."
    )

class EvaluationResult(BaseModel):
    """Schema for evaluation results."""
    input: str = Field(description="Input text or query")
    output: str = Field(description="Model output")
    score: float = Field(description="Evaluation score")
    feedback: str = Field(description="Evaluation feedback")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model(temperature: float = 0):
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=temperature
    )

def create_chain_with_evaluator():
    """Create a chain with built-in evaluation."""
    # Create model and prompt
    model = create_chat_model()
    
    # Create evaluator for criteria checking using our model
    criteria_evaluator = load_evaluator(
        EvaluatorType.CRITERIA,
        criteria={
            "relevance": "Is the response relevant to the question?",
            "accuracy": "Is the information accurate?",
            "completeness": "Is the response complete?"
        },
        llm=model  # Use the same AzureChatOpenAI instance for evaluation
    )
    
    # Create main prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant providing clear, accurate information."),
        ("human", "{input}")
    ])
    
    # Create the chain with evaluation (RunnablePassthrough is used to simply pass output)
    chain = prompt | model | RunnablePassthrough()
    
    return chain, criteria_evaluator

def evaluate_responses(
    evaluator, questions: List[str], responses: List[Any]
) -> List[EvaluationResult]:
    """
    Evaluate model responses against given questions.
    
    Args:
        evaluator: The evaluator to use.
        questions: List of input questions.
        responses: List of model responses.
        
    Returns:
        List[EvaluationResult]: Evaluation results.
    """
    results = []
    
    for question, response in zip(questions, responses):
        # Convert response to string if needed
        response_str = response.content if hasattr(response, "content") else str(response)
        
        # Evaluate the response
        eval_result = evaluator.evaluate_strings(
            prediction=response_str,
            input=question
        )
        
        # Calculate average score from criteria
        scores = [v.get('score', 0) for v in eval_result.get('criteria', {}).values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Create result object
        result = EvaluationResult(
            input=question,
            output=response_str,
            score=avg_score,
            feedback=eval_result.get('reasoning', '')
        )
        
        results.append(result)
    
    return results

def demonstrate_evaluation():
    """Demonstrate different evaluation capabilities."""
    try:
        print("\nDemonstrating LangChain Evaluation...\n")
        
        # Example 1: Basic Response Evaluation
        print("Example 1: Basic Response Evaluation")
        print("-" * 50)
        
        # Create chain and evaluator
        chain, evaluator = create_chain_with_evaluator()
        
        # Test questions
        questions = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What is the meaning of life?"
        ]
        
        # Get responses
        print("\nGenerating responses...")
        responses = []
        for question in questions:
            response = chain.invoke({"input": question})
            responses.append(response)
            print(f"\nQuestion: {question}")
            print(f"Response: {response}")
        
        # Evaluate responses
        print("\nEvaluating responses...")
        results = evaluate_responses(evaluator, questions, responses)
        
        for result in results:
            print(f"\nInput: {result.input}")
            print(f"Score: {result.score:.2f}")
            print(f"Feedback: {result.feedback}")
        print("=" * 50)
        
        # Example 2: Comparing Model Outputs
        print("\nExample 2: Comparing Model Outputs")
        print("-" * 50)
        
        # Create models with different temperatures
        model_0 = create_chat_model(temperature=0)
        model_1 = create_chat_model(temperature=0.7)
        
        # Create prompts
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
        
        # Create chains
        chain_0 = prompt | model_0 | RunnablePassthrough()
        chain_1 = prompt | model_1 | RunnablePassthrough()
        
        # Test question
        question = "Explain the concept of gravity in simple terms."
        
        # Get responses
        print("\nGenerating responses from different models...")
        response_0 = chain_0.invoke({"input": question})
        response_1 = chain_1.invoke({"input": question})
        
        print(f"\nQuestion: {question}")
        print(f"\nModel 0 (temp=0.0):\n{response_0}")
        print(f"\nModel 1 (temp=0.7):\n{response_1}")
        
        # Evaluate both responses
        print("\nEvaluating responses...")
        results = evaluate_responses(
            evaluator,
            [question, question],
            [response_0, response_1]
        )
        
        print("\nEvaluation Results:")
        print(f"Model 0 Score: {results[0].score:.2f}")
        print(f"Model 1 Score: {results[1].score:.2f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_evaluation()

if __name__ == "__main__":
    main()


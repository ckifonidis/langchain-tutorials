"""
LangChain Evaluation Example

This example demonstrates how to implement evaluation capabilities in LangChain,
showing different methods to assess model and chain performance. Compatible with
LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.evaluation import StringEvaluator  # if needed for other evaluations
from langchain_core.outputs import LLMResult

# Correct import for QAEvalChain
from langchain.evaluation.qa.eval_chain import QAEvalChain

# Load environment variables
load_dotenv()

# Check if required environment variables are available
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

class EvaluationMetrics(BaseModel):
    """Schema for evaluation metrics."""
    accuracy: float = Field(description="Accuracy score of the response")
    relevance: float = Field(description="Relevance score of the response")
    completeness: float = Field(description="Completeness score of the response")
    additional_metrics: Dict[str, Any] = Field(description="Additional evaluation metrics")

def create_model_and_prompt():
    """
    Create the language model and prompt template.
    
    Returns:
        Tuple[AzureChatOpenAI, ChatPromptTemplate]: Model and prompt template.
    """
    # Initialize the model
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question clearly and accurately:
    
    Question: {question}
    """)
    
    return model, prompt

def evaluate_response(model, question: str, predicted: str, reference: str, criteria: Optional[List[str]] = None) -> EvaluationMetrics:
    """
    Evaluate a model's response against a reference answer using QAEvalChain.
    
    Args:
        model: The language model to use for evaluation.
        question: The question that was asked.
        predicted: Model's predicted answer.
        reference: Reference (correct) answer.
        criteria: Optional list of evaluation criteria (not used by default).
        
    Returns:
        EvaluationMetrics: Evaluation results.
    """
    # Instantiate the QA evaluation chain using the provided model.
    eval_chain = QAEvalChain.from_llm(llm=model)
    
    # Prepare examples and predictions as separate lists.
    examples = [{"query": question, "answer": reference}]
    predictions = [{"result": predicted}]
    
    # Evaluate the examples against the predictions.
    evaluation_results = eval_chain.evaluate(examples, predictions)
    
    # For debugging: print the raw evaluation result.
    print("\n[DEBUG] Raw Evaluation Result:", evaluation_results)
    
    # Use the first result from the evaluation chain.
    evaluation = evaluation_results[0]
    
    # Extract the qualitative result.
    result_text = evaluation.get("results", "").strip().upper()
    
    # Map "CORRECT" to a numeric score of 1.0, otherwise 0.0.
    numeric_score = 1.0 if result_text == "CORRECT" else 0.0
    
    # Construct and return the metrics.
    metrics = EvaluationMetrics(
        accuracy=numeric_score,
        relevance=numeric_score,
        completeness=numeric_score,
        additional_metrics={"results": evaluation.get("results")}
    )
    
    return metrics

def demonstrate_evaluation():
    """Demonstrate different evaluation capabilities."""
    try:
        print("\nDemonstrating LangChain Evaluation...\n")
        
        # Create model and prompt.
        model, prompt = create_model_and_prompt()
        
        # Example 1: Basic Question Answering Evaluation.
        print("Example 1: Question Answering Evaluation")
        print("-" * 50)
        
        # Test cases with questions and their ground-truth answers.
        test_cases = [
            {
                "question": "What is the capital of France?",
                "reference": "Paris."
            },
            {
                "question": "What is machine learning?",
                "reference": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
            }
        ]
        
        for case in test_cases:
            # Format prompt and get model response.
            formatted_prompt = prompt.format(question=case["question"])
            response = model.invoke(formatted_prompt)
            
            # Evaluate the response using the evaluation chain.
            metrics = evaluate_response(
                model,
                question=case["question"],
                predicted=response.content,
                reference=case["reference"]
            )
            
            print(f"\nQuestion: {case['question']}")
            print(f"Predicted: {response.content}")
            print(f"Reference: {case['reference']}")
            print("\nEvaluation Metrics:")
            print(f"Accuracy: {metrics.accuracy:.2f}")
            print(f"Relevance: {metrics.relevance:.2f}")
            print(f"Completeness: {metrics.completeness:.2f}")
            
            if metrics.additional_metrics:
                print("\nAdditional Metrics:")
                for metric, value in metrics.additional_metrics.items():
                    print(f"{metric}: {value}")
            
            print("-" * 50)
        
        # Example 2: Custom Evaluation Criteria (if applicable).
        print("\nExample 2: Custom Evaluation Criteria")
        print("-" * 50)
        
        # (Note: Custom criteria are not automatically used by QAEvalChain in this version.)
        custom_criteria = ["factual_accuracy", "response_length", "clarity"]
        
        question = "Explain how photosynthesis works."
        reference = ("Photosynthesis is the process by which plants convert sunlight, water, "
                     "and carbon dioxide into oxygen and energy in the form of sugar (glucose).")
        
        formatted_prompt = prompt.format(question=question)
        response = model.invoke(formatted_prompt)
        
        metrics = evaluate_response(
            model,
            question=question,
            predicted=response.content,
            reference=reference,
            criteria=custom_criteria
        )
        
        print(f"\nQuestion: {question}")
        print(f"Predicted: {response.content}")
        print(f"Reference: {reference}")
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics.accuracy:.2f}")
        print(f"Relevance: {metrics.relevance:.2f}")
        print(f"Completeness: {metrics.completeness:.2f}")
        
        if metrics.additional_metrics:
            print("\nCustom Criteria Results:")
            for metric, value in metrics.additional_metrics.items():
                print(f"{metric}: {value}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the evaluation demonstration."""
    demonstrate_evaluation()

if __name__ == "__main__":
    main()
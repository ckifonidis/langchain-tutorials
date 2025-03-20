"""
LangChain Evaluation Example with openevals Integration (Using Azure GPT-4o)

This example demonstrates how to evaluate model outputs using a prebuilt evaluator
(LLM-as-a-judge from openevals) while using your Azure Chat model ("azure:gpt-4o").
Note that openevals is optimized for a model like "openai:o3-mini", so evaluation
feedback may be less robust with other models.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnablePassthrough

# Import openevals components
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# Load environment variables
load_dotenv()

# Check required environment variables for Azure Chat model
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}. Please add them to your .env file."
    )
    
# Create an AzureChatOpenAI instance using environment variables
azure_judge = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

class EvaluationResult(BaseModel):
    """Schema for evaluation results."""
    input: str = Field(description="Input text or query")
    output: str = Field(description="Model output")
    score: float = Field(description="Evaluation score")
    feedback: str = Field(description="Evaluation feedback")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model(temperature: float = 0) -> AzureChatOpenAI:
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=temperature
    )

def create_chain_with_evaluator() -> (RunnablePassthrough, AzureChatOpenAI):
    """Create a chain that generates responses from the LLM."""
    model = create_chat_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant providing clear, accurate information."),
        ("human", "{input}")
    ])
    chain = prompt | model | RunnablePassthrough()
    return chain, model  # Return the model so we can pass it to the evaluator

def create_prebuilt_evaluator():
    """
    Create a prebuilt LLM-as-a-judge evaluator from openevals.
    
    Here, we pass our Azure Chat model's configuration by specifying a model string.
    Instead of "openai:o3-mini", we use "azure:gpt-4o". (Note that evaluation
    reasoning might be less optimal.)
    """
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="gpt-4o",  # Use your Azure model identifier here
        judge=azure_judge,
    )
    return evaluator

def evaluate_responses(
    evaluator, questions: List[str], responses: List[str], reference_outputs: List[str]
) -> List[EvaluationResult]:
    """
    Evaluate model responses against given questions using the prebuilt evaluator.
    
    Args:
        evaluator: The openevals evaluator.
        questions: List of input questions.
        responses: List of model responses.
        reference_outputs: List of expected reference outputs.
        
    Returns:
        List[EvaluationResult]: Evaluation results.
    """
    results = []
    
    for question, response, reference in zip(questions, responses, reference_outputs):
        # Convert response to string if needed.
        response_str = response.content if hasattr(response, "content") else str(response)
        
        # Run the evaluator; openevals expects:
        #   - inputs: the question,
        #   - outputs: the model's response,
        #   - reference_outputs: the expected output.
        eval_result = evaluator(
            inputs=question,
            outputs=response_str,
            reference_outputs=reference
        )
        
        score = eval_result.get("score", 0)
        feedback = eval_result.get("feedback", "")
        
        result = EvaluationResult(
            input=question,
            output=response_str,
            score=score,
            feedback=feedback
        )
        results.append(result)
    
    return results

def demonstrate_evaluation():
    """Demonstrate different evaluation capabilities."""
    try:
        print("\nDemonstrating LangChain Evaluation with openevals (Azure GPT-4o)...\n")
        
        # Example 1: Basic Response Evaluation
        print("Example 1: Basic Response Evaluation")
        print("-" * 50)
        
        chain, model = create_chain_with_evaluator()
        evaluator = create_prebuilt_evaluator()
        
        questions = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What is the meaning of life?"
        ]
        reference_outputs = [
            "Paris",  # Expected answer for question 1
            ("Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. "
             "It occurs in two main stages: the light-dependent reactions and the Calvin cycle."),
            ("The meaning of life is a philosophical question that may have many answers depending on one's perspective; common answers include seeking happiness, fulfilling a divine purpose, or creating your own meaning.")
        ]
        
        print("\nGenerating responses...")
        responses = []
        for question in questions:
            response = chain.invoke({"input": question})
            responses.append(response)
            print(f"\nQuestion: {question}")
            print(f"Response: {response}")
        
        print("\nEvaluating responses...")
        results = evaluate_responses(evaluator, questions, responses, reference_outputs)
        
        for result in results:
            print(f"\nInput: {result.input}")
            print(f"Score: {result.score:.2f}")
            print(f"Feedback: {result.feedback}")
        print("=" * 50)
        
        # Example 2: Comparing Model Outputs
        print("\nExample 2: Comparing Model Outputs")
        print("-" * 50)
        
        model_0 = create_chat_model(temperature=0)
        model_1 = create_chat_model(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
        
        chain_0 = prompt | model_0 | RunnablePassthrough()
        chain_1 = prompt | model_1 | RunnablePassthrough()
        
        question = "Explain the concept of gravity in simple terms."
        
        print("\nGenerating responses from different models...")
        response_0 = chain_0.invoke({"input": question})
        response_1 = chain_1.invoke({"input": question})
        
        print(f"\nQuestion: {question}")
        print(f"\nModel 0 (temp=0.0):\n{response_0}")
        print(f"\nModel 1 (temp=0.7):\n{response_1}")
        
        reference_gravity = ("Gravity is the force by which a planet or other body draws objects toward its center. "
                             "It is what gives us weight and keeps the planets in orbit around the sun.")
        
        results = evaluate_responses(
            evaluator,
            [question, question],
            [response_0, response_1],
            [reference_gravity, reference_gravity]
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


#!/usr/bin/env python3
"""
LangChain Code Review System Example

This example demonstrates how to combine chains and evaluation capabilities to create
a sophisticated code review system that can analyze code quality, provide feedback,
and suggest improvements.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import re
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

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

class CodeMetrics(BaseModel):
    """Schema for code quality metrics."""
    complexity: int = Field(description="Code complexity score (0-10)")
    readability: int = Field(description="Code readability score (0-10)")
    maintainability: int = Field(description="Code maintainability score (0-10)")
    documentation: int = Field(description="Documentation quality score (0-10)")
    best_practices: int = Field(description="Best practices adherence score (0-10)")

class CodeReview(BaseModel):
    """Schema for code review results."""
    file_name: str = Field(description="Name of the reviewed file")
    language: str = Field(description="Programming language")
    metrics: CodeMetrics = Field(description="Code quality metrics")
    issues: List[str] = Field(description="Identified issues")
    suggestions: List[str] = Field(description="Improvement suggestions")
    examples: List[str] = Field(description="Example improvements")
    overall_score: float = Field(description="Overall code quality score")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_analysis_prompt() -> PromptTemplate:
    """Create the analysis prompt template."""
    return PromptTemplate(
        template="""Given the following code, analyze its structure and patterns:

Code:
{code}

Language: {language}

Provide a detailed analysis considering:
1. Code organization
2. Design patterns used
3. Potential complexity issues
4. Notable patterns or anti-patterns

Analysis:""",
        input_variables=["code", "language"]
    )

def create_metrics_prompt() -> PromptTemplate:
    """Create the metrics prompt template."""
    return PromptTemplate(
        template="""Based on the code analysis, calculate metrics:

Analysis:
{analysis}

Provide numeric scores (0-10) for:
1. Complexity (lower is better)
2. Readability
3. Maintainability
4. Documentation Quality
5. Best Practices Adherence

Format as JSON:
{{
    "complexity": X,
    "readability": X,
    "maintainability": X,
    "documentation": X,
    "best_practices": X
}}""",
        input_variables=["analysis"]
    )

def create_suggestions_prompt() -> PromptTemplate:
    """Create the suggestions prompt template."""
    return PromptTemplate(
        template="""Based on the analysis and metrics, suggest improvements:

Analysis: {analysis}
Metrics: {metrics}

Provide:
1. List of specific issues
2. Concrete improvement suggestions
3. Example improvements
4. Overall assessment

Format as JSON:
{{
    "issues": ["issue1", "issue2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "examples": ["example1", "example2", ...],
    "overall_score": X.X
}}""",
        input_variables=["analysis", "metrics"]
    )

def create_review_chain(llm: AzureChatOpenAI) -> Any:
    """Create a chain for reviewing code."""
    # Create prompts
    analysis_prompt = create_analysis_prompt()
    metrics_prompt = create_metrics_prompt()
    suggestions_prompt = create_suggestions_prompt()
    
    # Create individual chain functions
    def analyze(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis chain."""
        response = llm.invoke(
            analysis_prompt.format_prompt(
                code=inputs["code"],
                language=inputs["language"]
            ).to_string()
        )
        return {
            "code": inputs["code"],
            "language": inputs["language"],
            "analysis": response.content
        }
    
    def calculate_metrics(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run metrics chain."""
        response = llm.invoke(
            metrics_prompt.format_prompt(
                analysis=inputs["analysis"]
            ).to_string()
        )
        inputs["metrics"] = response.content
        return inputs
    
    def suggest_improvements(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run suggestions chain."""
        response = llm.invoke(
            suggestions_prompt.format_prompt(
                analysis=inputs["analysis"],
                metrics=inputs["metrics"]
            ).to_string()
        )
        inputs["suggestions"] = response.content
        return inputs
    
    # Create sequential chain pipeline using RunnableLambda
    chain = (
        RunnableLambda(analyze)
        | RunnableLambda(calculate_metrics)
        | RunnableLambda(suggest_improvements)
    )
    
    return chain

def create_evaluation_chain(llm: AzureChatOpenAI) -> LabeledCriteriaEvalChain:
    """Create a chain for evaluating review quality."""
    criteria = {
        "completeness": "The review covers all important aspects of code quality",
        "actionability": "Suggestions are specific and implementable",
        "clarity": "Feedback is clear and well-explained",
        "correctness": "Technical assessments are accurate"
    }
    
    return LabeledCriteriaEvalChain.from_llm(
        llm=llm,
        criteria=criteria,
        evaluation_template="""Review the following code review:

Code Review:
{input}

Evaluate the review based on:
- Completeness
- Actionability
- Clarity
- Correctness

Provide a score and explanation for each criterion."""
    )

def parse_json_output(output: str) -> dict:
    """
    Extract the JSON block from the output and parse it.
    This function first tries to locate a JSON code block delimited by ```json and ```.
    If not found, it falls back to extracting the first JSON-like substring.
    """
    output = output.strip()
    # Try to extract JSON block using regex
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: extract content between first '{' and last '}'
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            json_str = output[start:end+1]
        else:
            raise ValueError(f"Failed to locate JSON block in output: {output}")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from extracted text: {json_str}") from e

def review_code(code: str, language: str, file_name: str) -> CodeReview:
    """Review code using chains and evaluation."""
    try:
        print(f"\nProcessing file: {file_name}")
        
        # Initialize components
        llm = create_chat_model()
        review_chain = create_review_chain(llm)
        evaluation_chain = create_evaluation_chain(llm)
        
        # Execute review chain
        results = review_chain.invoke({
            "code": code,
            "language": language
        })
        
        # Parse the metrics and suggestions outputs using our helper
        metrics_dict = parse_json_output(results["metrics"])
        suggestions_dict = parse_json_output(results["suggestions"])
        
        # Create metrics object
        metrics = CodeMetrics(**metrics_dict)
        
        # Create the code review object
        review = CodeReview(
            file_name=file_name,
            language=language,
            metrics=metrics,
            issues=suggestions_dict["issues"],
            suggestions=suggestions_dict["suggestions"],
            examples=suggestions_dict["examples"],
            overall_score=suggestions_dict["overall_score"]
        )
        
        # Evaluate review quality by providing a reference string (using the input code)
        evaluation = evaluation_chain.evaluate_strings(
            reference=code,
            prediction=str(review),
            input=code
        )
        
        # Handle evaluation output flexibly
        if isinstance(evaluation, dict) and "criteria_scores" in evaluation:
            criteria_scores = evaluation["criteria_scores"]
        else:
            try:
                criteria_scores = parse_json_output(evaluation)
            except Exception:
                criteria_scores = {}
        
        print("\nReview Quality Evaluation:")
        for criterion, score in criteria_scores.items():
            print(f"{criterion}: {score}/5")
        
        return review
        
    except Exception as e:
        print(f"Error reviewing code: {str(e)}")
        raise

def demonstrate_code_review():
    """Demonstrate the Code Review System capabilities."""
    try:
        print("\nInitializing Code Review System...\n")
        
        # Example code to review
        example_code = """
def calculate_total(items):
    t = 0
    for i in items:
        t = t + i
    return t

def process_data(data):
    result = []
    for d in data:
        if d > 0:
            result.append(d * 2)
    return result

class DataHandler:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        self.items.append(item)
    
    def get_items(self):
        return self.items
"""
        
        # Review code
        review = review_code(
            code=example_code,
            language="Python",
            file_name="example.py"
        )
        
        # Display results
        print("\nCode Review Results:")
        print(f"File: {review.file_name}")
        print(f"Language: {review.language}")
        
        print("\nMetrics:")
        print(f"Complexity: {review.metrics.complexity}/10")
        print(f"Readability: {review.metrics.readability}/10")
        print(f"Maintainability: {review.metrics.maintainability}/10")
        print(f"Documentation: {review.metrics.documentation}/10")
        print(f"Best Practices: {review.metrics.best_practices}/10")
        
        print("\nIdentified Issues:")
        for issue in review.issues:
            print(f"- {issue}")
        
        print("\nSuggestions:")
        for suggestion in review.suggestions:
            print(f"- {suggestion}")
        
        print("\nExample Improvements:")
        for example in review.examples:
            print(f"- {example}")
        
        print(f"\nOverall Score: {review.overall_score}/10")
        print(f"Review Time: {review.timestamp}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Code Review System...")
    demonstrate_code_review()

if __name__ == "__main__":
    main()
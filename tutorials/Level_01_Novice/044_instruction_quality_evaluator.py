"""
LangChain Instruction Quality Evaluator Example

This example demonstrates how to combine prompt templates and evaluation capabilities
to generate and assess the quality of instructions, providing metrics and suggestions for improvement.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.evaluation import load_evaluator

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

class InstructionMetrics(BaseModel):
    """Schema for instruction quality metrics."""
    clarity_score: float = Field(description="Clarity score (0-100)")
    completeness_score: float = Field(description="Completeness score (0-100)")
    actionability_score: float = Field(description="Actionability score (0-100)")
    conciseness_score: float = Field(description="Conciseness score (0-100)")
    average_score: float = Field(description="Average of all scores")

class QualityIssue(BaseModel):
    """Schema for quality issues found."""
    category: str = Field(description="Issue category")
    description: str = Field(description="Issue description")
    severity: str = Field(description="Issue severity (Low/Medium/High)")
    suggestion: str = Field(description="Improvement suggestion")

class QualityAnalysis(BaseModel):
    """Schema for comprehensive instruction quality analysis."""
    instruction_id: str = Field(description="Unique instruction identifier")
    original_text: str = Field(description="Original instruction text")
    metrics: InstructionMetrics = Field(description="Quality metrics")
    issues: List[QualityIssue] = Field(description="Identified issues")
    improved_version: str = Field(description="Suggested improved version")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "instruction_id": "INST042",
                "original_text": "Configure the application settings by modifying the config file and restart the service",
                "metrics": {
                    "clarity_score": 65.0,
                    "completeness_score": 40.0,
                    "actionability_score": 80.0,
                    "conciseness_score": 90.0,
                    "average_score": 68.8
                },
                "issues": [
                    {
                        "category": "Completeness",
                        "description": "Missing config file location",
                        "severity": "High",
                        "suggestion": "Specify the full path to the configuration file"
                    },
                    {
                        "category": "Clarity",
                        "description": "Ambiguous service reference",
                        "severity": "Medium",
                        "suggestion": "Specify the exact service name and restart command"
                    },
                    {
                        "category": "Actionability",
                        "description": "Missing order of operations",
                        "severity": "Low",
                        "suggestion": "Clarify if service restart is required immediately after config change"
                    }
                ],
                "improved_version": "Edit the configuration file at /etc/myapp/config.yaml with the new settings, then restart the application service using 'sudo systemctl restart myapp.service'",
                "timestamp": "2024-03-19T15:30:00"
            }]
        }
    }

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model with a higher temperature for more creative output."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7
    )

def create_evaluator(llm: AzureChatOpenAI):
    """Create a criteria evaluator for instruction quality."""
    criteria = {
        "clarity": "The instruction is clear and unambiguous",
        "completeness": "All necessary information is provided",
        "actionability": "The instruction can be acted upon directly",
        "conciseness": "The instruction is concise without losing meaning"
    }
    return load_evaluator(
        evaluator="criteria",
        criteria=criteria,
        llm=llm
    )

def evaluate_instruction(
    instruction: str,
    llm: AzureChatOpenAI,
    parser: PydanticOutputParser,
    evaluator: any
) -> QualityAnalysis:
    """
    Evaluate instruction quality and suggest improvements.
    
    Args:
        instruction: The instruction to evaluate.
        llm: Language model for analysis.
        parser: Output parser for structured results.
        evaluator: Criteria evaluator.
        
    Returns:
        QualityAnalysis: Comprehensive quality analysis.
    """
    criteria = {
        "clarity": "The instruction is clear and unambiguous",
        "completeness": "All necessary information is provided",
        "actionability": "The instruction can be acted upon directly",
        "conciseness": "The instruction is concise without losing meaning"
    }
    
    evaluation_template = PromptTemplate.from_template(
        """Analyze the following instruction for clarity, completeness, actionability, and conciseness:

Instruction: {instruction}

Based on your analysis, assign scores (0-100) for:
- Clarity
- Completeness
- Actionability
- Conciseness

Then, identify at least three issues. For each issue, provide:
- category (e.g., Completeness, Clarity, Actionability)
- description
- severity (Low, Medium, High)
- suggestion

Finally, propose an improved version of the instruction.

Your response must be a JSON object that exactly follows this format:
{{
  "issues": [
      {{
         "category": "<string>",
         "description": "<string>",
         "severity": "<Low/Medium/High>",
         "suggestion": "<string>"
      }},
      {{
         "category": "<string>",
         "description": "<string>",
         "severity": "<Low/Medium/High>",
         "suggestion": "<string>"
      }},
      {{
         "category": "<string>",
         "description": "<string>",
         "severity": "<Low/Medium/High>",
         "suggestion": "<string>"
      }}
  ],
  "improved_version": "<string>"
}}
"""
    )
    
    eval_prompt_text = evaluation_template.format(instruction=instruction)
    message = HumanMessage(content=eval_prompt_text)
    
    metrics = {}
    for criterion in criteria.keys():
        result = evaluator.evaluate_strings(
            prediction=instruction,
            criteria=criterion,
            input=instruction
        )
        metrics[f"{criterion}_score"] = result.get("score", 0) * 100
    metrics["average_score"] = sum(
        v for k, v in metrics.items() if k.endswith("_score")
    ) / len(criteria)
    
    response = llm.invoke([message])
    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_content = "\n".join(lines).strip()
    try:
        suggestions = json.loads(raw_content) if raw_content else {"issues": [], "improved_version": instruction}
    except json.JSONDecodeError:
        suggestions = {"issues": [], "improved_version": instruction}
    
    analysis_id = f"INST{hash(instruction) % 1000:03d}"
    
    return QualityAnalysis(
        instruction_id=analysis_id,
        original_text=instruction,
        metrics=InstructionMetrics(**metrics),
        issues=suggestions.get("issues", []),
        improved_version=suggestions.get("improved_version", instruction)
    )

def demonstrate_instruction_evaluation():
    """Demonstrate instruction quality evaluation capabilities."""
    try:
        print("\nDemonstrating Instruction Quality Evaluation...\n")
        
        llm = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=QualityAnalysis)
        evaluator = create_evaluator(llm)
        
        print("Example: Technical Documentation Instruction")
        print("-" * 50)
        
        instruction = (
            "Configure the application settings by modifying the config file and restart the service"
        )
        
        analysis = evaluate_instruction(instruction, llm, parser, evaluator)
        
        print("\nInstruction Analysis:")
        print(f"ID: {analysis.instruction_id}")
        print(f"Original: {analysis.original_text}")
        print("\nQuality Metrics:")
        print(f"Clarity: {analysis.metrics.clarity_score:.1f}")
        print(f"Completeness: {analysis.metrics.completeness_score:.1f}")
        print(f"Actionability: {analysis.metrics.actionability_score:.1f}")
        print(f"Conciseness: {analysis.metrics.conciseness_score:.1f}")
        print(f"Average Score: {analysis.metrics.average_score:.1f}")
        print("\nIdentified Issues:")
        for issue in analysis.issues:
            print(f"\nCategory: {issue.category}")
            print(f"Description: {issue.description}")
            print(f"Severity: {issue.severity}")
            print(f"Suggestion: {issue.suggestion}")
        print(f"\nImproved Version:\n{analysis.improved_version}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Instruction Quality Evaluator...")
    demonstrate_instruction_evaluation()

if __name__ == "__main__":
    main()

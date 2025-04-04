#!/usr/bin/env python3
"""
LangChain HR Interview Assistant (105) (LangChain v3)

This example demonstrates an HR interview assistance system using three key concepts:
1. Evaluation: Assess candidate responses
2. Prompt Templates: Generate structured interview questions
3. Example Selectors: Choose relevant interview scenarios

It provides comprehensive interview support for HR teams in banking.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.evaluation import load_evaluator
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InterviewQuestion(BaseModel):
    """Schema for interview questions."""
    role: str = Field(description="Job role being interviewed for")
    category: str = Field(description="Question category (technical/behavioral)")
    question: str = Field(description="Interview question")
    expected_points: List[str] = Field(description="Key points to look for")

class CandidateResponse(BaseModel):
    """Schema for candidate responses."""
    question_id: str = Field(description="Question identifier")
    response: str = Field(description="Candidate's response")
    timestamps: Dict[str, datetime] = Field(description="Response timeline")

class EvaluationResult(BaseModel):
    """Schema for evaluation results."""
    score: float = Field(description="Response score (0-10)")
    strengths: List[str] = Field(description="Identified strengths")
    areas_for_improvement: List[str] = Field(description="Areas needing improvement")
    feedback: str = Field(description="Detailed feedback")

class HRInterviewAssistant:
    def __init__(self):
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Initialize embeddings for example selection
        deployment = os.getenv("AZURE_DEPLOYMENT")
        if not deployment:
            raise ValueError("AZURE_DEPLOYMENT environment variable must be set")
            
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=deployment,
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            chunk_size=1,
            model=os.getenv("AZURE_MODEL_NAME", "text-embedding-3-small")
        )
        
        self.setup_prompt_templates()

    def setup_prompt_templates(self):
        """Set up interview prompt templates."""
        # Define example format
        self.example_template = """
Role: {role}
Category: {category}
Question: {question}
Key Points: {points}
"""
        
        # Define base examples
        self.examples = [
            {
                "role": "Risk Analyst",
                "category": "technical",
                "question": "How would you evaluate VaR models for market risk assessment?",
                "points": "Model validation, Backtesting, Stress scenarios"
            },
            {
                "role": "Compliance Officer",
                "category": "technical",
                "question": "How do you ensure regulatory reporting accuracy?",
                "points": "Process controls, Data validation, Review procedures"
            },
            {
                "role": "Digital Banking Manager",
                "category": "behavioral",
                "question": "How do you manage digital transformation projects?",
                "points": "Change management, Stakeholder engagement, Risk mitigation"
            }
        ]
        
        # Create example prompt template
        self.example_prompt = PromptTemplate(
            input_variables=["role", "category", "question", "points"],
            template=self.example_template
        )
        
        # Create few-shot prompt template
        self.question_template = PromptTemplate(
            template="""You are an expert HR interviewer for a bank. Based on these example questions:

{examples}

Generate an appropriate interview question for:
Role: {role}
Category: {category}

Format your response as:
Question: [Your question here]
Key Points: [Key points to look for in the answer]""",
            input_variables=["examples", "role", "category"]
        )

    async def generate_question(self, role: str, category: str) -> InterviewQuestion:
        """Generate an interview question."""
        try:
            # Format examples
            formatted_examples = "\n\n".join([
                self.example_prompt.format(**example)
                for example in self.examples
                if example["category"] == category  # Filter by category
            ])
            
            # Generate question using examples
            result = await self.llm.ainvoke(
                [
                    SystemMessage(content="You are an expert HR interviewer specializing in banking roles."),
                    HumanMessage(content=self.question_template.format(
                        examples=formatted_examples,
                        role=role,
                        category=category
                    ))
                ]
            )
            
            # Extract question and points from response
            content = result.content
            question_parts = content.split("Key Points:")
            
            if len(question_parts) == 2:
                question = question_parts[0].replace("Question:", "").strip()
                points = [point.strip() for point in question_parts[1].strip().split(",")]
            else:
                question = content
                points = ["Technical knowledge", "Problem-solving", "Communication"]
            
            # Create structured question
            return InterviewQuestion(
                role=role,
                category=category,
                question=question,
                expected_points=points
            )
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            return None

    async def evaluate_response(self, question: InterviewQuestion, response: CandidateResponse) -> EvaluationResult:
        """Evaluate a candidate's response."""
        try:
            # Generate evaluation criteria
            criteria = {
                "completeness": f"Response addresses the key points: {', '.join(question.expected_points)}",
                "technical_accuracy": "Demonstrates technical understanding of banking concepts",
                "clarity": "Clear and well-structured communication"
            }
            
            # Evaluate response
            eval_results = await self.evaluator.aevaluate_strings(
                prediction=response.response,
                input=question.question,
                criteria=criteria
            )
            
            # Calculate score
            score = sum(eval_results["criteria_scores"].values()) / len(criteria) * 10
            
            return EvaluationResult(
                score=score,
                strengths=eval_results["criteria_scores"],
                areas_for_improvement=eval_results["criteria_missing"],
                feedback=eval_results["reasoning"]
            )
            
        except Exception as e:
            print(f"Error evaluating response: {str(e)}")
            return None

async def demonstrate_hr_assistant():
    print("\nHR Interview Assistant Demo")
    print("==========================\n")

    assistant = HRInterviewAssistant()

    # Example roles and categories
    interview_scenarios = [
        {"role": "Credit Risk Manager", "category": "technical"},
        {"role": "Digital Banking Product Owner", "category": "behavioral"}
    ]

    for scenario in interview_scenarios:
        print(f"Generating question for {scenario['role']}")
        print(f"Category: {scenario['category']}\n")
        
        question = await assistant.generate_question(**scenario)
        if question:
            print(f"Generated Question:")
            print(f"- {question.question}")
            print("\nExpected Points:")
            for point in question.expected_points:
                print(f"- {point}")
                
            # Simulate candidate response
            response = CandidateResponse(
                question_id="Q1",
                response="I would approach this by first analyzing historical data, then implementing statistical models, and finally validating results with stakeholders.",
                timestamps={
                    "start": datetime.now(),
                    "end": datetime.now()
                }
            )
            
            print("\nExample Response:")
            print(f"- {response.response}\n")
            
            # Evaluate response
            evaluation = await assistant.evaluate_response(question, response)
            if evaluation:
                print("Evaluation Results:")
                print(f"Score: {evaluation.score:.1f}/10")
                print("\nStrengths:")
                for key, value in evaluation.strengths.items():
                    print(f"- {key}: {value}")
                print("\nAreas for Improvement:")
                for area in evaluation.areas_for_improvement:
                    print(f"- {area}")
                print(f"\nFeedback: {evaluation.feedback}")
            
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_hr_assistant())
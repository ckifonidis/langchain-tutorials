#!/usr/bin/env python3
"""
LangChain Recipe Quality Evaluator Example (LangChain v3)

This example demonstrates how to combine chains and evaluation capabilities to create
a sophisticated recipe generation and evaluation system that can create recipes and
assess their quality using multiple criteria.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class RecipeIngredient(BaseModel):
    """Schema for recipe ingredients."""
    name: str = Field(description="Ingredient name")
    amount: str = Field(description="Quantity needed")
    unit: str = Field(description="Unit of measurement")
    notes: str = Field(description="Optional preparation notes")

class RecipeStep(BaseModel):
    """Schema for recipe steps."""
    order: int = Field(description="Step number")
    instruction: str = Field(description="Step instruction")
    duration: str = Field(description="Estimated duration")
    tips: str = Field(description="Optional cooking tips")

class Recipe(BaseModel):
    """Schema for complete recipes."""
    title: str = Field(description="Recipe title")
    servings: int = Field(description="Number of servings")
    prep_time: str = Field(description="Preparation time")
    cook_time: str = Field(description="Cooking time")
    difficulty: str = Field(description="Recipe difficulty level")
    ingredients: List[RecipeIngredient] = Field(description="List of ingredients")
    steps: List[RecipeStep] = Field(description="Cooking instructions")
    notes: str = Field(description="Additional recipe notes")
    timestamp: datetime = Field(default_factory=datetime.now)

class RecipeEvaluation(BaseModel):
    """Schema for recipe evaluation results."""
    completeness: int = Field(description="Score for recipe completeness (1-10)")
    clarity: int = Field(description="Score for instruction clarity (1-10)")
    feasibility: int = Field(description="Score for recipe feasibility (1-10)")
    nutrition: int = Field(description="Score for nutritional balance (1-10)")
    creativity: int = Field(description="Score for recipe creativity (1-10)")
    improvements: List[str] = Field(description="Suggested improvements")
    overall_score: float = Field(description="Overall recipe quality score")

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7  # Higher temperature for recipe creativity
    )

def create_recipe_generation_chain(llm: AzureChatOpenAI) -> RunnablePassthrough:
    """Create a chain for recipe generation."""
    # Recipe generation prompt
    prompt = PromptTemplate(
        template="""Create a detailed recipe based on these requirements and restrictions:

Requirements:
{requirements}

Dietary Restrictions:
{restrictions}

Return the recipe as a valid JSON object that exactly conforms to the following structure:

{{
  "title": "<recipe title>",
  "servings": <number of servings>,
  "prep_time": "<preparation time>",
  "cook_time": "<cooking time>",
  "difficulty": "<difficulty level>",
  "ingredients": [
    {{
      "name": "<ingredient name>",
      "amount": "<quantity>",
      "unit": "<unit of measure>",
      "notes": "<preparation notes>"
    }}
  ],
  "steps": [
    {{
      "order": <step number>,
      "instruction": "<step instruction>",
      "duration": "<estimated time>",
      "tips": "<cooking tips>"
    }}
  ],
  "notes": "<additional recipe notes>"
}}

Do not include any additional text or commentary. Ensure that the output is valid JSON.""",
        input_variables=["requirements", "restrictions"]
    )
    
    return prompt | llm

def create_evaluation_chain(llm: AzureChatOpenAI) -> RunnablePassthrough:
    """Create a chain for recipe evaluation."""
    evaluation_prompt = PromptTemplate(
        template="""Evaluate this recipe based on the following criteria:

Recipe:
{recipe}

Requirements:
{requirements}

Dietary Restrictions:
{restrictions}

For each criterion below, provide a score from 1-10 and brief explanation.
Return your evaluation as a JSON object with this exact structure:

{{
    "completeness": {{
        "score": <score>,
        "reason": "<explanation>"
    }},
    "clarity": {{
        "score": <score>,
        "reason": "<explanation>"
    }},
    "feasibility": {{
        "score": <score>,
        "reason": "<explanation>"
    }},
    "nutrition": {{
        "score": <score>,
        "reason": "<explanation>"
    }},
    "creativity": {{
        "score": <score>,
        "reason": "<explanation>"
    }},
    "improvements": [
        "<improvement suggestion>",
        "<improvement suggestion>",
        "<improvement suggestion>"
    ]
}}""",
        input_variables=["recipe", "requirements", "restrictions"]
    )
    
    return evaluation_prompt | llm

def clean_json(text: str) -> str:
    """Remove markdown code fences and extra whitespace from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

def generate_and_evaluate_recipe(requirements: str, restrictions: str) -> Tuple[Recipe, RecipeEvaluation]:
    """Generate and evaluate a recipe based on requirements and restrictions."""
    try:
        print("\nGenerating recipe...")
        
        # Initialize components
        llm = create_chat_model()
        recipe_chain = create_recipe_generation_chain(llm)
        eval_chain = create_evaluation_chain(llm)
        
        # Generate recipe
        recipe_result = recipe_chain.invoke({
            "requirements": requirements,
            "restrictions": restrictions
        })
        
        # Clean and parse recipe
        recipe_text = clean_json(recipe_result.content)
        recipe = Recipe.model_validate_json(recipe_text)
        
        # Evaluate recipe
        print("\nEvaluating recipe quality...")
        eval_result = eval_chain.invoke({
            "recipe": recipe_text,
            "requirements": requirements,
            "restrictions": restrictions
        })
        
        # Parse evaluation results
        eval_text = clean_json(eval_result.content)
        eval_data = json.loads(eval_text)
        
        # Create evaluation object
        evaluation = RecipeEvaluation(
            completeness=eval_data["completeness"]["score"],
            clarity=eval_data["clarity"]["score"],
            feasibility=eval_data["feasibility"]["score"],
            nutrition=eval_data["nutrition"]["score"],
            creativity=eval_data["creativity"]["score"],
            improvements=eval_data["improvements"],
            overall_score=sum([
                eval_data[k]["score"] for k in [
                    "completeness", "clarity", "feasibility",
                    "nutrition", "creativity"
                ]
            ]) / 5.0
        )
        
        return recipe, evaluation
        
    except Exception as e:
        print(f"Error generating recipe: {str(e)}")
        raise

def demonstrate_recipe_evaluation():
    """Demonstrate the Recipe Quality Evaluator capabilities."""
    try:
        print("\nInitializing Recipe Quality Evaluator...\n")
        
        # Example requirements
        requirements = """
Create a healthy dinner recipe that:
- Is high in protein
- Takes less than 45 minutes to prepare
- Uses fresh vegetables
- Serves 2 people
"""
        
        restrictions = """
Dietary restrictions:
- No dairy
- No nuts
- Low sodium
"""
        
        # Generate and evaluate recipe
        recipe, evaluation = generate_and_evaluate_recipe(requirements, restrictions)
        
        # Display results
        print("\nGenerated Recipe:")
        print(f"Title: {recipe.title}")
        print(f"Servings: {recipe.servings}")
        print(f"Prep Time: {recipe.prep_time}")
        print(f"Cook Time: {recipe.cook_time}")
        print(f"Difficulty: {recipe.difficulty}")
        
        print("\nIngredients:")
        for ingredient in recipe.ingredients:
            print(f"- {ingredient.amount} {ingredient.unit} {ingredient.name}")
            if ingredient.notes:
                print(f"  Note: {ingredient.notes}")
        
        print("\nInstructions:")
        for step in recipe.steps:
            print(f"{step.order}. {step.instruction}")
            print(f"   Time: {step.duration}")
            if step.tips:
                print(f"   Tip: {step.tips}")
        
        if recipe.notes:
            print(f"\nNotes: {recipe.notes}")
        
        print("\nRecipe Evaluation:")
        print(f"Completeness: {evaluation.completeness}/10")
        print(f"Clarity: {evaluation.clarity}/10")
        print(f"Feasibility: {evaluation.feasibility}/10")
        print(f"Nutrition: {evaluation.nutrition}/10")
        print(f"Creativity: {evaluation.creativity}/10")
        
        print("\nSuggested Improvements:")
        for improvement in evaluation.improvements:
            print(f"- {improvement}")
        
        print(f"\nOverall Score: {evaluation.overall_score:.1f}/10")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Recipe Quality Evaluator...")
    demonstrate_recipe_evaluation()

if __name__ == "__main__":
    main()

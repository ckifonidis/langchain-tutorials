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
from typing import List, Tuple, Any, Dict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.base import Chain  # Base chain class

# Load environment variables
load_dotenv()

# -----------------------------
# Define an EmptyMemory class
# -----------------------------

class EmptyMemory:
    memory_variables: List[str] = []

# -----------------------------
# Pydantic Models for Recipe
# -----------------------------

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
    """Schema for recipe evaluations."""
    completeness: int = Field(description="Score for recipe completeness (1-10)")
    clarity: int = Field(description="Score for instruction clarity (1-10)")
    feasibility: int = Field(description="Score for recipe feasibility (1-10)")
    nutrition: int = Field(description="Score for nutritional balance (1-10)")
    creativity: int = Field(description="Score for recipe creativity (1-10)")
    improvements: List[str] = Field(description="Suggested improvements")
    overall_score: float = Field(description="Overall recipe quality score")

# -----------------------------
# Helper Functions and Chains
# -----------------------------

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7  # Higher temperature for recipe creativity
    )

def clean_json(text: str) -> str:
    """Remove markdown code fences and extra whitespace from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

# --- DictWrapperChain as a subclass of Chain ---
class DictWrapperChain(Chain):
    """
    A chain that wraps a runnable (e.g. created by `prompt | llm`) so that its output is returned
    as a dictionary with a specified key.
    """
    _runnable: Any = PrivateAttr()
    _key: str = PrivateAttr()
    # Declare a public field "memory" (not as a property) so that SequentialChain can access it.
    memory: Any = Field(default_factory=lambda: EmptyMemory())

    def __init__(self, runnable: Any, key: str, **kwargs):
        super().__init__(**kwargs)
        self._runnable = runnable
        self._key = key
        mem = getattr(runnable, "memory", None)
        if not (mem and hasattr(mem, "memory_variables")):
            mem = EmptyMemory()
        self.memory = mem

    @property
    def input_keys(self) -> List[str]:
        return getattr(self._runnable, "input_keys", [])

    @property
    def output_keys(self) -> List[str]:
        return [self._key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = self._runnable.invoke(inputs)
        return {self._key: result}

    def call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._call(inputs)

def create_recipe_chain(llm: AzureChatOpenAI) -> SequentialChain:
    """Create a chain for recipe generation using the new runnable style."""
    # Recipe concept generation prompt.
    concept_prompt = PromptTemplate(
        template="""Generate a creative recipe concept based on these requirements:

Requirements:
{requirements}

Dietary Restrictions:
{restrictions}

Generate a recipe title and a brief description of the concept.""",
        input_variables=["requirements", "restrictions"]
    )
    # Use the new runnable style: prompt | llm
    concept_runnable = concept_prompt | llm

    # Detailed recipe creation prompt – output must be valid JSON.
    recipe_prompt = PromptTemplate(
        template="""Create a detailed recipe based on this concept:

Concept:
{concept}

Return the recipe as a valid JSON object that exactly conforms to the following structure:

{{
  "title": string,
  "servings": integer,
  "prep_time": string,
  "cook_time": string,
  "difficulty": string,
  "ingredients": [
    {{
      "name": string,
      "amount": string,
      "unit": string,
      "notes": string
    }}
  ],
  "steps": [
    {{
      "order": integer,
      "instruction": string,
      "duration": string,
      "tips": string
    }}
  ],
  "notes": string
}}

Do not include any additional text or commentary. Ensure that the output is valid JSON.
""",
        input_variables=["concept"]
    )
    recipe_runnable = recipe_prompt | llm

    # Wrap the runnables using DictWrapperChain.
    wrapped_concept = DictWrapperChain(concept_runnable, "concept")
    wrapped_recipe = DictWrapperChain(recipe_runnable, "recipe")

    # Combine the wrapped chains in sequence.
    return SequentialChain(
        chains=[wrapped_concept, wrapped_recipe],
        input_variables=["requirements", "restrictions"],
        output_variables=["concept", "recipe"]
    )

def create_evaluation_chain(llm: AzureChatOpenAI) -> LabeledCriteriaEvalChain:
    """Create a chain for recipe evaluation."""
    criteria = {
        "completeness": "Recipe includes all necessary ingredients and clear instructions",
        "clarity": "Instructions are easy to follow and well-organized",
        "feasibility": "Recipe is practical to make with common equipment and ingredients",
        "nutrition": "Recipe provides a balanced nutritional profile",
        "creativity": "Recipe shows originality and interesting flavor combinations"
    }
    return LabeledCriteriaEvalChain.from_llm(
        llm=llm,
        criteria=criteria,
        evaluation_template="""Evaluate this recipe based on these criteria:

Recipe:
{input}

Criteria:
- Completeness
- Clarity
- Feasibility
- Nutrition
- Creativity

Provide a score (1-10) and explanation for each criterion."""
    )

def generate_and_evaluate_recipe(requirements: str, restrictions: str) -> Tuple[Recipe, RecipeEvaluation]:
    """Generate and evaluate a recipe based on requirements and restrictions."""
    try:
        print("\nGenerating recipe...")
        llm = create_chat_model()
        recipe_chain = create_recipe_chain(llm)
        eval_chain = create_evaluation_chain(llm)
        
        # Generate the recipe.
        result = recipe_chain.invoke({
            "requirements": requirements,
            "restrictions": restrictions
        })
        
        # Expect the recipe chain to return a dictionary with a "recipe" key.
        recipe_text = result["recipe"].strip()
        recipe_text = clean_json(recipe_text)
        
        # Parse the JSON into a Recipe model.
        recipe = Recipe.model_validate_json(recipe_text)
        
        # Evaluate the recipe.
        print("\nEvaluating recipe quality...")
        evaluation = eval_chain.evaluate_strings(
            prediction=recipe_text,
            input=requirements,
            reference=restrictions
        )
        
        # Calculate overall score.
        scores = evaluation.get("criteria_scores", {})
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        recipe_eval = RecipeEvaluation(
            completeness=scores.get("completeness", 0),
            clarity=scores.get("clarity", 0),
            feasibility=scores.get("feasibility", 0),
            nutrition=scores.get("nutrition", 0),
            creativity=scores.get("creativity", 0),
            improvements=evaluation.get("improvement_suggestions", []),
            overall_score=overall_score
        )
        
        return recipe, recipe_eval
        
    except Exception as e:
        print(f"Error generating recipe: {str(e)}")
        raise

def demonstrate_recipe_evaluation():
    """Demonstrate the Recipe Quality Evaluator capabilities."""
    try:
        print("\nInitializing Recipe Quality Evaluator...\n")
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
        recipe, evaluation = generate_and_evaluate_recipe(requirements, restrictions)
        
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


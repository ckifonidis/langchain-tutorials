#!/usr/bin/env python3
"""
LangChain Recipe Recommender (LangChain v3)

This example demonstrates how to build a recipe recommendation system using chat models
and structured output parsing. It provides personalized recipe suggestions based on
ingredients and preferences.

Key concepts demonstrated:
1. Chat Models: Using chat-optimized models for natural interaction
2. Structured Output: Enforcing structured responses with output parsers
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

class RecipeStep(BaseModel):
    """Individual recipe step."""
    step_number: int = Field(description="Step number in the recipe")
    instruction: str = Field(description="Detailed cooking instruction")
    duration_minutes: Optional[int] = Field(description="Estimated time for this step")
    temperature_celsius: Optional[int] = Field(description="Cooking temperature if needed")

class Recipe(BaseModel):
    """Recipe recommendation with details."""
    name: str = Field(description="Recipe name")
    cuisine_type: str = Field(description="Type of cuisine")
    difficulty: str = Field(description="Easy/Medium/Hard")
    prep_time_minutes: int = Field(description="Preparation time")
    cook_time_minutes: int = Field(description="Cooking time")
    servings: int = Field(description="Number of servings")
    ingredients: List[str] = Field(description="List of required ingredients")
    steps: List[RecipeStep] = Field(description="Cooking instructions")
    dietary_info: List[str] = Field(description="Dietary tags (vegetarian, gluten-free, etc.)")
    tips: List[str] = Field(description="Cooking tips and suggestions")

class RecipeRecommender:
    """Recipe recommendation system using chat models."""

    def __init__(self):
        """Initialize the recipe recommender."""
        # Create chat model
        self.chat_model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.7  # Some creativity in recipes
        )
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=Recipe)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert chef and cooking instructor. 
Suggest recipes based on available ingredients and preferences.

Return your suggestion in a structured format with:
- Recipe name and cuisine type
- Difficulty and timing
- Ingredients list
- Step-by-step instructions
- Dietary information and tips

{format_instructions}"""),
            ("human", """Available ingredients: {ingredients}
Dietary preferences: {preferences}
Time available: {max_time} minutes""")
        ])
    
    def get_recipe(
        self,
        ingredients: List[str],
        preferences: List[str],
        max_time: int
    ) -> Recipe:
        """Get a recipe recommendation."""
        try:
            # Format inputs
            input_data = {
                "ingredients": ", ".join(ingredients),
                "preferences": ", ".join(preferences),
                "max_time": max_time,
                "format_instructions": self.parser.get_format_instructions()
            }
            
            # Create and invoke the recipe chain
            chain = self.prompt | self.chat_model | self.parser
            recipe = chain.invoke(input_data)
            
            return recipe
            
        except Exception as e:
            print(f"Error getting recipe: {str(e)}")
            raise

def demonstrate_recipe_recommender():
    """Demonstrate the recipe recommender."""
    print("\nLangChain Recipe Recommender")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Quick Vegetarian Meal",
            "ingredients": [
                "pasta", "tomatoes", "garlic", "olive oil", 
                "basil", "parmesan cheese"
            ],
            "preferences": ["vegetarian", "quick", "easy"],
            "max_time": 30
        },
        {
            "name": "Gluten-Free Dinner",
            "ingredients": [
                "chicken breast", "rice", "broccoli", "carrots", 
                "ginger", "soy sauce"
            ],
            "preferences": ["gluten-free", "healthy", "protein-rich"],
            "max_time": 45
        },
        {
            "name": "Dessert",
            "ingredients": [
                "eggs", "butter", "sugar", "chocolate", 
                "vanilla extract", "flour"
            ],
            "preferences": ["sweet", "baking", "indulgent"],
            "max_time": 60
        }
    ]
    
    # Create recommender
    recommender = RecipeRecommender()
    
    # Test each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 50)
        print("Ingredients:", ", ".join(scenario["ingredients"]))
        print("Preferences:", ", ".join(scenario["preferences"]))
        print(f"Time Available: {scenario['max_time']} minutes")
        
        try:
            recipe = recommender.get_recipe(
                scenario["ingredients"],
                scenario["preferences"],
                scenario["max_time"]
            )
            
            # Display recipe
            print("\nRecommended Recipe:")
            print(f"Name: {recipe.name}")
            print(f"Cuisine: {recipe.cuisine_type}")
            print(f"Difficulty: {recipe.difficulty}")
            print(f"Time: {recipe.prep_time_minutes + recipe.cook_time_minutes} minutes")
            print(f"Servings: {recipe.servings}")
            
            print("\nIngredients:")
            for ingredient in recipe.ingredients:
                print(f"- {ingredient}")
            
            print("\nInstructions:")
            for step in recipe.steps:
                duration = f" ({step.duration_minutes} mins)" if step.duration_minutes else ""
                temp = f" at {step.temperature_celsius}°C" if step.temperature_celsius else ""
                print(f"{step.step_number}. {step.instruction}{duration}{temp}")
            
            print("\nDietary Information:")
            for info in recipe.dietary_info:
                print(f"- {info}")
            
            print("\nCooking Tips:")
            for tip in recipe.tips:
                print(f"- {tip}")
            
        except Exception as e:
            print(f"Could not generate recipe: {str(e)}")
        
        if i < len(scenarios):
            print("\n" + "-" * 50)
        else:
            print("\n" + "=" * 50)

if __name__ == "__main__":
    demonstrate_recipe_recommender()
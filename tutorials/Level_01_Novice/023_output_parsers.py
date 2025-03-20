"""
LangChain Output Parsers Example

This example demonstrates how to use output parsers in LangChain to structure 
and validate model responses. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

class MovieReview(BaseModel):
    """Schema for movie review output."""
    title: str = Field(description="Title of the movie")
    year: int = Field(description="Release year of the movie")
    rating: float = Field(description="Rating out of 10")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")

class WeatherReport(BaseModel):
    """Schema for weather report output."""
    location: str = Field(description="Name of the location")
    temperature: float = Field(description="Current temperature in Celsius")
    conditions: str = Field(description="Weather conditions description")
    forecast: List[str] = Field(description="Weather forecast for next few days")
    last_updated: datetime = Field(description="Time of the report")

def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def demonstrate_output_parsers():
    """Demonstrate different output parser capabilities."""
    try:
        print("\nDemonstrating LangChain Output Parsers...\n")
        
        # Initialize model
        model = create_chat_model()
        
        # Example 1: Simple String Parser
        print("Example 1: Simple String Parser")
        print("-" * 50)
        
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "What is the capital of {country}?")
        ])
        
        # Create chain with string parser
        string_chain = simple_prompt | model | StrOutputParser()
        
        # Test with different countries
        countries = ["France", "Japan", "Brazil"]
        for country in countries:
            response = string_chain.invoke({"country": country})
            print(f"\nQuery: What is the capital of {country}?")
            print(f"Response: {response}")
        print("=" * 50)
        
        # Example 2: JSON Parser
        print("\nExample 2: JSON Parser")
        print("-" * 50)
        
        # Note the double curly braces to escape the JSON structure.
        json_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a movie critic. 
Provide movie reviews in JSON format with the following structure:
{{
    "title": "movie title",
    "year": release year,
    "rating": rating out of 10,
    "summary": "brief review"
}}"""),
            ("human", "Review the movie {movie_title}")
        ])
        
        # Create chain with JSON parser
        json_chain = json_prompt | model | JsonOutputParser()
        
        # Test with different movies
        movies = ["The Matrix", "Inception", "Avatar"]
        for movie in movies:
            response = json_chain.invoke({"movie_title": movie})
            print(f"\nMovie Review for: {movie}")
            print(f"Response: {response}")
        print("=" * 50)
        
        # Example 3: Pydantic Parser
        print("\nExample 3: Pydantic Parser")
        print("-" * 50)
        
        # Create Pydantic parser
        pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
        
        # Create prompt with formatting instructions.
        pydantic_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a detailed movie critic.
{PYDANTIC_FORMAT_INSTRUCTIONS}"""),
            ("human", "Provide a detailed review for the movie {movie_title}. Use the following schema:\n{schema}")
        ])
        
        # Create chain with Pydantic parser.
        pydantic_chain = pydantic_prompt | model | pydantic_parser
        
        # Prepare the schema string from the MovieReview model.
        schema = MovieReview.schema_json(indent=2)
        
        # Test with a movie
        movie = "The Dark Knight"
        response = pydantic_chain.invoke({"movie_title": movie, "schema": schema})
        print(f"\nStructured Movie Review for: {movie}")
        print(f"Title: {response.title}")
        print(f"Year: {response.year}")
        print(f"Rating: {response.rating}/10")
        print("\nPros:")
        for pro in response.pros:
            print(f"- {pro}")
        print("\nCons:")
        for con in response.cons:
            print(f"- {con}")
        print(f"\nSummary: {response.summary}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_output_parsers()

if __name__ == "__main__":
    main()

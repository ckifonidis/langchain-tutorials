"""
LangChain Structured Output Example

This example demonstrates how to get structured output from language models
using Pydantic models. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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

class MovieReview(BaseModel):
    """Schema for a movie review."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year of the movie")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    review: str = Field(description="Brief review of the movie")
    tags: List[str] = Field(description="List of genre or theme tags")
    director: Optional[str] = Field(
        description="The movie's director",
        default=None
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "title": "The Matrix",
                "year": 1999,
                "rating": 9,
                "review": "A groundbreaking sci-fi masterpiece with mind-bending concepts.",
                "tags": ["sci-fi", "action", "cyberpunk"],
                "director": "The Wachowskis"
            }]
        }
    }

def demonstrate_structured_output() -> None:
    """Demonstrate how to get structured output from LangChain."""
    try:
        # Initialize chat model
        chat = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Create a Pydantic output parser using the MovieReview schema.
        pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
        
        # Example 1: Basic movie review for 'Inception'
        print("\nExample 1: Basic Movie Review")
        system_msg = SystemMessage(content="""
You are a film critic who provides movie reviews.
When asked about a movie, respond with a JSON object that follows this schema exactly:
{
  "title": "movie title (string)",
  "year": release year (integer),
  "rating": rating (integer from 1 to 10),
  "review": "brief review (string)",
  "tags": ["list", "of", "genres or themes"],
  "director": "director name (string, optional)"
}
Please ensure that your response is valid JSON.
An example response is:
{
  "title": "Inception",
  "year": 2010,
  "rating": 9,
  "review": "Inception is a mind-bending thriller that explores dream manipulation with stunning visuals and a complex narrative. Christopher Nolan's direction and a stellar cast deliver an emotionally gripping experience.",
  "tags": ["Science Fiction", "Thriller", "Action", "Mind-bending", "Psychological"],
  "director": "Christopher Nolan"
}
""")
        human_msg = HumanMessage(content="Review the movie 'Inception'")
        
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        chain = prompt | chat | pydantic_parser
        
        response_inception = chain.invoke({"input": "Review the movie 'Inception'"})
        
        print("\nStructured Review for Inception:")
        print(f"Title: {response_inception.title}")
        print(f"Year: {response_inception.year}")
        print(f"Rating: {response_inception.rating}/10")
        print(f"Review: {response_inception.review}")
        print(f"Tags: {', '.join(response_inception.tags)}")
        if response_inception.director:
            print(f"Director: {response_inception.director}")
        
        # Example 2: Multiple movie reviews
        print("\nExample 2: Multiple Movie Reviews")
        
        # For 'The Matrix'
        system_msg_matrix = SystemMessage(content="""
You are a film critic. Provide a JSON review for the movie 'The Matrix' following this schema:
{
  "title": "movie title (string)",
  "year": release year (integer),
  "rating": rating (integer from 1 to 10),
  "review": "brief review (string)",
  "tags": ["list", "of", "genres or themes"],
  "director": "director name (string, optional)"
}
Ensure the response is valid JSON.
Example:
{
  "title": "The Matrix",
  "year": 1999,
  "rating": 9,
  "review": "A groundbreaking sci-fi masterpiece with innovative effects and deep philosophical themes.",
  "tags": ["sci-fi", "action", "cyberpunk"],
  "director": "The Wachowskis"
}
""")
        human_msg_matrix = HumanMessage(content="Review the movie 'The Matrix'")
        prompt_matrix = ChatPromptTemplate.from_messages([system_msg_matrix, human_msg_matrix])
        chain_matrix = prompt_matrix | chat | pydantic_parser
        response_matrix = chain_matrix.invoke({"input": "Review the movie 'The Matrix'"})
        
        # For 'Blade Runner'
        system_msg_blade = SystemMessage(content="""
You are a film critic. Provide a JSON review for the movie 'Blade Runner' following this schema:
{
  "title": "movie title (string)",
  "year": release year (integer),
  "rating": rating (integer from 1 to 10),
  "review": "brief review (string)",
  "tags": ["list", "of", "genres or themes"],
  "director": "director name (string, optional)"
}
Ensure the response is valid JSON.
Example:
{
  "title": "Blade Runner",
  "year": 1982,
  "rating": 8,
  "review": "A visually striking neo-noir sci-fi film that explores themes of identity and humanity.",
  "tags": ["sci-fi", "neo-noir", "dystopian"],
  "director": "Ridley Scott"
}
""")
        human_msg_blade = HumanMessage(content="Review the movie 'Blade Runner'")
        prompt_blade = ChatPromptTemplate.from_messages([system_msg_blade, human_msg_blade])
        chain_blade = prompt_blade | chat | pydantic_parser
        response_blade = chain_blade.invoke({"input": "Review the movie 'Blade Runner'"})
        
        print("\nComparison of Sci-Fi Classics:")
        for review in [response_matrix, response_blade]:
            print(f"\n{review.title} ({review.year})")
            print(f"Rating: {review.rating}/10")
            print(f"Review: {review.review}")
            print(f"Tags: {', '.join(review.tags)}")
            if review.director:
                print(f"Director: {review.director}")
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Structured Output...")
    demonstrate_structured_output()

if __name__ == "__main__":
    main()

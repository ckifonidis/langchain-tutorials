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
            temperature=0.7
        )
        
        # Example 1: Basic movie review
        print("\nExample 1: Basic Movie Review")
        system_msg = SystemMessage(content="""
            You are a film critic who provides movie reviews. 
            When asked about a movie, provide a structured review 
            following the specified format.
        """)
        
        human_msg = HumanMessage(content="Review the movie 'Inception'")
        
        response = chat.invoke(
            [system_msg, human_msg],
            model_kwargs={"response_format": {"type": "json"}},
            output_schema=MovieReview
        )
        
        print("\nStructured Review:")
        print(f"Title: {response.title}")
        print(f"Year: {response.year}")
        print(f"Rating: {response.rating}/10")
        print(f"Review: {response.review}")
        print(f"Tags: {', '.join(response.tags)}")
        if response.director:
            print(f"Director: {response.director}")
        
        # Example 2: Multiple movie reviews
        print("\nExample 2: Multiple Movie Reviews")
        human_msg2 = HumanMessage(content="""
            Compare and review both 'The Matrix' and 'Blade Runner' 
            as classic sci-fi films.
        """)
        
        # First movie
        response_matrix = chat.invoke(
            [
                SystemMessage(content="Review 'The Matrix' as a sci-fi classic."),
                human_msg2
            ],
            model_kwargs={"response_format": {"type": "json"}},
            output_schema=MovieReview
        )
        
        # Second movie
        response_blade = chat.invoke(
            [
                SystemMessage(content="Review 'Blade Runner' as a sci-fi classic."),
                human_msg2
            ],
            model_kwargs={"response_format": {"type": "json"}},
            output_schema=MovieReview
        )
        
        print("\nComparison of Sci-Fi Classics:")
        for review in [response_matrix, response_blade]:
            print(f"\n{review.title} ({review.year})")
            print(f"Rating: {review.rating}/10")
            print(f"Review: {review.review}")
            print(f"Tags: {', '.join(review.tags)}")
            if review.director:
                print(f"Director: {review.director}")
        
    except ValueError as ve:
        print(f"\nValidation error: {str(ve)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Structured Output...")
    demonstrate_structured_output()

if __name__ == "__main__":
    main()
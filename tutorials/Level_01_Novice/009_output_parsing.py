"""
LangChain Output Parsing Example

This example demonstrates how to use output parsers in LangChain to structure
model responses into specific formats. We'll create a movie review analyzer
that outputs structured data using Pydantic models and output parsing.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Check if required environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                    "Please add them to your .env file.")

class MovieReview(BaseModel):
    """Schema for a structured movie review."""
    title: str = Field(description="The title of the movie")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief summary of the movie")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    recommended: bool = Field(description="Whether the movie is recommended")

def create_review_analyzer() -> tuple[PydanticOutputParser, PromptTemplate]:
    """
    Create a movie review analyzer with structured output parsing.
    
    Returns:
        tuple: (parser, prompt_template) for analyzing movie reviews
    """
    # Create an output parser using the MovieReview schema
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    
    # Get format instructions for the model
    format_instructions = parser.get_format_instructions()
    
    # Create a prompt template that includes format instructions
    prompt = PromptTemplate(
        template="""Analyze the following movie review and provide a structured response.
        
Review: {review}

{format_instructions}

Remember to:
1. Extract the movie title if mentioned
2. Provide a 1-10 rating based on the sentiment
3. Write a brief summary
4. List key positive and negative points
5. Determine if the movie is recommended based on the overall review

Your response:""",
        input_variables=["review"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    return parser, prompt

def analyze_review(review_text: str, chat_model, parser: PydanticOutputParser, 
                  prompt: PromptTemplate) -> MovieReview:
    """
    Analyze a movie review and return structured data.
    
    Args:
        review_text: The text of the movie review
        chat_model: The language model to use
        parser: The output parser
        prompt: The prompt template
        
    Returns:
        MovieReview: Structured review data
    """
    # Format the prompt with the review text
    formatted_prompt = prompt.format(review=review_text)
    
    try:
        # Get response from the model
        response = chat_model.invoke(formatted_prompt)
        
        # Parse the response into a MovieReview object
        parsed_review = parser.parse(response.content)
        return parsed_review
        
    except Exception as e:
        print(f"Error analyzing review: {str(e)}")
        raise

def demonstrate_output_parsing():
    """Demonstrate how to use output parsing with example movie reviews."""
    try:
        # Initialize the chat model
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7  # Add some creativity to the reviews
        )
        
        # Create the review analyzer
        parser, prompt = create_review_analyzer()
        
        # Example reviews to analyze
        reviews = [
            """Just watched 'The Matrix' again and it's still mind-blowing after all these years! 
            The special effects were groundbreaking for its time and still hold up. Keanu Reeves 
            is perfect as Neo, and the philosophical themes really make you think. Some scenes 
            might be a bit confusing for first-time viewers, and the sequels didn't quite match 
            up, but the original is a masterpiece. Definitely a must-watch for any sci-fi fan!""",
            
            """'Gigli' is probably one of the worst movies I've ever seen. The chemistry between 
            the leads is non-existent, the plot makes no sense, and the dialogue is cringe-worthy. 
            The only positive thing I can say is that some scenes are unintentionally funny. 
            Save your time and watch something else."""
        ]
        
        print("\nDemonstrating LangChain Output Parsing...\n")
        
        for i, review in enumerate(reviews, 1):
            print(f"\nExample {i}: Analyzing review...")
            print("-" * 50)
            print("Original Review:", review)
            print("-" * 50)
            
            # Analyze the review
            parsed_review = analyze_review(review, chat_model, parser, prompt)
            
            # Display the structured results
            print("\nStructured Analysis:")
            print(f"Title: {parsed_review.title}")
            print(f"Rating: {parsed_review.rating}/10")
            print(f"Summary: {parsed_review.summary}")
            print("\nPros:")
            for pro in parsed_review.pros:
                print(f"- {pro}")
            print("\nCons:")
            for con in parsed_review.cons:
                print(f"- {con}")
            print(f"\nRecommended: {'Yes' if parsed_review.recommended else 'No'}")
            print("=" * 50)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_output_parsing()

if __name__ == "__main__":
    main()
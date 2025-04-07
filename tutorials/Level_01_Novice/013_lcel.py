"""
LangChain Expression Language (LCEL) Example

This example demonstrates how to use LCEL in LangChain to create and compose
chains of operations. Shows various LCEL patterns including sequential chains,
branches, and error handling. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
    """Schema for a movie review."""
    title: str = Field(description="The title of the movie")
    sentiment: str = Field(description="The sentiment of the review (positive/negative)")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief summary of the review")

class MovieReviewParser(BaseOutputParser[MovieReview]):
    """Parser for movie review output."""
    
    def parse(self, text: str) -> MovieReview:
        """Parse the output into a MovieReview object."""
        lines = text.strip().split("\n")
        data = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "rating":
                    value = int(value)
                
                data[key] = value
        
        return MovieReview(**data)

def create_review_chain():
    """
    Create a chain for processing movie reviews using LCEL.
    
    Returns:
        A runnable chain that processes movie reviews
    """
    # Create the model
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following movie review and provide a structured response.
    Include the movie title, sentiment (positive/negative), numerical rating from 1 to 10 (use just the number),
    and a brief summary.
    
    Format your response as:
    Title: <movie title>
    Sentiment: <positive/negative>
    Rating: <number between 1-10, no fractions or /10>
    Summary: <brief summary>
    
    Review: {review}
    """)
    
    # Create the parser
    parser = MovieReviewParser()
    
    # Create the chain
    chain = prompt | model | parser
    
    return chain

def create_translation_chain():
    """
    Create a chain for translating text using LCEL.
    
    Returns:
        A runnable chain that translates text
    """
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Translate the following text to {language}: {text}"
    )
    
    chain = prompt | model | StrOutputParser()
    
    return chain

def create_branching_chain():
    """
    Create a chain with branching logic using LCEL.
    
    Returns:
        A runnable chain that demonstrates branching
    """
    def route_by_language(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Route text to appropriate language model based on detected language."""
        text = input_dict["text"]
        # Simple language detection based on first word
        if text.lower().startswith(("the", "a", "in", "on")):
            return {"source_language": "English", "text": text}
        else:
            return {"source_language": "Unknown", "text": text}
    
    router = RunnableLambda(route_by_language)
    
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create prompts for different branches
    english_prompt = ChatPromptTemplate.from_template(
        "Analyze this English text: {text}"
    )
    
    unknown_prompt = ChatPromptTemplate.from_template(
        "Detect the language and analyze this text: {text}"
    )
    
    # Create the branching chain
    chain = router | {
        "english_analysis": RunnablePassthrough() | english_prompt | model | StrOutputParser(),
        "unknown_analysis": RunnablePassthrough() | unknown_prompt | model | StrOutputParser(),
        "original_input": RunnablePassthrough()
    }
    
    return chain

def demonstrate_lcel():
    """Demonstrate different LCEL patterns."""
    try:
        print("\nDemonstrating LangChain Expression Language (LCEL)...\n")
        
        # Example 1: Sequential Chain
        print("Example 1: Sequential Chain (Movie Review)")
        print("-" * 50)
        
        review_chain = create_review_chain()
        review_text = """
        The latest Batman movie was absolutely fantastic! The dark atmosphere
        and brilliant performances kept me on the edge of my seat throughout
        the entire film. While some scenes were a bit too intense, the
        overall experience was incredible.
        """
        
        result = review_chain.invoke({"review": review_text})
        print(f"Title: {result.title}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Rating: {result.rating}")
        print(f"Summary: {result.summary}")
        print("=" * 50)
        
        # Example 2: Translation Chain
        print("\nExample 2: Translation Chain")
        print("-" * 50)
        
        translation_chain = create_translation_chain()
        text = "Hello, world! How are you today?"
        
        result = translation_chain.invoke({
            "text": text,
            "language": "Spanish"
        })
        print(f"Original: {text}")
        print(f"Translated: {result}")
        print("=" * 50)
        
        # Example 3: Branching Chain
        print("\nExample 3: Branching Chain")
        print("-" * 50)
        
        branching_chain = create_branching_chain()
        
        # Test with English text
        english_text = "The weather is beautiful today"
        result = branching_chain.invoke({"text": english_text})
        
        print("English Input Analysis:")
        print(f"Source Language: {result['original_input']['source_language']}")
        print(f"Analysis: {result['english_analysis']}")
        
        # Test with unknown text
        unknown_text = "Bonjour le monde"
        result = branching_chain.invoke({"text": unknown_text})
        
        print("\nUnknown Input Analysis:")
        print(f"Source Language: {result['original_input']['source_language']}")
        print(f"Analysis: {result['unknown_analysis']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_lcel()

if __name__ == "__main__":
    main()
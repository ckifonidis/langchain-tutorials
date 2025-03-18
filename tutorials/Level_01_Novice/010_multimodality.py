"""
LangChain Multimodality Example

This example demonstrates how to work with multimodal inputs (text and images)
in LangChain, using Azure OpenAI's GPT-4 Vision model for image understanding 
and description. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
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

class ImageInput(BaseModel):
    """Schema for image input."""
    url: str = Field(
        description="URL or local path to the image",
        examples=["path/to/image.jpg", "https://example.com/image.jpg"]
    )

class ImageDescription(BaseModel):
    """Schema for structured image description."""
    main_subject: str = Field(description="The primary subject or focus of the image")
    setting: str = Field(description="The environment or context where the image was taken")
    colors: List[str] = Field(description="Dominant colors present in the image")
    objects: List[str] = Field(description="Notable objects or elements in the image")
    mood: str = Field(description="Overall mood or atmosphere of the image")
    
def create_image_message(image_path: str) -> dict:
    """
    Create a message dictionary for an image input.
    
    Args:
        image_path: Path or URL to the image
        
    Returns:
        Dictionary containing image data in the format expected by the model
    """
    # For local files, convert to base64
    if os.path.exists(image_path):
        import base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            }
    else:
        # For URLs, pass directly
        return {
            "type": "image_url",
            "image_url": image_path
        }

def analyze_image(image_path: str, chat_model) -> ImageDescription:
    """
    Analyze an image and return a structured description.
    
    Args:
        image_path: Path or URL to the image
        chat_model: The language model to use for analysis
        
    Returns:
        ImageDescription: Structured description of the image
        
    Raises:
        ValueError: If the image path is invalid
        Exception: For other errors during image analysis
    """
    try:
        # Create message with image content
        image_data = create_image_message(image_path)
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Please analyze this image and provide a structured description with the following information:
                    1. Main subject
                    2. Setting/environment
                    3. Dominant colors
                    4. Notable objects
                    5. Overall mood/atmosphere
                    Format your response as a valid JSON object with these fields."""
                },
                image_data
            ]
        )
        
        # Get model's analysis
        response = chat_model.invoke([message])
        
        try:
            # Parse response into structured format
            import json
            description_dict = json.loads(response.content)
            return ImageDescription(
                main_subject=description_dict["main_subject"],
                setting=description_dict["setting"],
                colors=description_dict["colors"],
                objects=description_dict["objects"],
                mood=description_dict["mood"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing model response: {str(e)}")
            print("Raw response:", response.content)
            raise
            
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        raise

def demonstrate_multimodality():
    """Demonstrate multimodal capabilities using example images."""
    try:
        # Initialize chat model with support for image inputs
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            max_tokens=1000
        )
        
        # Example images (one local, one URL)
        example_images = [
            "examples/city_street.jpg",  # Local file
            "https://example.com/landscape.jpg"  # URL
        ]
        
        print("\nDemonstrating LangChain Multimodality...\n")
        
        for i, image_path in enumerate(example_images, 1):
            print(f"\nExample {i}: Analyzing {'local image' if os.path.exists(image_path) else 'image URL'}")
            print("-" * 50)
            
            try:
                # Analyze the image
                description = analyze_image(image_path, chat_model)
                
                # Display results
                print("\nImage Analysis:")
                print(f"Main Subject: {description.main_subject}")
                print(f"Setting: {description.setting}")
                print("\nDominant Colors:")
                for color in description.colors:
                    print(f"- {color}")
                print("\nNotable Objects:")
                for obj in description.objects:
                    print(f"- {obj}")
                print(f"\nMood: {description.mood}")
                print("=" * 50)
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_multimodality()

if __name__ == "__main__":
    main()
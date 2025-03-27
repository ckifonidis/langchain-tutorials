#!/usr/bin/env python3
"""
LangChain Image Chat Analyzer (LangChain v3)

This example demonstrates building an image analysis chat system using multimodal
capabilities and message handling. It searches for, downloads, and analyzes images
based on user queries.

Key concepts demonstrated:
1. Multimodality: Processing both text and image inputs
2. Messages: Managing conversation flow and context
"""

import os
import base64
import json
import subprocess
from typing import List, Dict, Any
from datetime import datetime
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from urllib.parse import urlparse

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

class ImageAnalysis(BaseModel):
    """Image analysis results."""
    main_subjects: List[str] = Field(description="Main subjects in the image")
    scene_type: str = Field(description="Type of scene (indoor/outdoor/etc)")
    colors: List[str] = Field(description="Dominant colors")
    activities: List[str] = Field(description="Activities or actions shown")
    objects: List[str] = Field(description="Notable objects present")
    mood: str = Field(description="Overall mood or atmosphere")

class ImageSearchResult(BaseModel):
    """Image search result details."""
    title: str = Field(description="Image title")
    url: str = Field(description="Image URL")
    source: str = Field(description="Source website")
    thumbnail: str = Field(description="Thumbnail URL")

class ImageChatAnalyzer:
    """Image analysis system with chat capabilities."""
    
    def __init__(self):
        """Initialize the image chat analyzer."""
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ImageAnalysis)
        self.conversation_history = []
        
        # Setup images directory
        self.images_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'images'
        )
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def _generate_filename(self, content_description: str) -> str:
        """Generate a filename based on content description."""
        # Clean description for filename
        clean_desc = re.sub(r'[^a-zA-Z0-9\s]', '', content_description)
        clean_desc = clean_desc.strip().lower().replace(' ', '_')[:50]
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return f"{clean_desc}_{timestamp}"
    
    def _download_image(self, url: str, description: str = None) -> str:
        """Download image using curl."""
        try:
            # Extract extension from URL or use default
            parsed_url = urlparse(url)
            original_name = os.path.basename(parsed_url.path)
            ext = os.path.splitext(original_name)[1].lower()
            
            if not ext or ext not in ['.jpg', '.jpeg', '.png']:
                ext = '.jpg'
            
            # Generate descriptive filename
            base_name = self._generate_filename(description if description else "downloaded_image")
            filename = f"{base_name}{ext}"
            image_path = os.path.join(self.images_dir, filename)
            
            # Use curl to download
            command = [
                'curl', '-L', '-o', image_path,
                '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                '-H', 'Accept: image/webp,image/apng,image/*,*/*;q=0.8',
                '--connect-timeout', '10',
                '--max-time', '30',
                url
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                return image_path
            else:
                raise ValueError("Downloaded file is empty")
            
        except subprocess.CalledProcessError as e:
            print(f"Curl error: {e.stderr}")
            raise
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            raise
    
    def search_images(self, query: str, max_results: int = 3) -> List[ImageSearchResult]:
        """Search for images using DuckDuckGo."""
        try:
            print(f"\nSearching for images matching: {query}")
            
            # Initialize DuckDuckGo search
            with DDGS() as ddgs:
                # Search for images
                images = []
                for result in ddgs.images(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    size=None,
                    color=None,
                    type_image=None,
                    layout=None,
                    license_image=None,
                    max_results=max_results
                ):
                    # Create image result object
                    image = ImageSearchResult(
                        title=result.get('title', 'Untitled Image'),
                        url=result.get('image', ''),
                        source=result.get('source', 'Unknown'),
                        thumbnail=result.get('thumbnail', '')
                    )
                    images.append(image)
                    
                    if len(images) >= max_results:
                        break
            
            print(f"Found {len(images)} images")
            return images
            
        except Exception as e:
            print(f"Error during image search: {str(e)}")
            return []
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            raise
    
    def analyze_image(self, image_path: str) -> ImageAnalysis:
        """Analyze an image and return structured insights."""
        try:
            # Encode image
            image_data = self._encode_image(image_path)
            
            # Create system message
            system_message = SystemMessage(
                content="""You are an expert image analyst. Analyze the image and provide output in valid JSON format with these exact fields:
{
    "main_subjects": ["list", "of", "subjects"],
    "scene_type": "description of scene",
    "colors": ["list", "of", "colors"],
    "activities": ["list", "of", "activities"],
    "objects": ["list", "of", "objects"],
    "mood": "description of mood"
}
Return ONLY the JSON object, no other text.""")
            
            # Create message with image
            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this image in detail:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            # Get analysis
            messages = [system_message, human_message]
            response = self.llm.invoke(messages)

            # Parse JSON response
            try:
                # Find JSON content
                content = response.content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_response = json.loads(content[start:end])
                else:
                    raise ValueError("No JSON object found in response")
                
                print("\nParsed Analysis:", json.dumps(json_response, indent=2))
                analysis = ImageAnalysis(**json_response)
                
                # Add to conversation history after successful parsing
                self.conversation_history.extend([human_message, response])
                
                return analysis
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response: {response.content}")
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise
    
    def generate_similar_image(self, analysis: ImageAnalysis, original_path: str) -> None:
        """Request a creative description for generating a similar image."""
        try:
            # Create generation prompt from analysis
            prompt = f"""Based on this analysis:
- Scene Type: {analysis.scene_type}
- Main Subjects: {', '.join(analysis.main_subjects)}
- Colors: {', '.join(analysis.colors)}
- Objects: {', '.join(analysis.objects)}
- Mood: {analysis.mood}

Please generate an artistic interpretation that captures these elements.
The image should maintain the same mood and style but be a unique creation."""
            
            print("\nArtistic interpretation prompt:")
            print(f"{'-' * 50}\n{prompt}\n{'-' * 50}")
            
            # Create generation request
            messages = [
                SystemMessage(content="You are an expert visual artist and creative director."),
                HumanMessage(content=f"""Based on this prompt:
{prompt}

Please describe in detail how you would create an artistic interpretation.
Include specific details about composition, lighting, colors, and focal points.""")
            ]
            
            # Get creative description
            response = self.llm.invoke(messages)
            print("\nArtistic Vision:")
            print(f"{'-' * 50}\n{response.content}\n{'-' * 50}")
            
            # Save both prompt and description
            prompt_file = os.path.splitext(original_path)[0] + "_artistic_vision.txt"
            with open(prompt_file, "w") as f:
                f.write(f"Original Prompt:\n{prompt}\n\nArtistic Vision:\n{response.content}")
            
            print(f"\nSaved artistic vision to: {prompt_file}")
            
        except Exception as e:
            print(f"Error creating artistic interpretation: {str(e)}")
            raise
    
    def ask_about_image(self, question: str) -> str:
        """Ask a question about the previously analyzed image."""
        try:
            if not self.conversation_history:
                raise ValueError("No image has been analyzed yet")
            
            # Create question message
            question_message = HumanMessage(content=question)
            
            # Add to conversation history
            messages = self.conversation_history + [question_message]
            response = self.llm.invoke(messages)
            
            # Update history
            self.conversation_history.extend([question_message, response])
            
            return response.content
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            raise

def demonstrate_image_chat():
    """Demonstrate the image chat analyzer."""
    print("\nImage Chat Analyzer Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ImageChatAnalyzer()
    
    while True:
        try:
            # Get search query
            print("\nWhat kind of images would you like to analyze?")
            print("(or type 'exit' to quit)")
            query = input("> ").strip()
            
            if query.lower() == 'exit':
                break
            
            # Search for images
            images = analyzer.search_images(query, max_results=3)
            
            if not images:
                print("No suitable images found. Please try a different search.")
                continue
            
            print(f"\nFound {len(images)} images:")
            for i, image in enumerate(images, 1):
                print(f"\n{i}. {image.title}")
                print(f"Source: {image.source}")
                print(f"URL: {image.url}")
            
            # Select image to analyze
            print("\nWhich image would you like to analyze? (1-3)")
            while True:
                try:
                    choice = int(input("> ").strip())
                    if 1 <= choice <= len(images):
                        break
                    print(f"Please enter a number between 1 and {len(images)}")
                except ValueError:
                    print("Please enter a valid number")
            
            selected = images[choice - 1]
            print(f"\nDownloading image: {selected.title}")
            
            # Download and analyze image
            image_path = None
            try:
                image_path = analyzer._download_image(selected.url, selected.title)
                print(f"\nSaved image to: {image_path}")
                print("\nAnalyzing image...")
                analysis = analyzer.analyze_image(image_path)
                
                # Print analysis results
                print("\nAnalysis Results:")
                print("=" * 30)
                print(f"\nScene Type: {analysis.scene_type}")
                
                print("\nMain Subjects:")
                for subject in sorted(analysis.main_subjects):
                    print(f"- {subject}")
                
                print("\nColors:")
                for color in analysis.colors:
                    print(f"- {color}")
                
                print("\nActivities:")
                for activity in analysis.activities:
                    print(f"• {activity}")
                
                print("\nObjects:")
                for obj in analysis.objects:
                    print(f"- {obj}")
                
                print(f"\nMood: {analysis.mood}")
                
                # Generate artistic interpretation
                print("\n" + "=" * 30)
                print("\nCreating artistic interpretation...")
                analyzer.generate_similar_image(analysis, image_path)
                
                # Allow questions
                while True:
                    print("\nAsk a question about the image (or type 'next' for new image):")
                    question = input("> ").strip()
                    
                    if question.lower() == 'next':
                        break
                    
                    answer = analyzer.ask_about_image(question)
                    print(f"\nAnswer: {answer}")
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                print("Please try another image or search query")
                continue
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query")
            continue
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    demonstrate_image_chat()
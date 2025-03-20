"""
LangChain Prompt Templates Example

This example demonstrates how to create and use different types of prompt templates
in LangChain. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

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

class StoryInput(BaseModel):
    """Schema for story generation input."""
    protagonist: str = Field(description="Main character of the story")
    setting: str = Field(description="Where the story takes place")
    theme: str = Field(description="Main theme or message of the story")

def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7  # More creative for story generation
    )

def create_simple_prompt():
    """Create a simple prompt template."""
    return PromptTemplate.from_template(
        "Tell me a short story about {protagonist} in {setting} that teaches us about {theme}."
    )

def create_structured_chat_prompt():
    """Create a structured chat prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a creative storyteller who crafts engaging tales."),
        ("human", "I want a story with these elements:"),
        ("human", "Protagonist: {protagonist}"),
        ("human", "Setting: {setting}"),
        ("human", "Theme: {theme}"),
        ("human", "Please write a short, engaging story.")
    ])

def create_dynamic_chat_prompt(previous_stories: List[str] = None):
    """Create a dynamic chat prompt with message history."""
    messages = [
        SystemMessagePromptTemplate.from_template(
            "You are a creative storyteller who learns from previous stories."
        ),
        MessagesPlaceholder(variable_name="story_history"),
        HumanMessagePromptTemplate.from_template(
            """Write a new story using these elements, but make it different from previous ones:
            Protagonist: {protagonist}
            Setting: {setting}
            Theme: {theme}"""
        )
    ]
    return ChatPromptTemplate.from_messages(messages)

def demonstrate_prompt_templates():
    """Demonstrate different prompt template capabilities."""
    try:
        print("\nDemonstrating LangChain Prompt Templates...\n")
        
        # Initialize model
        model = create_chat_model()
        
        # Example 1: Simple Prompt Template
        print("Example 1: Simple Prompt Template")
        print("-" * 50)
        
        simple_prompt = create_simple_prompt()
        story_input = StoryInput(
            protagonist="a curious cat",
            setting="a bustling city",
            theme="curiosity and discovery"
        )
        
        # Format and display prompt
        formatted_prompt = simple_prompt.format(**story_input.model_dump())
        print("Formatted Prompt:")
        print(formatted_prompt)
        
        # Get response
        chain = simple_prompt | model | StrOutputParser()
        response = chain.invoke(story_input.model_dump())
        
        print("\nGenerated Story:")
        print(response)
        print("=" * 50)
        
        # Example 2: Structured Chat Prompt
        print("\nExample 2: Structured Chat Prompt")
        print("-" * 50)
        
        chat_prompt = create_structured_chat_prompt()
        
        # Format and display messages
        messages = chat_prompt.format_messages(**story_input.model_dump())
        print("Formatted Messages:")
        for msg in messages:
            print(f"{msg.type.capitalize()}: {msg.content}")
        
        # Get response
        chat_chain = chat_prompt | model | StrOutputParser()
        response = chat_chain.invoke(story_input.model_dump())
        
        print("\nGenerated Story:")
        print(response)
        print("=" * 50)
        
        # Example 3: Dynamic Chat Prompt with History
        print("\nExample 3: Dynamic Chat Prompt with History")
        print("-" * 50)
        
        # Create history
        history = [
            SystemMessage(content="Previous story for context..."),
            HumanMessage(content="Tell me a different story."),
        ]
        
        dynamic_prompt = create_dynamic_chat_prompt()
        
        # Format with history
        messages = dynamic_prompt.format_messages(
            story_history=history,
            **story_input.model_dump()
        )
        
        print("Prompt with History:")
        for msg in messages:
            print(f"{msg.type.capitalize()}: {msg.content}")
        
        # Get response
        dynamic_chain = dynamic_prompt | model | StrOutputParser()
        response = dynamic_chain.invoke({
            "story_history": history,
            **story_input.model_dump()
        })
        
        print("\nGenerated Story:")
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_prompt_templates()

if __name__ == "__main__":
    main()
"""
LangChain Content Classifier Example

This example demonstrates how to combine agents and output parsers to create
a sophisticated content classifier that can analyze text and provide structured
categorization with metadata.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class ContentMetadata(BaseModel):
    """Schema for content metadata."""
    topics: List[str] = Field(description="Main topics discussed")
    sentiment: str = Field(description="Overall sentiment (positive/neutral/negative)")
    formality: str = Field(description="Writing style formality (formal/casual)")
    complexity: int = Field(description="Text complexity score (1-10)")
    keywords: List[str] = Field(description="Key terms and phrases")

class ContentClassification(BaseModel):
    """Schema for content classification results."""
    category: str = Field(description="Primary content category")
    subcategories: List[str] = Field(description="Related subcategories")
    metadata: ContentMetadata = Field(description="Detailed content metadata")
    confidence: float = Field(description="Classification confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_metadata_parser() -> PydanticOutputParser:
    """Create parser for content metadata."""
    return PydanticOutputParser(pydantic_object=ContentMetadata)

def create_classification_parser() -> PydanticOutputParser:
    """Create parser for classification results."""
    return PydanticOutputParser(pydantic_object=ContentClassification)

def create_analysis_tool(llm: AzureChatOpenAI) -> Tool:
    """Create a tool for analyzing content metadata."""
    metadata_parser = create_metadata_parser()
    
    prompt = PromptTemplate(
        template="""Analyze the following content and extract metadata:

Content: {content}

Provide a detailed analysis including:
1. Main topics
2. Overall sentiment
3. Writing style formality
4. Complexity score
5. Key terms and phrases

{format_instructions}""",
        input_variables=["content"],
        partial_variables={"format_instructions": metadata_parser.get_format_instructions()}
    )
    
    def analyze_content(content: str) -> str:
        """Analyze content and extract metadata."""
        response = llm.invoke(
            prompt.format_prompt(content=content).to_string()
        )
        return response.content
    
    return Tool(
        name="analyze_content",
        description="Analyze content to extract metadata",
        func=analyze_content
    )

def create_classifier_tool(llm: AzureChatOpenAI) -> Tool:
    """Create a tool for content classification."""
    classification_parser = create_classification_parser()
    
    prompt = PromptTemplate(
        template="""Classify the following content with metadata:

Content: {content}
Metadata: {metadata}

Determine:
1. Primary category
2. Related subcategories
3. Classification confidence

{format_instructions}""",
        input_variables=["content", "metadata"],
        partial_variables={"format_instructions": classification_parser.get_format_instructions()}
    )
    
    def classify_content(content: str, metadata: str) -> str:
        """Classify content using metadata."""
        response = llm.invoke(
            prompt.format_prompt(
                content=content,
                metadata=metadata
            ).to_string()
        )
        return response.content
    
    return Tool(
        name="classify_content",
        description="Classify content using metadata",
        func=classify_content
    )

def create_classifier_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """Create an agent for content classification."""
    tools = [
        create_analysis_tool(llm),
        create_classifier_tool(llm)
    ]
    
    prompt = PromptTemplate(
        template="""You are a content classification expert.

Use the following format:
Thought: Consider what to do
Action: Choose a tool: {tool_names}
Action Input: Tool input
Observation: Tool output
... (repeat until classification is complete)
Final Answer: Detailed classification results

Task: Analyze and classify this content: {input}

{agent_scratchpad}""",
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tool_names": ", ".join([t.name for t in tools])}
    )
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def classify_content(content: str) -> ContentClassification:
    """Classify content using the agent-based system."""
    try:
        print(f"\nAnalyzing content ({len(content)} characters)...")
        
        # Initialize components
        llm = create_chat_model()
        agent = create_classifier_agent(llm)
        
        # Run classification
        result = agent.invoke({"input": content})
        
        # Parse final classification
        classification_parser = create_classification_parser()
        classification = classification_parser.parse(result["output"])
        
        return classification
        
    except Exception as e:
        print(f"Error classifying content: {str(e)}")
        raise

def demonstrate_classifier():
    """Demonstrate the Content Classifier capabilities."""
    try:
        print("\nInitializing Content Classifier...\n")
        
        # Example content to classify
        content = """
The Impact of Artificial Intelligence on Modern Healthcare

AI is revolutionizing healthcare delivery and patient outcomes. Machine learning 
algorithms are enhancing diagnostic accuracy, while natural language processing 
is improving clinical documentation. Healthcare providers are leveraging AI to 
analyze medical images, predict patient risks, and optimize treatment plans.

Recent studies show a 35% improvement in early disease detection when AI assists 
medical professionals. Additionally, automated systems have reduced administrative 
workload by 45%, allowing healthcare workers to focus more on patient care.

However, challenges remain regarding data privacy, algorithmic bias, and the need 
for human oversight. Healthcare organizations must carefully balance technological 
innovation with ethical considerations and regulatory compliance.
"""
        
        # Classify content
        classification = classify_content(content)
        
        # Display results
        print("\nClassification Results:")
        print(f"Category: {classification.category}")
        print("\nSubcategories:")
        for subcat in classification.subcategories:
            print(f"- {subcat}")
        
        print("\nMetadata:")
        print(f"Topics: {', '.join(classification.metadata.topics)}")
        print(f"Sentiment: {classification.metadata.sentiment}")
        print(f"Formality: {classification.metadata.formality}")
        print(f"Complexity: {classification.metadata.complexity}/10")
        
        print("\nKeywords:")
        for keyword in classification.metadata.keywords:
            print(f"- {keyword}")
        
        print(f"\nConfidence: {classification.confidence:.2%}")
        print(f"Timestamp: {classification.timestamp}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Content Classifier...")
    demonstrate_classifier()

if __name__ == "__main__":
    main()
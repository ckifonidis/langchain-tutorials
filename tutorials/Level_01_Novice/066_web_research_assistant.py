#!/usr/bin/env python3
"""
LangChain Web Research Assistant (LangChain v3)

This example demonstrates how to build a web research assistant using tool calling
and callbacks. It performs web searches, extracts information, and tracks progress
using callbacks.

Key concepts demonstrated:
1. Tool Calling: Using tools for web search and data extraction
2. Callbacks: Monitoring and tracking research progress
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

class ResearchProgress(BaseCallbackHandler):
    """Callback handler for tracking research progress."""
    
    def __init__(self):
        """Initialize progress tracker."""
        self.start_time = None
        self.current_step = None
        self.steps_completed = 0
        self.sources_found = 0
    
    def on_llm_start(self, *args, **kwargs):
        """Called when LLM starts processing."""
        if not self.start_time:
            self.start_time = datetime.now()
            print("\nStarting research process...")
    
    def on_tool_start(self, serialized: Dict[str, Any], *args, **kwargs):
        """Called when a tool starts being used."""
        tool_name = serialized.get('name', 'unknown tool')
        self.current_step = f"Searching with {tool_name}"
        print(f"\nStep {self.steps_completed + 1}: {self.current_step}")
    
    def on_tool_end(self, *args, **kwargs):
        """Called when a tool completes its task."""
        self.steps_completed += 1
        self.sources_found += 1
        print(f"Found {self.sources_found} sources")
    
    def on_chain_end(self, *args, **kwargs):
        """Called when a chain completes processing."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"\nResearch completed in {duration.total_seconds():.1f} seconds")
            print(f"Total sources analyzed: {self.sources_found}")

class ResearchSource(BaseModel):
    """Information source details."""
    title: str = Field(description="Source title or headline")
    url: str = Field(description="Source URL")
    snippet: str = Field(description="Relevant excerpt from source")
    date: str = Field(description="Publication date if available", default="Not specified")

class ResearchSummary(BaseModel):
    """Research summary with findings."""
    topic: str = Field(description="Research topic")
    main_findings: List[str] = Field(description="Main research findings")
    sources: List[ResearchSource] = Field(description="Information sources")
    key_insights: List[str] = Field(description="Key insights from research")

class WebResearchAssistant:
    """Web research assistant using tool calling and callbacks."""
    
    def __init__(self):
        """Initialize the research assistant."""
        # Create callback handler
        self.progress = ResearchProgress()
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0,
            callbacks=[self.progress]
        )
        
        # Initialize search tools
        self.search_results = DuckDuckGoSearchResults(
            max_results=5  # Limit results for better processing
        )
        
        # Initialize parser
        self.parser = PydanticOutputParser(pydantic_object=ResearchSummary)
    
    def _format_search_results(self, results: List[Dict]) -> List[ResearchSource]:
        """Format search results into ResearchSource objects."""
        sources = []
        for result in results:
            source = ResearchSource(
                title=result.get("title", "Untitled"),
                url=result.get("link", "No URL"),
                snippet=result.get("snippet", "No excerpt available"),
                date=result.get("date", "Not specified")
            )
            sources.append(source)
        return sources
    
    def research_topic(
        self,
        topic: str,
        max_sources: int = 3
    ) -> ResearchSummary:
        """Conduct web research on a topic."""
        try:
            # Get search results
            raw_results = self.search_results.invoke(topic)
            
            # Create research prompt
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research analyst. Analyze the search results and create a structured summary.

Format your response as a valid JSON object with these fields:
{{
    "topic": "research topic",
    "main_findings": ["finding 1", "finding 2", ...],
    "sources": [array of source objects],
    "key_insights": ["insight 1", "insight 2", ...]
}}

{format_instructions}"""),
                ("human", """Research Topic: {topic}

Search Results:
{results}

Create a structured analysis in the specified JSON format.""")
            ])
            
            # Create research chain
            chain = (
                research_prompt 
                | self.llm 
                | self.parser
            )
            
            summary = chain.invoke({
                "topic": topic,
                "results": raw_results,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return summary
            
        except Exception as e:
            print(f"Error conducting research: {str(e)}")
            raise

def demonstrate_research_assistant():
    """Demonstrate the web research assistant."""
    print("\nWeb Research Assistant Demo")
    print("=" * 50)
    
    # Test topics
    topics = [
        {
            "name": "Latest AI Developments",
            "query": "Most significant artificial intelligence developments 2024",
            "sources": 2
        },
        {
            "name": "Renewable Energy Progress",
            "query": "Latest breakthroughs in renewable energy technology 2024",
            "sources": 2
        }
    ]
    
    # Create assistant
    assistant = WebResearchAssistant()
    
    # Research each topic
    for topic in topics:
        print(f"\nResearching: {topic['name']}")
        print("-" * 50)
        
        try:
            results = assistant.research_topic(
                topic["query"],
                topic["sources"]
            )
            
            # Display results
            print("\nResearch Summary:")
            print(f"Topic: {results.topic}")
            
            print("\nMain Findings:")
            for finding in results.main_findings:
                print(f"- {finding}")
            
            print("\nKey Insights:")
            for insight in results.key_insights:
                print(f"- {insight}")
            
            print("\nSources:")
            for source in results.sources:
                print(f"\nTitle: {source.title}")
                print(f"URL: {source.url}")
                if source.date != "Not specified":
                    print(f"Date: {source.date}")
                print(f"Excerpt: {source.snippet}")
            
        except Exception as e:
            print(f"Research failed: {str(e)}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    demonstrate_research_assistant()
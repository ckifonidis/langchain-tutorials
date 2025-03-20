"""
LangChain Agents Example

This example demonstrates how to create and use agents in LangChain for task
automation and decision making. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, ToolException
from langchain.agents import create_openai_functions_agent, AgentExecutor

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

class Calculator(BaseTool):
    """Tool for performing basic calculations."""
    name: str = "calculator"
    description: str = "Useful for performing basic arithmetic operations"
    
    def _run(self, query: str) -> str:
        """Run the calculator."""
        try:
            # Safely evaluate basic arithmetic
            result = eval(query, {"__builtins__": {}}, {"abs": abs})
            return str(result)
        except Exception as e:
            raise ToolException(f"Error performing calculation: {str(e)}")

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    name: str = "weather"
    description: str = "Get the weather for a specific location"
    
    def _run(self, location: str) -> str:
        """Simulate getting weather information."""
        # This is a mock implementation
        weather_data = {
            "New York": "Sunny, 75°F",
            "London": "Rainy, 60°F",
            "Tokyo": "Cloudy, 70°F",
            "Paris": "Clear, 68°F"
        }
        return weather_data.get(location, f"Weather data not available for {location}")

def create_agent(tools: List[BaseTool]) -> AgentExecutor:
    """
    Create an agent with the specified tools.
    
    Args:
        tools: List of tools to provide to the agent
        
    Returns:
        AgentExecutor: The configured agent
    """
    # Initialize the model
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create the prompt template with both input and agent_scratchpad variables.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that can use tools to accomplish tasks.
Always try to use the most appropriate tool for the job.
If no tool is suitable, provide a direct response."""),
        ("human", "{input}\n{agent_scratchpad}")
    ])
    
    # Create the agent using OpenAI functions.
    agent = create_openai_functions_agent(model, tools, prompt)
    
    # Create the agent executor.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def demonstrate_agent_capabilities():
    """Demonstrate various agent capabilities."""
    try:
        print("\nDemonstrating LangChain Agents...\n")
        
        # Create tools
        calculator = Calculator()
        weather_tool = WeatherTool()
        tools = [calculator, weather_tool]
        
        # Create agent
        agent_executor = create_agent(tools)
        
        # Example 1: Using Calculator Tool
        print("Example 1: Using Calculator Tool")
        print("-" * 50)
        
        calc_questions = [
            "What is 15 * 7?",
            "Calculate 123 + 456",
            "What is the absolute value of -42?"
        ]
        
        for question in calc_questions:
            print(f"\nQuestion: {question}")
            response = agent_executor.invoke({"input": question})
            print(f"Response: {response['output']}")
        print("=" * 50)
        
        # Example 2: Using Weather Tool
        print("\nExample 2: Using Weather Tool")
        print("-" * 50)
        
        weather_questions = [
            "What's the weather like in New York?",
            "Tell me the weather in London",
            "What's the temperature in Tokyo?"
        ]
        
        for question in weather_questions:
            print(f"\nQuestion: {question}")
            response = agent_executor.invoke({"input": question})
            print(f"Response: {response['output']}")
        print("=" * 50)
        
        # Example 3: Complex Queries
        print("\nExample 3: Complex Queries")
        print("-" * 50)
        
        complex_questions = [
            "If it's 68°F in Paris, what's that temperature minus 32 divided by 1.8 in Celsius?",
            "The temperature in Tokyo is 70°F and in New York is 75°F. What's the difference?",
            "If it's 60°F in London, what would be half of that temperature?"
        ]
        
        for question in complex_questions:
            print(f"\nQuestion: {question}")
            response = agent_executor.invoke({"input": question})
            print(f"Response: {response['output']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_agent_capabilities()

if __name__ == "__main__":
    main()

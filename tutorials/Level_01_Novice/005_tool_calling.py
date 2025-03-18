"""
LangChain Tool Calling Example

This example demonstrates how to implement tool calling in LangChain,
showing how language models can programmatically use tools through
structured function calling. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
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

class WeatherInput(BaseModel):
    """Schema for weather tool input."""
    city: str = Field(
        description="The city to get weather for",
        examples=["New York", "London", "Tokyo"]
    )

class TimeInput(BaseModel):
    """Schema for time tool input."""
    format_type: str = Field(
        default="24h",
        description="Time format (12h or 24h)",
        pattern="^(12h|24h)$"
    )

def format_tool_schema(schema: BaseModel) -> Dict[str, Any]:
    """Format a Pydantic schema into OpenAI function parameter schema."""
    schema_dict = schema.model_json_schema()
    return {
        "name": schema.__name__.lower(),
        "parameters": schema_dict,
    }

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    name: str = Field(default="weather")
    description: str = Field(default="Get current weather information for a specific city")
    args_schema: type[BaseModel] = Field(default=WeatherInput)

    def _run(self, city: str) -> Dict[str, Any]:
        """
        Get current weather information for a specific city.
        
        Args:
            city: Name of the city to get weather for
            
        Returns:
            Dictionary containing temperature and weather condition
            
        Raises:
            ValueError: If city is not found in the database
        """
        weather_data = {
            "New York": {"temp": 22.5, "condition": "sunny"},
            "London": {"temp": 18.0, "condition": "cloudy"},
            "Tokyo": {"temp": 25.0, "condition": "rainy"},
            "Paris": {"temp": 20.0, "condition": "partly cloudy"}
        }
        
        city = city.strip().title()
        if city not in weather_data:
            available_cities = list(weather_data.keys())
            raise ValueError(
                f"No weather data available for {city}. "
                f"Available cities: {', '.join(available_cities)}"
            )
        
        return weather_data[city]

class TimeTool(BaseTool):
    """Tool for getting current time."""
    
    name: str = Field(default="time")
    description: str = Field(default="Get current time in specified format (12h or 24h)")
    args_schema: type[BaseModel] = Field(default=TimeInput)

    def _run(self, format_type: str = "24h") -> str:
        """
        Get current time in specified format.
        
        Args:
            format_type: Time format, either "12h" or "24h"
            
        Returns:
            Current time as string in specified format
            
        Raises:
            ValueError: If format_type is invalid
        """
        if format_type not in ["12h", "24h"]:
            raise ValueError('format_type must be either "12h" or "24h"')
        
        current_time = datetime.now()
        return current_time.strftime("%I:%M %p" if format_type == "12h" else "%H:%M")

def prepare_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """Prepare tools for Azure OpenAI format."""
    prepared_tools = []
    
    for tool in tools:
        schema = tool.args_schema.model_json_schema()
        
        # Ensure the schema is properly formatted
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
        
        function_def = {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters
        }
        
        prepared_tools.append({"type": "function", "function": function_def})
        
    print("DEBUG - Prepared tools:", prepared_tools)
    return prepared_tools

def handle_tool_call(tool_call: Dict[str, Any], tools: Dict[str, BaseTool]) -> ToolMessage:
    """Handle a tool call and create appropriate tool message."""
    if not tool_call:
        raise ValueError("Tool call cannot be None")
    
    print("DEBUG - Tool call received:", tool_call)
    print("DEBUG - Available tools:", list(tools.keys()))
    
    # Extract tool name from the response structure
    if isinstance(tool_call, dict):
        if "function" in tool_call:
            tool_name = tool_call["function"].get("name")
        elif "name" in tool_call:
            tool_name = tool_call.get("name")
        else:
            print("DEBUG - Invalid tool call structure:", tool_call)
            raise ValueError("Invalid tool call structure")
    else:
        tool_name = getattr(tool_call, "name", None)
        if not tool_name:
            raise ValueError("Could not extract tool name from response")
    
    print("DEBUG - Extracted tool name:", tool_name)
    
    if not tool_name or tool_name not in tools:
        raise ValueError(f"Unknown or invalid tool: {tool_name}")
        
    arguments = tool_call.get("function", {}).get("arguments", "{}") if isinstance(tool_call, dict) else "{}"
    if isinstance(arguments, str):
        try:
            import json
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
        except Exception as e:
            raise ValueError(f"Failed to parse tool arguments: {str(e)}")
    
    tool = tools[tool_name]
    result = tool._run(**arguments)
    print("DEBUG - Tool execution result:", result)
    return ToolMessage(content=str(result), tool_call_id=tool_call.get("id"))

def handle_chat_response(response, tools_dict, messages):
    """Handle chat model response and process any tool calls."""
    print("DEBUG - Processing response:", response)
    
    if not response:
        print("DEBUG - Empty response received")
        return
    
    if not hasattr(response, "tool_calls"):
        print("DEBUG - No tool_calls attribute found")
        return
    
    print("DEBUG - Tool calls found:", response.tool_calls)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_msg = handle_tool_call(tool_call, tools_dict)
            messages.extend([AIMessage(content="", tool_calls=[tool_call]), tool_msg])
    else:
        print("DEBUG - No tool calls to process")

def demonstrate_tool_calling() -> None:
    """
    Demonstrate how models can use tools through function calling.
    Shows weather queries, time formatting, and combined tool usage.
    """
    # Initialize tools and model
    print("\nDEBUG - Environment variables:")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    print(f"AZURE_OPENAI_ENDPOINT: {'Set' if endpoint else 'Not set'}")
    print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {'Set' if deployment else 'Not set'}")
    print(f"AZURE_OPENAI_API_VERSION: {api_version}")
    
    raw_tools = [WeatherTool(), TimeTool()]
    tools = prepare_tools(raw_tools)
    tools_dict = {tool.name: tool for tool in raw_tools}
    
    try:
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.0
        )
        print("\nDEBUG - Chat model initialized")
        print("DEBUG - Model deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
        print("DEBUG - API version:", os.getenv("AZURE_OPENAI_API_VERSION"))

        # Create system message that knows about available tools
        system_msg = SystemMessage(content="""
            Assistant with access to specific tools:
            1. Weather tool - Get current weather for: New York, London, Tokyo, Paris
            2. Time tool - Get current time in 12h or 24h format
            
            When using tools, format the results naturally in your response.
        """)
        
        # Example 1: Weather query
        print("\nExample 1: Weather Query")
        human_msg1 = HumanMessage(content="What's the weather like in Tokyo right now?")
        messages = [system_msg, human_msg1]
        
        # Configure the model for tool use
        response1 = chat_model.invoke(
            messages,
            tools=tools,
            tool_choice={"type": "auto"},
            stream=False
        )
        print("DEBUG - Response type:", type(response1))
        print("DEBUG - Response:", response1)
        
        if response1 and hasattr(response1, "tool_calls") and response1.tool_calls:
            print("DEBUG - Tool calls:", response1.tool_calls)
            for tool_call in response1.tool_calls:
                print("DEBUG - Processing tool call:", tool_call)
                print("DEBUG - Tool call type:", type(tool_call))
                tool_msg = handle_tool_call(tool_call, tools_dict)
                messages.extend([AIMessage(content="", tool_calls=[tool_call]), tool_msg])
        messages.append(response1)
        print("Response:", response1.content)
        
        # Example 2: Time query
        print("\nExample 2: Time Query")
        human_msg2 = HumanMessage(content="What time is it now in 12-hour format?")
        messages.append(human_msg2)
        
        response2 = chat_model.invoke(
            messages,
            tools=tools,
            tool_choice={"type": "auto"},
            stream=False
        )
        print("DEBUG - Response2 type:", type(response2))
        print("DEBUG - Response2:", response2)
        
        print("DEBUG - Checking tool_calls attribute:", hasattr(response2, "tool_calls"))
        if response2 and hasattr(response2, "tool_calls") and response2.tool_calls:
            for tool_call in response2.tool_calls:
                tool_msg = handle_tool_call(tool_call, tools_dict)
                messages.extend([AIMessage(content="", tool_calls=[tool_call]), tool_msg])
        messages.append(response2)
        print("Response:", response2.content)
        
        # Example 3: Combined query
        print("\nExample 3: Combined Query")
        human_msg3 = HumanMessage(content="What's the weather in London and what time is it?")
        messages.append(human_msg3)
        
        response3 = chat_model.invoke(
            messages,
            tools=tools,
            tool_choice={"type": "auto"},
            stream=False
        )
        
        if response3 and hasattr(response3, "tool_calls") and response3.tool_calls:
            for tool_call in response3.tool_calls:
                tool_msg = handle_tool_call(tool_call, tools_dict)
                messages.extend([AIMessage(content="", tool_calls=[tool_call]), tool_msg])
        messages.append(response3)
        print("Response:", response3.content)
        
    except ValueError as ve:
        print(f"\nValidation error: {str(ve)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Tool Calling...")
    demonstrate_tool_calling()

if __name__ == "__main__":
    main()

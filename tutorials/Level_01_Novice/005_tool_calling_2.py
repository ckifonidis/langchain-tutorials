"""
LangChain Tool Calling Example

This example demonstrates how to implement tool calling inpp LangChain,
showing how language models can programmatically use tools through
structured function calling.
"""

import os
import json
from typing import Union, Dict, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function

# Load environment variables from the .env file
load_dotenv()

# Check if required Azure OpenAI environment variables are available
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

class WeatherTool(BaseTool):
    """Simple weather information tool (simulated data)."""

    name = "weather"
    description = "Get current weather information for a specific city"

    def _run(self, city: str) -> str:
        """Simulate getting weather data for a city and return a formatted string."""
        weather_data = {
            "New York": {"temp": 22.5, "condition": "sunny"},
            "London": {"temp": 18.0, "condition": "cloudy"},
            "Tokyo": {"temp": 25.0, "condition": "rainy"},
            "Paris": {"temp": 20.0, "condition": "partly cloudy"}
        }
        city = city.strip().title()
        if city not in weather_data:
            return (f"No weather data available for {city}. "
                    f"Available cities: {', '.join(weather_data.keys())}.")
        data = weather_data[city]
        return f"In {city}, it's currently {data['condition']} with a temperature of {data['temp']}°C."

    async def _arun(self, city: str) -> str:
        return self._run(city)

class TimeTool(BaseTool):
    """Tool for getting current time in different formats."""

    name = "time"
    description = "Get current time in specified format (12h or 24h)"

    def _run(self, format_type: str = "24h") -> str:
        """Get current time in specified format."""
        current_time = datetime.now()
        if format_type.lower() == "12h":
            return current_time.strftime("%I:%M %p")
        return current_time.strftime("%H:%M")

    async def _arun(self, format_type: str = "24h") -> str:
        return self._run(format_type)

def init_chat_model():
    """Initialize the Azure OpenAI chat model with function calling."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

def process_response(messages, functions, tools, chat_model):
    """
    Iteratively check if the response contains a function call.
    If so, execute the corresponding tool, append its output as a message,
    and re-invoke the model until no function call remains.
    """
    response = chat_model.invoke(messages, functions=functions)
    while response.additional_kwargs and "function_call" in response.additional_kwargs:
        fc = response.additional_kwargs["function_call"]
        fname = fc.get("name")
        fargs_str = fc.get("arguments", "{}")
        try:
            fargs = json.loads(fargs_str)
        except Exception:
            fargs = {}
        tool_result = None
        for tool in tools:
            if tool.name == fname:
                tool_result = tool._run(**fargs)
                break
        if tool_result is not None:
            tool_msg = HumanMessage(content=tool_result)
            messages.append(tool_msg)
            response = chat_model.invoke(messages, functions=functions)
        else:
            break
    return response

def demonstrate_tool_calling():
    """Demonstrate how models can use tools through function calling."""
    # Initialize tools and model
    weather_tool = WeatherTool()
    time_tool = TimeTool()
    tools = [weather_tool, time_tool]

    chat_model = init_chat_model()

    # Convert tools to OpenAI functions using the new conversion function.
    functions = [convert_to_openai_function(t) for t in tools]

    # Create a system message that informs the assistant about the available tools.
    system_msg = SystemMessage(content="""
        You are a helpful assistant with access to tools for checking weather 
        and time. Use these tools when asked about weather conditions or 
        current time. Available cities for weather: New York, London, Tokyo, Paris.
    """)

    try:
        # Example 1: Weather query
        human_msg1 = HumanMessage(content="What's the weather like in Tokyo right now?")
        messages = [system_msg, human_msg1]
        response1 = process_response(messages, functions, tools, chat_model)
        print("\nFirst Example (Weather):", response1.content)

        # Example 2: Time query
        human_msg2 = HumanMessage(content="What time is it now in 12-hour format?")
        messages.append(response1)
        messages.append(human_msg2)
        response2 = process_response(messages, functions, tools, chat_model)
        print("\nSecond Example (Time):", response2.content)

        # Example 3: Combined query
        human_msg3 = HumanMessage(content="What's the weather in London and what time is it?")
        messages.append(response2)
        messages.append(human_msg3)
        response3 = process_response(messages, functions, tools, chat_model)
        print("\nThird Example (Combined):", response3.content)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def main():
    print("\nDemonstrating LangChain Tool Calling...")
    demonstrate_tool_calling()

if __name__ == "__main__":
    main()


# Expected Output:
# First Example (Weather): In Tokyo, it's currently rainy with a temperature of 25.0°C.
# Second Example (Time): The current time is 02:30 PM.
# Third Example (Combined): In London, it's cloudy with a temperature of 18.0°C, 
# and the current time is 02:30 PM.

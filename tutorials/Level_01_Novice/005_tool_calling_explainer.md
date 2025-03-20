# Understanding Tool Calling in LangChain

Welcome to this comprehensive guide on tool calling in LangChain! This tutorial explains how language models can use tools programmatically through structured function calling. We'll break down every component and concept to make it easily understandable.

## Core Concepts

1. **What are Tools?**
   Tools are functions that language models can use to:
   - **Perform Actions**: Get data or execute operations
   - **Access Information**: Retrieve external data
   - **Process Data**: Transform or analyze information
   - **Interact**: Interface with external systems

2. **Key Components**
   ```python
   from langchain_core.tools import BaseTool
   from langchain_openai import AzureChatOpenAI
   from langchain_core.messages import HumanMessage, SystemMessage
   from langchain_core.utils.function_calling import convert_to_openai_function
   ```

## Code Breakdown

1. **Tool Definition**
   ```python
   class WeatherTool(BaseTool):
       name: str = "weather"
       description: str = "Get current weather information for a specific city"
       
       def _run(self, city: str) -> str:
           weather_data = {
               "New York": {"temp": 22.5, "condition": "sunny"},
               "London": {"temp": 18.0, "condition": "cloudy"},
               # ... more cities
           }
           city = city.strip().title()
           if city not in weather_data:
               return f"No weather data available for {city}..."
           data = weather_data[city]
           return f"In {city}, it's currently {data['condition']}..."
   ```
   
   Key points:
   - Inherits from BaseTool
   - Defines name and description
   - Implements _run method
   - Handles input validation
   - Returns formatted output

2. **Message Processing**
   ```python
   def process_response(messages, functions, tools, chat_model):
       response = chat_model.invoke(messages, functions=functions)
       while response.additional_kwargs.get("function_call"):
           # Extract function call details
           fc = response.additional_kwargs["function_call"]
           fname = fc.get("name")
           fargs_str = fc.get("arguments", "{}")
           
           # Parse arguments and execute tool
           fargs = json.loads(fargs_str)
           tool_result = None
           for tool in tools:
               if tool.name == fname:
                   tool_result = tool._run(**fargs)
                   break
           
           # Append result and continue conversation
           if tool_result:
               messages.append(HumanMessage(content=tool_result))
               response = chat_model.invoke(messages, functions=functions)
   ```
   
   Process flow:
   1. Model generates response
   2. Check for function calls
   3. Extract function details
   4. Execute appropriate tool
   5. Add result to conversation
   6. Continue until complete

3. **Tool Registration**
   ```python
   # Initialize tools
   weather_tool = WeatherTool()
   time_tool = TimeTool()
   tools = [weather_tool, time_tool]

   # Convert to OpenAI functions
   functions = [convert_to_openai_function(t) for t in tools]
   ```
   
   Important steps:
   - Create tool instances
   - Collect tools in list
   - Convert to function format
   - Prepare for model use

## Example Usage

```python
# System setup
system_msg = SystemMessage(content="""
    You are a helpful assistant with access to tools for checking weather 
    and time. Use these tools when asked about weather conditions or 
    current time. Available cities: New York, London, Tokyo, Paris.
""")

# Weather query
human_msg = HumanMessage(content="What's the weather like in Tokyo?")
messages = [system_msg, human_msg]
response = process_response(messages, functions, tools, chat_model)

# Expected output:
# "In Tokyo, it's currently rainy with a temperature of 25.0°C."
```

## Advanced Features

1. **Asynchronous Support**
   ```python
   async def _arun(self, city: str) -> str:
       return self._run(city)  # Async version of tool execution
   ```

2. **Error Handling**
   ```python
   try:
       fargs = json.loads(fargs_str)
   except Exception:
       fargs = {}  # Fallback to empty args
   ```

3. **Multiple Tools**
   ```python
   # Handle combined queries
   "What's the weather in London and what time is it?"
   # Model will use both tools sequentially
   ```

## Best Practices

1. **Tool Definition**
   - Clear names and descriptions
   - Input validation
   - Error handling
   - Documentation

2. **Message Management**
   - Track conversation context
   - Handle tool outputs
   - Maintain message order
   - Clean responses

3. **System Messages**
   - Clear instructions
   - Tool availability
   - Usage guidelines
   - Limitations

## Common Patterns

1. **Sequential Tool Use**
   ```python
   # Weather check followed by time
   response1 = weather_tool._run("London")
   response2 = time_tool._run("12h")
   ```

2. **Error Recovery**
   ```python
   if city not in weather_data:
       return f"No weather data available for {city}. Available cities: ..."
   ```

## Example Outputs

```plaintext
First Example (Weather):
In Tokyo, it's currently rainy with a temperature of 25.0°C.

Second Example (Time):
The current time is 02:30 PM.

Third Example (Combined):
In London, it's cloudy with a temperature of 18.0°C, and the current time is 02:30 PM.
```

## Common Issues and Solutions

1. **Tool Not Found**
   ```python
   if tool_result is None:
       return f"Tool '{fname}' not available"
   ```

2. **Invalid Arguments**
   ```python
   try:
       fargs = json.loads(fargs_str)
   except json.JSONDecodeError:
       return "Invalid arguments provided"
   ```

3. **Response Processing**
   ```python
   # Handle empty or invalid responses
   if not response.additional_kwargs:
       return response.content
   ```

## Resources

1. **Official Documentation**
   - **Main Guide**: https://python.langchain.com/docs/concepts/tools/
   - **Overview**: https://python.langchain.com/docs/concepts/tools/#overview
   - **Key Concepts**: https://python.langchain.com/docs/concepts/tools/#key-concepts
   - **Tool Interface**: https://python.langchain.com/docs/concepts/tools/#tool-interface

2. **Tool Creation and Usage**
   - **@tool Decorator**: https://python.langchain.com/docs/concepts/tools/#create-tools-using-the-tool-decorator
   - **Direct Tool Usage**: https://python.langchain.com/docs/concepts/tools/#use-the-tool-directly
   - **Tool Inspection**: https://python.langchain.com/docs/concepts/tools/#inspect
   - **Schema Configuration**: https://python.langchain.com/docs/concepts/tools/#configuring-the-schema

3. **Advanced Features**
   - **Tool Artifacts**: https://python.langchain.com/docs/concepts/tools/#tool-artifacts
   - **Special Type Annotations**: https://python.langchain.com/docs/concepts/tools/#special-type-annotations
   - **InjectedToolArg**: https://python.langchain.com/docs/concepts/tools/#injectedtoolarg
   - **RunnableConfig**: https://python.langchain.com/docs/concepts/tools/#runnableconfig

4. **Additional Resources**
   - **InjectedState**: https://python.langchain.com/docs/concepts/tools/#injectedstate
   - **InjectedStore**: https://python.langchain.com/docs/concepts/tools/#injectedstore
   - **Best Practices**: https://python.langchain.com/docs/concepts/tools/#best-practices
   - **Toolkits**: https://python.langchain.com/docs/concepts/tools/#toolkits
   - **Interface Guide**: https://python.langchain.com/docs/concepts/tools/#interface
   - **Related Resources**: https://python.langchain.com/docs/concepts/tools/#related-resources

Remember:
- Define clear tool interfaces
- Handle errors gracefully
- Process responses carefully
- Maintain conversation context
- Document tool capabilities
- Test edge cases
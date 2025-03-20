# Understanding Agents in LangChain

Welcome to this comprehensive guide on using agents in LangChain! Agents are autonomous components that can use tools and make decisions to accomplish tasks. This tutorial will help you understand how to create and use agents effectively.

## Core Concepts

1. **What are Agents?**
   Think of agents as smart assistants that can:
   
   - **Make Decisions**: Choose appropriate tools for tasks
   - **Use Tools**: Execute actions using available tools
   - **Track Progress**: Use scratchpad to maintain context
   - **Handle Feedback**: Process results and adjust actions

2. **Tool Definition**
   ```python
   class Calculator(BaseTool):
       """Tool for performing basic calculations."""
       name: str = "calculator"  # Explicit str type
       description: str = "Useful for performing basic arithmetic operations"
       
       def _run(self, query: str) -> str:
           try:
               result = eval(query, {"__builtins__": {}}, {"abs": abs})
               return str(result)
           except Exception as e:
               raise ToolException(f"Error: {str(e)}")
   ```
   
   Key features:
   - Explicit type annotations
   - Clear description
   - Safe implementation
   - Error handling

3. **Agent Setup**
   ```python
   def create_agent(tools: List[BaseTool]) -> AgentExecutor:
       model = AzureChatOpenAI(
           azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
           openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
           azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
           api_key=os.getenv("AZURE_OPENAI_API_KEY"),
           temperature=0
       )
       
       # Include agent_scratchpad in prompt
       prompt = ChatPromptTemplate.from_messages([
           ("system", "You are a helpful AI assistant..."),
           ("human", "{input}\n{agent_scratchpad}")
       ])
       
       agent = create_openai_functions_agent(model, tools, prompt)
       return AgentExecutor(
           agent=agent,
           tools=tools,
           verbose=True,
           handle_parsing_errors=True
       )
   ```

## Implementation Breakdown

1. **Agent Prompt Template**
   ```python
   prompt = ChatPromptTemplate.from_messages([
       ("system", """You are a helpful AI assistant that can use tools to accomplish tasks.
Always try to use the most appropriate tool for the job.
If no tool is suitable, provide a direct response."""),
       ("human", "{input}\n{agent_scratchpad}")
   ])
   ```
   
   Important aspects:
   - System message defines behavior
   - agent_scratchpad tracks progress
   - Clear instructions
   - Flexible response options

2. **Tool Implementation**
   ```python
   class WeatherTool(BaseTool):
       name: str = "weather"  # Type annotation
       description: str = "Get the weather for a specific location"
       
       def _run(self, location: str) -> str:
           weather_data = {
               "New York": "Sunny, 75°F",
               "London": "Rainy, 60°F"
           }
           return weather_data.get(location, 
                                 f"Weather data not available for {location}")
   ```
   
   Key points:
   - Explicit type hints
   - Clear error messages
   - Predictable behavior
   - Documentation strings

3. **Agent Execution**
   ```python
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,     # Show thought process
       handle_parsing_errors=True  # Better error handling
   )
   
   response = agent_executor.invoke({
       "input": "What's the weather in New York?"
   })
   ```

## Best Practices

1. **Tool Design**
   ```python
   class CustomTool(BaseTool):
       # Always use explicit type annotations
       name: str = "tool_name"
       description: str = "Clear description"
       
       def _run(self, input_str: str) -> str:
           try:
               # Implementation
               return result
           except Exception as e:
               raise ToolException(str(e))
   ```
   
   Guidelines:
   - Use type hints
   - Provide clear descriptions
   - Handle errors properly
   - Document behavior

2. **Prompt Engineering**
   ```python
   system_message = """You are a helpful AI assistant that can use tools to accomplish tasks.
Always try to use the most appropriate tool for the job.
If no tool is suitable, provide a direct response."""
   ```
   
   Tips:
   - Clear instructions
   - Define behavior
   - Set expectations
   - Include examples

3. **Error Management**
   ```python
   try:
       response = agent_executor.invoke({"input": query})
   except Exception as e:
       print(f"Error during execution: {str(e)}")
       # Implement fallback behavior
   ```

## Example Output

When running `python 021_agents.py`, you'll see:

```
Demonstrating LangChain Agents...

Example 1: Using Calculator Tool
--------------------------------------------------
Question: What is 15 * 7?

> Entering new AgentExecutor chain...
Thought: I should use the calculator tool to perform this multiplication.
Action: calculator
Action Input: 15 * 7
Observation: 105
Thought: I have the result.
Final Answer: The result of 15 * 7 is 105.
> Finished chain.

Response: The result of 15 * 7 is 105.
```

## Common Patterns

1. **Complex Queries**
   ```python
   # Combining multiple tools
   question = "If it's 68°F in Paris, what's that in Celsius?"
   # Agent will use weather tool and calculator
   ```

2. **Error Recovery**
   ```python
   # Agent handles missing data gracefully
   response = agent_executor.invoke({
       "input": "Weather in Unknown City?"
   })
   ```

## Resources

1. **Official Documentation**
   - **Agents Guide**: https://python.langchain.com/docs/how_to/#agents
   - **Tools Guide**: https://python.langchain.com/docs/how_to/#tools
   - **OpenAI Functions**: https://python.langchain.com/docs/how_to/migrate_agent/

2. **Additional Resources**
   - **Custom Tools**: https://python.langchain.com/docs/how_to/custom_tools/
   - **AgentExecutor**: https://python.langchain.com/docs/how_to/migrate_agent/

## Real-World Applications

1. **Data Processing**
   - Calculations
   - Unit conversions
   - Data transformation

2. **Information Gathering**
   - Weather information
   - Data lookup
   - API interactions

3. **Task Automation**
   - Multi-step processes
   - Decision making
   - Data analysis

Remember: 
- Use explicit type hints
- Include agent_scratchpad
- Enable verbose mode for debugging
- Handle errors gracefully
- Test with various scenarios
- Document tool behavior clearly
# Tool Calling in LangChain

## Core Concepts

Tool calling in LangChain is a mechanism that allows models to interact with tools based on context and need. Key principles include:

1. Dynamic Tool Usage
   - Models decide when to use tools based on input relevance
   - Not every interaction requires tool use
   - Context-aware tool selection

   ```python
   from langchain.agents import Tool, AgentExecutor, initialize_agent
   from langchain.chat_models import ChatOpenAI
   
   # Define tools with clear purposes
   tools = [
       Tool(
           name="Search",
           func=search.run,
           description="Useful for searching the internet"
       ),
       Tool(
           name="Calculator",
           func=calculator.run,
           description="Useful for performing calculations"
       )
   ]
   
   # Initialize agent with dynamic tool selection
   agent = initialize_agent(
       tools,
       ChatOpenAI(temperature=0),
       agent="zero-shot-react-description",
       verbose=True
   )
   ```

2. Structured Output
   - Generates structured output matching defined schemas
   - Can be used even without actual tool invocation
   - Standardized format for tool interactions

   ```python
   from langchain.tools import StructuredTool
   from pydantic import BaseModel
   
   class SearchInput(BaseModel):
       query: str
       max_results: int = 5
   
   search_tool = StructuredTool.from_function(
       func=search_function,
       name="search",
       description="Search for information",
       args_schema=SearchInput
   )
   ```

## Implementation Approaches

1. Direct Tool Calling
   - Models generate tool-compatible outputs
   - Schema-based response formatting
   - Input validation and processing

   ```python
   from langchain.tools import format_tool_to_openai_function
   
   # Format tool for direct calling
   functions = [format_tool_to_openai_function(tool) for tool in tools]
   
   response = chat.predict_messages(
       messages,
       functions=functions,
       function_call={"name": "Search"}
   )
   ```

2. Agent-Based Tool Calling
   - Agents detect when tools should be called
   - Determine appropriate tool inputs
   - Handle tool response processing

   ```python
   from langchain.agents import AgentType, create_react_agent
   from langchain.prompts import PromptTemplate
   
   # Create agent with custom prompt
   prompt = PromptTemplate.from_template("""Answer the following questions as best you can:
   {input}
   Use these tools if needed: {tools}""")
   
   agent = create_react_agent(llm, tools, prompt)
   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   ```

## Core Features

1. Schema Definition
   - User-defined output schemas
   - Input parameter specifications
   - Response format definitions

   ```python
   from pydantic import BaseModel, Field
   
   class ToolOutput(BaseModel):
       result: str = Field(..., description="The result of the tool execution")
       confidence: float = Field(..., description="Confidence score of the result")
   
   class CustomTool(BaseTool):
       name = "custom_tool"
       description = "Custom tool with structured output"
       args_schema = SearchInput
       
       def _run(self, query: str, max_results: int) -> ToolOutput:
           # Tool implementation
           return ToolOutput(result="Found data", confidence=0.95)
   ```

2. Tool Detection
   - Automatic tool need identification
   - Context-based tool selection
   - Multiple tool coordination

   ```python
   from langchain.agents import load_tools, AgentType
   
   # Load and coordinate multiple tools
   tool_names = ["serpapi", "llm-math", "wikipedia"]
   tools = load_tools(tool_names, llm=llm)
   
   agent = initialize_agent(
       tools,
       llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       handle_parsing_errors=True
   )
   ```

## Best Practices

1. Tool Definition:
   - Clear schema specifications
   - Well-defined input parameters
   - Proper output handling

   ```python
   from typing import Optional, Type
   
   class BetterTool(BaseTool):
       name: str = "better_tool"
       description: str = "A well-defined tool with clear specifications"
       args_schema: Optional[Type[BaseModel]] = SearchInput
       
       def _run(self, query: str, max_results: int = 5) -> str:
           """Run the tool with proper error handling and validation."""
           try:
               # Tool logic
               return result
           except Exception as e:
               raise ToolException(f"Error running tool: {str(e)}")
   ```

2. Implementation Strategy:
   - Choose appropriate calling method
   - Handle errors gracefully
   - Validate inputs and outputs

## Resources

Documentation Links:
- [Tool Calling Concepts](https://python.langchain.com/docs/concepts/tool_calling/)
- [Chat Models with Tools](https://python.langchain.com/docs/how_to/tool_calling/)
- [Function Calling Guide](https://python.langchain.com/docs/how_to/function_calling/)
- [Tool Calling Agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)

## Implementation Considerations

1. Model Selection:
   - Choose models that support tool calling
   - Consider model capabilities
   - Evaluate performance requirements

   ```python
   from langchain.chat_models import ChatOpenAI
   
   # Model with tool calling capabilities
   chat = ChatOpenAI(
       model_name="gpt-3.5-turbo-0613",
       temperature=0,
       # Enable function calling
       model_kwargs={"functions": functions}
   )
   ```

2. Error Handling:
   - Handle invalid tool calls
   - Manage timeout scenarios
   - Process unexpected responses

   ```python
   from langchain.callbacks import TimeoutCallback
   
   # Set up timeout handling
   timeout_callback = TimeoutCallback(timeout=10.0)
   
   try:
       result = agent.run(
           input="Query requiring tools",
           callbacks=[timeout_callback]
       )
   except Exception as e:
       print(f"Tool calling error: {str(e)}")
   ```

3. Tool Integration:
   - Proper schema definition
   - Input validation
   - Response processing

## Use Cases

1. Function Execution:
   - API interactions
   - Database operations
   - System commands

   ```python
   # API interaction tool
   api_tool = Tool(
       name="api_call",
       func=lambda x: requests.get(x).json(),
       description="Make API calls to external services"
   )
   ```

2. Structured Responses:
   - Data extraction
   - Format conversion
   - Information processing

   ```python
   from langchain.output_parsers import PydanticOutputParser
   
   class ParsedResponse(BaseModel):
       data: str
       format: str
   
   parser = PydanticOutputParser(pydantic_object=ParsedResponse)
   ```

3. Agent Operations:
   - Multi-step tasks
   - Decision-based tool selection
   - Complex workflow management

   ```python
   from langchain.agents import create_json_agent
   from langchain.tools.json.tool import JsonSpec
   
   # Create specialized agent for JSON operations
   json_spec = JsonSpec(dict_={"data": {"key": "value"}})
   json_agent = create_json_agent(llm, json_spec)
# Understanding Tool Calling in LangChain

This document explains how to implement tool calling in LangChain, demonstrating how language models can programmatically interact with tools through structured function calling.

Note: For setup instructions and package requirements, please refer to `USAGE_GUIDE.md` in the root directory.

## Core Concepts

1. **Tool Calling Architecture**
   LangChain's tool calling system enables models to interact with external tools:
   
   - **Function-Based Tools**: Tools are defined as Python functions with clear input/output specifications, making them easy to create and maintain.
   
   - **Input Schemas**: Pydantic models define the structure and validation rules for tool inputs, ensuring type safety and proper documentation.
   
   - **Tool Decoration**: The `@tool` decorator simplifies tool creation by automatically handling function conversion and integration.

2. **Input Schema Design**
   Pydantic models provide structured input validation:
   
   - **Type Annotations**: Clear definition of expected input types and formats.
   
   - **Field Descriptions**: Documentation that helps models understand parameter purposes.
   
   - **Validation Rules**: Built-in checks for input correctness.

3. **Tool Integration**
   Tools connect with models through:
   
   - **Automatic Registration**: Tools are automatically formatted for model use.
   
   - **Context Awareness**: Models understand tool capabilities through descriptions.
   
   - **Result Processing**: Standardized handling of tool outputs.

## Implementation Breakdown

1. **Tool Definition**
   ```python
   @tool(args_schema=WeatherInput)
   def get_weather(city: str) -> Dict[str, Union[float, str]]:
       """Get current weather information for a specific city."""
       # Tool implementation
   ```
   This shows:
   - Clear tool identification
   - Parameter specification
   - Return type definition

2. **Tool Integration**
   ```python
   chat_model = AzureChatOpenAI(
       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
       model="gpt-4"  # Model must support function calling
   )
   
   # Prepare tools for use
   tools = [get_weather, get_time]
   ```
   This demonstrates:
   - Model configuration
   - Tool preparation
   - Function calling support

3. **Tool Usage**
   ```python
   response = chat_model.invoke(
       messages,
       tools=tools,
       tool_choice="auto"
   )
   ```
   This shows:
   - Message preparation
   - Tool availability
   - Automatic tool selection

## Best Practices

1. **Tool Design**
   
   - **Clear Documentation**:
     ```python
     @tool
     def get_data(query: str) -> Dict[str, Any]:
         """
         Retrieve specific data based on query.
         
         Args:
             query: Search string for data retrieval
             
         Returns:
             Dictionary containing retrieved data or error message
         """
     ```
   
   - **Type Safety**:
     ```python
     from typing import Optional
     
     @tool
     def process_item(
         item_id: int,
         options: Optional[Dict[str, str]] = None
     ) -> Dict[str, Any]:
         """Process item with optional parameters."""
     ```

2. **Error Handling**
   ```python
   @tool
   def safe_operation(input_data: str) -> Dict[str, Any]:
       try:
           result = process_data(input_data)
           return {"status": "success", "data": result}
       except Exception as e:
           return {"status": "error", "message": str(e)}
   ```

## Common Patterns

1. **Chained Tools**
   ```python
   @tool
   def combined_info(location: str) -> Dict[str, Any]:
       """Get both weather and time for a location."""
       weather = get_weather(location)
       time = get_time("24h")
       return {
           "weather": weather,
           "time": time
       }
   ```

2. **Input Validation**
   ```python
   class ValidatedInput(BaseModel):
       value: int = Field(..., ge=0, le=100)
       mode: str = Field(..., pattern="^(fast|slow)$")
   ```

## Resources

1. **Official Documentation**
   - Tool Decoration Guide
   - Input Schema Reference
   - Function Calling Documentation

2. **Additional Learning**
   - Pydantic Integration
   - Type Hint Usage
   - Error Handling Patterns

## Key Takeaways

1. **Modern Tool Design**
   - Use function decorators
   - Define clear schemas
   - Implement type safety

2. **Integration Practices**
   - Proper tool registration
   - Automatic tool choice
   - Result handling

3. **Advanced Features**
   - Input validation
   - Error management
   - Tool composition
# Understanding Basic Tools in LangChain

This document provides a comprehensive guide to creating and using custom tools in LangChain, focusing on proper implementation patterns that are compatible with LangChain v0.3 and Pydantic v2. We'll explore how tools enable language models to perform specific actions and computations in a structured, type-safe manner.

## Core Concepts

1. **Tool Architecture in LangChain**
   LangChain's tool system provides a structured way for language models to interact with external functions and services:
   
   - **Tool Definition and Structure**: Tools in LangChain are implemented as classes that inherit from BaseTool, providing a consistent interface for defining capabilities that language models can use. This structured approach ensures type safety, input validation, and clear documentation of tool functionality.
   
   - **Input Validation and Schema**: Tools use Pydantic v2 models to define and validate their inputs, ensuring that data passed to tools meets specified requirements. This validation layer prevents runtime errors and provides clear feedback when invalid inputs are provided.
   
   - **Execution Flow**: Tools implement a standardized execution pattern through the `_run` method, which processes validated inputs and returns structured outputs. This consistency makes tools predictable and reliable in their operation.

2. **Type Safety and Validation**
   Modern LangChain tools emphasize type safety and proper validation:
   
   - **Field Annotations**: Every tool attribute, including name and description, must be properly annotated with types using Pydantic's Field class. This ensures that tools are well-documented and their properties are validated at runtime.
   
   - **Input Schemas**: Tools define their input requirements using Pydantic models, which specify expected data types, validation rules, and example values. This makes tools self-documenting and helps prevent invalid inputs from reaching the tool's logic.
   
   - **Return Type Specification**: Tools clearly define their output types, making it easy to understand what kind of results to expect and how to handle them in your application.

3. **Error Handling and Safety**
   Robust error handling is crucial for reliable tool operation:
   
   - **Input Validation**: Tools perform thorough validation of inputs before processing, ensuring that only valid data is processed and providing clear error messages when validation fails.
   
   - **Execution Safety**: Tools implement safeguards against potentially harmful operations, such as restricting what mathematical expressions can be evaluated or limiting access to system resources.
   
   - **Error Communication**: When errors occur, tools provide detailed error messages that help developers understand what went wrong and how to fix it.

## Implementation Breakdown

1. **Calculator Tool Implementation**
   ```python
   class CalculatorInput(BaseModel):
       """Input schema for calculator tool."""
       expression: str = Field(
           description="The mathematical expression to evaluate",
           examples=["2 + 2", "10 * 5", "(25 - 5) / 4"]
       )

   class Calculator(BaseTool):
       """Tool that performs basic arithmetic calculations."""
       
       name: str = Field(default="calculator")
       description: str = Field(default="Performs basic arithmetic calculations")
       args_schema: type[BaseModel] = Field(default=CalculatorInput)

       def _run(self, expression: str) -> Dict[str, Any]:
           """Execute the calculator tool."""
           # Implementation details...
   ```

   This implementation showcases several important aspects:
   
   - **Schema Definition**: The CalculatorInput model clearly defines what inputs the tool accepts and provides examples for better understanding.
   
   - **Tool Structure**: The Calculator class properly inherits from BaseTool and defines all required attributes with appropriate type annotations.
   
   - **Method Implementation**: The _run method includes proper type hints and returns structured data that's easy to process.

2. **Safety Implementation**
   ```python
   def _run(self, expression: str) -> Dict[str, Any]:
       """
       Execute the calculator tool with safety checks.
       
       Args:
           expression: A mathematical expression to evaluate
           
       Returns:
           Dictionary containing the expression, result, and status
           
       Raises:
           ValueError: If the expression contains invalid characters or is unsafe
       """
       # Clean and validate the input
       expression = expression.strip()
       allowed_chars = set("0123456789+-*/(). ")
       
       # Check for invalid characters
       if not all(c in allowed_chars for c in expression):
           raise ValueError(
               "Invalid characters in expression. Only numbers and basic operators allowed."
           )
       
       try:
           # Create a restricted evaluation environment
           result = eval(expression, {"__builtins__": {}})
           return {
               "expression": expression,
               "result": result,
               "status": "success"
           }
       except Exception as e:
           raise ValueError(f"Invalid expression: {str(e)}")
   ```

   This security-focused implementation shows:
   
   - **Input Sanitization**: Carefully cleans and validates input before processing
   - **Character Restriction**: Only allows safe mathematical characters
   - **Restricted Evaluation**: Uses a limited execution environment
   - **Structured Output**: Returns results in a well-defined format

## Best Practices

1. **Tool Design Patterns**
   The following patterns ensure reliable and maintainable tools:
   
   - **Clear Naming**: Tools should have descriptive names that clearly indicate their purpose and functionality. This helps users understand what the tool does without needing to read detailed documentation.
   
   - **Comprehensive Documentation**: Tools should include detailed docstrings that explain their purpose, inputs, outputs, and any limitations or special considerations. This documentation helps users integrate tools effectively.
   
   - **Input Validation**: Implement thorough input validation using Pydantic models to catch invalid inputs early and provide helpful error messages that guide users to correct usage.
   
   - **Structured Output**: Return results in a consistent, well-documented format that makes it easy for users to process and handle tool outputs in their applications.

2. **Security Considerations**
   Implementing secure tools requires attention to several aspects:
   
   - **Input Sanitization**: Always clean and validate inputs to prevent injection attacks or other security vulnerabilities. This includes checking for invalid characters and ensuring inputs meet expected formats.
   
   - **Resource Limitations**: Implement appropriate limits on resource usage, such as computation time or memory consumption, to prevent denial-of-service attacks or resource exhaustion.
   
   - **Error Handling**: Provide detailed error messages for valid errors while avoiding exposure of sensitive information in error responses.

## Common Patterns

1. **Tool Registration and Usage**
   ```python
   # Initialize the tool
   calculator = Calculator()
   
   # Use the tool with validation
   try:
       result = calculator._run("25 * 4")
       print(f"Result: {result['result']}")
   except ValueError as e:
       print(f"Error: {str(e)}")
   ```

2. **Model Integration**
   ```python
   # Initialize chat model with tools
   chat_model = AzureChatOpenAI()
   
   # Create messages with tool context
   messages = [
       SystemMessage(content="You have access to a calculator tool."),
       HumanMessage(content="What is 15 * 7?")
   ]
   
   # Get model response with tool access
   response = chat_model.invoke(messages, tools=[calculator])
   ```

## Resources

1. **Official Documentation**
   -t**Tools Overview**: https://python.langchain.com/docs/concepts/tools/
    **Key Concepts**: https://python.langchain.com/docs/concepts/tools/#key-concepts
   - **Tool Interface**: https://python.langchain.com/docs/concepts/tools/#tool-interface
   - **Best pPactices**: https://python.langchain.com/docs/concepts/tools/#best-practices
 
2. **Tool Creation and Usage**
   - **Create Tools Using @tool**: https://python.langchain.com/docs/concepts/tools/#create-tools-using-the-tool-decorator
   - **Use the Tool Directly**: https://python.langchain.com/docs/concepts/tools/#use-the-tool-directly
   - **Configuring the Schema**: https://python.langchain.com/docs/concepts/tools/#configuring-the-schema
   - **Tool Artifacts**: https://python.langchain.com/docs/concepts/tools/#tool-artifacts

3. **Advanced Topics**
   - **Special Type Annotations**: https://python.langchain.com/docs/concepts/tools/#special-type-annotations
   - **Tool Binding**: https://python.langchain.com/docs/concepts/tools/#tool-binding
   - **Toolkits**: https://python.langchain.com/docs/concepts/tools/#toolkits
   - **InjectedTools**: https://python.langchain.com/docs/concepts/tools/#injectedtoolarg

## Key Takeaways

1. **Tool Implementation**
   - Always inherit from BaseTool
   - Use proper type annotations
   - Implement thorough input validation
   - Return structured, documented outputs

2. **Security and Reliability**
   - Validate all inputs
   - Implement appropriate safeguards
   - Handle errors gracefully
   - Document limitations and requirements

3. **Best Practices**
   - Follow consistent naming conventions
   - Provide comprehensive documentation
   - Implement proper error handling
   - Return well-structured results
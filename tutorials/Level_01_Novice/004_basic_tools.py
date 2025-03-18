"""
LangChain Basic Tools Example

This example demonstrates how to create and use custom tools in LangChain.
Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

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
        """
        Execute the calculator tool.
        
        Args:
            expression: The arithmetic expression to evaluate
            
        Returns:
            Dict containing the result and expression
            
        Raises:
            ValueError: If the expression is invalid or unsafe
        """
        # Clean and validate the input
        expression = expression.strip()
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        try:
            # Safely evaluate the expression
            result = eval(expression, {"__builtins__": {}})
            return {
                "expression": expression,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

def demonstrate_tool_usage() -> None:
    """Demonstrate how to use custom tools with LangChain."""
    try:
        # Initialize chat model and tool
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            model_kwargs={"model": "gpt-4"},
            temperature=0
        )
        
        calculator = Calculator()
        
        # System message that knows about the calculator
        system_msg = SystemMessage(content="""
            You are a helpful assistant with access to a calculator tool. 
            When asked about calculations, use the calculator tool to ensure accuracy.
        """)
        
        # Example 1: Simple calculation
        print("\nExample 1: Simple Calculation")
        human_msg1 = HumanMessage(content="What is 15 * 7?")
        messages = [system_msg, human_msg1]
        
        # The model can use the calculator tool
        result = calculator._run("15 * 7")
        print(f"Tool result: {result}")
        
        response = chat_model.invoke(messages)
        print(f"Model response: {response.content}")
        
        # Example 2: Complex calculation
        print("\nExample 2: Complex Calculation")
        human_msg2 = HumanMessage(content="What is (25 - 5) / 4?")
        messages = [system_msg, human_msg1, response, human_msg2]
        
        result = calculator._run("(25 - 5) / 4")
        print(f"Tool result: {result}")
        
        response = chat_model.invoke(messages)
        print(f"Model response: {response.content}")
        
    except ValueError as ve:
        print(f"\nValidation error: {str(ve)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Basic Tools...")
    demonstrate_tool_usage()

if __name__ == "__main__":
    main()
"""
LangChain Basic Tools Example

This example demonstrates how to create and use custom tools in LangChain.
Compatible with LangChain v0.3 and Pydantic v2.
"""

prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format exactly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Remember, once you have the result from a tool, you should move toward providing a Final Answer rather than using the tool again with the same input.

Begin!

{agent_scratchpad}

Question: {input}
Thought:"""


import os
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain.prompts import ChatPromptTemplate

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
            model="gpt-4o",
            temperature=0
        )
        
        calculator = Calculator()
        
        print("\nPart 1: Direct Calculator Tool Usage")
        print("-" * 40)
        
        # Example 1: Simple calculation with direct tool usage
        print("\nExample 1: Simple Calculation")
        expression1 = "12 * 1423"
        result = calculator._run(expression1)
        print(f"Tool result: {result}")
        
        # Example 2: Complex calculation with direct tool usage
        print("\nExample 2: Complex Calculation")
        expression2 = "(30 - 5) / 4"
        result = calculator._run(expression2)
        print(f"Tool result: {result}")
        
        print("\nPart 2: Agent-based Calculator Usage")
        print("-" * 40)
        
        # Create an agent with the calculator tool
        tools = [
            Tool(
                name="Calculator",
                func=calculator._run,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression."
            )
        ]

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the agent with improved settings
        agent = create_react_agent(
            llm=chat_model,
            tools=tools,
            prompt=prompt
        )
        
        # Create the agent executor with explicit safeguards
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # Prevent infinite loops
            early_stopping_method="generate"  # Helps handle edge cases
        )
        
        # Example 1: Simple calculation with agent
        print("\nExample 1: Simple Calculation (via Agent)")
        result = agent_executor.invoke({"input": f"What is {expression1}?"})
        print(f"Agent response: {result['output']}")
        
        # Example 2: Complex calculation with agent
        print("\nExample 2: Complex Calculation (via Agent)")
        result = agent_executor.invoke({"input": f"What is {expression2}?"})
        print(f"Agent response: {result['output']}")
        
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
"""
LangChain Portfolio Management Assistant Example

This example demonstrates how to combine tool calling and memory management to create
a portfolio management assistant that can analyze investments and maintain context
of investment discussions.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.utils.function_calling import convert_to_openai_function

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

class PortfolioAnalyzer(BaseTool):
    """Tool for analyzing investment portfolios."""
    
    name: str = "portfolio_analyzer"
    description: str = "Analyze portfolio composition and provide insights"
    
    def _run(self, portfolio: dict) -> str:
        """
        Analyze a portfolio's composition and return insights.
        
        Args:
            portfolio: A dictionary of asset allocation percentages.
        
        Returns:
            str: Analysis and recommendations.
        """
        total = sum(portfolio.values())
        if not (99.5 <= total <= 100.5):  # Allow small rounding differences
            return f"Error: Portfolio allocations must sum to 100% (current: {total}%)"
        
        # Basic portfolio analysis (simplified for example)
        risk_score = 0
        risk_factors = {
            "stocks": 3,     # Higher risk
            "bonds": 2,      # Medium risk
            "cash": 1,       # Low risk
            "reits": 3,      # Higher risk
            "commodities": 4 # Highest risk
        }
        
        for asset, percentage in portfolio.items():
            risk_score += (percentage / 100) * risk_factors.get(asset.lower(), 2)
        
        risk_level = (
            "Low" if risk_score < 1.5 else
            "Medium" if risk_score < 2.5 else
            "High"
        )
        
        # Generate analysis
        analysis = [
            f"Portfolio Risk Level: {risk_level} (Score: {risk_score:.2f})",
            "Allocation Analysis:"
        ]
        
        for asset, percentage in portfolio.items():
            analysis.append(f"- {asset.title()}: {percentage:.1f}%")
        
        # Add recommendations
        analysis.append("Recommendations:")
        if risk_score > 2.5 and portfolio.get("bonds", 0) < 20:
            analysis.append("- Consider increasing bond allocation for better stability")
        if portfolio.get("cash", 0) < 5:
            analysis.append("- Maintain at least 5% cash for emergencies")
        if portfolio.get("stocks", 0) > 70:
            analysis.append("- High stock exposure: Consider diversification")
        
        return "\n".join(analysis)
    
    async def _arun(self, portfolio: dict) -> str:
        return self._run(portfolio)

class PortfolioRebalancer(BaseTool):
    """Tool for suggesting portfolio rebalancing actions."""
    
    name: str = "portfolio_rebalancer"
    description: str = "Suggest actions to rebalance a portfolio to target allocations"
    
    def _run(self, current: dict, target: dict) -> str:
        """
        Suggest rebalancing actions.
        
        Args:
            current: Current portfolio allocations.
            target: Target portfolio allocations.
        
        Returns:
            str: Rebalancing suggestions.
        """
        actions = ["Rebalancing Actions Required:"]
        
        for asset in set(current.keys()) | set(target.keys()):
            current_pct = current.get(asset, 0)
            target_pct = target.get(asset, 0)
            diff = target_pct - current_pct
            
            if abs(diff) >= 1:  # Only suggest changes for >1% difference
                action = "increase" if diff > 0 else "decrease"
                actions.append(
                    f"- {asset.title()}: {action} by {abs(diff):.1f}% (Current: {current_pct:.1f}% → Target: {target_pct:.1f}%)"
                )
        
        if len(actions) == 1:
            return "No significant rebalancing required"
            
        return "\n".join(actions)
    
    async def _arun(self, current: dict, target: dict) -> str:
        return self._run(current, target)

def create_chat_model():
    """Initialize the Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def process_response(messages, functions, tools, chat_model, debug: bool = True):
    """
    Process response and handle any function calls.
    
    Iteratively check if the response contains a function call.
    If so, execute the corresponding tool, append its output as a message,
    and re-invoke the model until no function call remains.
    
    The debug flag enables printing of intermediate information.
    """
    iteration = 0
    max_iterations = 10  # Prevent infinite loops
    
    response = chat_model.invoke(messages, functions=functions)
    if debug:
        print(f"[DEBUG] Initial response additional_kwargs: {response.additional_kwargs}")
    
    while response.additional_kwargs.get("function_call") and iteration < max_iterations:
        iteration += 1
        fc = response.additional_kwargs["function_call"]
        if debug:
            print(f"[DEBUG] Iteration {iteration}: function_call = {fc}")
            
        fname = fc.get("name")
        fargs_str = fc.get("arguments", "{}")
        try:
            fargs = json.loads(fargs_str)
        except Exception as e:
            if debug:
                print(f"[DEBUG] Error parsing function arguments: {e}")
            fargs = {}
            
        tool_result = None
        for tool in tools:
            if tool.name == fname:
                if debug:
                    print(f"[DEBUG] Executing tool: {fname} with arguments: {fargs}")
                tool_result = tool._run(**fargs)
                if debug:
                    print(f"[DEBUG] Tool result: {tool_result}")
                break
                
        if tool_result is not None:
            tool_msg = HumanMessage(content=tool_result)
            messages.append(tool_msg)
            response = chat_model.invoke(messages, functions=functions)
            if debug:
                print(f"[DEBUG] New response additional_kwargs: {response.additional_kwargs}")
        else:
            if debug:
                print(f"[DEBUG] No tool found for function name: {fname}")
            break
            
    if iteration >= max_iterations and debug:
        print("[DEBUG] Maximum iterations reached; exiting loop.")
        
    return response

def demonstrate_portfolio_management():
    """Demonstrate portfolio management capabilities."""
    try:
        # Initialize components
        chat_model = create_chat_model()
        
        # Create tools
        analyzer = PortfolioAnalyzer()
        rebalancer = PortfolioRebalancer()
        tools = [analyzer, rebalancer]
        
        # Convert tools to functions
        functions = [convert_to_openai_function(t) for t in tools]
        
        # Initialize memory using ConversationBufferMemory with new recommended usage.
        memory = ConversationBufferMemory(memory_key="history")
        
        # Example 1: Portfolio Analysis
        print("\nExample 1: Portfolio Analysis")
        print("-" * 50)
        
        # Save initial context
        memory.save_context(
            {"input": "I have a portfolio I'd like to analyze"},
            {"output": "I can help analyze your portfolio. Please share your current allocations."}
        )
        
        # Create system message with conversation history
        history = memory.load_memory_variables({}).get("history", "")
        system_msg = SystemMessage(content=f"""
You are a portfolio management assistant with access to portfolio analysis and rebalancing tools.
Use these tools to help clients understand and optimize their investments.
Previous conversation context:
{history}
""")
        
        # Define current portfolio.
        portfolio = {
            "stocks": 70,
            "bonds": 20,
            "cash": 5,
            "reits": 5
        }
        
        # IMPORTANT: Instruct the assistant to call the tool explicitly with parameters.
        human_msg = HumanMessage(content=f"Call the portfolio_analyzer function with parameters: {{\"portfolio\": {json.dumps(portfolio)}}}")
        
        messages = [system_msg, human_msg]
        response = process_response(messages, functions, tools, chat_model, debug=True)
        print("\nPortfolio Analysis:", response.content)
        
        # Example 2: Portfolio Rebalancing
        print("\nExample 2: Portfolio Rebalancing")
        print("-" * 50)
        
        # Save previous interaction to memory
        memory.save_context(
            {"input": human_msg.content},
            {"output": response.content}
        )
        
        # Define target allocation.
        target_portfolio = {
            "stocks": 60,
            "bonds": 30,
            "cash": 5,
            "reits": 5
        }
        
        human_msg = HumanMessage(content=f"Call the portfolio_rebalancer function with parameters: {{\"current\": {json.dumps(portfolio)}, \"target\": {json.dumps(target_portfolio)}}}")
        
        # Update system message with new conversation context.
        history = memory.load_memory_variables({}).get("history", "")
        system_msg = SystemMessage(content=f"""
You are a portfolio management assistant with access to portfolio analysis and rebalancing tools.
Use these tools to help clients understand and optimize their investments.
Previous conversation context:
{history}
""")
        
        messages = [system_msg, human_msg]
        response = process_response(messages, functions, tools, chat_model, debug=True)
        print("\nRebalancing Suggestions:", response.content)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Portfolio Management Assistant...")
    demonstrate_portfolio_management()

if __name__ == "__main__":
    main()

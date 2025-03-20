"""
LangChain Financial Advisor Bot Example

This example demonstrates how to combine structured output parsing and memory
management to create a financial advisor chatbot that provides formatted
recommendations while maintaining conversation context.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import HumanMessage, SystemMessage

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

class FinancialRecommendation(BaseModel):
    """Schema for structured financial recommendations."""
    investment_type: str = Field(description="Type of investment recommendation")
    risk_level: str = Field(description="Risk level (Low, Medium, High)")
    time_horizon: str = Field(description="Recommended investment time horizon")
    expected_return: float = Field(description="Expected annual return percentage")
    minimum_investment: float = Field(description="Minimum investment amount")
    key_benefits: List[str] = Field(description="Key benefits of this recommendation")
    considerations: List[str] = Field(description="Important considerations or risks")
    suggested_allocation: Optional[Dict[str, float]] = Field(
        description="Suggested portfolio allocation",
        default=None
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "investment_type": "Diversified ETF Portfolio",
                "risk_level": "Medium",
                "time_horizon": "5-7 years",
                "expected_return": 7.5,
                "minimum_investment": 5000.00,
                "key_benefits": [
                    "Broad market exposure",
                    "Low management fees",
                    "High liquidity"
                ],
                "considerations": [
                    "Market volatility risk",
                    "No guaranteed returns",
                    "Requires periodic rebalancing"
                ],
                "suggested_allocation": {
                    "stocks": 60.0,
                    "bonds": 30.0,
                    "cash": 10.0
                }
            }]
        }
    }

def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7
    )

def create_financial_advisor():
    """Create a financial advisor with memory and structured output."""
    # Initialize the model
    chat_model = create_chat_model()
    
    # Create memory for maintaining conversation context
    memory = ConversationSummaryMemory(
        llm=chat_model,
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create parser for structured recommendations
    parser = PydanticOutputParser(pydantic_object=FinancialRecommendation)
    
    # Create the system message template
    system_template = """You are an experienced financial advisor who provides 
structured investment recommendations. Consider the chat history and current query 
to provide personalized advice.

Chat History:
{chat_history}

When recommending investments, respond with a JSON object that follows this schema:
{format_instructions}

Ensure your recommendation considers the client's previous interactions and risk profile.
"""
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    
    return chat_model, memory, parser, prompt

def demonstrate_financial_advisor():
    """Demonstrate the financial advisor bot capabilities."""
    try:
        print("\nDemonstrating Financial Advisor Bot...\n")
        
        # Create advisor components
        chat_model, memory, parser, prompt = create_financial_advisor()
        
        # Example 1: Initial Risk Profile
        print("Example 1: Initial Risk Profile Discussion")
        print("-" * 50)
        
        # Save initial interaction to memory
        memory.save_context(
            {"input": "I'm interested in investing for retirement in about 10 years."},
            {"output": "I understand you're planning for retirement in 10 years. That's a good time horizon for a balanced investment approach."}
        )
        
        # Get recommendation
        query = "What investment strategy would you recommend for my retirement goal?"
        
        # Format the prompt using memory and parser instructions.
        formatted_text = prompt.format(
            chat_history=memory.load_memory_variables({})["chat_history"],
            format_instructions=parser.get_format_instructions(),
            input=query
        )
        # Since formatted_text is a string, we wrap it in a SystemMessage.
        messages = [SystemMessage(content=formatted_text)]
        
        # Get and parse response
        response = chat_model.invoke(messages)
        recommendation = parser.parse(response.content)
        
        print("\nStructured Recommendation:")
        print(f"Investment Type: {recommendation.investment_type}")
        print(f"Risk Level: {recommendation.risk_level}")
        print(f"Time Horizon: {recommendation.time_horizon}")
        print(f"Expected Return: {recommendation.expected_return}%")
        print(f"Minimum Investment: ${recommendation.minimum_investment:,.2f}")
        print("\nKey Benefits:")
        for benefit in recommendation.key_benefits:
            print(f"- {benefit}")
        print("\nConsiderations:")
        for consideration in recommendation.considerations:
            print(f"- {consideration}")
        if recommendation.suggested_allocation:
            print("\nSuggested Allocation:")
            for asset, percentage in recommendation.suggested_allocation.items():
                print(f"- {asset.title()}: {percentage}%")
        
        # Example 2: Follow-up Question
        print("\nExample 2: Follow-up Discussion")
        print("-" * 50)
        
        # Save the previous recommendation to memory
        memory.save_context(
            {"input": query},
            {"output": f"Recommended a {recommendation.investment_type} strategy with {recommendation.risk_level} risk level."}
        )
        
        # Process follow-up question
        follow_up = "What if I wanted a more conservative approach?"
        
        formatted_text = prompt.format(
            chat_history=memory.load_memory_variables({})["chat_history"],
            format_instructions=parser.get_format_instructions(),
            input=follow_up
        )
        messages = [SystemMessage(content=formatted_text)]
        response = chat_model.invoke(messages)
        conservative_rec = parser.parse(response.content)
        
        print("\nUpdated Conservative Recommendation:")
        print(f"Investment Type: {conservative_rec.investment_type}")
        print(f"Risk Level: {conservative_rec.risk_level}")
        print(f"Expected Return: {conservative_rec.expected_return}%")
        if conservative_rec.suggested_allocation:
            print("\nConservative Allocation:")
            for asset, percentage in conservative_rec.suggested_allocation.items():
                print(f"- {asset.title()}: {percentage}%")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Financial Advisor Bot...")
    demonstrate_financial_advisor()

if __name__ == "__main__":
    main()

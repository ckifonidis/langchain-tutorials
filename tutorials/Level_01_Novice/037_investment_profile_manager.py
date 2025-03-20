"""
LangChain Investment Profile Manager Example

This example demonstrates how to combine memory management and output parsing to create
a system that tracks client investment preferences and generates personalized,
structured investment recommendations.

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory

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

class RiskProfile(BaseModel):
    """Schema for client risk profile."""
    risk_tolerance: str = Field(description="Risk tolerance level (Conservative, Moderate, Aggressive)")
    investment_horizon: str = Field(description="Investment time horizon")
    income_category: str = Field(description="Income level category")
    investment_experience: str = Field(description="Level of investment experience")
    financial_goals: List[str] = Field(description="List of financial goals")
    investment_restrictions: List[str] = Field(description="Investment restrictions or preferences")

class PortfolioRecommendation(BaseModel):
    """Schema for portfolio recommendations."""
    asset_allocation: Dict[str, float] = Field(description="Recommended asset allocation percentages")
    suggested_products: List[Dict[str, str]] = Field(description="Suggested investment products")
    rebalancing_frequency: str = Field(description="Recommended rebalancing frequency")
    min_investment: float = Field(description="Minimum investment amount")
    risk_level: str = Field(description="Portfolio risk level")
    expected_return: float = Field(description="Expected annual return percentage")

class InvestmentProfile(BaseModel):
    """Schema for comprehensive investment profile."""
    client_id: str = Field(description="Unique client identifier")
    profile: RiskProfile = Field(description="Client's risk profile")
    recommendation: PortfolioRecommendation = Field(description="Portfolio recommendations")
    next_review_date: datetime = Field(description="Next profile review date")
    notes: List[str] = Field(description="Important notes about client preferences")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "client_id": "INV001",
                "profile": {
                    "risk_tolerance": "Moderate",
                    "investment_horizon": "5-10 years",
                    "income_category": "High",
                    "investment_experience": "Intermediate",
                    "financial_goals": [
                        "Retirement planning",
                        "College fund",
                        "Wealth preservation"
                    ],
                    "investment_restrictions": [
                        "No tobacco companies",
                        "Prefer ESG investments"
                    ]
                },
                "recommendation": {
                    "asset_allocation": {
                        "stocks": 60.0,
                        "bonds": 30.0,
                        "cash": 5.0,
                        "alternatives": 5.0
                    },
                    "suggested_products": [
                        {
                            "type": "ETF",
                            "category": "US Large Cap",
                            "suggestion": "Total Market Index Fund"
                        },
                        {
                            "type": "ETF",
                            "category": "Corporate Bonds",
                            "suggestion": "Investment Grade Bond Fund"
                        }
                    ],
                    "rebalancing_frequency": "Quarterly",
                    "min_investment": 25000.0,
                    "risk_level": "Moderate",
                    "expected_return": 7.5
                },
                "next_review_date": "2024-09-19T00:00:00",
                "notes": [
                    "Prefers sustainable investments",
                    "Interested in tax-efficient strategies",
                    "Planning to increase contributions annually"
                ]
            }]
        }
    }

def create_chat_model():
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def update_investment_profile(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    memory: ConversationSummaryMemory,
    client_data: Dict
) -> InvestmentProfile:
    """
    Generate or update an investment profile based on client data.
    
    Args:
        chat_model: The chat model to use
        parser: The output parser for structured profiles
        memory: Conversation memory for context
        client_data: Current client information
        
    Returns:
        InvestmentProfile: Updated investment profile and recommendations
    """
    # Get format instructions from parser
    format_instructions = parser.get_format_instructions()
    
    # Load conversation history
    history = memory.load_memory_variables({}).get("history", "")
    
    # Build system message
    system_text = (
        "You are a professional investment advisor. Review the client information "
        "and conversation history to create or update their investment profile "
        "and recommendations.\n\n"
        f"Previous conversation context:\n{history}\n\n"
        "Respond with a JSON object that exactly follows this schema (no additional text):"
        f"\n\n{format_instructions}\n"
    )
    
    # Create messages
    system_msg = SystemMessage(content=system_text)
    human_msg = HumanMessage(
        content=f"Analyze this client's information and provide recommendations: {json.dumps(client_data)}"
    )
    
    # Get profile
    response = chat_model.invoke([system_msg, human_msg])
    profile = parser.parse(response.content)
    
    # Update memory with current interaction
    memory.save_context(
        {"input": f"Client data update for {client_data.get('client_id', 'Unknown')}"},
        {"output": "Generated investment profile and recommendations"}
    )
    
    return profile

def demonstrate_profile_management():
    """Demonstrate investment profile management capabilities."""
    try:
        print("\nDemonstrating Investment Profile Management...\n")
        
        # Initialize components
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=InvestmentProfile)
        memory = ConversationSummaryMemory(llm=chat_model, memory_key="history")
        
        # Example 1: New Client Profile
        print("Example 1: Creating New Investment Profile")
        print("-" * 50)
        
        new_client_data = {
            "client_id": "INV001",
            "age": 35,
            "employment": "Technology Sector",
            "annual_income": 150000,
            "current_savings": 100000,
            "risk_preferences": "Moderate",
            "investment_goals": [
                "Retirement planning",
                "College fund for future children"
            ],
            "investment_restrictions": [
                "No tobacco companies",
                "Prefer sustainable investments"
            ]
        }
        
        profile1 = update_investment_profile(chat_model, parser, memory, new_client_data)
        
        print("\nInvestment Profile:")
        print(f"Client ID: {profile1.client_id}")
        print("\nRisk Profile:")
        print(f"Risk Tolerance: {profile1.profile.risk_tolerance}")
        print(f"Investment Horizon: {profile1.profile.investment_horizon}")
        print(f"Investment Experience: {profile1.profile.investment_experience}")
        print("\nFinancial Goals:")
        for goal in profile1.profile.financial_goals:
            print(f"- {goal}")
        
        print("\nRecommended Portfolio:")
        print("\nAsset Allocation:")
        for asset, allocation in profile1.recommendation.asset_allocation.items():
            print(f"- {asset.title()}: {allocation:.1f}%")
        
        print("\nSuggested Products:")
        for product in profile1.recommendation.suggested_products:
            print(f"- {product['category']}: {product['suggestion']}")
        
        print(f"\nExpected Return: {profile1.recommendation.expected_return:.1f}%")
        print(f"Risk Level: {profile1.recommendation.risk_level}")
        
        # Example 2: Profile Update
        print("\nExample 2: Updating Existing Profile")
        print("-" * 50)
        
        update_data = {
            "client_id": "INV001",
            "profile_update": {
                "risk_tolerance": "Moderate-Aggressive",
                "financial_goals": [
                    "Retirement planning",
                    "College fund",
                    "Home purchase in 5 years"
                ],
                "investment_amount_change": 50000,  # Additional investment
                "new_preferences": [
                    "Interest in international markets",
                    "Consider real estate investment"
                ]
            }
        }
        
        profile2 = update_investment_profile(chat_model, parser, memory, update_data)
        
        print("\nUpdated Investment Profile:")
        print(f"Client ID: {profile2.client_id}")
        print("\nUpdated Risk Profile:")
        print(f"Risk Tolerance: {profile2.profile.risk_tolerance}")
        print(f"Investment Horizon: {profile2.profile.investment_horizon}")
        
        print("\nUpdated Financial Goals:")
        for goal in profile2.profile.financial_goals:
            print(f"- {goal}")
        
        print("\nNew Asset Allocation:")
        for asset, allocation in profile2.recommendation.asset_allocation.items():
            print(f"- {asset.title()}: {allocation:.1f}%")
        
        print("\nUpdated Recommendations:")
        for product in profile2.recommendation.suggested_products:
            print(f"- {product['category']}: {product['suggestion']}")
        
        print("\nNotes:")
        for note in profile2.notes:
            print(f"- {note}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Investment Profile Manager...")
    demonstrate_profile_management()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Campaign Personalizer (122) (LangChain v3)

This example demonstrates marketing campaign personalization using:
1. Customer Analysis: Segment understanding
2. Content Generation: Personalized messaging
3. Channel Selection: Optimal delivery

It helps marketing teams create targeted financial product campaigns.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProductType(str, Enum):
    """Financial product types."""
    SAVINGS = "savings"
    CHECKING = "checking"
    INVESTMENT = "investment"
    CREDIT_CARD = "credit_card"
    PERSONAL_LOAN = "personal_loan"
    MORTGAGE = "mortgage"

class CustomerSegment(str, Enum):
    """Customer segments."""
    YOUNG_PROFESSIONAL = "young_professional"
    FAMILY = "family"
    SENIOR = "senior"
    STUDENT = "student"
    BUSINESS = "business"
    HIGH_NET_WORTH = "high_net_worth"

class Channel(str, Enum):
    """Marketing channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEB = "web"
    SOCIAL = "social"

class MarketingPreference(BaseModel):
    """Customer marketing preferences."""
    preferred_channels: List[Channel] = Field(description="Preferred channels")
    contact_frequency: str = Field(description="Contact frequency")
    interests: List[str] = Field(description="Product interests")
    opted_out: List[Channel] = Field(description="Opted-out channels")

class CustomerProfile(BaseModel):
    """Customer marketing profile."""
    customer_id: str = Field(description="Customer ID")
    segment: CustomerSegment = Field(description="Customer segment")
    products: List[ProductType] = Field(description="Current products")
    preferences: MarketingPreference = Field(description="Marketing preferences")
    metadata: Dict = Field(default_factory=dict)

class Campaign(BaseModel):
    """Marketing campaign details."""
    campaign_id: str = Field(description="Campaign ID")
    product: ProductType = Field(description="Target product")
    segments: List[CustomerSegment] = Field(description="Target segments")
    channels: List[Channel] = Field(description="Available channels")
    content: str = Field(description="Campaign content")

class CampaignPersonalizer:
    """Marketing campaign personalization system."""

    def __init__(self):
        """Initialize personalizer."""
        logger.info("Starting campaign personalizer...")
        
        # Setup chat model
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        logger.info("Chat model ready")
        
        # Create personalization chain
        personalizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a marketing campaign personalizer for a bank.
Given customer profiles and campaign details, create personalized recommendations.

Your analysis should include:

1. Customer Understanding
- Segment analysis
- Current products
- Preferences and interests
- Contact history

2. Campaign Fit
- Product relevance
- Segment alignment
- Channel suitability
- Timing considerations

3. Content Personalization
- Key messages
- Value propositions
- Unique benefits
- Call to action

4. Channel Strategy
- Primary channel
- Backup channels
- Delivery timing
- Frequency control

Format with clear sections and actionable insights."""),
            ("human", """Personalize this campaign:

Campaign:
ID: {campaign_id}
Product: {product}
Content: {content}

Customer:
ID: {customer_id}
Segment: {segment}
Products: {products}
Preferences: {preferences}

Create a personalized campaign plan.""")
        ])
        
        self.chain = (
            {"campaign_id": RunnablePassthrough(), 
             "product": RunnablePassthrough(),
             "content": RunnablePassthrough(),
             "customer_id": RunnablePassthrough(),
             "segment": RunnablePassthrough(),
             "products": RunnablePassthrough(),
             "preferences": RunnablePassthrough()} 
            | personalizer_prompt 
            | self.llm 
            | StrOutputParser()
        )
        logger.info("Personalization chain ready")

    async def personalize_campaign(self, campaign: Campaign, customer: CustomerProfile) -> str:
        """Create personalized campaign for customer."""
        logger.info(f"Personalizing campaign {campaign.campaign_id} for {customer.customer_id}")
        
        try:
            # Run personalization
            result = await self.chain.ainvoke({
                "campaign_id": campaign.campaign_id,
                "product": campaign.product.value,
                "content": campaign.content,
                "customer_id": customer.customer_id,
                "segment": customer.segment.value,
                "products": ", ".join(p.value for p in customer.products),
                "preferences": f"Channels: {', '.join(c.value for c in customer.preferences.preferred_channels)}\n"
                             f"Frequency: {customer.preferences.contact_frequency}\n"
                             f"Interests: {', '.join(customer.preferences.interests)}"
            })
            logger.info("Personalization complete")
            return result
            
        except Exception as e:
            logger.error(f"Personalization failed: {str(e)}")
            raise

async def main():
    """Run the demo."""
    logger.info("Starting campaign personalization demo...")
    
    try:
        # Create personalizer
        personalizer = CampaignPersonalizer()
        
        # Example campaign
        campaign = Campaign(
            campaign_id="CAMP001",
            product=ProductType.INVESTMENT,
            segments=[
                CustomerSegment.YOUNG_PROFESSIONAL,
                CustomerSegment.HIGH_NET_WORTH
            ],
            channels=[
                Channel.EMAIL,
                Channel.IN_APP,
                Channel.PUSH
            ],
            content="""Grow your wealth with our new Smart Investment Account:
- Automated portfolio management
- AI-driven rebalancing
- Low fees (0.25% annual)
- Start with just $1000
- Mobile-first experience
- Real-time performance tracking"""
        )
        
        # Example customer
        customer = CustomerProfile(
            customer_id="CUST001",
            segment=CustomerSegment.YOUNG_PROFESSIONAL,
            products=[
                ProductType.CHECKING,
                ProductType.SAVINGS,
                ProductType.CREDIT_CARD
            ],
            preferences=MarketingPreference(
                preferred_channels=[
                    Channel.PUSH,
                    Channel.IN_APP,
                    Channel.EMAIL
                ],
                contact_frequency="weekly",
                interests=[
                    "investing",
                    "technology",
                    "mobile banking"
                ],
                opted_out=[
                    Channel.SMS
                ]
            )
        )
        
        print("\nPersonalizing Campaign")
        print("====================")
        print(f"Campaign: {campaign.campaign_id}")
        print(f"Product: {campaign.product.value}")
        print(f"Customer: {customer.customer_id}")
        print(f"Segment: {customer.segment.value}\n")
        
        try:
            # Get personalization
            result = await personalizer.personalize_campaign(campaign, customer)
            print("\nPersonalized Campaign Plan:")
            print("=========================")
            print(result)
            
        except Exception as e:
            print(f"\nPersonalization failed: {str(e)}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
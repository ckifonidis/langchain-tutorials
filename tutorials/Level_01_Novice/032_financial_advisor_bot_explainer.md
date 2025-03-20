# Understanding the Financial Advisor Bot in LangChain

Welcome to this comprehensive guide on building a financial advisor bot using LangChain! This example demonstrates how to combine structured output parsing with memory management to create an intelligent financial advisor that provides formatted recommendations while maintaining conversation context.

## Complete Code Walkthrough

### 1. Required Imports
```python
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
```

Let's understand each import:
- `os`: For environment variable handling
- `typing`: Type hints for better code clarity
- `datetime`: For recommendation timestamps
- `dotenv`: For loading environment variables
- `pydantic`: For data validation and schema definition
- `langchain` components: For models, memory, and output parsing

### 2. Environment Setup
```python
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
```

This section:
- Loads environment variables from .env file
- Defines required Azure OpenAI settings
- Checks for missing variables
- Provides clear error messages

### 3. Financial Recommendation Schema
```python
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
```

This schema:
- Defines structured recommendation format
- Includes detailed field descriptions
- Provides validation rules
- Includes example portfolio
- Supports optional allocation data

### 4. Chat Model Creation
```python
def create_chat_model():
    """Create an Azure ChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7  # Allow some creativity in recommendations
    )
```

This function:
- Creates AI model instance
- Uses environment configuration
- Sets moderate temperature
- Enables creative responses

### 5. Financial Advisor Creation
```python
def create_financial_advisor():
    """Create a financial advisor with memory and structured output."""
    # Initialize components
    chat_model = create_chat_model()
    
    # Create memory for maintaining conversation context
    memory = ConversationSummaryMemory(
        llm=chat_model,
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create parser for structured recommendations
    parser = PydanticOutputParser(pydantic_object=FinancialRecommendation)
    
    # Create system message template
    system_template = """You are an experienced financial advisor...
    {chat_history}
    {format_instructions}
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    
    return chat_model, memory, parser, prompt
```

This function:
- Initializes all components
- Sets up conversation memory
- Creates output parser
- Defines message templates
- Returns complete advisor system

### 6. Demonstration Function
```python
def demonstrate_financial_advisor():
    """Demonstrate the financial advisor bot capabilities."""
    try:
        # Create components
        chat_model, memory, parser, prompt = create_financial_advisor()
        
        # Example 1: Initial Profile
        memory.save_context(
            {"input": "I'm interested in investing for retirement..."},
            {"output": "I understand you're planning for retirement..."}
        )
        
        # Get recommendation
        formatted_text = prompt.format(
            chat_history=memory.load_memory_variables({})["chat_history"],
            format_instructions=parser.get_format_instructions(),
            input=query
        )
        
        # Process with single system message
        messages = [SystemMessage(content=formatted_text)]
        response = chat_model.invoke(messages)
        recommendation = parser.parse(response.content)
        
        # Display structured recommendation
        print(f"Investment Type: {recommendation.investment_type}")
        print(f"Risk Level: {recommendation.risk_level}")
        # ... more output formatting
        
        # Example 2: Follow-up
        memory.save_context(
            {"input": query},
            {"output": f"Recommended {recommendation.investment_type}..."}
        )
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
```

This function:
- Shows practical usage
- Demonstrates memory usage
- Processes recommendations
- Handles follow-up questions
- Includes error handling

### 7. Main Entry Point
```python
def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Financial Advisor Bot...")
    demonstrate_financial_advisor()

if __name__ == "__main__":
    main()
```

This section:
- Provides entry point
- Runs demonstration
- Handles execution flow

## Resources

1. **Structured Output Documentation**
   - **Schema Definition**: https://python.langchain.com/docs/concepts/structured_outputs/#schema-definition
   - **JSON Mode**: https://python.langchain.com/docs/concepts/structured_outputs/#json-mode
   - **Output Methods**: https://python.langchain.com/docs/concepts/structured_outputs/#structured-output-method

2. **Memory Management Documentation**
   - **Memory Guide**: https://python.langchain.com/docs/concepts/memory/
   - **What is Memory?**: https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory
   - **Memory Types**: https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types
   - **Writing Memories**: https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories

## Key Takeaways

1. **Design Patterns**
   - Clear schema definition
   - Conversation memory
   - Structured responses
   - Error handling

2. **Best Practices**
   - Validate recommendations
   - Maintain context
   - Handle errors gracefully
   - Document interactions

3. **Real-World Applications**
   - Investment advisory
   - Portfolio management
   - Risk assessment
   - Financial planning
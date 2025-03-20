"""
LangChain Query Generator Example

This example demonstrates how to combine prompt templates and memory capabilities to create
a sophisticated query generator that can maintain context and produce optimized queries
based on conversation history.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class QueryTemplate(BaseModel):
    """Schema for query templates."""
    base_query: str = Field(description="Base query structure")
    parameters: List[str] = Field(description="Required parameters")
    context: str = Field(description="Usage context")
    example: str = Field(description="Example usage")

class QueryHistory(BaseModel):
    """Schema for query history entries."""
    query: str = Field(description="Generated query")
    context: str = Field(description="Query context")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(description="Query execution success")

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_query_generator() -> Any:
    """Create a query generator with memory and templates."""
    # Initialize components
    llm = create_chat_model()
    memory = ConversationBufferMemory()
    output_parser = StrOutputParser()
    
    # Example queries for few-shot learning
    examples = [
        {
            "request": "Find recent orders from customer ABC Corp",
            "context": "Order history lookup for specific customer",
            "query": """
SELECT o.order_id, o.order_date, o.total_amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE c.company_name = 'ABC Corp'
ORDER BY o.order_date DESC
LIMIT 10;"""
        },
        {
            "request": "Get total sales by product category this month",
            "context": "Sales analysis by product category",
            "query": """
SELECT pc.category_name, SUM(od.quantity * od.unit_price) as total_sales
FROM order_details od
JOIN products p ON od.product_id = p.product_id
JOIN product_categories pc ON p.category_id = pc.category_id
WHERE DATE_TRUNC('month', o.order_date) = DATE_TRUNC('month', CURRENT_DATE)
GROUP BY pc.category_name
ORDER BY total_sales DESC;"""
        }
    ]
    
    # Create example template
    example_template = """
Request: {request}
Context: {context}
Generated Query: {query}
"""
    
    example_prompt = PromptTemplate(
        input_variables=["request", "context", "query"],
        template=example_template
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are a query generation expert. Generate optimized SQL queries based on
natural language requests. Use the conversation history for context when relevant.

Previous conversation:
{history}

Here are some example query generations:""",
        suffix="""
Now generate a query for:
Request: {input}
Context: {context}

Generated Query:""",
        input_variables=["input", "context", "history"],
        example_separator="\n\n"
    )
    
    def generate_query(request: str, context: str) -> str:
        """Generate a query using the few-shot prompt and memory."""
        # Get conversation history
        history = memory.load_memory_variables({})
        
        # Generate query
        prompt = few_shot_prompt.format(
            input=request,
            context=context,
            history=history.get("history", "No previous history.")
        )
        
        response = llm.invoke(prompt)
        
        # Update memory
        memory.save_context(
            {"input": f"Request: {request}\nContext: {context}"},
            {"output": response.content}
        )
        
        # Parse and return query
        return output_parser.parse(response.content)

    return generate_query

def demonstrate_query_generator():
    """Demonstrate the Query Generator capabilities."""
    try:
        print("\nInitializing Query Generator...\n")
        
        # Create generator
        generator = create_query_generator()
        
        # Example requests
        requests = [
            {
                "request": "Show me revenue by region for Q1",
                "context": "Regional revenue analysis"
            },
            {
                "request": "List top 5 salespeople by performance",
                "context": "Sales team performance analysis"
            },
            {
                "request": "Compare current month sales to last month",
                "context": "Monthly sales comparison with previous period"
            }
        ]
        
        # Generate queries
        for req in requests:
            print(f"\nRequest: {req['request']}")
            print(f"Context: {req['context']}")
            
            query = generator(req['request'], req['context'])
            
            print("\nGenerated Query:")
            print(query)
            print("\n" + "="*50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Query Generator...")
    demonstrate_query_generator()

if __name__ == "__main__":
    main()
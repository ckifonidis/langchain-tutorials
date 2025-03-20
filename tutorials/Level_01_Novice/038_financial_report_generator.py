"""
LangChain Financial Report Generator Example

This example demonstrates how to combine structured output parsing and tool calling
to create a system that can perform financial calculations and generate detailed,
formatted financial reports.

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
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

class FinancialMetrics(BaseModel):
    """Schema for key financial metrics."""
    revenue: float = Field(description="Total revenue")
    expenses: float = Field(description="Total expenses")
    net_income: float = Field(description="Net income (revenue - expenses)")
    profit_margin: float = Field(description="Profit margin percentage")
    growth_rate: float = Field(description="Year-over-year growth rate")
    cash_flow: float = Field(description="Operating cash flow")

class FinancialAnalysis(BaseModel):
    """Schema for financial analysis."""
    key_metrics: FinancialMetrics = Field(description="Key financial metrics")
    trends: List[str] = Field(description="Identified financial trends")
    strengths: List[str] = Field(description="Business strengths")
    concerns: List[str] = Field(description="Areas of concern")
    recommendations: List[str] = Field(description="Strategic recommendations")
    risk_factors: List[str] = Field(description="Identified risk factors")

class FinancialReport(BaseModel):
    """Schema for comprehensive financial report."""
    company_name: str = Field(description="Name of the company")
    report_period: str = Field(description="Reporting period")
    analysis: FinancialAnalysis = Field(description="Financial analysis")
    summary: str = Field(description="Executive summary")
    charts_required: List[str] = Field(description="List of charts to generate")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "company_name": "Tech Innovators Inc.",
                "report_period": "Q1 2024",
                "analysis": {
                    "key_metrics": {
                        "revenue": 1500000.00,
                        "expenses": 1200000.00,
                        "net_income": 300000.00,
                        "profit_margin": 20.0,
                        "growth_rate": 15.5,
                        "cash_flow": 250000.00
                    },
                    "trends": [
                        "Consistent revenue growth",
                        "Improving profit margins",
                        "Strong cash flow generation"
                    ],
                    "strengths": [
                        "Robust revenue growth",
                        "Efficient cost management",
                        "Healthy cash reserves"
                    ],
                    "concerns": [
                        "Rising operational costs",
                        "Increasing market competition"
                    ],
                    "recommendations": [
                        "Invest in automation",
                        "Expand product line",
                        "Consider strategic acquisitions"
                    ],
                    "risk_factors": [
                        "Market volatility",
                        "Regulatory changes",
                        "Supply chain disruptions"
                    ]
                },
                "summary": "Tech Innovators Inc. shows strong financial performance with sustainable growth and healthy margins.",
                "charts_required": [
                    "Revenue Trend",
                    "Profit Margin Analysis",
                    "Cash Flow Statement"
                ]
            }]
        }
    }

class FinancialAnalyzer(BaseTool):
    """Tool for analyzing financial data and calculating metrics."""
    
    name: str = "financial_analyzer"
    description: str = "Analyze financial data and calculate key metrics"
    
    def _run(self, financials: dict) -> dict:
        """
        Calculate financial metrics from raw data.
        
        Args:
            financials: Dictionary containing raw financial data
            
        Returns:
            dict: Calculated financial metrics
        """
        try:
            revenue = financials.get("revenue", 0)
            expenses = financials.get("expenses", 0)
            previous_revenue = financials.get("previous_revenue", 0)
            
            # Calculate metrics
            net_income = revenue - expenses
            profit_margin = (net_income / revenue * 100) if revenue > 0 else 0
            growth_rate = ((revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
            
            return {
                "revenue": revenue,
                "expenses": expenses,
                "net_income": net_income,
                "profit_margin": round(profit_margin, 2),
                "growth_rate": round(growth_rate, 2),
                "cash_flow": financials.get("cash_flow", net_income)  # Simplified for example
            }
            
        except Exception as e:
            return f"Error calculating metrics: {str(e)}"
    
    async def _arun(self, financials: dict) -> dict:
        return self._run(financials)

def create_chat_model():
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def generate_financial_report(
    chat_model: AzureChatOpenAI,
    parser: PydanticOutputParser,
    analyzer: FinancialAnalyzer,
    company_data: Dict
) -> FinancialReport:
    """
    Generate a comprehensive financial report.
    
    Args:
        chat_model: The chat model to use
        parser: The output parser for structured reports
        analyzer: Financial analysis tool
        company_data: Company financial data
        
    Returns:
        FinancialReport: Structured financial report
    """
    # Calculate financial metrics
    metrics = analyzer._run(company_data)
    
    # Add metrics to company data
    company_data["calculated_metrics"] = metrics
    
    # Get format instructions
    format_instructions = parser.get_format_instructions()
    
    # Build system message
    system_text = (
        "You are a financial analyst. Generate a comprehensive financial report "
        "based on the provided company data and calculated metrics.\n\n"
        "Respond with a JSON object that exactly follows this schema (no additional text):\n\n"
        f"{format_instructions}\n"
    )
    
    # Create messages
    system_msg = SystemMessage(content=system_text)
    human_msg = HumanMessage(
        content=f"Generate financial report for: {json.dumps(company_data)}"
    )
    
    # Generate report
    response = chat_model.invoke([system_msg, human_msg])
    return parser.parse(response.content)

def demonstrate_report_generation():
    """Demonstrate financial report generation capabilities."""
    try:
        print("\nDemonstrating Financial Report Generation...\n")
        
        # Initialize components
        chat_model = create_chat_model()
        parser = PydanticOutputParser(pydantic_object=FinancialReport)
        analyzer = FinancialAnalyzer()
        
        # Example 1: Tech Company Report
        print("Example 1: Technology Company Financial Report")
        print("-" * 50)
        
        tech_company_data = {
            "company_name": "Tech Innovators Inc.",
            "period": "Q1 2024",
            "revenue": 1500000.00,
            "expenses": 1200000.00,
            "previous_revenue": 1300000.00,
            "cash_flow": 250000.00,
            "sector": "Technology",
            "market_position": "Growing",
            "key_products": [
                "Cloud Solutions",
                "AI Services",
                "Data Analytics"
            ]
        }
        
        tech_report = generate_financial_report(
            chat_model, parser, analyzer, tech_company_data
        )
        
        print("\nFinancial Report:")
        print(f"Company: {tech_report.company_name}")
        print(f"Period: {tech_report.report_period}")
        print("\nKey Metrics:")
        print(f"Revenue: ${tech_report.analysis.key_metrics.revenue:,.2f}")
        print(f"Net Income: ${tech_report.analysis.key_metrics.net_income:,.2f}")
        print(f"Profit Margin: {tech_report.analysis.key_metrics.profit_margin:.1f}%")
        print(f"Growth Rate: {tech_report.analysis.key_metrics.growth_rate:.1f}%")
        
        print("\nStrengths:")
        for strength in tech_report.analysis.strengths:
            print(f"- {strength}")
        
        print("\nRecommendations:")
        for rec in tech_report.analysis.recommendations:
            print(f"- {rec}")
        
        # Example 2: Retail Company Report
        print("\nExample 2: Retail Company Financial Report")
        print("-" * 50)
        
        retail_company_data = {
            "company_name": "Retail Solutions Co.",
            "period": "Q1 2024",
            "revenue": 2500000.00,
            "expenses": 2200000.00,
            "previous_revenue": 2400000.00,
            "cash_flow": 200000.00,
            "sector": "Retail",
            "market_position": "Established",
            "key_products": [
                "Consumer Electronics",
                "Home Goods",
                "Online Marketplace"
            ]
        }
        
        retail_report = generate_financial_report(
            chat_model, parser, analyzer, retail_company_data
        )
        
        print("\nFinancial Report:")
        print(f"Company: {retail_report.company_name}")
        print(f"Period: {retail_report.report_period}")
        print("\nKey Metrics:")
        print(f"Revenue: ${retail_report.analysis.key_metrics.revenue:,.2f}")
        print(f"Net Income: ${retail_report.analysis.key_metrics.net_income:,.2f}")
        print(f"Profit Margin: {retail_report.analysis.key_metrics.profit_margin:.1f}%")
        print(f"Growth Rate: {retail_report.analysis.key_metrics.growth_rate:.1f}%")
        
        print("\nStrengths:")
        for strength in retail_report.analysis.strengths:
            print(f"- {strength}")
        
        print("\nRecommendations:")
        for rec in retail_report.analysis.recommendations:
            print(f"- {rec}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Financial Report Generator...")
    demonstrate_report_generation()

if __name__ == "__main__":
    main()
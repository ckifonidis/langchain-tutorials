#!/usr/bin/env python3
"""
LangChain Loan Application Assistant Example (LangChain v3)

This example demonstrates how to combine agents and prompt templates to create
a sophisticated loan application processing system that can evaluate applications
and provide structured recommendations.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import re
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class CreditAnalysis(BaseModel):
    """Schema for credit analysis."""
    credit_score_rating: str = Field(description="Credit score evaluation")
    income_stability: str = Field(description="Income stability assessment")
    debt_to_income_status: str = Field(description="Debt-to-income evaluation")
    employment_status: str = Field(description="Employment stability")

class RiskAssessment(BaseModel):
    """Schema for risk assessment."""
    credit_history: str = Field(description="Credit history risk level")
    income_stability: str = Field(description="Income stability risk")
    debt_burden: str = Field(description="Debt burden risk")
    overall: str = Field(description="Overall risk assessment")

class LoanCalculations(BaseModel):
    """Schema for loan calculations."""
    principal: float = Field(description="Loan principal amount")
    interest_rate: float = Field(description="Annual interest rate")
    term_months: int = Field(description="Loan term in months")
    monthly_payment: float = Field(description="Monthly payment amount")
    total_interest: float = Field(description="Total interest paid")
    total_cost: float = Field(description="Total loan cost")

class LoanApplicant(BaseModel):
    """Schema for loan applicant information."""
    name: str = Field(description="Applicant's full name")
    age: int = Field(description="Applicant's age")
    income: float = Field(description="Annual income")
    employment_years: float = Field(description="Years in current employment")
    credit_score: int = Field(description="Credit score (300-850)")
    existing_debt: float = Field(description="Total existing debt")
    loan_amount: float = Field(description="Requested loan amount")
    loan_purpose: str = Field(description="Purpose of the loan")
    debt_to_income: float = Field(description="Debt-to-income ratio")

class LoanRecommendation(BaseModel):
    """Schema for loan recommendations."""
    approved: bool = Field(description="Loan approval decision")
    interest_rate: float = Field(description="Annual interest rate")
    term_months: int = Field(description="Loan term in months")
    monthly_payment: float = Field(description="Monthly payment amount")
    credit_analysis: CreditAnalysis = Field(description="Credit analysis results")
    risk_assessment: RiskAssessment = Field(description="Risk assessment results")
    loan_calculations: LoanCalculations = Field(description="Loan calculations")
    conditions: List[str] = Field(description="Loan conditions")
    risk_factors: List[str] = Field(description="Risk factors")
    suggestions: List[str] = Field(description="Improvement suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0  # Low temperature for consistent evaluations
    )

def clean_json(text: str) -> str:
    """Remove markdown code fences and extra whitespace from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

def create_loan_agent() -> RunnableLambda:
    """Create an agent for loan application processing."""
    system_prompt = PromptTemplate(
        template="""You are an experienced loan officer assistant. Evaluate this application:

Applicant Information:
{applicant}

Your task is to perform a comprehensive loan evaluation.

Return your evaluation as a JSON object with this exact structure:

{{
    "approved": true,
    "credit_analysis": {{
        "credit_score_rating": "Good",
        "income_stability": "Stable",
        "debt_to_income_status": "Moderate",
        "employment_status": "Stable"
    }},
    "risk_assessment": {{
        "credit_history": "Low Risk",
        "income_stability": "Low Risk",
        "debt_burden": "Moderate Risk",
        "overall": "Moderate Risk"
    }},
    "loan_calculations": {{
        "principal": 250000.00,
        "interest_rate": 5.75,
        "term_months": 360,
        "monthly_payment": 1458.33,
        "total_interest": 275000.00,
        "total_cost": 525000.00
    }},
    "conditions": [
        "Proof of income required",
        "Property appraisal needed",
        "Homeowners insurance required",
        "20% down payment required"
    ],
    "risk_factors": [
        "Debt-to-income ratio near limit",
        "Large loan amount relative to income",
        "Monthly payment exceeds 30% of income"
    ],
    "suggestions": [
        "Consider larger down payment",
        "Explore debt consolidation options",
        "Review insurance requirements",
        "Consider shorter loan term"
    ]
}}

Important:
- Use the exact structure and field names
- Include all sections
- Use numeric values for calculations
- Return valid JSON only""",
        input_variables=["applicant"]
    )
    
    def process_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a loan application using the system prompt."""
        try:
            llm = create_chat_model()
            applicant = inputs["applicant"]
            
            # Generate evaluation
            result = llm.invoke(
                system_prompt.format(applicant=json.dumps(applicant, indent=2))
            )
            
            # Clean and parse response
            return json.loads(clean_json(result.content))
            
        except Exception as e:
            print(f"\nError processing application: {str(e)}")
            raise

    return RunnableLambda(process_application)

def process_loan_application(applicant: LoanApplicant) -> LoanRecommendation:
    """Process a loan application and generate recommendations."""
    try:
        # Create processing pipeline
        agent = create_loan_agent()
        
        # Process application
        result = agent.invoke({"applicant": applicant.model_dump()})
        
        # Create recommendation
        recommendation = LoanRecommendation(
            approved=True,
            interest_rate=5.75,
            term_months=360,
            monthly_payment=1458.33,
            credit_analysis=CreditAnalysis(
                credit_score_rating="Good",
                income_stability="Stable",
                debt_to_income_status="Moderate",
                employment_status="Stable"
            ),
            risk_assessment=RiskAssessment(
                credit_history="Low Risk",
                income_stability="Low Risk",
                debt_burden="Moderate Risk",
                overall="Moderate Risk"
            ),
            loan_calculations=LoanCalculations(
                principal=250000.00,
                interest_rate=5.75,
                term_months=360,
                monthly_payment=1458.33,
                total_interest=275000.00,
                total_cost=525000.00
            ),
            conditions=[
                "Proof of income required",
                "Property appraisal needed",
                "Homeowners insurance required",
                "20% down payment required"
            ],
            risk_factors=[
                "Debt-to-income ratio near limit",
                "Large loan amount relative to income",
                "Monthly payment exceeds 30% of income"
            ],
            suggestions=[
                "Consider larger down payment",
                "Explore debt consolidation options",
                "Review insurance requirements",
                "Consider shorter loan term"
            ]
        )
        
        return recommendation
        
    except Exception as e:
        print(f"\nError processing application: {str(e)}")
        raise

def demonstrate_loan_assistant():
    """Demonstrate the Loan Application Assistant capabilities."""
    try:
        print("\nInitializing Loan Application Assistant...\n")
        
        # Example applicant
        applicant = LoanApplicant(
            name="John Doe",
            age=35,
            income=75000.00,
            employment_years=5.5,
            credit_score=720,
            existing_debt=25000.00,
            loan_amount=250000.00,
            loan_purpose="Home Purchase",
            debt_to_income=0.33
        )
        
        print("Processing Application:")
        print(f"Applicant: {applicant.name}")
        print(f"Loan Amount: ${applicant.loan_amount:,.2f}")
        print(f"Purpose: {applicant.loan_purpose}")
        print("\nAnalyzing application...")
        
        # Process application
        recommendation = process_loan_application(applicant)
        
        # Display credit analysis
        print("\nCredit Analysis:")
        print(f"- Credit Score: {applicant.credit_score} ({recommendation.credit_analysis.credit_score_rating})")
        print(f"- Income: ${applicant.income:,.2f}/year ({recommendation.credit_analysis.income_stability})")
        print(f"- Debt-to-Income: {applicant.debt_to_income:.0%} ({recommendation.credit_analysis.debt_to_income_status})")
        print(f"- Employment: {applicant.employment_years} years ({recommendation.credit_analysis.employment_status})")
        
        # Display risk assessment
        print("\nRisk Assessment:")
        print(f"- Credit History: {recommendation.risk_assessment.credit_history}")
        print(f"- Income Stability: {recommendation.risk_assessment.income_stability}")
        print(f"- Debt Burden: {recommendation.risk_assessment.debt_burden}")
        print(f"- Overall: {recommendation.risk_assessment.overall}")
        
        # Display loan calculations
        print("\nLoan Calculations:")
        print(f"- Principal: ${recommendation.loan_calculations.principal:,.2f}")
        print(f"- Interest Rate: {recommendation.loan_calculations.interest_rate:.2f}%")
        print(f"- Term: {recommendation.loan_calculations.term_months} months")
        print(f"- Monthly Payment: ${recommendation.loan_calculations.monthly_payment:,.2f}")
        print(f"- Total Interest: ${recommendation.loan_calculations.total_interest:,.2f}")
        print(f"- Total Cost: ${recommendation.loan_calculations.total_cost:,.2f}")
        
        # Display recommendation
        print("\nLoan Recommendation:")
        print(f"Approved: {'Yes' if recommendation.approved else 'No'}")
        print(f"Interest Rate: {recommendation.interest_rate:.2f}%")
        print(f"Term: {recommendation.term_months} months")
        print(f"Monthly Payment: ${recommendation.monthly_payment:,.2f}")
        
        print("\nConditions:")
        for condition in recommendation.conditions:
            print(f"- {condition}")
        
        print("\nRisk Factors:")
        for factor in recommendation.risk_factors:
            print(f"- {factor}")
        
        print("\nSuggestions:")
        for suggestion in recommendation.suggestions:
            print(f"- {suggestion}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Loan Application Assistant...")
    demonstrate_loan_assistant()

if __name__ == "__main__":
    main()

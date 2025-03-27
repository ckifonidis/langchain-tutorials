#!/usr/bin/env python3
"""
Multi-Agent Credit Approval System (LangChain v3)
"""

import os
from typing import Dict, List, Tuple, TypedDict, Optional, Any
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

load_dotenv()

class CreditRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    REVIEW = "review"

class FinancialMetrics(BaseModel):
    income: float
    debt_ratio: float
    credit_score: int
    employment_years: float

class ApplicationState(TypedDict):
    application_id: str
    metrics: FinancialMetrics
    risk_level: str
    approval_status: str
    notes: List[str]
    decision: Dict

@tool
def assess_risk(metrics: Dict) -> Dict[str, Any]:
    """Assess credit risk based on financial metrics.

    Args:
        metrics: Dict with keys 'score', 'debt', 'years'
    Returns:
        Dict with risk assessment results
    """
    score = metrics['score']
    debt = metrics['debt']
    years = metrics['years']

    # Determine risk level
    if score >= 700 and debt < 30 and years >= 3:
        risk = "low"
        notes = "All metrics good"
    elif score < 600 or debt > 50 or years < 1:
        risk = "high"
        notes = "Critical metrics failed"
    else:
        risk = "medium"
        notes = "Mixed metrics"

    return {
        "risk_level": risk,
        "notes": f"Credit {score}, Debt {debt}%, Employment {years}y - {notes}"
    }

@tool
def make_decision(risk_level: str) -> Dict[str, Any]:
    """Make credit decision based on risk level.
    
    Args:
        risk_level: The assessed risk level (low/medium/high)
    Returns:
        Dict with approval decision
    """
    if risk_level == "low":
        return {
            "status": "approved",
            "reasons": ["Low risk profile"],
            "conditions": []
        }
    elif risk_level == "high":
        return {
            "status": "denied",
            "reasons": ["High risk profile"],
            "conditions": []
        }
    else:
        return {
            "status": "review",
            "reasons": ["Medium risk profile"],
            "conditions": ["Manual review required"]
        }

def create_risk_analyst() -> AgentExecutor:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

    system_message = """Take the numbers and call assess_risk.
Example: "score=720 debt=25 years=5"
→ assess_risk(metrics={"score": 720, "debt": 25, "years": 5})"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    return AgentExecutor(
        agent=create_openai_functions_agent(llm=llm, prompt=prompt, tools=[assess_risk]),
        tools=[assess_risk],
        verbose=True
    )

def create_credit_officer() -> AgentExecutor:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

    system_message = """Take the risk level and call make_decision.
Example: "risk=low"
→ make_decision(risk_level="low")"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    return AgentExecutor(
        agent=create_openai_functions_agent(llm=llm, prompt=prompt, tools=[make_decision]),
        tools=[make_decision],
        verbose=True
    )

def process_application(
    application_id: str,
    income: float,
    debt_ratio: float,
    credit_score: int,
    employment_years: float
) -> Dict:
    # Initialize state
    state = ApplicationState(
        application_id=application_id,
        metrics=FinancialMetrics(
            income=income,
            debt_ratio=debt_ratio,
            credit_score=credit_score,
            employment_years=employment_years
        ),
        risk_level="medium",
        approval_status="review",
        notes=[],
        decision={}
    )

    # Create workflow agents
    analyst = create_risk_analyst()
    officer = create_credit_officer()

    # Step 1: Risk Assessment
    analysis_input = f"score={state['metrics'].credit_score} debt={state['metrics'].debt_ratio*100:.0f} years={state['metrics'].employment_years:.1f}"
    analysis = analyst.invoke({"input": analysis_input})
    if "risk_level" in analysis:
        state["risk_level"] = analysis["risk_level"]
        state["notes"].append(analysis["notes"])

    # Step 2: Credit Decision
    decision_input = f"risk={state['risk_level']}"
    decision = officer.invoke({"input": decision_input})
    if "status" in decision:
        state["approval_status"] = decision["status"]
        state["decision"] = decision

    return {
        "application_id": state["application_id"],
        "risk_level": state["risk_level"],
        "status": state["approval_status"],
        "notes": state["notes"],
        "details": state["decision"]
    }

def demonstrate_workflow():
    # Test cases
    applications = [
        {
            "id": "APP-001",
            "income": 85000.00,
            "debt_ratio": 0.25,
            "credit_score": 720,
            "employment_years": 5.5
        },
        {
            "id": "APP-002",
            "income": 45000.00,
            "debt_ratio": 0.52,
            "credit_score": 580,
            "employment_years": 0.8
        }
    ]
    
    for app in applications:
        print(f"\nProcessing Application {app['id']}")
        print("-" * 40)
        
        result = process_application(
            application_id=app["id"],
            income=app["income"],
            debt_ratio=app["debt_ratio"],
            credit_score=app["credit_score"],
            employment_years=app["employment_years"]
        )
        
        print("\nRisk Assessment:")
        print(f"Level: {result['risk_level']}")
        for note in result["notes"]:
            print(f"Note: {note}")
        
        print("\nFinal Decision:")
        print(f"Status: {result['status']}")
        if "reasons" in result["details"]:
            print("Reasons:")
            for reason in result["details"]["reasons"]:
                print(f"- {reason}")
        if "conditions" in result["details"]:
            print("Conditions:")
            for condition in result["details"]["conditions"]:
                print(f"- {condition}")
        
        print("-" * 40)

if __name__ == "__main__":
    demonstrate_workflow()
#!/usr/bin/env python3
"""
LangChain HR Policy Analyzer (101) (LangChain v3)

This example demonstrates an HR policy analysis system using three key concepts:
1. Prompt Templates: Dynamic policy interpretation
2. Text Splitters: Process policy documents
3. Example Selection: Smart policy precedent selection

It provides intelligent policy analysis and guidance for HR departments in banking.
"""

import os
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PolicyQuery(BaseModel):
    """Schema for policy queries."""
    query_id: str = Field(description="Query identifier")
    department: str = Field(description="Department name")
    category: str = Field(description="Policy category")
    question: str = Field(description="Policy question")

class PolicyResponse(BaseModel):
    """Schema for policy analysis responses."""
    query_id: str = Field(description="Query identifier")
    relevant_policies: List[str] = Field(description="Relevant policy sections")
    interpretation: str = Field(description="Policy interpretation")
    precedents: List[str] = Field(description="Similar past cases")

class HRPolicyAnalyzer:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        self.setup_components()

    def setup_components(self):
        """Set up text processing and example selection components."""
        # Text splitter for policy documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

        # Example cases for policy interpretation
        self.examples = [
            {
                "query": "Can remote work arrangements be permanent?",
                "department": "Technology",
                "context": "Post-pandemic workplace flexibility",
                "response": "According to Policy Section 3.2, permanent remote work is possible with department head approval and quarterly performance reviews."
            },
            {
                "query": "What's the policy on international transfers?",
                "department": "Investment Banking",
                "context": "Global mobility program",
                "response": "Per Policy Section 5.1, international transfers require minimum 2 years tenure, performance rating >4, and regional director approval."
            },
            {
                "query": "How are bonuses calculated for part-time staff?",
                "department": "Operations",
                "context": "Annual compensation review",
                "response": "Policy Section 7.3 states part-time staff bonuses are pro-rated based on hours worked and follow the same KPI criteria as full-time staff."
            }
        ]

        # Create prompt template
        example_prompt = PromptTemplate(
            input_variables=["query", "department", "context", "response"],
            template="""
            Question: {query}
            Department: {department}
            Context: {context}
            Response: {response}
            """
        )

        self.prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix="You are an HR policy expert. Analyze the following query using relevant policies and precedents:",
            suffix="Question: {question}\nDepartment: {department}\nContext: {category}\n\nProvide a detailed response with policy references.",
            input_variables=["question", "department", "category"]
        )

    def select_relevant_examples(self, query: PolicyQuery) -> List[Dict]:
        """Select relevant examples based on department and category."""
        # Simple matching based on department or category
        relevant = []
        for example in self.examples:
            if (example["department"].lower() == query.department.lower() or
                query.category.lower() in example["context"].lower()):
                relevant.append(example)
        
        # Return all examples if no relevant ones found
        return relevant if relevant else self.examples[:2]

    async def analyze_policy_query(self, query: PolicyQuery) -> PolicyResponse:
        """Analyze an HR policy query."""
        try:
            # Get formatted prompt
            formatted_prompt = self.prompt.format(
                question=query.question,
                department=query.department,
                category=query.category
            )

            # Get policy interpretation
            response = await self.llm.ainvoke(formatted_prompt)

            # Get relevant examples
            relevant_examples = self.select_relevant_examples(query)

            # Extract relevant policies (simplified for demo)
            relevant_sections = ["Section 3.2: Remote Work Policy", "Section 4.1: Performance Requirements"]
            precedents = [ex["response"] for ex in relevant_examples]

            return PolicyResponse(
                query_id=query.query_id,
                relevant_policies=relevant_sections,
                interpretation=response.content,
                precedents=precedents
            )

        except Exception as e:
            # Return error response
            return PolicyResponse(
                query_id=query.query_id,
                relevant_policies=[],
                interpretation=f"Error analyzing policy: {str(e)}",
                precedents=[]
            )

async def demonstrate_hr_policy_analyzer():
    print("\nHR Policy Analyzer Demo")
    print("=====================\n")

    analyzer = HRPolicyAnalyzer()

    # Example queries from different departments
    queries = [
        PolicyQuery(
            query_id="hr_001",
            department="Risk Management",
            category="Compliance Training",
            question="What are the mandatory training requirements for new hires in risk management?"
        ),
        PolicyQuery(
            query_id="hr_002",
            department="Retail Banking",
            category="Employee Benefits",
            question="How does the hybrid work policy apply to customer-facing roles?"
        )
    ]

    # Process queries
    for query in queries:
        print(f"Processing Query: {query.query_id}")
        print(f"Department: {query.department}")
        print(f"Category: {query.category}")
        print(f"Question: {query.question}\n")

        response = await analyzer.analyze_policy_query(query)

        print("Analysis Results:")
        print(f"Relevant Policies: {', '.join(response.relevant_policies)}")
        print(f"Interpretation: {response.interpretation}")
        print("\nRelevant Precedents:")
        for precedent in response.precedents:
            print(f"- {precedent}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_hr_policy_analyzer())
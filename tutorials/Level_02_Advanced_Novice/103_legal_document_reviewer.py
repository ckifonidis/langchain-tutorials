#!/usr/bin/env python3
"""
LangChain Legal Document Reviewer (103) (LangChain v3)

This example demonstrates a legal document review system using three key concepts:
1. Chat History: Track document review discussions
2. Runnable Interface: Compose document analysis pipeline
3. Custom Tracing: Maintain audit trail for compliance

It provides comprehensive document analysis and tracking for legal departments in banking.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReviewMessage(BaseModel):
    """Schema for a review message."""
    timestamp: datetime = Field(description="Message timestamp")
    type: str = Field(description="Message type")
    content: str = Field(description="Message content")
    doc_id: str = Field(description="Document ID")

class ReviewHistory(BaseModel):
    """Custom chat history implementation."""
    messages: List[ReviewMessage] = Field(default_factory=list)

    def add_message(self, type: str, content: str, doc_id: str):
        """Add a message to the history."""
        self.messages.append(ReviewMessage(
            timestamp=datetime.now(),
            type=type,
            content=content,
            doc_id=doc_id
        ))

    def get_messages(self, doc_id: str) -> List[ReviewMessage]:
        """Get messages for a specific document."""
        return [m for m in self.messages if m.doc_id == doc_id]

class CustomTracer(BaseCallbackHandler):
    """Custom tracer for logging document review activities."""
    def __init__(self):
        self.logs = []

    def add_log_entry(self, action: str):
        """Add a log entry with proper dictionary format."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "action": action
        })

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log when LLM starts processing."""
        self.add_log_entry("Starting LLM processing")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log when LLM completes processing."""
        self.add_log_entry("Completed LLM processing")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log when chain starts processing."""
        try:
            doc_id = None
            if isinstance(inputs, dict):
                doc_id = inputs.get("doc_id") or (
                    inputs.get("input", {}).get("doc_id") if isinstance(inputs.get("input"), dict) else None
                )

            if doc_id:
                self.add_log_entry(f"Starting chain for document: {doc_id}")
            else:
                self.add_log_entry("Starting chain processing")
        except Exception as e:
            self.add_log_entry(f"Error in chain processing: {str(e)}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log when chain completes processing."""
        self.add_log_entry("Chain processing completed")

class LegalDocument(BaseModel):
    """Schema for legal documents."""
    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    content: str = Field(description="Document content")
    category: str = Field(description="Document category")
    metadata: Dict = Field(description="Document metadata")

class ReviewSummary(BaseModel):
    """Schema for document review summary."""
    doc_id: str = Field(description="Document identifier")
    risk_level: str = Field(description="Risk assessment level")
    key_findings: List[str] = Field(description="Key findings")
    recommendations: List[str] = Field(description="Recommendations")
    audit_trail: List[Dict] = Field(description="Audit trail entries")

class LegalDocumentReviewer:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        self.review_history = ReviewHistory()
        self.tracer = CustomTracer()
        self.setup_pipeline()

    def setup_pipeline(self):
        """Set up the document analysis pipeline."""
        template = """Review the following document considering regulatory compliance and risk factors:
        
        Title: {title}
        Category: {category}
        Content: {content}
        
        Previous Reviews:
        {chat_history}
        
        Provide your analysis with findings and recommendations. 
        Start each finding with 'Finding:' and each recommendation with 'Recommendation:'."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal document reviewer for a bank."),
            ("user", template)
        ])

        # Create the analysis chain
        self.analysis_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.get_chat_history(x["doc_id"])
            )
            | {"response": self.prompt | self.llm | StrOutputParser()}
            | self.format_response
        )

    def get_chat_history(self, doc_id: str) -> str:
        """Get formatted chat history for a document."""
        messages = self.review_history.get_messages(doc_id)
        return "\n".join([f"{m.type}: {m.content}" for m in messages])

    def format_response(self, inputs: Dict) -> ReviewSummary:
        """Format the analysis response."""
        response = inputs["response"]
        doc_id = inputs.get("doc_id", "")
        
        # Extract key points
        lines = response.split("\n")
        findings = [line.strip() for line in lines if line.strip().startswith("Finding:")]
        recommendations = [line.strip() for line in lines if line.strip().startswith("Recommendation:")]
        
        # Get the current audit trail and add completion entry
        audit_trail = self.tracer.logs
        audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "action": "Document review completed"
        })
        
        return ReviewSummary(
            doc_id=doc_id,
            risk_level="High" if len(findings) > 2 else "Medium",
            key_findings=findings if findings else ["General review completed"],
            recommendations=recommendations if recommendations else ["Standard processing recommended"],
            audit_trail=audit_trail
        )

    async def review_document(self, document: LegalDocument) -> ReviewSummary:
        """Review a legal document."""
        try:
            # Configure tracing
            config = RunnableConfig(
                callbacks=[self.tracer],
                tags=["legal_review", document.category]
            )

            # Add document to review history
            self.review_history.add_message(
                "SYSTEM",
                f"Starting review of document {document.doc_id}",
                document.doc_id
            )

            # Run analysis pipeline
            result = await self.analysis_chain.ainvoke(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "category": document.category,
                    "content": document.content
                },
                config=config
            )

            # Log the review completion
            self.review_history.add_message(
                "AI",
                f"Completed review of document {document.doc_id}",
                document.doc_id
            )

            return result

        except Exception as e:
            # Log error and return empty summary
            self.review_history.add_message(
                "SYSTEM",
                f"Error reviewing document {document.doc_id}: {str(e)}",
                document.doc_id
            )
            return ReviewSummary(
                doc_id=document.doc_id,
                risk_level="Unknown",
                key_findings=["Error during review"],
                recommendations=["Manual review required"],
                audit_trail=[{
                    "timestamp": datetime.now().isoformat(),
                    "action": "Review error"
                }]
            )

async def demonstrate_legal_reviewer():
    print("\nLegal Document Reviewer Demo")
    print("===========================\n")

    reviewer = LegalDocumentReviewer()

    # Example documents
    documents = [
        LegalDocument(
            doc_id="legal_001",
            title="Credit Card Terms Update",
            content="""Updated terms for premium credit card services including:
            1. Annual fee structure changes
            2. New rewards program terms
            3. Modified dispute resolution process""",
            category="Terms and Conditions",
            metadata={"department": "Retail Banking", "priority": "High"}
        ),
        LegalDocument(
            doc_id="legal_002",
            title="Third-Party Vendor Agreement",
            content="""Service agreement for cloud infrastructure provider covering:
            1. Data security requirements
            2. Service level agreements
            3. Compliance obligations""",
            category="Vendor Contracts",
            metadata={"department": "IT", "priority": "Medium"}
        )
    ]

    # Process documents
    for doc in documents:
        print(f"Reviewing Document: {doc.title}")
        print(f"Category: {doc.category}")
        print(f"Department: {doc.metadata['department']}\n")

        result = await reviewer.review_document(doc)
        
        print("Review Summary:")
        print(f"Risk Level: {result.risk_level}")
        print("\nKey Findings:")
        for finding in result.key_findings:
            print(f"- {finding}")
        print("\nRecommendations:")
        for recommendation in result.recommendations:
            print(f"- {recommendation}")
        print("\nAudit Trail:")
        for entry in result.audit_trail:
            print(f"- {entry['timestamp']}: {entry['action']}")
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_legal_reviewer())
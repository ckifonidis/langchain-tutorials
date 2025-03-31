#!/usr/bin/env python3
"""
LangChain Multimodal Document Processor (LangChain v3)

This example demonstrates a multimodal document processing system using three key concepts:
1. multimodality: Handle text and image data
2. runnable_interface: Composable processing pipeline
3. key_methods: Flexible processing patterns (invoke, stream, batch)

It provides robust document processing capabilities for banking/fintech applications.
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage,
    AIMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableConfig
)

# Load environment variables
load_dotenv(".env")

class DocumentType(str, Enum):
    """Document types supported."""
    CHECK = "check"
    RECEIPT = "receipt"
    INVOICE = "invoice"
    ID_CARD = "id_card"
    OTHER = "other"

class DocumentContent(BaseModel):
    """Document content with multimodal data."""
    text: Optional[str] = Field(description="Text content", default=None)
    image_data: Optional[str] = Field(description="Base64 encoded image", default=None)
    doc_type: DocumentType = Field(description="Document type")
    metadata: Dict = Field(description="Document metadata")
    
class ProcessingResult(BaseModel):
    """Processing result with extracted information."""
    doc_id: str = Field(description="Document identifier")
    extracted_data: Dict = Field(description="Extracted information")
    confidence: float = Field(description="Confidence score")
    processing_time: float = Field(description="Processing time in seconds")

class MultimodalProcessor:
    """Multimodal document processor using runnable interface."""
    
    def __init__(self):
        """Initialize processor."""
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Create system templates
        self.check_template = """Analyze this check image and extract:
        1. Check amount
        2. Date
        3. Payee
        4. Bank name
        Provide the information in a structured format."""
        
        self.receipt_template = """Analyze this receipt and extract:
        1. Total amount
        2. Date
        3. Merchant
        4. Items purchased
        Provide the information in a structured format."""
        
        self.invoice_template = """Analyze this invoice and extract:
        1. Invoice number
        2. Total amount
        3. Due date
        4. Line items
        Provide the information in a structured format."""
        
        self.id_card_template = """Analyze this ID card and extract:
        1. Name
        2. ID number
        3. Date of birth 
        4. Expiry date
        Do not include sensitive personal information."""
        
        # Create processing chain
        self.chain = (
            RunnablePassthrough.assign(
                timestamp=lambda _: datetime.now().isoformat()
            )
            | self._create_prompt
            | self.llm
            | self._parse_response
        )
    
    def _create_prompt(self, inputs: Dict) -> List[Any]:
        """Create appropriate prompt based on document type."""
        doc = inputs["document"]
        template = getattr(
            self,
            f"{doc.doc_type.value}_template",
            "Analyze this document and extract key information."
        )

        messages = [
            SystemMessage(content=template)
        ]
        
        content = []
                
        # Add text content if available
        if doc.text:
            content.append({
                "type": "text",
                "text": f"Text content:\n{doc.text}"
            })
        
        # Add image content if available
        if doc.image_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{doc.image_data}"
                }
            })
        
        # Add content as human message
        messages.append(HumanMessage(content=content))
        
        
        return messages
    
    def _parse_response(self, response: AIMessage) -> Dict:
        """Parse LLM response into structured data."""
        try:
            # Simple parsing - could be enhanced with output parsers
            content = response.content.strip()
            lines = response.content.strip().split("\n")
            data = {}
            
            # Try parsing as JSON first
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback to key-value parsing
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        data[key.strip()] = value.strip().strip(',"')
            
            return data
        except Exception as e:
            return {"error": f"Error parsing response: {str(e)}"}
    
    async def process(
        self,
        document: DocumentContent,
        run_id: Optional[str] = None
    ) -> ProcessingResult:
        """Process document using invoke pattern."""
        try:
            start_time = datetime.now()
            
            # Process with chain
            result = await self.chain.ainvoke(
                {
                    "document": document,
                    "run_id": run_id or datetime.now().isoformat()
                }
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                doc_id=run_id or datetime.now().isoformat(),
                extracted_data=result,
                confidence=0.85,  # Could be calculated based on LLM confidence
                processing_time=duration
            )
            
        except Exception as e:
            return ProcessingResult(
                doc_id=run_id or datetime.now().isoformat(),
                extracted_data={"error": str(e)},
                confidence=0.0,
                processing_time=0.0
            )
    
    async def stream_process(
        self,
        document: DocumentContent,
        run_id: Optional[str] = None
    ) -> AsyncIterator[Dict]:
        """Process document using stream pattern."""
        try:
            start_time = datetime.now()
            
            # Stream with chain
            async for chunk in self.chain.astream(
                {
                    "document": document,
                    "run_id": run_id or datetime.now().isoformat()
                }
            ):
                yield {
                    "chunk": chunk.content if hasattr(chunk, "content") else str(chunk),
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            yield {"error": str(e)}
    
    async def batch_process(
        self,
        documents: List[DocumentContent]
    ) -> List[ProcessingResult]:
        """Process documents using batch pattern."""
        results = []
        
        # Process each document
        for doc in documents:
            try:
                result = await self.process(doc)
                results.append(result)
            except Exception as e:
                results.append(
                    ProcessingResult(
                        doc_id=datetime.now().isoformat(),
                        extracted_data={"error": str(e)},
                        confidence=0.0,
                        processing_time=0.0
                    )
                )
        
        return results

async def demonstrate_processor():
    """Demonstrate the multimodal processor."""
    print("\nMultimodal Document Processor Demo")
    print("=================================\n")
    
    # Create processor
    processor = MultimodalProcessor()
    
    # Sample check image (base64 encoded)
    sample_check = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    # Test documents
    documents = [
        DocumentContent(
            text="Pay to the order of John Smith\nAmount: $1,500.00\nDate: 2025-03-27",
            image_data=sample_check,
            doc_type=DocumentType.CHECK,
            metadata={"source": "mobile_deposit"}
        ),
        DocumentContent(
            text="INVOICE\nInvoice #: 12345\nAmount Due: $2,750.00\nDue Date: 2025-04-15",
            doc_type=DocumentType.INVOICE,
            metadata={"source": "email"}
        )
    ]
    
    try:
        print("1. Processing Single Document")
        print("-" * 40)
        
        result = await processor.process(documents[0])
        print(f"Document ID: {result.doc_id}")
        print(f"Extracted Data: {result.extracted_data}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print("-" * 40)
        
        print("\n2. Streaming Document")
        print("-" * 40)
        
        async for chunk in processor.stream_process(documents[1]):
            print(f"Chunk: {chunk}")
        print("-" * 40)
        
        print("\n3. Batch Processing")
        print("-" * 40)
        
        results = await processor.batch_process(documents)
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"ID: {result.doc_id}")
            print(f"Data: {result.extracted_data}")
        print("-" * 40)
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_processor())
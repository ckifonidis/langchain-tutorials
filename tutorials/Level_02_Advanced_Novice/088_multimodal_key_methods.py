#!/usr/bin/env python3
"""
LangChain Key Methods with Azure OpenAI (LangChain v3)

This example demonstrates efficient method handling using three key concepts:
1. key_methods: Invoke, stream, and batch patterns
2. Azure OpenAI: Model interaction and embeddings
3. structured output: Type-safe response handling

It provides robust data processing capabilities for banking/fintech applications.
"""

import os
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv(".env")

class ProcessingMode(str, Enum):
    """Processing modes."""
    INVOKE = "invoke"
    STREAM = "stream"
    BATCH = "batch"

class ProcessingRequest(BaseModel):
    """Processing request."""
    content: str = Field(description="Content to process")
    mode: ProcessingMode = Field(description="Processing mode")
    batch_size: Optional[int] = Field(description="Batch size for batch mode", default=None)

class ProcessingResult(BaseModel):
    """Processing result."""
    content: str = Field(description="Processed content")
    mode: ProcessingMode = Field(description="Processing mode used")
    metadata: Dict = Field(description="Processing metadata")
    embedding: Optional[List[float]] = Field(description="Content embedding", default=None)

SYSTEM_TEMPLATE = """You are a specialized financial analyst AI assistant. Your role is to:
1. Analyze financial transaction data
2. Identify patterns and anomalies
3. Provide actionable insights
4. Use precise financial terminology
5. Consider risk and compliance

Format your responses in a clear, structured manner with sections and bullet points when appropriate."""

SAMPLE_TRANSACTION_DATA = """
Transaction Data Sample:
- Date: 2025-03-27
- Amount: $15,750
- Type: Wire Transfer
- Source Account: Business Checking #1234
- Destination: Investment Account #5678
- Category: Investment
- Location: New York, NY
- Status: Completed
"""

class DataProcessor:
    """Data processing with key methods."""
    
    def __init__(self):
        """Initialize processor with Azure OpenAI."""
        # Initialize Azure Chat OpenAI for regular requests
        self.chat_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
            streaming=False
        )
        
        # Initialize Azure Chat OpenAI for streaming
        self.streaming_chat_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            model=os.getenv("AZURE_MODEL_NAME", "text-embedding-3-small")
        )
    
    async def process_invoke(self, request: ProcessingRequest) -> ProcessingResult:
        """Process using invoke pattern."""
        try:
            # Create messages
            messages = [
                SystemMessage(content=SYSTEM_TEMPLATE),
                HumanMessage(content=f"{request.content}\n{SAMPLE_TRANSACTION_DATA}")
            ]
            
            # Get response
            response = self.chat_model.invoke(messages)
            
            # Get embedding
            embedding = self.embeddings.embed_query(request.content)
            
            return ProcessingResult(
                content=response.content,
                mode=ProcessingMode.INVOKE,
                metadata={"timestamp": datetime.now().isoformat()},
                embedding=embedding
            )
        
        except Exception as e:
            raise ValueError(f"Error in invoke processing: {str(e)}")
    
    async def process_stream(self, request: ProcessingRequest) -> None:
        """Process using stream pattern."""
        try:
            # Create messages
            messages = [
                SystemMessage(content=SYSTEM_TEMPLATE),
                HumanMessage(content=f"{request.content}\n{SAMPLE_TRANSACTION_DATA}")
            ]
            
            # Get streaming response using streaming model
            await self.streaming_chat_model.ainvoke(messages)
        
        except Exception as e:
            raise ValueError(f"Error in stream processing: {str(e)}")
    
    async def process_batch(self, request: ProcessingRequest) -> List[ProcessingResult]:
        """Process using batch pattern."""
        try:
            analyses = [
                "Analyze this transaction for potential fraud indicators.",
                "Provide a risk assessment of this transaction.",
                "Suggest compliance checks for this transaction."
            ]
            
            # Create batch of messages
            messages_batch = [
                [
                    SystemMessage(content=SYSTEM_TEMPLATE),
                    HumanMessage(content=f"{analysis}\n{SAMPLE_TRANSACTION_DATA}")
                ]
                for analysis in analyses[:request.batch_size or len(analyses)]
            ]
            
            # Get batch response
            responses = await self.chat_model.abatch(messages_batch)
            
            # Get embeddings for all variations
            embeddings = self.embeddings.embed_documents([
                analysis for analysis in analyses[:request.batch_size or len(analyses)]
            ])
            
            return [
                ProcessingResult(
                    content=response.content,
                    mode=ProcessingMode.BATCH,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "batch_index": i,
                        "analysis_type": analyses[i]
                    },
                    embedding=embedding
                )
                for i, (response, embedding) in enumerate(zip(responses, embeddings))
            ]
        
        except Exception as e:
            raise ValueError(f"Error in batch processing: {str(e)}")

async def demonstrate_processor():
    """Demonstrate the processor."""
    print("\nKey Methods Processor Demo")
    print("=========================\n")
    
    # Create processor
    processor = DataProcessor()
    
    # Test data
    base_content = "Analyze this transaction for unusual patterns or potential risks."
    
    try:
        # 1. Invoke Pattern
        print("1. Invoke Pattern:")
        print("-" * 40)
        
        invoke_request = ProcessingRequest(
            content=base_content,
            mode=ProcessingMode.INVOKE
        )
        
        invoke_result = await processor.process_invoke(invoke_request)
        print(f"Response: {invoke_result.content}")
        print(f"Embedding Size: {len(invoke_result.embedding)}")
        print("-" * 40)
        
        # 2. Stream Pattern
        print("\n2. Stream Pattern:")
        print("-" * 40)
        
        stream_request = ProcessingRequest(
            content=base_content,
            mode=ProcessingMode.STREAM
        )
        
        print("Streaming response:")
        await processor.process_stream(stream_request)
        print("\n" + "-" * 40)
        
        # 3. Batch Pattern
        print("\n3. Batch Pattern:")
        print("-" * 40)
        
        batch_request = ProcessingRequest(
            content=base_content,
            mode=ProcessingMode.BATCH,
            batch_size=3
        )
        
        batch_results = await processor.process_batch(batch_request)
        for i, result in enumerate(batch_results, 1):
            print(f"\nBatch {i}:")
            print(f"Analysis Type: {result.metadata['analysis_type']}")
            print(f"Response: {result.content}")
            print(f"Embedding Size: {len(result.embedding)}")
        
        print("-" * 40)
    
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demonstrate_processor())
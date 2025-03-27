#!/usr/bin/env python3
"""
LangChain Document Summarizer (LangChain v3)

This example demonstrates how to build a document summarization system using text splitters
and key method extraction. It processes long documents into meaningful summaries while
identifying key concepts and methods.

Key concepts demonstrated:
1. Text Splitters: Processing long documents into manageable chunks
2. Key Methods: Extracting and organizing important concepts
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

class KeyConcept(BaseModel):
    """Key concept or method identified in the text."""
    name: str = Field(description="Name of the concept or method")
    category: str = Field(description="Category (concept/method/term)")
    description: str = Field(description="Brief description")
    importance: int = Field(description="Importance score (1-5)")
    related_concepts: List[str] = Field(description="Related concepts or methods")

class DocumentSummary(BaseModel):
    """Summary of a document section."""
    title: str = Field(description="Section title or identifier")
    word_count: int = Field(description="Number of words in original text")
    main_points: List[str] = Field(description="Main points from the text")
    key_concepts: List[KeyConcept] = Field(description="Key concepts identified")
    summary: str = Field(description="Concise summary of the section")

class DocumentSummarizer:
    """Document summarization system using text splitting and key method extraction."""
    
    def __init__(self):
        """Initialize the document summarizer."""
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize chat model
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0
        )
        
        # Create summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst and summarizer.
Analyze the given text to identify:
1. Main points and themes
2. Key concepts and methods
3. Important relationships and connections

Provide a structured summary following this format:
{format_instructions}"""),
            ("human", "Text to analyze:\n{text}")
        ])
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=DocumentSummary)
    
    def process_text(
        self,
        text: str,
        title: str = "Document Section"
    ) -> DocumentSummary:
        """Process a section of text."""
        try:
            # Count words
            word_count = len(text.split())
            
            # Create summary chain
            chain = (
                self.summary_prompt 
                | self.llm 
                | self.parser
            )
            
            # Generate summary
            summary = chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Update title and word count
            summary.title = title
            summary.word_count = word_count
            
            return summary
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            raise
    
    def summarize_document(
        self,
        document: str,
        title: str = "Document"
    ) -> List[DocumentSummary]:
        """Summarize a complete document."""
        try:
            # Split document into chunks
            chunks = self.text_splitter.split_text(document)
            
            # Process each chunk
            summaries = []
            for i, chunk in enumerate(chunks, 1):
                chunk_title = f"{title} - Section {i}"
                summary = self.process_text(chunk, chunk_title)
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            print(f"Error summarizing document: {str(e)}")
            raise

def demonstrate_summarizer():
    """Demonstrate the document summarizer."""
    print("\nDemonstrating Document Summarizer")
    print("=" * 50)
    
    # Test documents
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": """Machine Learning is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms improve through experience.

Key concepts in machine learning include supervised learning, where the algorithm learns from labeled training data, and unsupervised learning, where it finds patterns in unlabeled data. The choice between these approaches depends on the available data and the problem being solved.

Common applications of machine learning include image recognition, natural language processing, and recommendation systems. These systems use various algorithms like decision trees, neural networks, and support vector machines.

The machine learning process typically involves data collection, preprocessing, model selection, training, and evaluation. Each step requires careful consideration and domain expertise to achieve optimal results."""
        },
        {
            "title": "Data Preprocessing Techniques",
            "content": """Data preprocessing is a crucial step in any data analysis or machine learning pipeline. Raw data often contains inconsistencies, missing values, and noise that can affect model performance.

Common preprocessing steps include data cleaning, where missing or incorrect values are handled; data transformation, where data is converted into a suitable format; and feature scaling, where numerical values are normalized or standardized.

Feature engineering is another important aspect, involving the creation of new features from existing data to improve model performance. This might include combining features, creating interaction terms, or encoding categorical variables.

The choice of preprocessing techniques depends on factors like data type, model requirements, and the specific problem being solved. It's important to document all preprocessing steps to ensure reproducibility."""
        }
    ]
    
    # Create summarizer
    summarizer = DocumentSummarizer()
    
    # Process each document
    for doc in documents:
        print(f"\nAnalyzing: {doc['title']}")
        print("-" * 50)
        
        try:
            summaries = summarizer.summarize_document(
                doc["content"],
                doc["title"]
            )
            
            # Display summaries
            for summary in summaries:
                print(f"\nSection: {summary.title}")
                print(f"Word Count: {summary.word_count}")
                
                print("\nMain Points:")
                for point in summary.main_points:
                    print(f"- {point}")
                
                print("\nKey Concepts:")
                for concept in summary.key_concepts:
                    print(f"\n{concept.name} ({concept.category}, Importance: {concept.importance})")
                    print(f"Description: {concept.description}")
                    print(f"Related: {', '.join(concept.related_concepts)}")
                
                print("\nSummary:")
                print(summary.summary)
                print("-" * 50)
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            
        print("=" * 50)

if __name__ == "__main__":
    demonstrate_summarizer()
"""
LangChain Document Analysis System Example

This example demonstrates how to combine text splitting capabilities with structured output
parsing to create a sophisticated document analysis system that can process large documents
and provide structured analysis results.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

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

class TextSegment(BaseModel):
    """Schema for analyzed text segments."""
    segment_id: str = Field(description="Unique segment identifier")
    content: str = Field(description="Segment content")
    word_count: int = Field(description="Number of words in segment")
    key_topics: List[str] = Field(description="Main topics identified")
    sentiment: str = Field(description="Overall sentiment (positive/neutral/negative)")
    importance_score: float = Field(description="Segment importance (0-1)")

class DocumentAnalysis(BaseModel):
    """Schema for complete document analysis."""
    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    summary: str = Field(description="Overall summary")
    segments: List[TextSegment] = Field(description="Analyzed segments")
    total_words: int = Field(description="Total word count")
    main_themes: List[str] = Field(description="Main document themes")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with optimized settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

def create_segment_analyzer(llm: AzureChatOpenAI) -> tuple[PromptTemplate, PydanticOutputParser]:
    """Create the segment analyzer with output parser."""
    parser = PydanticOutputParser(pydantic_object=TextSegment)
    
    prompt = PromptTemplate(
        template="""Analyze the following text segment and provide a structured analysis.
        
        Text Segment:
        {text}
        
        Respond with a structured analysis following this format:
        {format_instructions}
        
        Analysis should include:
        1. A unique segment identifier
        2. The actual content
        3. Word count
        4. Key topics (up to 5)
        5. Overall sentiment
        6. Importance score (0-1)""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

def analyze_text_segment(
    segment: str,
    segment_id: str,
    llm: AzureChatOpenAI,
    prompt: PromptTemplate,
    parser: PydanticOutputParser
) -> TextSegment:
    """Analyze a single text segment."""
    try:
        # Format prompt with segment
        formatted_prompt = prompt.format(text=segment)
        
        # Get model response
        messages = [
            SystemMessage(content="You are an expert text analyst."),
            HumanMessage(content=formatted_prompt)
        ]
        response = llm.invoke(messages)
        
        # Parse response into TextSegment
        analysis = parser.parse(response.content)
        analysis.segment_id = segment_id
        
        return analysis
    except Exception as e:
        print(f"Error analyzing segment {segment_id}: {str(e)}")
        # Return default analysis for failed segments
        return TextSegment(
            segment_id=segment_id,
            content=segment[:100] + "...",
            word_count=len(segment.split()),
            key_topics=["Analysis Failed"],
            sentiment="neutral",
            importance_score=0.0
        )

def analyze_document(content: str, title: str) -> DocumentAnalysis:
    """Analyze a complete document by splitting and analyzing segments."""
    try:
        # Initialize components
        llm = create_chat_model()
        splitter = create_text_splitter()
        prompt, parser = create_segment_analyzer(llm)
        
        # Split document into segments
        segments = splitter.split_text(content)
        
        # Analyze each segment
        analyzed_segments = []
        total_words = 0
        all_topics = []
        
        for i, segment in enumerate(segments):
            segment_id = f"SEG{i+1:03d}"
            analysis = analyze_text_segment(segment, segment_id, llm, prompt, parser)
            
            analyzed_segments.append(analysis)
            total_words += analysis.word_count
            all_topics.extend(analysis.key_topics)
        
        # Generate document summary based on segment analyses
        summary_prompt = f"""Summarize this document based on its analyzed segments:
        Title: {title}
        Number of segments: {len(segments)}
        Total words: {total_words}
        Main topics identified: {', '.join(set(all_topics))}
        """
        
        summary_message = HumanMessage(content=summary_prompt)
        summary = llm.invoke([summary_message])
        
        # Create final analysis
        doc_id = f"DOC{hash(content) % 1000:03d}"
        main_themes = list(set(all_topics))[:5]  # Top 5 unique themes
        
        return DocumentAnalysis(
            doc_id=doc_id,
            title=title,
            summary=summary.content,
            segments=analyzed_segments,
            total_words=total_words,
            main_themes=main_themes
        )
        
    except Exception as e:
        print(f"Error analyzing document: {str(e)}")
        raise

def demonstrate_document_analysis():
    """Demonstrate the Document Analysis System capabilities."""
    try:
        print("\nInitializing Document Analysis System...\n")
        
        # Example document
        title = "Artificial Intelligence in Healthcare"
        content = """
        Artificial Intelligence (AI) is revolutionizing healthcare delivery and patient care. 
        Machine learning algorithms are improving diagnostic accuracy and speed, while natural 
        language processing is making medical records more accessible and useful.

        One key application is in medical imaging, where AI systems can analyze X-rays, MRIs, 
        and CT scans with remarkable precision. These systems can detect abnormalities that 
        might be missed by human observers and provide rapid preliminary assessments.

        Another significant impact is in personalized medicine. AI analyzes vast amounts of 
        patient data to help determine the most effective treatments for individual patients, 
        considering their genetic makeup, lifestyle, and environmental factors.

        However, challenges remain. Data privacy and security are major concerns, as healthcare 
        information is highly sensitive. There's also the need for regulatory frameworks to 
        ensure AI systems are safe and effective.

        Looking ahead, AI is expected to continue transforming healthcare through improved 
        prediction of health risks, automated administrative tasks, and enhanced drug 
        discovery processes. The future of healthcare will likely see even greater 
        integration of AI technologies.
        """
        
        # Analyze document
        analysis = analyze_document(content, title)
        
        # Display results
        print(f"Document Analysis Results:")
        print(f"ID: {analysis.doc_id}")
        print(f"Title: {analysis.title}")
        print("\nSummary:")
        print(analysis.summary)
        print("\nMain Themes:")
        for theme in analysis.main_themes:
            print(f"- {theme}")
        
        print("\nSegment Analysis:")
        for segment in analysis.segments:
            print(f"\nSegment {segment.segment_id}:")
            print(f"Word Count: {segment.word_count}")
            print(f"Sentiment: {segment.sentiment}")
            print(f"Importance: {segment.importance_score:.2f}")
            print("Key Topics:", ", ".join(segment.key_topics))
            print("-" * 50)
        
        print(f"\nTotal Words: {analysis.total_words}")
        print(f"Analysis Timestamp: {analysis.timestamp}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Document Analysis System...")
    demonstrate_document_analysis()

if __name__ == "__main__":
    main()
"""
LangChain Markdown Analysis System Example

This example demonstrates how to create a system that can read and analyze multiple markdown
files, extracting titles and content, and providing structured analysis results. It builds
upon the document analysis capabilities with specialized markdown handling.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import glob
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

class MarkdownSection(BaseModel):
    """Schema for analyzed markdown sections."""
    section_id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    word_count: int = Field(description="Number of words in section")
    key_topics: List[str] = Field(description="Main topics identified")
    sentiment: str = Field(description="Overall sentiment (positive/neutral/negative)")
    importance_score: float = Field(description="Section importance (0-1)")

class MarkdownAnalysis(BaseModel):
    """Schema for complete markdown document analysis."""
    file_path: str = Field(description="Path to the markdown file")
    file_name: str = Field(description="Name of the markdown file")
    title: str = Field(description="Document title")
    sections: List[MarkdownSection] = Field(description="Analyzed sections")
    total_words: int = Field(description="Total word count")
    main_themes: List[str] = Field(description="Main document themes")
    summary: str = Field(description="Overall summary")
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchAnalysis(BaseModel):
    """Schema for batch analysis of multiple markdown files."""
    batch_id: str = Field(description="Unique batch identifier")
    files_analyzed: int = Field(description="Number of files analyzed")
    analyses: List[MarkdownAnalysis] = Field(description="Individual file analyses")
    common_themes: List[str] = Field(description="Themes common across documents")
    batch_summary: str = Field(description="Overall batch analysis summary")
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

def extract_markdown_content(file_path: str) -> tuple[str, str]:
    """Extract title and content from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Extract title (first # heading)
    title = ""
    content = ""
    for line in lines:
        if not title and line.startswith("# "):
            title = line.strip("# ").strip()
        else:
            content += line
    
    return title, content.strip()

def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with optimized settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    )

def create_section_analyzer(llm: AzureChatOpenAI) -> tuple[PromptTemplate, PydanticOutputParser]:
    """Create the section analyzer with output parser."""
    parser = PydanticOutputParser(pydantic_object=MarkdownSection)
    
    prompt = PromptTemplate(
        template="""Analyze the following markdown section and provide a structured analysis.
        
        Title: {title}
        Content: {text}
        
        Respond with a structured analysis following this format:
        {format_instructions}
        
        Analysis should include:
        1. A unique section identifier
        2. The section title
        3. The actual content
        4. Word count
        5. Key topics (up to 5)
        6. Overall sentiment
        7. Importance score (0-1)""",
        input_variables=["title", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

def analyze_markdown_file(file_path: str, llm: AzureChatOpenAI) -> MarkdownAnalysis:
    """Analyze a single markdown file."""
    try:
        # Extract content
        title, content = extract_markdown_content(file_path)
        
        # Split content
        splitter = create_text_splitter()
        sections = splitter.split_text(content)
        
        # Create analyzer
        prompt, parser = create_section_analyzer(llm)
        
        # Analyze sections
        analyzed_sections = []
        all_topics = []
        total_words = 0
        
        for i, section in enumerate(sections):
            try:
                # Format prompt
                formatted_prompt = prompt.format(
                    title=title,
                    text=section
                )
                
                # Get model response
                messages = [
                    SystemMessage(content="You are an expert markdown analyst."),
                    HumanMessage(content=formatted_prompt)
                ]
                response = llm.invoke(messages)
                
                # Parse response
                analysis = parser.parse(response.content)
                analysis.section_id = f"SEC{i+1:03d}"
                
                analyzed_sections.append(analysis)
                total_words += analysis.word_count
                all_topics.extend(analysis.key_topics)
                
            except Exception as e:
                print(f"Error analyzing section {i+1}: {str(e)}")
                continue
        
        # Generate summary
        summary_prompt = f"""Summarize this markdown document:
        Title: {title}
        Number of sections: {len(sections)}
        Total words: {total_words}
        Main topics: {', '.join(set(all_topics))}
        """
        
        summary_message = HumanMessage(content=summary_prompt)
        summary = llm.invoke([summary_message])
        
        # Create analysis result
        return MarkdownAnalysis(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            title=title,
            sections=analyzed_sections,
            total_words=total_words,
            main_themes=list(set(all_topics))[:5],
            summary=summary.content
        )
        
    except Exception as e:
        print(f"Error analyzing file {file_path}: {str(e)}")
        raise

def analyze_markdown_batch(directory: str) -> BatchAnalysis:
    """Analyze all markdown files in a directory."""
    try:
        # Initialize components
        llm = create_chat_model()
        
        # Find all markdown files
        md_files = glob.glob(os.path.join(directory, "*.md"))
        if not md_files:
            raise ValueError(f"No markdown files found in {directory}")
        
        # Analyze each file
        analyses = []
        all_themes = []
        
        for file_path in md_files:
            try:
                analysis = analyze_markdown_file(file_path, llm)
                analyses.append(analysis)
                all_themes.extend(analysis.main_themes)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Find common themes
        theme_count = {}
        for theme in all_themes:
            theme_count[theme] = theme_count.get(theme, 0) + 1
        
        common_themes = [
            theme for theme, count in theme_count.items()
            if count > len(analyses) / 2  # Theme appears in more than half the files
        ]
        
        # Generate batch summary
        batch_summary_prompt = f"""Provide a comparative analysis of these markdown documents:
        Number of files analyzed: {len(analyses)}
        Common themes: {', '.join(common_themes)}
        Individual titles: {', '.join(a.title for a in analyses)}
        """
        
        summary_message = HumanMessage(content=batch_summary_prompt)
        batch_summary = llm.invoke([summary_message])
        
        # Create batch analysis
        return BatchAnalysis(
            batch_id=f"BATCH{datetime.now().strftime('%Y%m%d%H%M%S')}",
            files_analyzed=len(analyses),
            analyses=analyses,
            common_themes=common_themes,
            batch_summary=batch_summary.content
        )
        
    except Exception as e:
        print(f"Error in batch analysis: {str(e)}")
        raise

def demonstrate_markdown_analysis():
    """Demonstrate the Markdown Analysis System capabilities."""
    try:
        print("\nInitializing Markdown Analysis System...\n")
        
        # Analyze markdown files in the examples directory
        example_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples"
        )
        
        print(f"Analyzing markdown files in: {example_dir}\n")
        batch_analysis = analyze_markdown_batch(example_dir)
        
        # Display results
        print(f"Batch Analysis Results:")
        print(f"Batch ID: {batch_analysis.batch_id}")
        print(f"Files Analyzed: {batch_analysis.files_analyzed}")
        
        print("\nCommon Themes:")
        for theme in batch_analysis.common_themes:
            print(f"- {theme}")
        
        print("\nIndividual File Analyses:")
        for analysis in batch_analysis.analyses:
            print(f"\nFile: {analysis.file_name}")
            print(f"Title: {analysis.title}")
            print(f"Word Count: {analysis.total_words}")
            print("\nMain Themes:")
            for theme in analysis.main_themes:
                print(f"- {theme}")
            print("\nSummary:")
            print(analysis.summary)
            print("-" * 50)
        
        print("\nBatch Summary:")
        print(batch_analysis.batch_summary)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Markdown Analysis System...")
    demonstrate_markdown_analysis()

if __name__ == "__main__":
    main()
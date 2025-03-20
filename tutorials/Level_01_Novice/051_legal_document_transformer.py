"""
LangChain Legal Document Transformer Example

This example demonstrates how to create a multi-agent system that transforms and enhances
legal documents using specialized agents for different aspects like legal expertise,
language enhancement, structure formatting, and final review.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
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

# Define the base prompt template with required variables and format instructions
BASE_AGENT_TEMPLATE = """You are a specialized AI agent.

You have access to the following tools:
{tools}

Use the following format EXACTLY (include ONLY these sections):
Thought: Think about what to do
Action: Choose a tool from [{tool_names}]
Action Input: Input for the tool
Observation: Tool output
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: Your final response

Given task: {input}

{agent_scratchpad}"""

# Document transformation template
DOCUMENT_TEMPLATE = """# {title}

## Executive Summary
{summary}

## Article 1: Definitions
{definitions}

## Article 2: License Grant
{license_grant}

## Article 3: Terms and Conditions
{terms_conditions}

## Article 4: Rights and Obligations
{rights_obligations}

## Article 5: Term and Termination
{term_termination}

## Article 6: Warranties and Disclaimers
{warranties_disclaimers}

## Article 7: Limitations and Liabilities
{limitations_liabilities}

## Article 8: General Provisions
{general_provisions}

## Article 9: Governing Law
{governing_law}

## Article 10: Signature
{signature_block}"""

class DocumentSection(BaseModel):
    """Schema for document sections."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    analysis: Dict[str, Any] = Field(description="Section analysis")
    suggestions: List[str] = Field(description="Improvement suggestions")

class TransformedDocument(BaseModel):
    """Schema for transformed document."""
    original_path: str = Field(description="Path to original document")
    transformed_path: str = Field(description="Path to transformed document")
    sections: List[DocumentSection] = Field(description="Document sections")
    improvements_made: List[str] = Field(description="List of improvements")
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

def create_legal_expert_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """Create an agent specialized in legal expertise."""
    tools = [
        Tool(
            name="validate_legal_terms",
            func=lambda x: "Legal terms validated: compliance confirmed",
            description="Validate legal terminology and compliance"
        ),
        Tool(
            name="check_regulations",
            func=lambda x: "Regulatory check completed: requirements met",
            description="Check regulatory requirements"
        )
    ]
    
    prompt = PromptTemplate(
        template=BASE_AGENT_TEMPLATE + """

Remember to provide your final answer in a clear, structured format.""",
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_language_enhancement_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """Create an agent specialized in language enhancement."""
    tools = [
        Tool(
            name="improve_clarity",
            func=lambda x: "Language improved: enhanced readability",
            description="Improve text clarity and readability"
        ),
        Tool(
            name="standardize_terms",
            func=lambda x: "Terms standardized: consistent terminology",
            description="Standardize terminology usage"
        )
    ]
    
    prompt = PromptTemplate(
        template=BASE_AGENT_TEMPLATE + """

Remember to provide your final answer in a clear, structured format.""",
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_structure_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """Create an agent specialized in document structure."""
    tools = [
        Tool(
            name="format_section",
            func=lambda x: "Section formatted: proper structure applied",
            description="Format document section"
        ),
        Tool(
            name="validate_structure",
            func=lambda x: "Structure validated: consistent formatting",
            description="Validate document structure"
        )
    ]
    
    prompt = PromptTemplate(
        template=BASE_AGENT_TEMPLATE + """

Remember to provide your final answer in a clear, structured format.""",
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def create_review_agent(llm: AzureChatOpenAI) -> AgentExecutor:
    """Create an agent specialized in final review."""
    tools = [
        Tool(
            name="review_document",
            func=lambda x: "Document reviewed: quality standards met",
            description="Review complete document"
        ),
        Tool(
            name="validate_improvements",
            func=lambda x: "Improvements validated: enhancements confirmed",
            description="Validate implemented improvements"
        )
    ]
    
    prompt = PromptTemplate(
        template=BASE_AGENT_TEMPLATE + """

Remember to provide your final answer in a clear, structured format.""",
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

def read_markdown_file(file_path: str) -> str:
    """Read content from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_markdown_file(content: str, file_path: str) -> None:
    """Save content to a markdown file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def transform_document(input_path: str) -> TransformedDocument:
    """Transform a legal document using multiple specialized agents."""
    try:
        print(f"\nProcessing document: {input_path}")
        
        # Initialize components
        llm = create_chat_model()
        legal_expert = create_legal_expert_agent(llm)
        language_enhancer = create_language_enhancement_agent(llm)
        structure_expert = create_structure_agent(llm)
        reviewer = create_review_agent(llm)
        
        # Read original document
        original_content = read_markdown_file(input_path)
        
        # Process each section
        sections = []
        improvements = []
        
        # Extract sections (simple split by ##)
        raw_sections = original_content.split('##')
        title = raw_sections[0].strip('# \n')
        
        for i, section in enumerate(raw_sections[1:], 1):
            # Process section
            section_content = section.strip()
            section_title = section_content.split('\n')[0]
            section_text = '\n'.join(section_content.split('\n')[1:]).strip()
            
            print(f"\nProcessing section {i}: {section_title}")
            
            try:
                # Legal analysis
                legal_result = legal_expert.invoke({
                    "input": f"Analyze the legal aspects of this text:\n\n{section_text}",
                    "agent_scratchpad": ""
                })
                
                # Language enhancement
                language_result = language_enhancer.invoke({
                    "input": f"Improve the clarity and readability of this text:\n\n{section_text}",
                    "agent_scratchpad": ""
                })
                
                # Structure formatting
                structure_result = structure_expert.invoke({
                    "input": f"Format and structure this text:\n\n{section_text}",
                    "agent_scratchpad": ""
                })
                
                # Collect improvements
                section_improvements = [
                    legal_result.get('output', ''),
                    language_result.get('output', ''),
                    structure_result.get('output', '')
                ]
                
                sections.append(DocumentSection(
                    title=section_title,
                    content=section_text,
                    analysis={
                        "legal": legal_result,
                        "language": language_result,
                        "structure": structure_result
                    },
                    suggestions=section_improvements
                ))
                
                improvements.extend(section_improvements)
                
            except Exception as e:
                print(f"Error processing section {i}: {str(e)}")
                continue
        
        # Generate transformed document using template
        transformed_content = DOCUMENT_TEMPLATE.format(
            title=title,
            summary="This is an enhanced version of the original agreement.",
            definitions="[Enhanced definitions section]",
            license_grant="[Enhanced license grant section]",
            terms_conditions="[Enhanced terms and conditions]",
            rights_obligations="[Enhanced rights and obligations]",
            term_termination="[Enhanced term and termination]",
            warranties_disclaimers="[Enhanced warranties and disclaimers]",
            limitations_liabilities="[Enhanced limitations and liabilities]",
            general_provisions="[Enhanced general provisions]",
            governing_law="[Enhanced governing law section]",
            signature_block="[Enhanced signature block]"
        )
        
        # Review final document
        try:
            review_result = reviewer.invoke({
                "input": f"Review the complete document for quality and consistency:\n\n{transformed_content}",
                "agent_scratchpad": ""
            })
        except Exception as e:
            print(f"Error during final review: {str(e)}")
            review_result = {"output": "Review failed"}
        
        # Save transformed document
        output_path = input_path.replace('.md', '_transformed.md')
        save_markdown_file(transformed_content, output_path)
        
        return TransformedDocument(
            original_path=input_path,
            transformed_path=output_path,
            sections=sections,
            improvements_made=improvements
        )
        
    except Exception as e:
        print(f"Error transforming document: {str(e)}")
        raise

def demonstrate_document_transformation():
    """Demonstrate the Legal Document Transformer capabilities."""
    try:
        print("\nInitializing Legal Document Transformer...\n")
        
        # Process example document
        input_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "license_agreement.md"
        )
        
        # Transform document
        result = transform_document(input_path)
        
        # Display results
        print("\nTransformation Results:")
        print(f"Original Document: {result.original_path}")
        print(f"Transformed Document: {result.transformed_path}")
        
        print("\nImprovements Made:")
        for i, improvement in enumerate(result.improvements_made, 1):
            print(f"{i}. {improvement}")
        
        print("\nSection Analysis:")
        for section in result.sections:
            print(f"\nSection: {section.title}")
            print("Suggestions:")
            for suggestion in section.suggestions:
                print(f"- {suggestion}")
            print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Legal Document Transformer...")
    demonstrate_document_transformation()

if __name__ == "__main__":
    main()
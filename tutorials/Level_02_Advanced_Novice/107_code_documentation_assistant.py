#!/usr/bin/env python3
"""
LangChain Code Documentation Assistant (107) (LangChain v3)

This example demonstrates a code documentation assistant using three key concepts:
1. Document Loaders: Handle code repositories and docs
2. Retrievers: Find relevant code examples
3. Runnable Interface: Create composable pipelines

It provides comprehensive code documentation support for development teams in banking.
"""

import os
import textwrap
from typing import List, Dict, Optional, Union
from pathlib import Path
import ast
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CodeBlock(BaseModel):
    """Schema for code blocks."""
    code: str = Field(description="Source code")
    language: str = Field(description="Programming language")
    path: str = Field(description="File path")
    dependencies: List[str] = Field(description="Required dependencies")

class DocumentationSection(BaseModel):
    """Schema for documentation sections."""
    title: str = Field(description="Section title")
    content: str = Field(description="Documentation content")
    examples: List[CodeBlock] = Field(description="Code examples")
    references: List[str] = Field(description="Related documentation")

class CodeAnalysis(BaseModel):
    """Schema for code analysis results."""
    class_name: str = Field(description="Class name")
    methods: List[str] = Field(description="Method names")
    dependencies: List[str] = Field(description="Class dependencies")
    complexity: Dict[str, int] = Field(description="Complexity metrics")

class CodeDocumentationAssistant:
    def __init__(self, repo_path: str):
        if not os.path.exists(repo_path):
            # Create a temporary directory for the sample
            repo_path = "_temp_code"
            os.makedirs(repo_path, exist_ok=True)
            with open(os.path.join(repo_path, "sample.py"), "w") as f:
                f.write("# Sample code repository")

        self.repo_path = repo_path
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\n\n"]
        )
        
        # Initialize document retriever
        self.setup_retriever()
        
        # Setup runnable pipelines
        self.setup_pipelines()

    def setup_retriever(self):
        """Initialize the document retriever."""
        try:
            # Load and split documents
            docs = []
            for file_path in Path(self.repo_path).rglob("*.py"):
                try:
                    loader = TextLoader(str(file_path))
                    doc = loader.load()[0]
                    doc.metadata["source"] = str(file_path)
                    docs.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
            
            # Create a default document if no files found
            if not docs:
                docs = [Document(
                    page_content="# Sample Python code\nclass Sample:\n    pass",
                    metadata={"source": "default.py"}
                )]
            
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            
            # Initialize retriever
            self.retriever = BM25Retriever.from_documents(splits)
            
        except Exception as e:
            print(f"Error setting up retriever: {str(e)}")
            # Initialize with empty documents as fallback
            self.retriever = BM25Retriever.from_documents([
                Document(
                    page_content="# Empty repository",
                    metadata={"source": "empty.py"}
                )
            ])

    def setup_pipelines(self):
        """Setup runnable pipelines for code analysis."""
        # Pipeline for analyzing code structure
        self.code_analyzer = (
            RunnablePassthrough()
            .assign(
                ast_tree=lambda x: ast.parse(textwrap.dedent(x["code"]))
            )
            .assign(
                analysis=lambda x: self._analyze_code(x["ast_tree"])
            )
        )

    def _analyze_code(self, ast_tree) -> CodeAnalysis:
        """Analyze Python code structure."""
        try:
            classes = []
            methods = []
            dependencies = []
            complexity = {"classes": 0, "methods": 0, "lines": 0}
            
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    complexity["classes"] += 1
                elif isinstance(node, ast.FunctionDef):
                    methods.append(node.name)
                    complexity["methods"] += 1
                elif isinstance(node, ast.Import):
                    dependencies.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
            
            complexity["lines"] = len(ast.unparse(ast_tree).splitlines())
            
            return CodeAnalysis(
                class_name=classes[0] if classes else "NoClass",
                methods=methods,
                dependencies=list(set(dependencies)),
                complexity=complexity
            )
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            return CodeAnalysis(
                class_name="ErrorAnalyzing",
                methods=[],
                dependencies=[],
                complexity={"classes": 0, "methods": 0, "lines": 0}
            )

    async def _generate_documentation(self, code: str, analysis: CodeAnalysis, 
                                   examples: List[Document]) -> DocumentationSection:
        """Generate documentation from code."""
        try:
            # Prepare examples context
            examples_context = "\n".join(f"Example {i+1}:\n{ex.page_content}"
                                       for i, ex in enumerate(examples[:2]))  # Limit to 2 examples
            
            # Generate documentation with context
            prompt = f"""
            Analyze this banking software code and create documentation:

            Code:
            {textwrap.dedent(code)}

            Analysis:
            - Classes: {analysis.class_name}
            - Methods: {', '.join(analysis.methods)}
            - Dependencies: {', '.join(analysis.dependencies)}
            
            Similar Examples:
            {examples_context}
            
            Create documentation focusing on:
            1. Purpose and functionality
            2. Usage examples
            3. Dependencies
            4. Security considerations
            
            Format the response with clear sections and bullet points.
            """
            
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert banking software documentation writer."),
                HumanMessage(content=textwrap.dedent(prompt))
            ])
            
            # Extract examples from the code
            code_blocks = [
                CodeBlock(
                    code=code,
                    language="python",
                    path=examples[0].metadata["source"] if examples else "sample.py",
                    dependencies=analysis.dependencies
                )
            ]
            
            return DocumentationSection(
                title=f"Documentation: {analysis.class_name}",
                content=response.content,
                examples=code_blocks,
                references=[ex.metadata["source"] for ex in examples]
            )
        except Exception as e:
            print(f"Error generating documentation: {str(e)}")
            return DocumentationSection(
                title="Error",
                content=f"Failed to generate documentation: {str(e)}",
                examples=[],
                references=[]
            )

    async def get_documentation(self, code: str, analysis: CodeAnalysis) -> DocumentationSection:
        """Get documentation asynchronously."""
        # Get similar examples
        similar_examples = self.retriever.invoke(code)
        
        # Generate documentation
        return await self._generate_documentation(code, analysis, similar_examples)

    async def document_code(self, code: str) -> DocumentationSection:
        """Generate documentation for provided code."""
        try:
            # Run analysis pipeline
            analysis_result = self.code_analyzer.invoke({"code": code})
            
            # Generate documentation asynchronously
            documentation = await self.get_documentation(
                code=code,
                analysis=analysis_result["analysis"]
            )
            
            return documentation
            
        except Exception as e:
            print(f"Error documenting code: {str(e)}")
            return DocumentationSection(
                title="Error",
                content=f"Failed to process code: {str(e)}",
                examples=[],
                references=[]
            )

async def demonstrate_documentation_assistant():
    print("\nCode Documentation Assistant Demo")
    print("=================================\n")

    # Example banking code with proper indentation
    sample_code = '''
    class TransactionProcessor:
        def __init__(self, account_service):
            self.account_service = account_service
            self.logger = logging.getLogger(__name__)

        async def process_transaction(self, transaction: Transaction) -> bool:
            """Process a banking transaction with security checks."""
            try:
                # Validate transaction
                if not await self.validate_transaction(transaction):
                    return False
                
                # Update account balance
                await self.account_service.update_balance(
                    transaction.account_id,
                    transaction.amount
                )
                
                self.logger.info(f"Transaction processed: {transaction.id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Transaction failed: {str(e)}")
                return False
    '''

    # Initialize assistant with repository path
    assistant = CodeDocumentationAssistant("./banking_repo")

    print("Documenting sample code...")
    print("\nSource Code:")
    print(sample_code)
    print("\nGenerating documentation...\n")

    # Handle indentation and generate documentation
    documentation = await assistant.document_code(textwrap.dedent(sample_code))
    if documentation:
        print(documentation.title)
        print()
        print("Content:")
        print(documentation.content)
        
        print("\nCode Examples:")
        for example in documentation.examples:
            print(f"\nLanguage: {example.language}")
            print(f"Dependencies: {', '.join(example.dependencies)}")
        
        print("\nReferences:")
        for ref in documentation.references:
            print(f"- {ref}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_documentation_assistant())
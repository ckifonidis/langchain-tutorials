#!/usr/bin/env python3
"""
LangChain Code Review Assistant (LangChain v3)

This example demonstrates building a code review assistant using tracing and
retrievers. It analyzes code, finds similar patterns, and provides structured
feedback with detailed traces of the analysis process.

Key concepts demonstrated:
1. Tracing: Tracking and logging analysis steps
2. Retrievers: Finding similar code patterns and issues
"""

import os
import json
import glob
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")

print("\nChecking Azure OpenAI configuration...")
if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT]):
    raise ValueError("Missing required Azure OpenAI configuration in .env file")

print(f"Using embeddings deployment: {AZURE_EMB_DEPLOYMENT}")

class CodeIssue(BaseModel):
    """Identified code issue."""
    severity: str = Field(description="Issue severity (high/medium/low)")
    category: str = Field(description="Issue category (style/performance/security/etc)")
    description: str = Field(description="Issue description")
    line_numbers: List[int] = Field(description="Affected line numbers")
    suggestion: str = Field(description="Suggested fix")

class CodeReview(BaseModel):
    """Complete code review results."""
    file_path: str = Field(description="Reviewed file path")
    issues: List[CodeIssue] = Field(description="Identified issues")
    patterns: List[Dict[str, str]] = Field(description="Similar patterns found")
    summary: str = Field(description="Review summary")
    metrics: Dict[str, Any] = Field(description="Code metrics")

    @property
    def has_critical_issues(self) -> bool:
        """Check if review has any critical/high severity issues."""
        return any(issue.severity.lower() == "high" for issue in self.issues)

    @property
    def issues_by_severity(self) -> Dict[str, int]:
        """Count issues by severity."""
        return {
            severity: len([i for i in self.issues if i.severity.lower() == severity])
            for severity in ["high", "medium", "low"]
        }

class CodeReviewAssistant:
    """Code review system with tracing and pattern matching."""
    
    def __init__(self, max_reviews: int = 50):
        """Initialize the code review assistant."""
        # Setup reviews directory relative to script
        self.reviews_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'code_reviews'
        )
        self.max_reviews = max_reviews
        print(f"\nUsing reviews directory: {self.reviews_dir}")
        
        # Setup LLM with tracing
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0,
            callbacks=[ConsoleCallbackHandler()]
        )
        
        # Setup storage before embeddings
        self._init_storage()
        
        # Setup embeddings with embeddings-specific deployment
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMB_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            chunk_size=1
        )
        print("Configured embeddings for pattern matching")
        
        # Initialize file store and splitter
        self.store = LocalFileStore(self.reviews_dir)
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Setup retriever
        self.retriever = self._initialize_retriever()
    
    def _init_storage(self):
        """Initialize storage and cleanup old files."""
        try:
            # Create reviews directory if needed
            if not os.path.exists(self.reviews_dir):
                os.makedirs(self.reviews_dir)
                print(f"Created reviews directory: {self.reviews_dir}")
            
            # Cleanup old review files if too many
            review_files = glob.glob(os.path.join(self.reviews_dir, "review_*.json"))
            if len(review_files) > self.max_reviews:
                review_files.sort(key=os.path.getctime)  # Sort by creation time
                files_to_remove = review_files[:-self.max_reviews]
                for file in files_to_remove:
                    os.remove(file)
                print(f"Cleaned up {len(files_to_remove)} old review files")
            
        except Exception as e:
            print(f"Error initializing storage: {str(e)}")
            raise
    
    def _initialize_retriever(self):
        """Initialize the pattern retriever."""
        try:
            # Load or create vector store
            store_path = os.path.join(self.reviews_dir, "vectorstore")
            if os.path.exists(store_path):
                print("Loading pattern database...")
                vectorstore = FAISS.load_local(
                    store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                print("Creating new pattern store...")
                vectorstore = FAISS.from_texts(
                    ["initial pattern store"],
                    embedding=self.embeddings
                )
                print("Saving pattern store...")
                vectorstore.save_local(store_path)
            
            # Create retriever
            return ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=self.store,
                child_splitter=self.splitter,
                search_kwargs={"k": 3}
            )
            
        except Exception as e:
            print(f"Error initializing retriever: {str(e)}")
            raise
    
    def find_patterns(self, code: str) -> List[Dict[str, str]]:
        """Find similar code patterns."""
        try:
            print("\nSearching for similar patterns...")
            # Search for similar patterns
            docs = self.retriever.get_relevant_documents(code)
            if not docs:
                print("No similar patterns found in database")
                print("This is normal for the first review")
                return []
            
            patterns = []
            for doc in docs:
                pattern = {
                    'content': doc.page_content[:200],  # Limit content length
                    'similarity': "high" if doc.metadata.get('score', 0) > 0.8 else "medium",
                    'source': doc.metadata.get('source', 'unknown')
                }
                patterns.append(pattern)
            
            print(f"Found {len(patterns)} similar patterns")
            return patterns
            
        except Exception as e:
            print(f"Error finding patterns: {str(e)}")
            return []
    
    def analyze_code(self, file_path: str) -> CodeReview:
        """Analyze code file and provide review."""
        try:
            print(f"\nAnalyzing: {file_path}")
            print("=" * 50)
            
            # Read code file
            with open(file_path, 'r') as f:
                code = f.read()
            
            print(f"Analyzing {len(code.splitlines())} lines of code...")
            
            # Find similar patterns
            similar_patterns = self.find_patterns(code)
            
            # Create review prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert code reviewer. Always provide output in this exact JSON format:
{{
    "issues": [
        {{
            "severity": "high/medium/low",
            "category": "security/performance/style",
            "description": "Issue description",
            "line_numbers": [line numbers affected],
            "suggestion": "How to fix"
        }}
    ],
    "metrics": {{
        "complexity": float,
        "maintainability": float
    }},
    "summary": "Overall review summary"
}}"""),
                ("human", """Please analyze this code:

{input}

Similar patterns found:
{patterns}

Return a JSON object containing your review. No additional text.""")
            ])
            
            # Create review chain using LCEL
            review_chain = prompt | self.llm.with_config(
                callbacks=[ConsoleCallbackHandler()],
                verbose=True
            )
            
            print("\nGenerating review...")
            # Run chain
            try:
                response = review_chain.invoke({
                    "input": code,
                    "patterns": json.dumps(similar_patterns, indent=2)
                })
                print("\nReceived LLM response")
            except Exception as e:
                print(f"Chain error: {str(e)}")
                raise ValueError("Failed to generate review")
            
            print("\nParsing review results...")
            # Extract JSON from response
            content = response.content
            # Remove code block markers if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            # Strip and validate JSON
            json_str = content.strip()
            if not json_str:
                raise ValueError("Empty response from LLM")
            
            try:
                review_data = json.loads(json_str)
                print(f"\nFound {len(review_data.get('issues', []))} potential issues")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw content:\n{content}")
                raise ValueError("Invalid JSON in response")
            
            # Create review object
            review = CodeReview(
                file_path=file_path,
                patterns=similar_patterns,
                **review_data
            )
            
            print("\nSaving review...")
            # Save review
            self._save_review(review)
            
            return review
            
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            raise
    
    def _save_review(self, review: CodeReview):
        """Save review results."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.basename(review.file_path)
            review_file = f"review_{base_name}_{timestamp}.json"
            review_path = os.path.join(self.reviews_dir, review_file)
            
            # Save review
            with open(review_path, 'w') as f:
                json.dump(review.model_dump(), f, indent=2)
            
            print(f"\nSaved review to: {review_path}")
            
            # Add to retriever's knowledge
            print("\nUpdating pattern database...")
            
            # Create review summary text
            review_text = f"""
Code Review Summary
==================
File: {os.path.basename(review.file_path)}
Total Issues: {len(review.issues)}

Severity Distribution:
High Severity:   {review.issues_by_severity['high']}
Medium Severity: {review.issues_by_severity['medium']}
Low Severity:    {review.issues_by_severity['low']}

Summary:
{review.summary}
"""
            
            # Create metadata
            metadata = {
                "file": review.file_path,
                "issues": len(review.issues),
                "has_critical": review.has_critical_issues,
                "review_date": timestamp
            }
            
            # Create document
            doc = Document(
                page_content=review_text,
                metadata=metadata
            )
            
            # Update pattern store
            print("Adding review to pattern database...")
            self.retriever.vectorstore.add_documents([doc])
            
        except Exception as e:
            print(f"Error in save_review: {str(e)}")
            raise

def demonstrate_code_review():
    """Demonstrate the code review assistant."""
    print("\nCode Review Assistant Demo")
    print("=" * 50)
    print("Starting code review process...")
    
    print("\nSetting up sample code...")
    # Create sample code directory
    sample_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'sample_code'
    )
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Create sample files
    samples = {
        'login.py': '''def check_password(username, password):
    """Check user password."""
    stored = get_password(username)  # Gets plain text password
    return password == stored
''',
        'data_processor.py': '''def process_data(data):
    """Process user data."""
    results = []
    for item in data:
        results.append(transform(item))
    return results
''',
        'api_client.py': '''def call_api(url, data):
    """Call external API."""
    response = requests.post(url, json=data)
    return response.json()
'''
    }
    
    # Write sample files
    for name, content in samples.items():
        file_path = os.path.join(sample_dir, name)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created sample file: {name}")
    
    print("\nInitializing code review assistant...")
    try:
        reviewer = CodeReviewAssistant()
    except Exception as e:
        print(f"Failed to initialize reviewer: {str(e)}")
        return

    # List of files to review
    files = [
        os.path.join(sample_dir, name)
        for name in samples.keys()
    ]

    for file_path in files:
        try:
            print(f"\nReviewing: {file_path}")
            print("-" * 50)
            
            review = reviewer.analyze_code(file_path)
            
            print("\nCode Review Results:")
            print("=" * 30)
            print(f"File: {os.path.basename(review.file_path)}")
            
            # Print issue counts
            counts = review.issues_by_severity
            print(f"\nIssue Summary:")
            print(f"HIGH: {counts['high']} | MEDIUM: {counts['medium']} | LOW: {counts['low']}")

            print("\nIssues Found:")
            if review.issues:
                for i, issue in enumerate(review.issues, 1):
                    print(f"\nIssue {i}:")
                    print(f"- Severity: {issue.severity.upper()}")
                    print(f"- Category: {issue.category}")
                    print(f"- Lines: {', '.join(map(str, issue.line_numbers))}")
                    print(f"- Problem: {issue.description}")
                    print(f"- Solution: {issue.suggestion}")
                if review.has_critical_issues:
                    print("\n⚠️ Critical issues found! Please review carefully.")
            else:
                print("No issues found")
            
            print("\nSimilar Patterns:")
            if review.patterns:
                for i, pattern in enumerate(review.patterns, 1):
                    print(f"\nPattern {i}:")
                    print(f"Match ({pattern['similarity']}): {pattern['content']}")
                    print(f"From: {pattern['source']}")
            else:
                print("No similar patterns found")
            
            print("\nCode Quality Metrics:")
            print(json.dumps(review.metrics, indent=2))

            print("\nOverall Assessment:")
            print(review.summary)
            print("=" * 30)
            
        except Exception as e:
            print(f"Error reviewing {file_path}: {str(e)}")
        
        print("\n" + "=" * 50)

    print("\nReviews are stored in:")
    print(os.path.abspath(reviewer.reviews_dir))
    print("\nSample files are in:")
    print(sample_dir)

if __name__ == "__main__":
    demonstrate_code_review()
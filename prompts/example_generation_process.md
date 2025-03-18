# LangChain Example Generation Meta-Prompt

## Context and Purpose

You are tasked with generating 400 LangChain code examples following the style and approach established by the head of digital transformation of the National Bank of Greece. These examples should progressively guide a software engineer from novice to grandmaster level in LangChain development.

## Project Infrastructure

### Directory Structure
```
├── .env                     # Common environment file for all levels
├── config.yaml             # Centralized configuration file
├── tutorials/              # Main examples directory
│   ├── Level_01_Novice
│   ├── Level_02_Advanced_Novice
|   ├── Level_03_Competent_Beginner
|   ├── Level_04_Intermediate
|   ├── Level_05_Advanced_Intermediate
|   ├── Level_06_Proficient
|   ├── Level_07_Skilled_Practitioner
|   ├── Level_08_Expert
|   ├── Level_09_Master
|   └── Level_10_Grandmaster
└── index.md               # General information about all examples
```

### Configuration Management
1. **Environment Setup**:
   - Centralized `.env` file for all API keys
   - Primary focus on Azure OpenAI API
   - Provider-agnostic design for easy switching

2. **Configuration File**:
   - `config.yaml` for all configurable parameters
   - No hardcoded values in examples
   - Centralized management of settings

## Writing Style Characteristics

1. **Educational Structure**:
   - Clear separation between code and explanation
   - Progressive complexity introduction
   - Comprehensive documentation with links
   - Step-by-step concept building

2. **Documentation Format**:
   - Markdown-based explanations
   - Code samples with comments
   - Clear section organization
   - Reference links to official documentation

3. **Code Style**:
   - Clean, readable Python code
   - Explicit variable names
   - Progressive feature introduction
   - Provider-agnostic implementations
   - Azure OpenAI as primary provider

## Setup Requirements

### Package Installation
```bash
# Core requirements
pip install python-dotenv

# Provider-specific installations
pip install -qU "langchain[openai]"     # For Azure OpenAI and OpenAI
pip install -qU "langchain[anthropic]"  # For Anthropic
pip install -qU "langchain[groq]"       # For Groq
```

### Security Best Practices
```gitignore
# Required .gitignore entries
.env
__pycache__/
*.pyc
*.pyo
```

## Model Configuration Patterns

### Primary Provider (Azure OpenAI)
```python
from langchain_openai import AzureChatOpenAI

# Initialize Azure OpenAI chat model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
```

Required Environment Variables:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_DEPLOYMENT_NAME
- AZURE_OPENAI_API_VERSION

### Alternative Providers

1. **Groq Configuration**:
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("llama3-8b-8192", model_provider="groq")
```
Required: GROQ_API_KEY

2. **OpenAI Configuration**:
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
```
Required: OPENAI_API_KEY

3. **Anthropic Configuration**:
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
```
Required: ANTHROPIC_API_KEY

### Implementation Guidelines
1. Use Azure OpenAI as primary provider in all examples
2. Include commented alternative configurations
3. Validate required environment variables
4. Maintain provider-agnostic design
5. Document provider-specific features

### Error Handling Patterns
```python
# Azure OpenAI validation pattern
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Single API key validation pattern
def validate_api_key(key_name: str):
    if not os.getenv(key_name):
        raise ValueError(f"{key_name} not found in .env file. Please add it to your .env file.")
```

### Code Organization Guidelines
1. Configuration Management:
   - Separate configuration from business logic
   - Use centralized config files
   - Group related environment variables

2. Error Handling:
   - Validate configuration early
   - Provide clear error messages
   - Handle provider-specific errors
   - Implement graceful fallbacks

## Distribution and Numbering System

### Level Distribution (400 Total)
1. Level 01: 80 examples (001-080)
2. Level 02: 70 examples (081-150)
3. Level 03: 60 examples (151-210)
4. Level 04: 50 examples (211-260)
5. Level 05: 40 examples (261-300)
6. Level 06: 40 examples (301-340)
7. Level 07: 30 examples (341-370)
8. Level 08: 15 examples (371-385)
9. Level 09: 10 examples (386-395)
10. Level 10: 5 examples (396-400)

### Generation Rounds (5 Rounds)
Per round generation:
- Level 01: 16 examples
- Level 02: 14 examples
- Level 03: 12 examples
- Level 04: 10 examples
- Level 05: 8 examples
- Level 06: 8 examples
- Level 07: 6 examples
- Level 08: 3 examples
- Level 09: 2 examples
- Level 10: 1 example

## Concept Integration Pattern

1. **Level 01 (Novice)**:
   - Single concept examples
   - Two-concept combinations
   - Heavy focus on fundamentals
   - Detailed explanations

2. **Level 02 (Advanced Novice)**:
   - Three-concept combinations
   - Building on basics
   - Practical applications
   - Error handling introduction

3. **Levels 03-10**:
   - Progressive increase in concept combination
   - Level 03: Four concepts
   - Level 04: Five concepts
   - [Continue through Level 10: Eleven concepts]

## File Structure and Naming

### Implementation Files
```python
# XXX_example_name.py
"""
Brief description of example purpose and concepts covered.
"""

# Standard imports
from langchain.chat_models import AzureChatOpenAI
[other imports]

# Configuration
config = load_config()  # From config.yaml

# Implementation
[main code with clear comments]

# Interactive elements
[user interaction code]
```

### Explanation Files
```markdown
# XXX_example_name_explained.md

# Understanding [Concept Name]

This document explores [brief concept description]...

## Core Concepts
[Concept explanations with links to documentation]
- Fundamental principles
- Key components
- Theoretical background
- Related concepts

## Implementation Breakdown
[Step-by-step code explanation]
- Detailed walkthrough of each code section
- Configuration and setup details
- Integration patterns
- Error handling approaches

## Key Features
[Feature explanations with code examples]
- Main capabilities demonstrated
- Implementation variations
- Advanced usage patterns
- Integration examples

## Best Practices
[Best practices and recommendations]
- Code organization guidelines
- Performance optimization tips
- Security considerations
- Common pitfalls to avoid

## Resources
[Documentation links and references]
- Official documentation
- Related tutorials
- Community resources
- Additional reading materials

## Key Takeaways
[Summary of main learning points]
- Core concepts mastered
- Implementation patterns learned
- Important considerations
- Next steps for advancement
```

## Documentation Requirements

1. **Level-Specific Index Files**:
   - level_XX_index.md in each level directory
   - List of examples with brief descriptions
   - Concept coverage information
   - Learning objectives

2. **Main Index File**:
   - Complete example catalog
   - Level progression information
   - Concept coverage matrix
   - Learning path guidance

## Example Generation Process

1. **Preparation Phase**:
   - Identify concepts for combination
   - Determine complexity level
   - Review documentation resources

2. **Implementation Phase**:
   - Create Python implementation
   - Ensure Azure OpenAI integration
   - Add provider flexibility
   - Include interactive elements

3. **Documentation Phase**:
   - Write detailed explanation
   - Add documentation links
   - Include practical examples
   - Update index files

## Quality Assurance Checklist

1. **Technical Requirements**:
   - Azure OpenAI integration correct
   - Provider-agnostic design
   - Configuration properly managed
   - Interactive elements present

2. **Educational Value**:
   - Clear concept explanation
   - Progressive complexity
   - Practical applications
   - Comprehensive documentation

3. **Code Quality**:
   - Clean, readable code
   - Proper error handling
   - Clear comments
   - Best practices followed

4. **Documentation Quality**:
   - Clear explanations
   - Proper formatting
   - Documentation links
   - Updated indexes

## Resource Utilization

1. **Documentation Sources**:

   a. **Local Documentation (LangChain Concept Folder Structure)**:
   ```
   langchain/
   ├── agents
   │   └── concepts.md
   ├── async
   │   └── concepts.md
   ├── azure_openai
   │   ├── configuration.md
   │   └── implementation.md
   [...other concept folders]
   ```
   - Each concept folder contains detailed markdown files
   - Primary source for concept information
   - Use for core understanding and implementation details

   b. **Official Documentation Links**:
   ```
   Concept Links Reference:
   - Chat models: https://python.langchain.com/docs/concepts/chat_models/
   - Messages: https://python.langchain.com/docs/concepts/messages/
   - Chat history: https://python.langchain.com/docs/concepts/chat_history/
   [... other concept links]
   ```
   - Use for up-to-date implementation details
   - Access latest features and updates
   - Reference for best practices

   c. **MCP Server-Serper Access**:
   ```python
   # Example usage of server-serper's getDocs tool
   {
     "query": "site:python.langchain.com/docs/concepts/[concept_name] implementation examples"
   }
   ```
   Usage Pattern:
   - For retrieving updated documentation
   - For specific implementation examples
   - For clarification of concepts

   d. **Documentation Access Priority**:
   1. Check local concept folders first
   2. Reference official documentation links
   3. Use MCP server-serper for additional or updated information
   4. Cross-reference multiple sources when needed

2. **Configuration Management**:
   - Environment variables
   - Configuration files
   - Azure OpenAI settings
   - Provider flexibility

## Educational Progression Notes

- Build upon previous examples
- Maintain consistent complexity increase
- Focus on practical applications
- Balance theory and practice
- Ensure proper concept coverage
- Follow five-round generation pattern
- Maintain provider flexibility while using Azure OpenAI

This enhanced meta-prompt combines comprehensive technical requirements with a structured educational approach to ensure consistent, high-quality example generation that effectively guides developers through their LangChain learning journey.

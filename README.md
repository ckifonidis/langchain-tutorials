# LangChain Tutorial Examples Usage Guide

This guide explains how to set up your environment to run the LangChain tutorial examples.

## Package Requirements

All examples use LangChain v0.3 and compatible package versions:

### Core Packages
```plaintext
langchain>=0.3.0,<0.4.0
langchain-core>=0.3.0,<0.4.0
langchain-community>=0.3.0,<0.4.0
langchain-openai>=0.2.0,<0.3.0
```

### Dependencies
```plaintext
pydantic>=2,<3
python-dotenv>=1.0.0,<2.0.0
typing-extensions>=4.7.1
packaging>=23.2
PyYAML>=5.3.1
requests>=2.31.0
aiohttp>=3.8.0
tenacity>=8.1.0,<9.0.0
```

## Important Notes

1. **Python Version**: Python 3.9+ is required (3.8 is no longer supported)
2. **Pydantic**: All packages use Pydantic v2 internally
3. **Package Order**: Core packages must be installed in the correct order to avoid dependency conflicts

## Quick Start

### Ubuntu/Linux
0. Enter "tutorials" folder:
    ```bash
    cd tutorials
    ```

1. Run the setup script:
   ```bash
   chmod +x setup_ubuntu.sh
   ./setup_ubuntu.sh
   ```

2. Activate the virtual environment:
   ```bash
   source langchain_env/bin/activate
   ```

### Windows
0. Enter "tutorials" folder:
    ```bash
    cd tutorials\
    ```

1. Run the setup script:
   ```powershell
   .\setup_windows.ps1
   ```

2. Activate the virtual environment:
   ```powershell
   .\langchain_env\Scripts\activate
   ```

## Manual Setup

If you prefer to set up manually:

### Ubuntu/Linux

```bash
# Create virtual environment
python -m venv langchain_env

# Activate virtual environment
source langchain_env/bin/activate

# Install base dependencies first
pip install \
    'pydantic>=2,<3' \
    'typing-extensions>=4.7.1' \
    'python-dotenv>=1.0.0,<2.0.0'

# Install core dependencies
pip install \
    'packaging>=23.2' \
    'PyYAML>=5.3.1' \
    'requests>=2.31.0' \
    'aiohttp>=3.8.0' \
    'tenacity>=8.1.0,<9.0.0'

# Install LangChain core packages
pip install \
    'langchain-core>=0.3.0,<0.4.0' \
    'langchain>=0.3.0,<0.4.0' \
    'langchain-community>=0.3.0,<0.4.0'

# Install integration packages
pip install \
    'langchain-openai>=0.2.0,<0.3.0'
```

### Windows

```powershell
# Create virtual environment
python -m venv langchain_env

# Activate virtual environment
.\langchain_env\Scripts\activate

# Install base dependencies first
pip install `
    'pydantic>=2,<3' `
    'typing-extensions>=4.7.1' `
    'python-dotenv>=1.0.0,<2.0.0'

# Install core dependencies
pip install `
    'packaging>=23.2' `
    'PyYAML>=5.3.1' `
    'requests>=2.31.0' `
    'aiohttp>=3.8.0' `
    'tenacity>=8.1.0,<9.0.0'

# Install LangChain core packages
pip install `
    'langchain-core>=0.3.0,<0.4.0' `
    'langchain>=0.3.0,<0.4.0' `
    'langchain-community>=0.3.0,<0.4.0'

# Install integration packages
pip install `
    'langchain-openai>=0.2.0,<0.3.0'
```

## Environment Configuration

1. Create a `.env` file in the root directory:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

2. Replace the placeholders with your actual Azure OpenAI credentials.

## Verification

Test your setup by running:

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel

print('Installed package versions:')
import langchain
import langchain_core
import langchain_community
import langchain_openai
print(f'langchain: {langchain.__version__}')
print(f'langchain-core: {langchain_core.__version__}')
print(f'langchain-community: {langchain_community.__version__}')
print(f'langchain-openai: {langchain_openai.__version__}')
```

## Troubleshooting

### Common Issues

1. **Package Conflicts**
   - Install packages in the order shown above
   - Pydantic v1 is no longer supported
   - Remove any existing conflicting packages:
     ```bash
     pip uninstall langchain langchain-core langchain-openai langchain-community
     ```

2. **Python Version**
   - Python 3.8 is no longer supported
   - Use Python 3.9 or higher
   - Recommended: Python 3.9, 3.10, or 3.11

3. **Installation Errors**
   - Check your Python version: `python --version`
   - Upgrade pip: `pip install --upgrade pip`
   - Use a fresh virtual environment
   - Install packages in the correct order

### Getting Help

If you encounter issues:
1. Verify Python version is 3.9+
2. Check package versions with `pip freeze`
3. Ensure virtual environment is active
4. Try creating a fresh environment
5. Follow the installation order exactly

## Running Examples

1. Ensure virtual environment is activated
2. Navigate to example directory: `cd tutorials/Level_01_Novice/`
3. Run any example: `python 001_simple_chat_model.py`

Each example is self-contained and includes error handling for common issues.

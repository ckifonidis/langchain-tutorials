#!/bin/bash

# LangChain Tutorial Setup Script for Ubuntu/Linux
# This script creates a virtual environment and installs required packages
# Compatible with LangChain v0.3 and above

echo "Setting up LangChain Tutorial Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment already exists
if [ -d "langchain_env" ]; then
    echo "Directory 'langchain_env' already exists."
    read -p "Do you want to remove it and create a fresh environment? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf langchain_env
    else
        echo "Please remove or rename the existing 'langchain_env' directory and try again."
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv langchain_env

# Activate virtual environment
echo "Activating virtual environment..."
source langchain_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base dependencies first
echo "Installing base dependencies..."
pip install \
    'pydantic>=2,<3' \
    'typing-extensions>=4.7.1' \
    'python-dotenv>=1.0.0,<2.0.0'

# Install core dependencies
echo "Installing core dependencies..."
pip install \
    'packaging>=23.2' \
    'PyYAML>=5.3.1' \
    'requests>=2.31.0' \
    'aiohttp>=3.8.0' \
    'tenacity>=8.1.0,<9.0.0'

# Install LangChain core packages
echo "Installing LangChain core packages..."
pip install \
    'langchain-core>=0.3.0,<0.4.0' \
    'langchain>=0.3.0,<0.4.0' \
    'langchain-community>=0.3.0,<0.4.0'

# Install integration packages
echo "Installing integration packages..."
pip install \
    'langchain-openai>=0.2.0,<0.3.0'

# Verify installation
echo "Verifying installation..."
python3 -c "
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
print('\nAll packages loaded successfully!')
"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
EOL
    echo "Created .env file. Please update it with your Azure OpenAI credentials."
fi

echo "
Setup complete! To activate the environment:
$ source langchain_env/bin/activate

To deactivate when done:
$ deactivate
"
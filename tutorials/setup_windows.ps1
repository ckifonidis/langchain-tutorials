# LangChain Tutorial Setup Script for Windows
# This script creates a virtual environment and installs required packages
# Compatible with LangChain v0.3 and above

Write-Host "Setting up LangChain Tutorial Environment..." -ForegroundColor Green

# Check if Python 3 is installed
try {
    $pythonVersion = python --version 2>&1
    if (-not $pythonVersion.ToString().StartsWith("Python 3")) {
        Write-Host "Python 3 is required but not installed. Please install Python 3 and try again." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Python 3 is required but not installed. Please install Python 3 and try again." -ForegroundColor Red
    exit 1
}

# Check if virtual environment already exists
if (Test-Path "langchain_env") {
    Write-Host "Directory 'langchain_env' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove it and create a fresh environment? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force langchain_env
    } else {
        Write-Host "Please remove or rename the existing 'langchain_env' directory and try again." -ForegroundColor Red
        exit 1
    }
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv langchain_env

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\langchain_env\Scripts\Activate

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install base dependencies first
Write-Host "Installing base dependencies..." -ForegroundColor Green
pip install `
    'pydantic>=2,<3' `
    'typing-extensions>=4.7.1' `
    'python-dotenv>=1.0.0,<2.0.0'

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Green
pip install `
    'packaging>=23.2' `
    'PyYAML>=5.3.1' `
    'requests>=2.31.0' `
    'aiohttp>=3.8.0' `
    'tenacity>=8.1.0,<9.0.0'

# Install LangChain core packages
Write-Host "Installing LangChain core packages..." -ForegroundColor Green
pip install `
    'langchain-core>=0.3.0,<0.4.0' `
    'langchain>=0.3.0,<0.4.0' `
    'langchain-community>=0.3.0,<0.4.0'

# Install integration packages
Write-Host "Installing integration packages..." -ForegroundColor Green
pip install `
    'langchain-openai>=0.2.0,<0.3.0'

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Green
python -c "
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
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Green
    @"
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "Created .env file. Please update it with your Azure OpenAI credentials." -ForegroundColor Yellow
}

Write-Host @"

Setup complete! To activate the environment:
PS> .\langchain_env\Scripts\Activate

To deactivate when done:
PS> deactivate
"@ -ForegroundColor Green
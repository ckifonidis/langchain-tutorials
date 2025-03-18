# Azure OpenAI API Integration with LangChain

## Key Concepts

1. API Configuration
- AZURE_OPENAI_API_KEY: API key for authentication
- AZURE_OPENAI_ENDPOINT: Base URL for Azure OpenAI resource
- OPENAI_API_VERSION: API version (e.g., 2024-02-15-preview)

2. Deployment Configuration
- Each model requires its own deployment in Azure
- Deployment name is specified when creating the chat model
- Maps to specific OpenAI models like gpt-35-turbo

3. Authentication Options
- API Key authentication (simpler)
- Azure Active Directory (AAD) authentication (more secure)

## Environment Configuration
```bash
# Required environment variables
export AZURE_OPENAI_API_KEY=<your-api-key>
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export OPENAI_API_VERSION=2024-02-15-preview
```

## Model Instantiation Pattern
```python
from langchain_openai import AzureChatOpenAI

# Proper initialization with all required parameters
llm = AzureChatOpenAI(
    deployment_name="your-deployment",  # Your Azure deployment name
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0.7
)
```

## Alternative Authentication (AAD)
```python
from azure.identity import DefaultAzureCredential

# Get Azure credentials
credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

# Initialize with AAD authentication
llm = AzureChatOpenAI(
    deployment_name="your-deployment",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_ad_token=token.token,
    temperature=0.7
)
```

## Invoke Usage Patterns
The `invoke` method can be used in several ways:

1. Direct string input:
```python
result = llm.invoke("What are the best practices for writing clean code?")
```

2. Message list input:
```python
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is clean code?")
]
result = llm.invoke(messages)
```

3. With system messages for context:
```python
messages = [
    SystemMessage(content="You are an expert software developer"),
    HumanMessage(content="Explain SOLID principles")
]
result = llm.invoke(messages)
```

4. In chat loops with history:
```python
messages = [
    SystemMessage(content="You are a software architect")
]
while True:
    user_input = input("Ask: ")
    messages.append(HumanMessage(content=user_input))
    result = llm.invoke(messages)
    messages.append(AIMessage(content=result.content))
```

## Error Handling
```python
try:
    # Verify environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing variables: {', '.join(missing)}")
        
    # Initialize and use the model
    llm = AzureChatOpenAI(...)
    result = llm.invoke(message)
    
except Exception as e:
    print(f"Error: {str(e)}")
```

## Best Practices

1. Environment Variables:
   - Use .env files for local development
   - Use secure key vaults in production
   - Never commit API keys to version control

2. Deployment Names:
   - Use meaningful deployment names
   - Document model versions
   - Keep track of deployment regions

3. Error Handling:
   - Always validate environment variables
   - Implement proper error handling
   - Handle rate limits appropriately

4. Security:
   - Prefer AAD authentication in production
   - Rotate API keys regularly
   - Monitor usage and costs

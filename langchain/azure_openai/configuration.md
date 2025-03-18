# Azure OpenAI Configuration in LangChain

## Environment Setup

### Required Environment Variables
```bash
# The API key for your Azure OpenAI resource
AZURE_OPENAI_API_KEY=your_api_key_here

# The base URL for your Azure OpenAI resource
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# The API version to use
OPENAI_API_VERSION=2023-12-01-preview
```

## Authentication Methods

### 1. API Key Authentication
```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name="your-deployment",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
```

### 2. Azure Active Directory (AAD) Authentication
```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

llm = AzureChatOpenAI(
    deployment_name="your-deployment",
    azure_ad_token=token.token,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
```

## Model Deployment

### Deployment Configuration
1. Create a deployment in Azure OpenAI Studio
2. Choose the model (e.g., gpt-35-turbo)
3. Note the deployment name
4. Use the deployment name in LangChain

### Model Parameters
```python
llm = AzureChatOpenAI(
    deployment_name="your-deployment",
    temperature=0.7,           # Controls randomness (0-1)
    max_tokens=None,          # Max tokens in response
    top_p=1,                  # Nucleus sampling parameter
    frequency_penalty=0,      # Reduces word repetition
    presence_penalty=0,       # Encourages new topics
    timeout=None,            # Request timeout
    max_retries=2           # Number of retry attempts
)
```

## Best Practices

1. Environment Management
   - Use .env files for local development
   - Use secure key vaults in production
   - Rotate API keys regularly

2. Error Handling
   - Implement proper error handling
   - Handle rate limits
   - Set appropriate timeouts
   - Use retries for transient failures

3. Cost Management
   - Monitor token usage
   - Use caching when appropriate
   - Set token limits for responses

4. Security
   - Use AAD authentication in production
   - Implement least privilege access
   - Regularly audit access logs

## Common Issues

1. Authentication Errors
   - Check API key validity
   - Verify endpoint URL
   - Confirm API version compatibility

2. Deployment Issues
   - Verify deployment name
   - Check deployment status
   - Confirm model availability

3. Performance Issues
   - Monitor response times
   - Check rate limits
   - Optimize prompt length
   - Use appropriate timeouts

## Resources

1. Official Documentation
   - Azure OpenAI Service Documentation
   - LangChain Azure Integration Guide
   - Azure Identity Documentation

2. Security Guidelines
   - Azure Security Best Practices
   - OpenAI Security Guidelines
   - LangChain Security Recommendations

3. Cost Management
   - Azure Pricing Calculator
   - Token Usage Guidelines
   - Optimization Strategies
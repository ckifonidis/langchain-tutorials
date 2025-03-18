# LLM Provider Configuration Guide

## Environment Setup

Each provider requires specific environment variables and configuration. Below is a comprehensive guide for each supported provider.

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)
```

### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export OPENAI_API_VERSION="2024-02-15-preview"
```

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name="your-deployment",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7,
    max_tokens=1000
)
```

### Groq
```bash
export GROQ_API_KEY="your-api-key"
```

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="mixtral-8x7b-v1",
    temperature=0.7,
    max_tokens=500
)
```

### Mistral AI
```bash
export MISTRAL_API_KEY="your-api-key"
```

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7,
    max_tokens=500
)
```

### Cohere
```bash
export COHERE_API_KEY="your-api-key"
```

```python
from langchain_cohere import ChatCohere

llm = ChatCohere(
    model="command",
    temperature=0.7,
    max_tokens=500
)
```

## Common Configuration Parameters

### 1. Temperature
- Range: 0.0 to 1.0
- Default: 0.7
- Purpose: Controls response randomness
- Higher values: More creative
- Lower values: More deterministic

### 2. Max Tokens
- Purpose: Limits response length
- Default: Varies by model
- Consider: Model's context window
- Note: Affects cost directly

### 3. Request Timeout
- Purpose: Sets maximum wait time
- Default: Usually 60 seconds
- Adjust based on:
  * Network conditions
  * Model complexity
  * Response length

### 4. Retry Settings
```python
llm = ChatOpenAI(
    max_retries=3,
    retry_delay=1,
    timeout=30
)
```

## Security Best Practices

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Never commit keys to code
   - Use secret management in production

2. **Request Limits**
   - Implement rate limiting
   - Monitor usage
   - Set up alerts
   - Use try-catch blocks

3. **Error Handling**
   ```python
   try:
       response = llm.invoke(messages)
   except Exception as e:
       logger.error(f"Model error: {str(e)}")
       # Implement fallback strategy
   ```

## Production Considerations

1. **High Availability**
   - Implement fallback providers
   - Use multiple deployments
   - Monitor service health
   - Handle rate limits gracefully

2. **Cost Management**
   - Track token usage
   - Implement caching
   - Use cheaper models when possible
   - Set up cost alerts

3. **Performance Optimization**
   - Batch requests when possible
   - Use streaming for long responses
   - Implement response caching
   - Monitor latency

4. **Monitoring**
   ```python
   from langchain.callbacks import StdOutCallbackHandler

   llm = ChatOpenAI(
       callbacks=[StdOutCallbackHandler()],
       # other parameters
   )
   ```

## Provider-Specific Notes

### Azure OpenAI
- Requires deployment setup
- Supports AAD authentication
- Regional availability varies
- Enterprise features available

### Anthropic
- Longer context windows
- Different response style
- Constitutional AI features
- Custom prompt format

### Groq
- Optimized for speed
- Limited model selection
- Simple API interface
- Cost-effective for high volume

### Mistral AI
- Strong technical performance
- Growing model selection
- European provider
- GDPR compliant

### Cohere
- Task-specific models
- Built-in classification
- Custom fine-tuning
- Enterprise support available
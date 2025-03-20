# Understanding the Message Sentiment Analyzer in LangChain

Welcome to this comprehensive guide on building a message sentiment analyzer using LangChain! This example demonstrates how to combine embedding models with semantic analysis to create a sophisticated system that can analyze message sentiment using vector similarity and provide detailed evaluation results.

## Complete Code Walkthrough

### 1. Required Imports and Technical Foundation

```python
import os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import numpy as np

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openevals.llm import create_llm_as_judge
```

Our sentiment analysis system combines two powerful capabilities:

1. Embedding Models: The `AzureOpenAIEmbeddings` component transforms text into high-dimensional vectors, enabling semantic comparison through mathematical operations. These embeddings capture the underlying meaning and context of messages, making sentiment analysis more nuanced and accurate.

2. Vector Mathematics: Using NumPy (`np`), we perform sophisticated vector operations like cosine similarity calculations to determine how semantically similar messages are to reference examples of different sentiment categories.

### 2. Environment Configuration

```python
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
```

The implementation requires specific Azure configurations for both the chat model and embeddings:
- Chat model settings for general language understanding
- Embedding model settings for semantic vector generation
- API configuration for Azure services integration

### 3. Cosine Similarity Implementation

```python
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
```

This function demonstrates sophisticated vector mathematics:
1. Vector Normalization: Converting input lists to NumPy arrays for efficient computation
2. Zero Vector Handling: Preventing division by zero with explicit checks
3. Cosine Calculation: Computing the normalized dot product between vectors
4. Score Normalization: Ensuring results are in the range [0, 1]

### 4. Sentiment Analysis Process

```python
def analyze_sentiment(messages: List[str], embeddings: AzureOpenAIEmbeddings) -> AnalysisResults:
```

The analysis function shows advanced semantic processing:

1. Reference Examples:
```python
sentiment_examples = {
    "Positive": [
        "I'm really happy with this!",
        "Great job, thank you so much!",
        "This exceeds my expectations!"
    ],
    "Neutral": [
        "The product works as described.",
        "I received the delivery today.",
        "Let me know when it's ready."
    ],
    "Negative": [
        "This is not what I expected.",
        "I'm disappointed with the service.",
        "There are several issues to fix."
    ]
}
```

2. Embedding Computation:
```python
ref_embeddings = {
    category: [embeddings.embed_query(example) for example in examples]
    for category, examples in sentiment_examples.items()
}

msg_embedding = embeddings.embed_query(message)
```

3. Similarity Analysis:
```python
sim_values = [cosine_similarity(msg_embedding, ref_emb) for ref_emb in emb_list]
max_sim = max(sim_values) if sim_values else 0.0
scores[f"{category.lower()}_score"] = max_sim
```

### 5. Result Formatting

```python
output = f"Batch ID: {results.batch_id}\n\n"
for analysis in results.messages:
    output += (
        f"Message ID: {analysis.message_id}\n"
        f"Text: {analysis.text}\n"
        f"Dominant Sentiment: {analysis.sentiment.dominant_sentiment} "
        f"(Confidence: {analysis.confidence_score:.4f})\n"
        f"Scores:\n"
        f"  Positive: {analysis.sentiment.positive_score:.4f}\n"
        f"  Neutral:  {analysis.sentiment.neutral_score:.4f}\n"
        f"  Negative: {analysis.sentiment.negative_score:.4f}\n"
        f"{'-'*50}\n"
    )
```

## Expected Output

When running the sentiment analyzer with example messages, you'll see detailed output like this:

```plaintext
Batch ID: BATCH865

Message ID: MSG001
Text: I absolutely love this product! It's exactly what I needed.
Dominant Sentiment: Positive (Confidence: 0.8765)
Scores:
  Positive: 0.8765
  Neutral:  0.2134
  Negative: 0.1023
--------------------------------------------------

Message ID: MSG002
Text: The delivery arrived on time as scheduled.
Dominant Sentiment: Neutral (Confidence: 0.9123)
Scores:
  Positive: 0.2314
  Neutral:  0.9123
  Negative: 0.1432
--------------------------------------------------

Batch Summary:
Avg positive score: 0.5540
Avg neutral score: 0.5629
Avg negative score: 0.1228
Avg confidence: 0.8944
```

## Resources

### LangChain Documentation
- Embedding Models: https://python.langchain.com/docs/modules/data_connection/text_embedding/
- Azure Integration: https://python.langchain.com/docs/integrations/platforms/azure
- Vector Operations: https://python.langchain.com/docs/guides/embeddings/vector_operations
- Semantic Similarity: https://python.langchain.com/docs/guides/embeddings/semantic_similarity

### Azure OpenAI Documentation
- Embedding Service: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/embeddings
- API Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
- Model Deployment: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource
- Service Limits: https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits

### Additional Resources
- NumPy Documentation: https://numpy.org/doc/stable/reference/routines.linalg.html
- Vector Mathematics: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Performance Optimization: https://python.langchain.com/docs/guides/deployment/optimization

### GitHub Repositories
- LangChain: https://github.com/langchain-ai/langchain
- NumPy: https://github.com/numpy/numpy
- Azure OpenAI SDK: https://github.com/Azure/azure-sdk-for-python

### Related Documentation
- Python Type Hints: https://docs.python.org/3/library/typing.html
- Pydantic Models: https://docs.pydantic.dev/latest/
- Environment Variables: https://python-dotenv.readthedocs.io/

## Best Practices

### 1. Reference Example Management
For effective semantic comparison:
```python
def create_balanced_examples() -> Dict[str, List[str]]:
    """Create diverse reference examples."""
    return {
        "Positive": [
            "Strong positive with enthusiasm",
            "Moderate positive with specifics",
            "Subtle positive with context"
        ],
        "Neutral": [
            "Pure factual statement",
            "Technical description",
            "Objective update"
        ],
        "Negative": [
            "Strong negative with details",
            "Moderate criticism",
            "Subtle disappointment"
        ]
    }
```

### 2. Similarity Processing
For reliable sentiment scoring:
```python
def process_similarities(
    similarities: Dict[str, List[float]],
    threshold: float = 0.3
) -> Dict[str, float]:
    """Process similarity scores with thresholds."""
    max_sims = {
        category: max(scores) if scores else 0.0
        for category, scores in similarities.items()
    }
    
    # Apply threshold
    filtered = {
        k: v for k, v in max_sims.items()
        if v >= threshold
    }
    
    # Default to neutral if no strong matches
    if not filtered:
        return {"neutral": 1.0}
    
    # Normalize remaining scores
    total = sum(filtered.values())
    return {k: v/total for k, v in filtered.items()}
```

Remember when implementing semantic sentiment analysis:
- Maintain diverse reference examples
- Handle vector operations efficiently
- Implement proper error handling
- Use appropriate thresholds
- Consider computational costs
- Cache embeddings when possible
- Monitor similarity distributions
- Test with various inputs
- Document edge cases
- Track analysis accuracy
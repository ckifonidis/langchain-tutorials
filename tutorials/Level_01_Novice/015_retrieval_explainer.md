# Understanding Retrieval in LangChain

Welcome to this comprehensive guide on implementing retrieval capabilities in LangChain! This tutorial shows how to build a document retrieval system using Azure OpenAI embeddings and FAISS vector store.

## Prerequisites

Before starting, you need to install the FAISS package:

```bash
# For CPU-only support:
pip install faiss-cpu

# For GPU support (with CUDA):
pip install faiss-gpu
```

## Core Concepts

1. **Modern Document Retrieval**
   Think of this as a smart library system:
   
   - **Document Storage**: Documents are organized in FAISS (Facebook AI Similarity Search) for efficient retrieval.
   
   - **Embeddings**: Using Azure OpenAI's text-embedding-3-small model to convert text into numerical vectors.
   
   - **Authentication**: Direct API key authentication with Azure OpenAI services.
   
   - **Similarity Search**: Finding relevant documents by comparing their vector representations.

2. **Azure OpenAI Setup**
   Essential configuration:
   
   - **Embedding Endpoint**: Complete URL for the embeddings service endpoint
   - **API Key**: Your Azure OpenAI API key
   - **Deployment**: text-embedding-3-small-3 model deployment
   - **API Version**: Using preview version (2024-12-01-preview)

3. **FAISS Integration**
   Vector store capabilities:
   
   - **Vector Storage**: Efficient storage of document embeddings
   - **Fast Search**: Quick similarity matching using optimized algorithms
   - **Metadata Handling**: Document categorization and attributes
   - **CPU/GPU Support**: Can run on either CPU or GPU

## Implementation Breakdown

1. **Environment Setup**
   ```python
   # .env file configuration
   AZURE_EMBEDDING_ENDPOINT="https://your-resource.openai.azure.com/openai/deployments/text-embedding-3-small-3/embeddings"
   AZURE_API_KEY="your-api-key"
   AZURE_DEPLOYMENT="text-embedding-3-small-3"
   ```
   
   This provides:
   - Complete embeddings endpoint
   - Secure API key storage
   - Model deployment name

2. **Azure OpenAI Client**
   ```python
   def get_embeddings_client() -> AzureOpenAI:
       """Create an Azure OpenAI client for embeddings."""
       client = AzureOpenAI(
           api_version="2024-12-01-preview",
           azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
           api_key=os.getenv("AZURE_API_KEY")
       )
       return client
   ```
   
   Features:
   - Preview API version
   - Direct endpoint configuration
   - Secure API key usage

3. **Document Embedding**
   ```python
   def embed_documents(texts: List[str]) -> List[List[float]]:
       """Generate embeddings for a list of texts."""
       client = get_embeddings_client()
       response = client.embeddings.create(
           input=texts,
           model=os.getenv("AZURE_DEPLOYMENT")
       )
       return [item.embedding for item in response.data]
   ```
   
   Key aspects:
   - Batch text processing
   - Model deployment usage
   - Vector extraction

4. **FAISS Store Creation**
   ```python
   # Extract texts and compute embeddings
   texts = [doc.page_content for doc in documents]
   embeddings = embed_documents(texts)
   
   # Create FAISS index
   index = FAISS.from_embeddings(
       embedding=embed_documents,                    # Function for future queries
       text_embeddings=list(zip(texts, embeddings)), # Pre-computed embeddings
       metadatas=[doc.metadata for doc in documents]
   )
   ```
   
   Important points:
   - Efficient indexing
   - Precomputed embeddings
   - Metadata association
   - Future query handling

## Best Practices

1. **Environment Variables**
   ```python
   # Required variables
   required_vars = [
       "AZURE_EMBEDDING_ENDPOINT",
       "AZURE_API_KEY",
       "AZURE_DEPLOYMENT"
   ]
   
   # Validation
   missing_vars = [var for var in required_vars if not os.getenv(var)]
   if missing_vars:
       raise ValueError(f"Missing variables: {', '.join(missing_vars)}")
   ```

2. **Error Handling**
   ```python
   try:
       embeddings = embed_documents(texts)
   except Exception as e:
       print(f"Error generating embeddings: {str(e)}")
       raise
   ```

3. **Document Structure**
   ```python
   doc = Document(
       page_content="Clear content here",
       metadata={
           "category": "topic",
           "difficulty": "level"
       }
   )
   ```

## Performance Considerations

1. **FAISS Options**
   - CPU vs GPU: Choose based on your hardware and requirements
   - Index Types: Different index types for different scale needs
   - Memory Usage: Monitor memory consumption with large document sets

2. **Batch Processing**
   ```python
   # Process documents in batches if needed
   def process_large_document_set(documents: List[Document], batch_size: int = 100):
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i + batch_size]
           texts = [doc.page_content for doc in batch]
           embeddings = embed_documents(texts)
           # Process batch...
   ```

## Resources

1. **Azure OpenAI**
   - **API Guide**: https://learn.microsoft.com/azure/ai-services/openai/reference
   - **Embeddings Models**: https://learn.microsoft.com/azure/ai-services/openai/concepts/models#embeddings-models

2. **FAISS**
   - **Documentation**: https://faiss.ai/
   - **Installation Guide**: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
   - **Python Interface**: https://faiss.ai/python_reference.html

## Example Output

When running `python 015_retrieval.py`, you'll see:

```
Demonstrating LangChain Retrieval...

Creating document store...
Document store created with sample documents.

Example 1: Basic Similarity Search
--------------------------------------------------
Query: What is machine learning?

Results:
1. Document: Machine learning is a subset of artificial intelligence that enables systems to learn from data.
   Score: 0.8523
   Category: ai
   Difficulty: intermediate

2. Document: Deep learning is part of machine learning based on artificial neural networks.
   Score: 0.7845
   Category: ai
   Difficulty: advanced
==================================================
```

## Real-World Applications

1. **Search Systems**
   - Semantic document search
   - Knowledge base queries
   - Content recommendation

2. **Content Organization**
   - Automated categorization
   - Similar content grouping
   - Document clustering

3. **Research Tools**
   - Academic paper analysis
   - Similar research finding
   - Literature review

Remember: 
- Install appropriate FAISS version (CPU/GPU)
- Secure your API keys
- Monitor embedding costs
- Consider caching for frequently accessed embeddings
- Test with various document types and sizes
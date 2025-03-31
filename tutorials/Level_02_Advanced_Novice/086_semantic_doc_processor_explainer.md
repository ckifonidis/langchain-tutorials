# Semantic Document Processor with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated document processing system by combining three key LangChain v3 concepts:
1. Document Loaders: Multi-format document handling
2. Text Splitters: Intelligent chunking strategies
3. Retrievers: Hybrid semantic and keyword search

The system provides efficient document processing with combined search capabilities.

### Real-World Application Value
- Format flexibility
- Smart chunking
- Hybrid search
- Score weighting
- Source tracking

### System Architecture Overview
```
Documents → Loaders → Text Splitters → Vector Store   → Semantic Search   → Combined Results
                                    ↘ BM25 Store → Keyword Search  ↗
```

## Core LangChain Concepts

### 1. Document Loading
```python
# Initialize document loaders
self.loaders = {
    DocumentType.TEXT: TextLoader,
    DocumentType.CSV: CSVLoader,
    DocumentType.JSON: JSONLoader,
    DocumentType.PDF: UnstructuredPDFLoader
}

def load_document(self, file_path: str, doc_type: DocumentType) -> List[Document]:
    loader_class = self.loaders.get(doc_type)
    if not loader_class:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    loader = loader_class(file_path)
    docs = loader.load()
    
    # Add source metadata
    for doc in docs:
        doc.metadata["source"] = os.path.basename(file_path)
    
    return docs
```

Features:
- Multiple formats
- Source tracking
- Error handling
- Metadata enrichment

### 2. Text Splitting
```python
# Initialize text splitters
self.splitters = {
    ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ),
    ChunkingStrategy.CHARACTER: CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ),
    ChunkingStrategy.TOKEN: TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
}

def process_documents(self, docs: List[Document], strategy: ChunkingStrategy):
    splitter = self.splitters.get(strategy)
    chunks = splitter.split_documents(docs)
    self.documents.extend(chunks)
    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
    self.bm25_retriever = BM25Retriever.from_documents(chunks)
```

Benefits:
- Multiple strategies
- Size control
- Overlap handling
- Index creation

### 3. Hybrid Search
```python
def search_documents(self, query: str) -> List[SearchResult]:
    # Get semantic results
    semantic_results = self.vector_store.similarity_search_with_score(
        query,
        k=3
    )
    
    # Get keyword results
    keyword_results = self.bm25_retriever.get_relevant_documents(query)[:3]
    
    # Combine with weights
    for doc, score in semantic_results:
        search_results.append(
            SearchResult(
                content=doc.page_content,
                score=0.7 * (1.0 - score),  # Semantic weight
                metadata=doc.metadata,
                source=doc.metadata.get("source", "unknown")
            )
        )
```

Advantages:
- Combined search
- Score weighting
- Result ranking
- Source tracking

## Implementation Components

### 1. Document Processing
```python
class DocumentProcessor:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )
        
        self.splitters = {...}
        self.loaders = {...}
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
```

Key elements:
- Azure integration
- Format support
- Strategy selection
- Storage management

### 2. Result Management
```python
class SearchResult(BaseModel):
    content: str = Field(description="Matched content")
    score: float = Field(description="Relevance score")
    metadata: Dict = Field(description="Document metadata")
    source: str = Field(description="Source document")

# Score and rank results
search_results.sort(key=lambda x: x.score, reverse=True)
return search_results[:5]  # Return top 5
```

Features:
- Type validation
- Score tracking
- Metadata handling
- Source tracking

### 3. Search Implementation
```python
def search_documents(self, query: str) -> List[SearchResult]:
    # Get results from both retrievers
    semantic_results = self.vector_store.similarity_search_with_score(...)
    keyword_results = self.bm25_retriever.get_relevant_documents(...)
    
    # Combine and normalize scores
    for doc, score in semantic_results:
        score = 0.7 * (1.0 - score)  # Convert distance to similarity
    
    for i, doc in enumerate(keyword_results):
        score = 0.3 * (1.0 - (i / len(keyword_results)))
```

Capabilities:
- Dual search
- Score normalization
- Result merging
- Rank preservation

## Advanced Features

### 1. Score Normalization
```python
# Semantic scores (70% weight)
score = 0.7 * (1.0 - score)  # Convert distance to similarity

# Keyword scores (30% weight)
score = 0.3 * (1.0 - (i / len(keyword_results)))
```

Implementation:
- Weight balancing
- Distance conversion
- Position scoring
- Range normalization

### 2. Source Tracking
```python
# Add source during loading
doc.metadata["source"] = os.path.basename(file_path)

# Include in results
SearchResult(
    content=doc.page_content,
    score=score,
    metadata=doc.metadata,
    source=doc.metadata.get("source", "unknown")
)
```

Features:
- Metadata enrichment
- Source preservation
- Error handling
- Clear tracking

### 3. Error Management
```python
try:
    loader_class = self.loaders.get(doc_type)
    if not loader_class:
        raise ValueError(f"Unsupported document type: {doc_type}")
    
    loader = loader_class(file_path)
    docs = loader.load()
except Exception as e:
    raise ValueError(f"Error loading document: {str(e)}")
```

Strategies:
- Type validation
- Clear messages
- Error wrapping
- Safe defaults

## Expected Output

### 1. Document Processing
```text
Processing Documents:
----------------------------------------
Loading: report.txt
Created 1 chunks
Loading: policy.txt
Created 1 chunks
Loading: email.txt
Created 1 chunks

Total Documents: 3
```

### 2. Search Results
```text
Search: security requirements
----------------------------------------
Result 1
Score: 0.92
Source: policy.txt
Content: Corporate Security Policy
1. Data Protection...
```

## Best Practices

### 1. Document Handling
- Format validation
- Source tracking
- Metadata enrichment
- Error checking

### 2. Search Implementation
- Combined methods
- Score normalization
- Result ranking
- Error handling

### 3. Result Management
- Type safety
- Score tracking
- Source tracking
- Clear formatting

## References

### 1. LangChain Core Concepts
- [Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

### 2. Implementation Guides
- [Azure OpenAI](https://python.langchain.com/docs/integrations/text_embedding/azure_openai)
- [FAISS Store](https://python.langchain.com/docs/integrations/vectorstores/faiss)
- [BM25 Search](https://python.langchain.com/docs/integrations/retrievers/bm25)

### 3. Additional Resources
- [Metadata Guide](https://python.langchain.com/docs/modules/data_connection/document_loaders/metadata)
- [Score Normalization](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/normalize_scores)
- [Error Handling](https://python.langchain.com/docs/guides/debugging)
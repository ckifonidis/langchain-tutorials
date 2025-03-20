# Understanding Text Splitters in LangChain

Welcome to this comprehensive guide on text splitting in LangChain! Text splitting is crucial for handling large documents and preparing text for processing with language models. We'll explore different methods for breaking down text into manageable chunks.

## Core Concepts

1. **What are Text Splitters?**
   Think of text splitters like a smart document divider:
   
   - **Chunking**: Breaking long texts into smaller, manageable pieces
   - **Boundaries**: Finding natural places to split text
   - **Size Control**: Managing chunk sizes effectively
   - **Metadata**: Tracking information about each chunk

2. **Available Splitters**
   LangChain provides three main splitter types:
   
   ```python
   from langchain.text_splitter import (
       RecursiveCharacterTextSplitter,
       CharacterTextSplitter,
       TokenTextSplitter
   )
   ```

   Each serves a different purpose:
   - **RecursiveCharacterTextSplitter**: Smart splitting using multiple separators
   - **CharacterTextSplitter**: Simple splitting on specific characters
   - **TokenTextSplitter**: Splitting based on token count (no overlap)

3. **Chunk Metadata**
   Each chunk maintains important information:
   ```python
   class TextChunk(BaseModel):
       content: str = Field(description="The content of the text chunk")
       chunk_index: int = Field(description="Index of the chunk in sequence")
       metadata: Dict[str, Any] = Field(description="Metadata about the chunk")
   ```

## Implementation Breakdown

1. **Recursive Character Splitting**
   ```python
   def split_text_recursive(text: str, chunk_size: int = 100, chunk_overlap: int = 20):
       splitter = RecursiveCharacterTextSplitter(
           chunk_size=chunk_size,
           chunk_overlap=chunk_overlap,
           length_function=len,
           separators=["\n\n", "\n", ".", " ", ""]
       )
       
       documents = splitter.create_documents([text])
       
       # Track metadata for each chunk
       chunks = []
       for i, doc in enumerate(documents):
           chunk = TextChunk(
               content=doc.page_content,
               chunk_index=i,
               metadata={
                   "length": len(doc.page_content),
                   "start_char": i * (chunk_size - chunk_overlap)
               }
           )
           chunks.append(chunk)
   ```
   
   Features:
   - Multiple separator levels
   - Configurable overlap
   - Position tracking
   - Length metadata

2. **Character Splitting**
   ```python
   def split_text_character(text: str, chunk_size: int = 100, separator: str = "\n"):
       splitter = CharacterTextSplitter(
           chunk_size=chunk_size,
           separator=separator
       )
       
       documents = splitter.create_documents([text])
       
       chunks = []
       for i, doc in enumerate(documents):
           chunk = TextChunk(
               content=doc.page_content,
               chunk_index=i,
               metadata={
                   "length": len(doc.page_content),
                   "separator": separator
               }
           )
           chunks.append(chunk)
   ```
   
   Benefits:
   - Simple implementation
   - Predictable splits
   - Separator tracking

3. **Token-Based Splitting**
   ```python
   def split_text_token(text: str, chunk_size: int = 50):
       # Note: chunk_overlap=0 for token-based splitting
       splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
       
       documents = splitter.create_documents([text])
       
       chunks = []
       for i, doc in enumerate(documents):
           chunk = TextChunk(
               content=doc.page_content,
               chunk_index=i,
               metadata={
                   "length": len(doc.page_content),
                   "tokens": chunk_size
               }
           )
           chunks.append(chunk)
   ```
   
   Key points:
   - Token-aware splitting
   - No overlap configuration
   - Token count tracking

## Best Practices

1. **Choosing the Right Splitter**
   - Use RecursiveCharacterTextSplitter for general text
   - Use CharacterTextSplitter for simple, structured text
   - Use TokenTextSplitter for model-specific requirements

2. **Configuration Guidelines**
   ```python
   # For recursive splitting
   recursive_chunks = split_text_recursive(
       text,
       chunk_size=200,     # Larger chunks for context
       chunk_overlap=50    # Overlap for continuity
   )
   
   # For character splitting
   char_chunks = split_text_character(
       text,
       chunk_size=200,
       separator="\n"      # Natural paragraph breaks
   )
   
   # For token splitting
   token_chunks = split_text_token(
       text,
       chunk_size=50      # Model-appropriate token size
   )
   ```

3. **Metadata Usage**
   ```python
   def analyze_chunks(chunks: List[TextChunk]):
       for chunk in chunks:
           print(f"Chunk {chunk.chunk_index}:")
           print(f"Length: {chunk.metadata['length']}")
           if 'start_char' in chunk.metadata:
               print(f"Starts at: {chunk.metadata['start_char']}")
   ```

## Example Output

When running `python 017_text_splitters.py`, you'll see:

```
Demonstrating LangChain Text Splitters...

Original Text Length: 1245

Example 1: Recursive Character Splitting
--------------------------------------------------
Number of chunks: 4
Chunk 1:
Length: 198
Content: Machine Learning and Artificial Intelligence...

Example 2: Simple Character Splitting
--------------------------------------------------
Number of chunks: 5
Chunk 1:
Length: 180
Content: Machine Learning and Artificial Intelligence...

Example 3: Token-based Splitting
--------------------------------------------------
Number of chunks: 7
Chunk 1:
Length: 156
Content: Machine Learning and Artificial Intelligence...
```

## Real-World Applications

1. **Document Processing**
   - Long document splitting
   - Chapter separation
   - Section analysis

2. **Model Input Preparation**
   - Token-aware chunking
   - Context window management
   - Batch processing

3. **Content Analysis**
   - Section-by-section analysis
   - Topic segmentation
   - Content classification

## Resources

1. **Documentation**
   - **Text Splitters**: https://python.langchain.com/docs/concepts/text_splitters/
   - **Token Splitting**: https://python.langchain.com/docs/how_to/split_by_token/
   - **Examples**: https://python.langchain.com/docs/how_to/#text-splitters

2. **Additional Resources**
   - **Custom Splitters**: https://python.langchain.com/docs/how_to/#text-splitters
   - **Best Practices**: https://python.langchain.com/docs/how_to/semantic-chunker/
   
                         https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

Remember: 
- Choose splitters based on your needs
- Configure chunk sizes appropriately
- Use metadata for tracking
- Test with representative content
- Monitor chunk distributions
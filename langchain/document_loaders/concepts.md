# Document Loaders in LangChain

## Core Concepts

Document loaders in LangChain are designed to:

1. Basic Functionality
   - Load documents into standard Document format
   - Support various data sources
   - Handle multiple file types

   ```python
   from langchain.document_loaders import TextLoader
   from langchain.schema import Document
   
   # Basic text file loading
   loader = TextLoader("path/to/file.txt")
   documents = loader.load()
   
   # Create custom document
   doc = Document(
       page_content="This is a sample document",
       metadata={"source": "custom", "date": "2025-03-16"}
   )
   ```

2. Integration Support
   - Hundreds of data source integrations
   - Standardized loading interface
   - Customizable loading behavior

   ```python
   from langchain.document_loaders import (
       PyPDFLoader,
       WebBaseLoader,
       NotionDirectoryLoader
   )
   
   # PDF loading
   pdf_loader = PyPDFLoader("document.pdf")
   pdf_docs = pdf_loader.load()
   
   # Web page loading
   web_loader = WebBaseLoader("https://example.com")
   web_docs = web_loader.load()
   ```

## Implementation Types

1. Standard Loaders
   - DirectoryLoader for file system
   - File-specific loaders
   - Web-based loaders

   ```python
   from langchain.document_loaders import DirectoryLoader
   
   # Load all text files in a directory
   directory_loader = DirectoryLoader(
       "path/to/directory",
       glob="**/*.txt",
       show_progress=True
   )
   docs = directory_loader.load()
   ```

2. Custom Loaders
   - Custom implementation support
   - Line-by-line processing
   - Specialized format handling

   ```python
   from langchain.document_loaders.base import BaseLoader
   
   class CustomLoader(BaseLoader):
       def __init__(self, file_path: str):
           self.file_path = file_path
           
       def load(self) -> List[Document]:
           with open(self.file_path, 'r') as file:
               text = file.read()
           metadata = {"source": self.file_path}
           return [Document(page_content=text, metadata=metadata)]
   ```

## Key Features

1. Document Processing
   - Standard format conversion
   - Metadata handling
   - Content extraction

   ```python
   from langchain.document_loaders import JSONLoader
   import json
   
   # Custom JSON loading with metadata
   loader = JSONLoader(
       file_path="data.json",
       jq_schema=".content",
       metadata_func=lambda meta: {
           "title": meta.get("title"),
           "date": meta.get("date")
       }
   )
   ```

2. Loading Capabilities
   - Batch processing
   - Directory traversal
   - Format detection

   ```python
   from langchain.document_loaders import UnstructuredFileLoader
   
   # Automatic format detection and loading
   loader = UnstructuredFileLoader(
       "document.docx",
       mode="elements",
       strategy="fast"
   )
   elements = loader.load()
   ```

## Best Practices

1. Loader Selection:
   - Choose appropriate loader type
   - Consider data source format
   - Handle specific requirements

   ```python
   def get_appropriate_loader(file_path: str):
       if file_path.endswith('.pdf'):
           return PyPDFLoader(file_path)
       elif file_path.endswith('.txt'):
           return TextLoader(file_path)
       elif file_path.endswith('.html'):
           return WebBaseLoader(file_path)
       else:
           return UnstructuredFileLoader(file_path)
   ```

2. Implementation Strategy:
   - Proper error handling
   - Resource management
   - Performance optimization

## Resources

Documentation Links:
- [Document Loaders Concepts](https://python.langchain.com/docs/concepts/document_loaders/)
- [Document Loaders Integration](https://python.langchain.com/docs/integrations/document_loaders/)
- [Directory Loading Guide](https://python.langchain.com/docs/how_to/document_loader_directory/)
- [Custom Loader Guide](https://python.langchain.com/docs/how_to/document_loader_custom/)

## Implementation Considerations

1. Format Support:
   - File type compatibility
   - Content extraction methods
   - Metadata handling

   ```python
   from langchain.document_loaders import CSVLoader
   
   # CSV loading with custom handling
   loader = CSVLoader(
       file_path="data.csv",
       csv_args={
           "delimiter": ",",
           "quotechar": '"',
           "fieldnames": ["title", "content"]
       }
   )
   ```

2. Performance:
   - Batch processing efficiency
   - Memory management
   - Loading optimization

   ```python
   from typing import List, Generator
   
   def batch_load_documents(docs: List[str], batch_size: int = 100) -> Generator:
       for i in range(0, len(docs), batch_size):
           batch = docs[i:i + batch_size]
           yield [get_appropriate_loader(doc).load() for doc in batch]
   ```

3. Error Handling:
   - File access issues
   - Format validation
   - Content extraction errors

## Common Use Cases

1. File System Loading:
   - Directory processing
   - Multiple file handling
   - Recursive traversal

   ```python
   from langchain.document_loaders import DirectoryLoader
   
   # Recursive directory loading
   loader = DirectoryLoader(
       "data/",
       glob="**/*.{txt,pdf,docx}",
       recursive=True,
       show_progress=True,
       use_multithreading=True
   )
   ```

2. Specialized Loading:
   - Custom format support
   - Line-by-line processing
   - Format-specific extraction

   ```python
   from langchain.document_loaders import LineByLineLoader
   
   # Line-by-line processing
   loader = LineByLineLoader(
       file_path="large_file.txt",
       encoding="utf-8",
       skip_rows=1
   )
   ```

3. Web Content:
   - URL processing
   - Web scraping
   - API integration

   ```python
   from langchain.document_loaders import SeleniumURLLoader
   
   # Web scraping with Selenium
   urls = ["https://example1.com", "https://example2.com"]
   loader = SeleniumURLLoader(
       urls=urls,
       continue_on_failure=True
   )
   docs = loader.load()
   ```

## Integration Patterns

1. Data Pipeline:
   - Content extraction
   - Format conversion
   - Document processing

   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   # Document processing pipeline
   def process_documents(file_path: str):
       loader = get_appropriate_loader(file_path)
       documents = loader.load()
       
       splitter = RecursiveCharacterTextSplitter(
           chunk_size=1000,
           chunk_overlap=200
       )
       return splitter.split_documents(documents)
   ```

2. Error Management:
   - Exception handling
   - Validation procedures
   - Recovery strategies

3. Resource Control:
   - Memory efficiency
   - Processing optimization
   - Batch management

## Advanced Features

1. Custom Processing:
   - Format-specific handling
   - Metadata extraction
   - Content transformation

   ```python
   from langchain.document_loaders import PyMuPDFLoader
   
   # Advanced PDF processing
   class EnhancedPDFLoader(PyMuPDFLoader):
       def extract_metadata(self, page):
           return {
               "page_number": page.number,
               "rotation": page.rotation,
               "xrefs": page.get_xrefs()
           }
   ```

2. Pipeline Integration:
   - Document processing chains
   - Format conversion flows
   - Content enhancement
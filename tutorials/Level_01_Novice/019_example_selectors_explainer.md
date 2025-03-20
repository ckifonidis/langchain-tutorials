# Understanding Example Selectors in LangChain

Welcome to this comprehensive guide on using example selectors in LangChain! Example selectors help you choose the most relevant examples for few-shot prompting dynamically. This tutorial will help you understand and implement different selection strategies.

## Core Concepts

1. **What are Example Selectors?**
   Think of example selectors as smart librarians:
   
   - **Dynamic Selection**: Choose relevant examples based on your current needs
   - **Multiple Strategies**: Different ways to pick examples (semantic or length-based)
   - **Efficient Usage**: Optimize example selection for your use case
   - **Smart Filtering**: Automatically choose the most appropriate examples

2. **Required Configuration**
   ```python
   # Environment Variables
   AZURE_OPENAI_API_KEY="your-api-key"
   AZURE_OPENAI_ENDPOINT="your-endpoint"
   AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
   AZURE_EMBEDDING_ENDPOINT="your-embedding-endpoint"
   AZURE_API_KEY="your-api-key"
   AZURE_DEPLOYMENT="text-embedding-3-small-3"
   ```

3. **Example Structure**
   ```python
   examples = [
       {
           "question": "What is Python?",
           "answer": "Python is a high-level programming language...",
           "difficulty": "beginner",
           "category": "programming"
       }
   ]
   ```

## Implementation Breakdown

1. **Semantic Similarity Selector**
   ```python
   def create_semantic_selector(examples: List[Dict[str, str]], k: int = 2):
       # Initialize embeddings with correct configuration
       embeddings = AzureOpenAIEmbeddings(
           azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
           api_key=os.getenv("AZURE_API_KEY"),
           deployment=os.getenv("AZURE_DEPLOYMENT")
       )
       
       # Create selector using FAISS
       selector = SemanticSimilarityExampleSelector.from_examples(
           examples,
           embeddings,
           FAISS,
           k=k,
           input_keys=["question"]
       )
       
       return selector
   ```
   
   Features:
   - Uses embeddings for similarity
   - FAISS vector store integration
   - Configurable number of examples
   - Semantic understanding

2. **Length-Based Selector**
   ```python
   def create_length_based_selector(examples: List[Dict[str, str]], 
                                  max_length: int = 1000):
       # Use dictionary format for example template
       example_prompt = {"template": "Q: {question}\nA: {answer}"}
       
       return LengthBasedExampleSelector(
           examples=examples,
           example_prompt=example_prompt,
           max_length=max_length,
           length_function=len,
           input_keys=["question", "answer"]
       )
   ```
   
   Key aspects:
   - Simple template format
   - Length control
   - Multiple input fields
   - No embeddings needed

3. **Example Pool Management**
   ```python
   def create_example_pool():
       return [
           {
               "question": "What is Python?",
               "answer": "Python is a high-level...",
               "difficulty": "beginner",
               "category": "programming"
           },
           # More examples with different categories and difficulties
       ]
   ```

## Best Practices

1. **Semantic Selection Setup**
   ```python
   semantic_selector = create_semantic_selector(examples)
   
   # Select examples based on input question
   selected = semantic_selector.select_examples({
       "question": "How do I start learning Python?"
   })
   ```

2. **Length-Based Selection**
   ```python
   length_selector = create_length_based_selector(
       examples,
       max_length=200  # Adjust based on needs
   )
   
   # Select examples based on length
   selected = length_selector.select_examples({})
   ```

3. **Error Handling**
   ```python
   try:
       selected_examples = selector.select_examples({"question": question})
   except Exception as e:
       print(f"Selection error: {str(e)}")
       # Implement fallback strategy
   ```

## Example Output

When running `python 019_example_selectors.py`, you'll see:

```
Demonstrating LangChain Example Selectors...

Example 1: Semantic Similarity Selection
--------------------------------------------------
Input Question: How do I start learning Python?

Selected Examples:
1. Q: What is Python?
   A: Python is a high-level, interpreted programming language...
2. Q: What is version control?
   A: Version control is a system...

Example 2: Length-Based Selection
--------------------------------------------------
Max Length: 100
Number of examples selected: 1

Selected Examples:
1. Q: What is Python?
   A: Python is a high-level programming language...
```

## Common Patterns

1. **Question-Answer Format**
   ```python
   example_prompt = {
       "template": "Q: {question}\nA: {answer}"
   }
   ```

2. **Dynamic Selection**
   ```python
   # Adjust selection based on context
   if context == "programming":
       k = 3  # More programming examples
   else:
       k = 2  # Standard number
   ```

## Resources

1. **Official Documentation**
   - **Example Selectors**: https://python.langchain.com/docs/how_to/example_selectors/
   - **Vector Stores**: https://python.langchain.com/docs/integrations/vectorstores/
   - **Azure OpenAI**: https://learn.microsoft.com/azure/ai-services/openai/reference

2. **Additional Resources**
   - **Embeddings Guide**: https://python.langchain.com/docs/integrations/text_embedding/
   - **Best Practices**: https://python.langchain.com/docs/concepts/prompt_templates/

## Real-World Applications

1. **Documentation Systems**
   - Relevant example selection
   - Context-aware responses
   - Dynamic help systems

2. **Educational Tools**
   - Adaptive learning
   - Example difficulty matching
   - Concept explanation

3. **Support Systems**
   - Similar case finding
   - Solution matching
   - Context-aware responses

Remember: 
- Choose appropriate selector type
- Configure embeddings correctly
- Maintain diverse examples
- Handle errors gracefully
- Monitor selection quality
- Test with various inputs
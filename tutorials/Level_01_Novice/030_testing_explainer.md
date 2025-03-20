# Understanding Testing in LangChain

Welcome to this comprehensive guide on testing in LangChain! Testing helps ensure your applications work correctly and reliably. This tutorial will help you understand how to implement effective testing strategies.

## Core Concepts

1. **What is Testing?**
   Think of testing as quality assurance that:
   
   - **Validates**: Ensures correct behavior
   - **Verifies**: Checks component integration
   - **Catches**: Identifies issues early
   - **Documents**: Shows expected behavior

2. **Key Components**
   ```python
   import pytest
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_openai import AzureChatOpenAI
   from langchain_core.output_parsers import StrOutputParser
   ```

3. **Testing Types**
   ```python
   class QueryResult(BaseModel):
       query: str = Field(description="Input query")
       response: str = Field(description="Model response")
       timestamp: datetime = Field(default_factory=datetime.now)
   ```

## Implementation Breakdown

1. **Unit Testing**
   ```python
   def test_prompt_template():
       prompt = ChatPromptTemplate.from_messages([
           ("system", "You are a helpful assistant."),
           ("human", "{question}")
       ])
       
       formatted = prompt.format_messages(
           question="What is Python?"
       )
       
       assert len(formatted) == 2
       assert formatted[0].content == "You are a helpful assistant."
   ```
   
   Features:
   - Component isolation
   - Input validation
   - Output verification
   - Error checking

2. **Integration Testing**
   ```python
   def test_chain_integration():
       # Create chain
       chain = create_qa_chain()
       
       # Test response
       response = chain.invoke({
           "question": "What is the capital of France?"
       })
       
       # Verify result
       assert "Paris" in response.lower()
       assert len(response) > 0
   ```
   
   Benefits:
   - Component interaction
   - End-to-end flow
   - Response validation
   - Error handling

3. **Fact Checking**
   ```python
   class FactChecker:
       def __init__(self):
           self.facts = {
               "capital_france": "Paris",
               "earth_satellite": "Moon"
           }
       
       def verify_fact(self, category: str, statement: str):
           known_fact = self.facts.get(category, "").lower()
           return known_fact in statement.lower()
   ```

## Best Practices

1. **Test Setup**
   ```python
   @pytest.fixture
   def chat_model():
       return AzureChatOpenAI(
           deployment=os.getenv("DEPLOYMENT_NAME"),
           temperature=0
       )
   
   @pytest.fixture
   def qa_chain(chat_model):
       prompt = ChatPromptTemplate.from_messages([
           ("system", "You are a helpful assistant."),
           ("human", "{question}")
       ])
       return prompt | chat_model | StrOutputParser()
   ```

2. **Error Testing**
   ```python
   def test_error_handling():
       with pytest.raises(ValueError):
           # Test missing required input
           chain.invoke({})
       
       with pytest.raises(KeyError):
           # Test invalid variable
           prompt.format(nonexistent_var="test")
   ```

3. **Response Validation**
   ```python
   def validate_response(response: str, criteria: Dict[str, Any]):
       validations = {
           "length": len(response) > criteria["min_length"],
           "relevance": any(kw in response.lower() 
                          for kw in criteria["keywords"]),
           "format": criteria["pattern"].match(response) is not None
       }
       return all(validations.values())
   ```

## Example Output

When running `python 030_testing.py`, you'll see:

```
Demonstrating LangChain Testing...

Example 1: Unit Testing Components
--------------------------------------------------
Testing Prompt Formatting:
Number of messages: 2
System message: You are a helpful assistant.
Human message: What is Python?

Example 2: Integration Testing
--------------------------------------------------
Question: What is the capital of France?
Response: Paris is the capital of France.
Fact Check: ✓
```

## Common Patterns

1. **Test Case Structure**
   ```python
   class TestQAChain:
       def test_basic_query(self, qa_chain):
           response = qa_chain.invoke({
               "question": "test question"
           })
           assert response is not None
       
       def test_complex_query(self, qa_chain):
           # Test more complex scenarios
           pass
   ```

2. **Mock Responses**
   ```python
   @pytest.fixture
   def mock_model():
       class MockModel:
           def invoke(self, messages):
               return "Mock response"
       return MockModel()
   ```

## Resources

1. **Official Documentation**
   - **Testing Guide**: https://python.langchain.com/docs/concepts/testing/
   - **Standard Tests**: https://python.langchain.com/docs/contributing/how_to/integrations/standard_tests/
   - **Langchain Testing API Reference**: https://python.langchain.com/api_reference/standard_tests/index.html

2. **Additional Resources**
   - **Pytest**: https://docs.pytest.org/
   - **Chat Model UnitTests**: https://python.langchain.com/api_reference/standard_tests/unit_tests/langchain_tests.unit_tests.chat_models.ChatModelUnitTests.html
   - **Embeddings UnitTests**: https://python.langchain.com/api_reference/standard_tests/unit_tests/langchain_tests.unit_tests.embeddings.EmbeddingsUnitTests.html
   - **Tools UnitTests**: https://python.langchain.com/api_reference/standard_tests/unit_tests/langchain_tests.unit_tests.tools.ToolsUnitTests.html

## Real-World Applications

1. **Quality Assurance**
   - Response validation
   - Format checking
   - Error handling
   - Performance testing

2. **Regression Testing**
   - Version updates
   - Configuration changes
   - Component updates
   - Integration testing

3. **Continuous Integration**
   - Automated testing
   - Pipeline integration
   - Deployment checks
   - Quality gates

Remember: 
- Test components individually
- Validate integrations
- Check error handling
- Monitor performance
- Document test cases
- Update tests regularly
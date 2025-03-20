# Understanding Output Parsers in LangChain

Welcome to this comprehensive guide on using output parsers in LangChain! Output parsers help you convert model responses into structured, usable data. This tutorial will help you understand different parsing methods and their applications.

## Core Concepts

1. **What are Output Parsers?**
   Think of output parsers as translators that:
   
   - **Structure Data**: Convert raw text to organized formats
   - **Validate Responses**: Ensure correct formatting
   - **Transform Output**: Convert between data types
   - **Handle Errors**: Manage parsing failures

2. **Essential Imports**
   ```python
   from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
   from langchain_core.output_parsers import (
       StrOutputParser,
       JsonOutputParser,
       PydanticOutputParser
   )
   ```

3. **Output Schemas**
   ```python
   class MovieReview(BaseModel):
       title: str = Field(description="Title of the movie")
       year: int = Field(description="Release year")
       rating: float = Field(description="Rating out of 10")
       pros: List[str] = Field(description="Positive aspects")
       cons: List[str] = Field(description="Negative aspects")
       summary: str = Field(description="Brief review")
   ```

## Implementation Breakdown

1. **String Parser**
   ```python
   simple_prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant."),
       ("human", "What is the capital of {country}?")
   ])
   
   string_chain = simple_prompt | model | StrOutputParser()
   response = string_chain.invoke({"country": "France"})
   ```
   
   Features:
   - Basic text handling
   - No special formatting
   - Direct string output
   - Simple integration

2. **JSON Parser with Escaped Templates**
   ```python
   json_prompt = ChatPromptTemplate.from_messages([
       ("system", """Provide reviews in JSON format:
       {{
           "title": "movie title",
           "year": release year,
           "rating": rating out of 10
       }}"""),  # Note the double braces for escaping
       ("human", "Review {movie_title}")
   ])
   
   json_chain = json_prompt | model | JsonOutputParser()
   ```
   
   Key points:
   - Double braces for escaping
   - Proper JSON structure
   - Clear formatting
   - Template variables

3. **Pydantic Parser with Schema**
   ```python
   # Create parser
   pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
   
   # Get schema string
   schema = MovieReview.schema_json(indent=2)
   
   # Create prompt with schema
   pydantic_prompt = ChatPromptTemplate.from_messages([
       ("system", f"{PYDANTIC_FORMAT_INSTRUCTIONS}"),
       ("human", """Provide a review for {movie_title}.
       Use this schema:
       {schema}""")
   ])
   ```
   
   Benefits:
   - Schema validation
   - Clear structure
   - Type checking
   - Format guidance

## Best Practices

1. **Schema Design**
   ```python
   class WeatherReport(BaseModel):
       location: str = Field(description="Location name")
       temperature: float = Field(description="Temperature in Celsius")
       conditions: str = Field(description="Weather conditions")
       last_updated: datetime = Field(description="Report timestamp")
   ```

2. **Template Escaping**
   ```python
   # Use double braces for JSON templates
   template = """Format as JSON:
   {{
       "key": "value",
       "number": 123
   }}"""
   ```

3. **Schema Integration**
   ```python
   # Include schema in prompt
   prompt = ChatPromptTemplate.from_messages([
       ("system", format_instructions),
       ("human", "Use schema:\n{schema}")
   ])
   
   # Invoke with schema
   response = chain.invoke({
       "input_data": data,
       "schema": model.schema_json()
   })
   ```

## Example Output

When running `python 023_output_parsers.py`, you'll see:

```
Demonstrating LangChain Output Parsers...

Example 1: Simple String Parser
--------------------------------------------------
Query: What is the capital of France?
Response: Paris is the capital of France.

Example 2: JSON Parser
--------------------------------------------------
Movie Review for: The Matrix
Response: {
    "title": "The Matrix",
    "year": 1999,
    "rating": 9.5,
    "summary": "Groundbreaking sci-fi masterpiece..."
}

Example 3: Pydantic Parser
--------------------------------------------------
Structured Movie Review for: The Dark Knight
Title: The Dark Knight
Year: 2008
Rating: 9.8/10
Pros:
- Outstanding performance by Heath Ledger
- Complex plot and character development
```

## Common Patterns

1. **Schema Generation**
   ```python
   # Get schema for prompt
   schema = model_class.schema_json(indent=2)
   
   # Include in format instructions
   instructions = f"{format_base}\nSchema: {schema}"
   ```

2. **Parser Chaining**
   ```python
   # Create processing chain
   chain = (
       prompt 
       | model 
       | parser
   )
   ```

## Resources

1. **Official Documentation**
   - **Output Parsers**: https://python.langchain.com/docs/concepts/output_parsers/
   - **Format Instructions**: https://python.langchain.com/docs/how_to/#output-parsers
   - **Pydantic Integration**: https://python.langchain.com/docs/how_to/output_parser_structured/

2. **Additional Resources**
   - **Pydantic v2**: https://docs.pydantic.dev/latest/
   - **JSON Schemas**: https://json-schema.org/understanding-json-schema/

## Real-World Applications

1. **Data Processing**
   - API responses
   - Data extraction
   - Format conversion

2. **Content Generation**
   - Structured content
   - Formatted documents
   - Data preparation

3. **Integration Tasks**
   - API integration
   - Database storage
   - Service communication

Remember: 
- Escape JSON templates
- Include schemas in prompts
- Use format instructions
- Handle parsing errors
- Validate outputs
- Test edge cases
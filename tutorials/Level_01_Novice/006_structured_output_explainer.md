# Understanding Structured Output in LangChain

Welcome to this comprehensive guide on using structured output in LangChain! This tutorial explains how to get structured, validated responses from language models using Pydantic models.

## Core Concepts

1. **What is Structured Output?**
   Think of structured output as a way to:
   - **Format**: Get consistent response structures
   - **Validate**: Ensure data types are correct
   - **Parse**: Convert raw responses to usable objects
   - **Standardize**: Maintain consistent output patterns

2. **Key Components**
   ```python
   from pydantic import BaseModel, Field
   from langchain_core.output_parsers import PydanticOutputParser
   from langchain_core.messages import HumanMessage, SystemMessage
   from langchain_core.prompts import ChatPromptTemplate
   ```

## Implementation Breakdown

1. **Defining the Schema**
   ```python
   class MovieReview(BaseModel):
       title: str = Field(description="The title of the movie")
       year: int = Field(description="The release year")
       rating: int = Field(description="Rating from 1-10", ge=1, le=10)
       review: str = Field(description="Brief review")
       tags: List[str] = Field(description="Genre or theme tags")
       director: Optional[str] = Field(
           description="Movie's director",
           default=None
       )
   
       model_config = {
           "json_schema_extra": {
               "examples": [{
                   "title": "The Matrix",
                   "year": 1999,
                   "rating": 9,
                   # ... more example data
               }]
           }
       }
   ```
   
   Key points:
   - Type annotations
   - Field descriptions
   - Validation rules
   - Example data
   - Optional fields

2. **Setting Up the Parser**
   ```python
   # Create parser from schema
   pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
   
   # Create system message with format instructions
   system_msg = SystemMessage(content="""
   You are a film critic. Respond with a JSON object that follows this schema:
   {
     "title": "movie title (string)",
     "year": release year (integer),
     "rating": rating (integer from 1-10),
     ...
   }
   """)
   ```
   
   Important aspects:
   - Schema association
   - Format instructions
   - Clear expectations
   - Response structure

3. **Creating the Chain**
   ```python
   # Build the chain
   prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
   chain = prompt | chat | pydantic_parser
   
   # Use the chain
   response = chain.invoke({"input": "Review the movie 'Inception'"})
   ```

## Example Usage

1. **Basic Review**
   ```python
   # Get a single review
   response = chain.invoke({"input": "Review 'Inception'"})
   print(f"Title: {response.title}")
   print(f"Rating: {response.rating}/10")
   print(f"Review: {response.review}")
   ```

2. **Multiple Reviews**
   ```python
   # Compare multiple movies
   for movie in ["The Matrix", "Blade Runner"]:
       response = chain.invoke({"input": f"Review '{movie}'"})
       print(f"\n{response.title} ({response.year})")
       print(f"Rating: {response.rating}/10")
   ```

## Best Practices

1. **Schema Design**
   ```python
   class MovieReview(BaseModel):
       # Use clear field names
       title: str = Field(description="...")
       # Add validation
       rating: int = Field(ge=1, le=10)
       # Provide defaults when appropriate
       director: Optional[str] = Field(default=None)
   ```

2. **System Messages**
   ```python
   system_msg = SystemMessage(content="""
   Ensure your response:
   - Uses the exact field names
   - Provides appropriate data types
   - Includes all required fields
   - Follows JSON format
   """)
   ```

3. **Error Handling**
   ```python
   try:
       response = chain.invoke(input_data)
   except Exception as e:
       print(f"Error parsing response: {str(e)}")
       # Handle the error appropriately
   ```

## Example Output

When running `python 006_structured_output.py`, you'll see:

```
Demonstrating LangChain Structured Output...

Example 1: Basic Movie Review
Structured Review for Inception:
Title: Inception
Year: 2010
Rating: 9/10
Review: Inception is a mind-bending thriller...
Tags: Science Fiction, Thriller, Action...
Director: Christopher Nolan

Example 2: Multiple Movie Reviews
Comparison of Sci-Fi Classics:

The Matrix (1999)
Rating: 9/10
Tags: sci-fi, action, cyberpunk
Director: The Wachowskis

Blade Runner (1982)
Rating: 8/10
Tags: sci-fi, neo-noir, dystopian
Director: Ridley Scott
```

## Common Patterns

1. **Field Validation**
   ```python
   rating: int = Field(
       description="Rating from 1-10",
       ge=1,  # Greater than or equal to 1
       le=10  # Less than or equal to 10
   )
   ```

2. **Optional Fields**
   ```python
   tags: Optional[List[str]] = Field(
       description="Optional tags",
       default_factory=list
   )
   ```

3. **Example Data**
   ```python
   model_config = {
       "json_schema_extra": {
           "examples": [
               # Provide clear examples
           ]
       }
   }
   ```

## Common Issues and Solutions

1. **Type Mismatches**
   ```python
   # Problem: Model returns string for year
   "year": "1999"
   
   # Solution: Add type conversion
   year: int = Field(description="Year as integer")
   ```

2. **Missing Fields**
   ```python
   # Problem: Required field missing
   # Solution: Add default or make optional
   director: Optional[str] = Field(default=None)
   ```

3. **Invalid Values**
   ```python
   # Problem: Rating out of range
   # Solution: Add validation
   rating: int = Field(ge=1, le=10)
   ```

## Resources

1. **Official Documentation**
   - **Main Guide**: https://python.langchain.com/docs/concepts/structured_outputs/
   - **Overview**: https://python.langchain.com/docs/concepts/structured_outputs/#overview
   - **Key Concepts**: https://python.langchain.com/docs/concepts/structured_outputs/#key-concepts
   - **Recommended Usage**: https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage
   - **Schema Definition**: https://python.langchain.com/docs/concepts/structured_outputs/#schema-definition
   - **Returning Structured Output**: https://python.langchain.com/docs/concepts/structured_outputs/#returning-structured-output
   - **JSON Mode**: https://python.langchain.com/docs/concepts/structured_outputs/#json-mode
   - **Structured Output Method**: https://python.langchain.com/docs/concepts/structured_outputs/#structured-output-method

Remember:
- Define clear schemas
- Provide examples
- Handle errors gracefully
- Validate inputs
- Document fields
- Test edge cases
# Understanding Prompt Templates in LangChain

Welcome to this comprehensive guide on using prompt templates in LangChain! Prompt templates help you create consistent, reusable, and dynamic prompts for language models. This tutorial will help you understand different types of templates and their applications.

## Core Concepts

1. **What are Prompt Templates?**
   Think of prompt templates as reusable forms:
   
   - **Variables**: Placeholders for dynamic content
   - **Structure**: Consistent prompt formatting
   - **Reusability**: Templates for common patterns
   - **Type Safety**: Validated inputs and outputs

2. **Types of Templates**
   ```python
   from langchain_core.prompts import (
       ChatPromptTemplate,
       HumanMessagePromptTemplate,
       SystemMessagePromptTemplate,
       MessagesPlaceholder
   )
   from langchain.prompts import PromptTemplate
   ```

3. **Input Schema**
   ```python
   class StoryInput(BaseModel):
       protagonist: str = Field(description="Main character")
       setting: str = Field(description="Story location")
       theme: str = Field(description="Main message")
   ```

## Implementation Breakdown

1. **Simple Prompt Template**
   ```python
   def create_simple_prompt():
       return PromptTemplate.from_template(
           "Tell me a short story about {protagonist} in {setting} "
           "that teaches us about {theme}."
       )
   
   # Usage
   prompt = create_simple_prompt()
   formatted = prompt.format(
       protagonist="a curious cat",
       setting="a bustling city",
       theme="curiosity"
   )
   ```
   
   Features:
   - Simple variable substitution
   - Single string template
   - Easy to understand
   - Quick to implement

2. **Structured Chat Prompt**
   ```python
   def create_structured_chat_prompt():
       return ChatPromptTemplate.from_messages([
           ("system", "You are a creative storyteller..."),
           ("human", "I want a story with these elements:"),
           ("human", "Protagonist: {protagonist}"),
           ("human", "Setting: {setting}"),
           ("human", "Theme: {theme}")
       ])
   ```
   
   Benefits:
   - Role-based messages
   - Clear structure
   - Multiple components
   - Better context control

3. **Dynamic Chat Prompt**
   ```python
   def create_dynamic_chat_prompt(previous_stories: List[str] = None):
       messages = [
           SystemMessagePromptTemplate.from_template(
               "You are a creative storyteller..."
           ),
           MessagesPlaceholder(variable_name="story_history"),
           HumanMessagePromptTemplate.from_template(
               """Write a new story using:
               Protagonist: {protagonist}
               Setting: {setting}
               Theme: {theme}"""
           )
       ]
       return ChatPromptTemplate.from_messages(messages)
   ```

## Best Practices

1. **Template Organization**
   ```python
   # Group related templates
   class StoryTemplates:
       simple = PromptTemplate.from_template(...)
       structured = ChatPromptTemplate.from_messages([...])
       dynamic = create_dynamic_chat_prompt()
   ```

2. **Input Validation**
   ```python
   class StoryInput(BaseModel):
       protagonist: str = Field(description="Main character")
       setting: str = Field(description="Story location")
       theme: str = Field(description="Main message")
   
   # Use with template
   story_input = StoryInput(
       protagonist="a curious cat",
       setting="a bustling city",
       theme="curiosity"
   )
   ```

3. **Chain Construction**
   ```python
   # Create processing chain
   chain = (
       prompt 
       | model 
       | StrOutputParser()
   )
   
   # Use the chain
   response = chain.invoke(story_input.model_dump())
   ```

## Example Output

When running `python 022_prompt_templates.py`, you'll see:

```
Demonstrating LangChain Prompt Templates...

Example 1: Simple Prompt Template
--------------------------------------------------
Formatted Prompt:
Tell me a short story about a curious cat in a bustling city that teaches us about curiosity and discovery.

Generated Story:
In the heart of a bustling city, a curious cat named Whiskers...

Example 2: Structured Chat Prompt
--------------------------------------------------
System: You are a creative storyteller who crafts engaging tales.
Human: I want a story with these elements:
Human: Protagonist: a curious cat
Human: Setting: a bustling city
Human: Theme: curiosity and discovery
```

## Common Patterns

1. **Template Reuse**
   ```python
   base_template = "Consider {input}"
   templates = {
       "story": base_template + " and write a story",
       "summary": base_template + " and write a summary",
       "analysis": base_template + " and provide analysis"
   }
   ```

2. **Dynamic Assembly**
   ```python
   def build_prompt(components: List[str]):
       return ChatPromptTemplate.from_messages([
           ("system", system_message),
           *[(role, text) for role, text in components]
       ])
   ```

## Resources

1. **Official Documentation**
   - **Prompts Guide**: https://python.langchain.com/docs/concepts/prompt_templates/
   - **Templates**: https://python.langchain.com/docs/how_to/#prompt-templates
   - **Langchain Messages**: https://python.langchain.com/docs/concepts/messages/#langchain-messages

2. **Additional Resources**
   - **Examples**: https://python.langchain.com/docs/concepts/messages/#langchain-messages
   - **Best Practices**: https://python.langchain.com/docs/how_to/

## Real-World Applications

1. **Content Generation**
   - Story creation
   - Article writing
   - Description generation

2. **Conversation Flows**
   - Chat bots
   - Customer service
   - Virtual assistants

3. **Data Processing**
   - Analysis requests
   - Format conversion
   - Summary generation

Remember: 
- Keep templates modular
- Validate inputs
- Use appropriate template types
- Include clear context
- Handle errors gracefully
- Document template purposes
# Adaptive Learning System with LangChain: Complete Guide

## Introduction

This implementation demonstrates a sophisticated learning system by combining three key LangChain v3 concepts:
1. Example Selectors: Smart selection of relevant learning examples
2. LangChain Expression Language (LCEL): Dynamic chain composition
3. Prompt Templates: Structured and dynamic prompting

The system provides personalized learning experiences through intelligent example selection and structured feedback.

### Real-World Application Value
- Dynamic example selection
- Personalized feedback
- Structured evaluation
- Error resilience
- Type safety
- Educational adaptability

### System Architecture Overview
```
Student Input → AdaptiveLearningSystem → Evaluation Chain
                ↓                      ↓
         Example Selection     Structured Response
                ↓                      ↓
          Topic Matching        Detailed Feedback
```

## Core LangChain Concepts

### 1. Example Selectors
```python
self.topic_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2  # Maximum number of examples
)

self.few_shot_prompt = FewShotPromptTemplate(
    example_selector=self.topic_selector,
    example_prompt=example_prompt,
    prefix="Here are some similar examples:\n\n",
    suffix="\nNow try this question: {input}",
    input_variables=["input"]
)
```

Features:
- Length-based selection
- Dynamic filtering
- Context relevance
- Maximum example control

### 2. LCEL Composition
```python
self.learning_chain = (
    RunnablePassthrough.assign(examples=get_examples)
    | format_feedback
    | self.feedback_prompt
    | self.llm
)
```

Benefits:
- Clean composition
- Data transformation
- Error handling
- Async support

### 3. Prompt Templates
```python
feedback_template = """
Evaluate the student's answer for:
Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}
Similar Examples: {examples}

Provide your evaluation as a valid JSON object with these fields:
{{
    "is_correct": {is_correct},
    "feedback": "detailed feedback string",
    "next_steps": ["step1", "step2", ...],
    "relevant_examples": ["example1", "example2", ...]
}}"""
```

Advantages:
- Structured prompts
- Dynamic variables
- JSON formatting
- Clear instructions

## Implementation Components

### 1. Learning Models
```python
class LearningExample(BaseModel):
    topic: TopicArea
    level: DifficultyLevel
    question: str
    answer: str
    explanation: str
    hints: List[str]
```

Key elements:
- Type validation
- Required fields
- Field descriptions
- Clear structure

### 2. Response Handling
```python
class LearningResponse(BaseModel):
    is_correct: bool
    feedback: str
    next_steps: List[str]
    relevant_examples: List[str]
```

Features:
- Standardized responses
- Type safety
- Required feedback
- Action guidance

### 3. System Integration
```python
async def evaluate_answer(
    self,
    question: str,
    correct_answer: str,
    student_answer: str,
    topic: TopicArea,
    level: DifficultyLevel
) -> LearningResponse:
    """Evaluate student's answer with context."""
```

Capabilities:
- Async processing
- Type hints
- Error handling
- Context awareness

## Advanced Features

### 1. Error Management
```python
try:
    json_data = extract_json(result.content)
    return LearningResponse.model_validate(json_data)
except Exception as e:
    return LearningResponse(
        is_correct=False,
        feedback=f"Error evaluating answer: {str(e)}",
        next_steps=["Contact system administrator"],
        relevant_examples=[]
    )
```

Implementation:
- Clean error handling
- Graceful degradation
- User feedback
- System stability

### 2. Example Database
```python
LEARNING_EXAMPLES = {
    TopicArea.PYTHON: {
        DifficultyLevel.BEGINNER: [
            LearningExample(...),
            LearningExample(...)
        ],
        DifficultyLevel.INTERMEDIATE: [
            LearningExample(...)
        ]
    }
}
```

Features:
- Organized structure
- Topic categorization
- Difficulty levels
- Extensible design

### 3. JSON Processing
```python
def extract_json(text: str) -> dict:
    """Extract JSON from text, handling code blocks."""
    match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
```

Strategies:
- Regex matching
- Error handling
- Clean extraction
- Format validation

## Expected Output

### 1. Successful Evaluation
```text
Topic: python
Level: beginner
Question: What is the output of: print('Hello ' + 'World')?
Student Answer: HelloWorld
----------------------------------------
Correct: False
Feedback: [Detailed feedback with explanation]
Next Steps:
- Review string concatenation
- Practice with examples
```

### 2. Error Case
```text
Topic: sql
Level: beginner
Question: Write a query to select all columns
----------------------------------------
Correct: False
Feedback: Error processing request
Next Steps:
- Contact administrator
```

## Best Practices

### 1. Error Handling
- Graceful degradation
- User feedback
- System recovery
- Error logging

### 2. Response Processing
- Type validation
- JSON parsing
- Clean formatting
- Clear feedback

### 3. Example Management
- Topic organization
- Difficulty levels
- Clear structure
- Easy extension

## References

### 1. LangChain Core Concepts
- [Example Selectors](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)
- [Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
- [LCEL](https://python.langchain.com/docs/expression_language)

### 2. Implementation Guides
- [Async Operations](https://python.langchain.com/docs/expression_language/cookbook/async_parallel)
- [Error Handling](https://python.langchain.com/docs/guides/debugging)
- [JSON Parsing](https://python.langchain.com/docs/modules/model_io/output_parsers/)

### 3. Additional Resources
- [Type Hints](https://python.langchain.com/docs/guides/safety)
- [Pydantic Models](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)
- [Chain Composition](https://python.langchain.com/docs/expression_language/cookbook)
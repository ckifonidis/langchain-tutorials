# Understanding the Code Review System in LangChain

Welcome to this comprehensive guide on building a sophisticated Code Review System using LangChain! This example demonstrates how to combine chains with evaluation capabilities to create an intelligent system that analyzes code quality, provides structured feedback, and suggests concrete improvements.

## Complete Code Walkthrough

### 1. Core Components and System Architecture

```python
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_core.runnables import RunnableLambda
```

The system integrates multiple sophisticated components:

1. **Model Integration**:
   - Uses Azure OpenAI for consistent, high-quality analysis
   - Configures temperature=0 for deterministic responses
   - Handles environment variables securely

2. **Data Models**:
```python
class CodeMetrics(BaseModel):
    complexity: int = Field(description="Code complexity score (0-10)")
    readability: int = Field(description="Code readability score (0-10)")
    # ... additional metrics
```

The Pydantic models ensure type safety and data validation:
- Strong typing for all fields
- Field descriptions for clarity
- Default value handling
- Input validation

### 2. Prompt Engineering and Chain Design

The system uses three specialized prompts:

1. **Analysis Prompt**:
```python
def create_analysis_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""Given the following code, analyze its structure and patterns:
        [Template content...]""",
        input_variables=["code", "language"]
    )
```

Each prompt is carefully designed for:
- Clear instructions
- Structured output requirements
- Consistent formatting
- JSON response formatting

2. **Metrics Calculation**:
```python
def create_metrics_prompt() -> PromptTemplate:
    template="""Based on the code analysis, calculate metrics:
    [Template content with double braces for JSON]"""
```

Note the use of double braces (`{{`, `}}`) in JSON templates to:
- Escape format strings
- Prevent template variable conflicts
- Ensure proper JSON structure

### 3. Chain Implementation and Composition

```python
def create_review_chain(llm: AzureChatOpenAI) -> Any:
    def analyze(inputs: Dict[str, Any]) -> Dict[str, Any]:
        response = llm.invoke(
            analysis_prompt.format_prompt(
                code=inputs["code"],
                language=inputs["language"]
            ).to_string()
        )
        return {
            "code": inputs["code"],
            "language": inputs["language"],
            "analysis": response.content
        }
```

The chain implementation demonstrates sophisticated composition:

1. **Function Design**:
   - Clear input/output typing
   - State preservation between steps
   - Error handling at each stage

2. **Data Flow**:
   - Maintains context through chain steps
   - Accumulates results progressively
   - Handles complex nested data

### 4. JSON Parsing and Error Handling

```python
def parse_json_output(output: str) -> dict:
    """Extract and parse JSON from model output."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback extraction
        start = output.find('{')
        end = output.rfind('}')
```

The JSON parsing demonstrates robust error handling:

1. **Multiple Parsing Strategies**:
   - Primary: Code block extraction
   - Fallback: Direct JSON detection
   - Error recovery mechanisms

2. **Error Handling**:
```python
try:
    return json.loads(json_str)
except json.JSONDecodeError as e:
    raise ValueError(f"Failed to parse JSON: {json_str}") from e
```

### 5. Evaluation System

```python
def create_evaluation_chain(llm: AzureChatOpenAI) -> LabeledCriteriaEvalChain:
    """Create a chain for evaluating review quality."""
    criteria = {
        "completeness": "The review covers all important aspects",
        "actionability": "Suggestions are specific and implementable",
        # ... additional criteria
    }
```

The evaluation system provides comprehensive quality assessment:

1. **Evaluation Criteria**:
   - Completeness checking
   - Actionability assessment
   - Clarity validation
   - Technical accuracy

2. **Score Processing**:
```python
if isinstance(evaluation, dict) and "criteria_scores" in evaluation:
    criteria_scores = evaluation["criteria_scores"]
else:
    try:
        criteria_scores = parse_json_output(evaluation)
    except Exception:
        criteria_scores = {}
```

## Expected Output

When running the Code Review System, you'll see detailed output like this:

```plaintext
Demonstrating LangChain Code Review System...

Processing file: example.py

Code Review Results:
File: example.py
Language: Python

Metrics:
Complexity: 3/10 (Lower is better)
Readability: 7/10
Maintainability: 6/10
Documentation: 2/10
Best Practices: 5/10

Identified Issues:
- Missing function docstrings
- Variable 't' is not descriptive
- No type hints used
- No error handling
- Simple list implementation could be optimized

Suggestions:
- Add comprehensive docstrings to all functions
- Use descriptive variable names (e.g., 'total' instead of 't')
- Implement type hints for better code clarity
- Add input validation and error handling
- Consider using list comprehension for process_data

Example Improvements:
- def calculate_total(items: List[float]) -> float:
    """Calculate the sum of all items in the list."""
    return sum(items)
- def process_data(data: List[float]) -> List[float]:
    """Double all positive numbers in the data list."""
    return [d * 2 for d in data if d > 0]

Review Quality Evaluation:
completeness: 4.5/5
actionability: 5/5
clarity: 4.5/5
correctness: 5/5

Overall Score: 8.2/10
```

## Best Practices

### 1. Chain Configuration
For optimal results:
```python
def configure_review_system(
    llm: AzureChatOpenAI,
    review_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Configure the review system with best practices."""
    return {
        "temperature": 0,  # Ensure consistency
        "max_tokens": 2000,  # Adequate response length
        "evaluation_criteria": {...},  # Comprehensive criteria
        "error_handling": "strict"  # Robust error handling
    }
```

### 2. Prompt Engineering
For reliable output:
```python
def design_prompt(
    instruction: str,
    format_spec: Dict[str, Any]
) -> str:
    """Design prompts following best practices."""
    return f"""
    {instruction}
    
    Format requirements:
    1. Use valid JSON
    2. Include all required fields
    3. Follow numerical ranges
    4. Provide specific examples
    """
```

Remember when implementing code review systems:
- Validate all inputs thoroughly
- Handle JSON parsing robustly
- Maintain consistent scoring
- Provide actionable feedback
- Include concrete examples
- Ensure repeatable results
- Document expectations
- Handle edge cases
- Log processing steps
- Monitor quality metrics

## References

### Chain Documentation
- Chain Composition: [https://python.langchain.com/docs/expression_language/]
- Running Chains: [https://python.langchain.com/docs/expression_language/interface]
- Error Handling: [https://python.langchain.com/docs/expression_language/error_handling]

### Evaluation Documentation
- Criteria Evaluation: [https://python.langchain.com/docs/guides/evaluation/comparison/criteria]
- Quality Control: [https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain]
- Custom Evaluators: [https://python.langchain.com/docs/guides/evaluation/string/string_evaluator]

### JSON Processing
- Output Parsing: [https://python.langchain.com/docs/modules/model_io/output_parsers/]
- Structured Output: [https://python.langchain.com/docs/modules/model_io/output_parsers/structured]
- Error Recovery: [https://python.langchain.com/docs/modules/model_io/output_parsers/retry]
# Credit Approval Workflow

This example demonstrates a multi-agent credit approval system with strictly controlled tool calls.

## Key Components

1. Financial Analyst Agent:
   - Receives exact credit metrics
   - Uses strict thresholds for risk levels
   - Must call `assess_risk` immediately
   - No questions allowed

2. Credit Officer Agent:
   - Receives risk assessment
   - Uses shown risk level
   - Must call `make_decision` immediately
   - No questions allowed

3. Workflow Control:
   - Strict input formats
   - Strong validation
   - Clear error handling
   - Full debugging

## Fixed Implementation

1. Analyst Prompt:
   - Shows input format first
   - Step-by-step rules
   - Example tool call
   - NO QUESTIONS warning

2. Officer Prompt:
   - Shows case format first
   - Clear decision mapping
   - Example tool call
   - NO QUESTIONS warning

3. Tool Result Handling:
   - Multiple extraction methods
   - Better string cleaning
   - Clear debug output
   - Strong validation

## Usage

```python
# Example credit application
app = {
    "id": "APP-001",
    "income": 85000.00,
    "debt_ratio": 0.25,
    "credit_score": 720,
    "employment_years": 5.5
}

# Process application
result = process_application(
    application_id=app["id"],
    income=app["income"],
    debt_ratio=app["debt_ratio"],
    credit_score=app["credit_score"],
    employment_years=app["employment_years"]
)
```

## Debug Output

```
DEBUG: Sending analysis request:
Review credit application metrics:
- Income: $85,000.00
- Debt Ratio: 25.00%
- Credit Score: 720
- Employment: 5.5 years

DEBUG: Got agent result
DEBUG: Found risk_level in response
```

## Common Issues

1. Agent Asking Questions:
   - Fixed with explicit prompts
   - NO QUESTIONS warnings
   - Example formats shown

2. Wrong Number Usage:
   - Fixed with EXACT emphasis
   - Step-by-step rules
   - Clear thresholds

3. Tool Result Extraction:
   - Better response handling
   - String cleaning
   - Clear error paths

## Key Improvements

1. Agent Control:
   - Strict prompts
   - Clear instructions
   - No ambiguity
   - Forced tool calls

2. Data Handling:
   - Input validation
   - Error handling
   - Debug logging
   - Clear outputs

3. Process Flow:
   - Step-by-step rules
   - Clear thresholds
   - Strong validation
   - Error recovery

## Best Practices

1. Input Formats:
   - Show exact format
   - Use real examples
   - Clear structure
   - Strong typing

2. Tool Calls:
   - Immediate execution
   - No questions
   - Clear parameters
   - Strong validation

3. Error Handling:
   - Clear error paths
   - Default behaviors
   - Debug logging
   - Recovery options
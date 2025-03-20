# Understanding Evaluation with OpenEvals in LangChain

Welcome to this comprehensive guide on using OpenEvals with LangChain for model evaluation! This tutorial explains how to use pre-built evaluators to assess your language model outputs systematically.

## Prerequisites

Before starting, install the required packages:
```bash
pip install openevals
```

## Core Concepts

1. **What is OpenEvals?**
   OpenEvals is a toolkit that provides:
   - Pre-built evaluation metrics
   - LLM-as-a-judge capabilities
   - Standardized evaluation framework
   - Consistent scoring mechanisms

2. **Key Components**
   ```python
   from openevals.llm import create_llm_as_judge
   from openevals.prompts import CORRECTNESS_PROMPT
   from langchain_openai import AzureChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   ```

3. **Evaluation Schema**
   ```python
   class EvaluationResult(BaseModel):
       input: str = Field(description="Input text or query")
       output: str = Field(description="Model output")
       score: float = Field(description="Evaluation score")
       feedback: str = Field(description="Evaluation feedback")
       timestamp: datetime = Field(default_factory=datetime.now)
   ```

## Code Breakdown

1. **Setting Up the Judge Model**
   ```python
   azure_judge = AzureChatOpenAI(
       azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
       openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
       api_key=os.getenv("AZURE_OPENAI_API_KEY"),
       temperature=0  # Keep temperature at 0 for consistent evaluation
   )
   ```
   
   What's happening:
   - Creates an Azure-based evaluation model
   - Uses environment variables for configuration
   - Sets temperature to 0 for deterministic outputs
   - Will serve as our judge for evaluations

2. **Creating the Response Chain**
   ```python
   def create_chain_with_evaluator():
       model = create_chat_model()
       prompt = ChatPromptTemplate.from_messages([
           ("system", "You are a helpful assistant..."),
           ("human", "{input}")
       ])
       chain = prompt | model | RunnablePassthrough()
       return chain, model
   ```
   
   Key points:
   - Creates the model for generating responses
   - Sets up a basic prompt template
   - Uses LCEL for chain composition
   - Returns both chain and model for flexibility

3. **Setting Up the Evaluator**
   ```python
   def create_prebuilt_evaluator():
       evaluator = create_llm_as_judge(
           prompt=CORRECTNESS_PROMPT,
           feedback_key="correctness",
           model="gpt-4o",
           judge=azure_judge
       )
       return evaluator
   ```
   
   Important aspects:
   - Uses OpenEvals' pre-built evaluator
   - Configures for correctness assessment
   - Uses Azure GPT-4 as judge
   - Maintains consistency in evaluation

4. **Evaluation Process**
   ```python
   def evaluate_responses(evaluator, questions, responses, reference_outputs):
       results = []
       for question, response, reference in zip(questions, responses, reference_outputs):
           # Convert response to string if needed
           response_str = response.content if hasattr(response, "content") else str(response)
           
           # Run evaluation
           eval_result = evaluator(
               inputs=question,
               outputs=response_str,
               reference_outputs=reference
           )
           
           # Process results
           score = eval_result.get("score", 0)
           feedback = eval_result.get("feedback", "")
           
           results.append(EvaluationResult(
               input=question,
               output=response_str,
               score=score,
               feedback=feedback
           ))
       return results
   ```
   
   Step by step:
   1. Takes questions, responses, and reference outputs
   2. Processes each response for evaluation
   3. Runs the evaluator on each case
   4. Collects scores and feedback
   5. Returns structured results

## Example Usage

```python
# Setup
chain, model = create_chain_with_evaluator()
evaluator = create_prebuilt_evaluator()

# Questions and references
questions = [
    "What is the capital of France?",
    "Explain how photosynthesis works."
]
reference_outputs = [
    "Paris",
    "Photosynthesis is the process by which plants convert light energy..."
]

# Generate and evaluate responses
responses = [chain.invoke({"input": q}) for q in questions]
results = evaluate_responses(evaluator, questions, responses, reference_outputs)

# View results
for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Feedback: {result.feedback}")
```

## Best Practices

1. **Reference Output Quality**
   ```python
   reference_outputs = [
       # Detailed and accurate reference
       ("Photosynthesis is the process by which green plants, algae, "
        "and some bacteria convert light energy into chemical energy..."),
       # Clear and concise reference
       "Paris"
   ]
   ```

2. **Error Handling**
   ```python
   try:
       eval_result = evaluator(...)
   except Exception as e:
       print(f"Evaluation error: {str(e)}")
       return default_result
   ```

3. **Score Interpretation**
   ```python
   def interpret_score(score: float) -> str:
       if score >= 0.9: return "Excellent"
       elif score >= 0.7: return "Good"
       elif score >= 0.5: return "Fair"
       else: return "Needs Improvement"
   ```

## Advanced Features

1. **Temperature Comparison**
   ```python
   # Compare different temperature settings
   model_0 = create_chat_model(temperature=0)
   model_1 = create_chat_model(temperature=0.7)
   
   # Evaluate both models
   results_0 = evaluate_responses(...)
   results_1 = evaluate_responses(...)
   ```

2. **Custom Evaluation Criteria**
   ```python
   evaluator = create_llm_as_judge(
       prompt=custom_prompt,
       feedback_key="custom_metric",
       model="gpt-4o"
   )
   ```

## Resources


1. **Official Documentation**
   - **Evaluation Guide**: https://docs.smith.langchain.com/evaluation/concepts
   - **OpenEvals GitHub**: https://github.com/langchain-ai/openevals

2. **Additional Resources**
   - **Testing**: https://python.langchain.com/docs/contributing/how_to/testing/
   - **Debugging**: https://python.langchain.com/docs/how_to/debugging/

## Real-World Applications

1. **Model Comparison**
   - Different model versions
   - Temperature settings
   - Prompt variations
   - Architecture changes

2. **Quality Assurance**
   - Response accuracy
   - Content appropriateness
   - Output consistency
   - Error detection

3. **Continuous Improvement**
   - Performance monitoring
   - Model selection
   - Prompt refinement
   - System optimization

Remember:
- Always provide clear reference outputs
- Use appropriate temperature settings
- Handle errors gracefully
- Document evaluation criteria
- Monitor evaluation patterns
- Keep feedback specific
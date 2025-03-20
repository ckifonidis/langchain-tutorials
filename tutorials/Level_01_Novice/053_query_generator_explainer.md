# Understanding the Query Generator with Prompt Templates and Memory

This comprehensive guide explores how to build a sophisticated Query Generator by combining LangChain's prompt templates with memory capabilities. We'll create a system that can generate optimized SQL queries while maintaining conversation context and using few-shot learning.

## Complete Code Walkthrough

### 1. Core Components and System Architecture

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
```

The system integrates several sophisticated components:

1. **Prompt Template System**:
   - Few-shot learning setup
   - Example management
   - Dynamic template generation
   - Context injection

2. **Memory Management**:
   - Conversation history tracking
   - Context persistence
   - State management
   - History retrieval

### 2. Data Models and Schema Design

```python
class QueryTemplate(BaseModel):
    """Schema for query templates."""
    base_query: str = Field(description="Base query structure")
    parameters: List[str] = Field(description="Required parameters")
    context: str = Field(description="Usage context")
    example: str = Field(description="Example usage")
```

The models demonstrate:

1. **Template Structure**:
   - Base query patterns
   - Parameter definitions
   - Context information
   - Example documentation

2. **History Tracking**:
```python
class QueryHistory(BaseModel):
    query: str = Field(description="Generated query")
    context: str = Field(description="Query context")
    timestamp: datetime = Field(default_factory=datetime.now)
```

### 3. Few-Shot Learning Implementation

```python
examples = [
    {
        "request": "Find recent orders from customer ABC Corp",
        "context": "Order history lookup for specific customer",
        "query": "SELECT o.order_id, o.order_date..."
    },
    # Additional examples
]

example_prompt = PromptTemplate(
    input_variables=["request", "context", "query"],
    template=example_template
)
```

The few-shot setup demonstrates:

1. **Example Design**:
   - Clear request-query mapping
   - Context documentation
   - Pattern demonstration
   - Progressive complexity

2. **Template Construction**:
```python
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a query generation expert...",
    suffix="Now generate a query for..."
)
```

### 4. Memory Integration

```python
def generate_query(request: str, context: str) -> str:
    """Generate a query using the few-shot prompt and memory."""
    history = memory.load_memory_variables({})
    prompt = few_shot_prompt.format(
        input=request,
        context=context,
        history=history.get("history", "No previous history.")
    )
```

The memory system showcases:

1. **Context Management**:
   - History loading
   - State preservation
   - Context injection
   - Memory updates

2. **Query Generation**:
```python
memory.save_context(
    {"input": f"Request: {request}\nContext: {context}"},
    {"output": response.content}
)
```

## Expected Output

When running the Query Generator, you'll see output like this:

```plaintext
Demonstrating LangChain Query Generator...

Initializing Query Generator...

Request: Show me revenue by region for Q1
Context: Regional revenue analysis

Generated Query:
SELECT 
    r.region_name,
    SUM(od.quantity * od.unit_price) as revenue
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN regions r ON c.region_id = r.region_id
WHERE EXTRACT(QUARTER FROM o.order_date) = 1
    AND EXTRACT(YEAR FROM o.order_date) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY r.region_name
ORDER BY revenue DESC;
==================================================

Request: List top 5 salespeople by performance
Context: Sales team performance analysis

Generated Query:
SELECT 
    e.first_name,
    e.last_name,
    COUNT(o.order_id) as total_orders,
    SUM(od.quantity * od.unit_price) as total_sales
FROM employees e
JOIN orders o ON e.employee_id = o.employee_id
JOIN order_details od ON o.order_id = od.order_id
WHERE e.title LIKE '%Sales%'
GROUP BY e.employee_id, e.first_name, e.last_name
ORDER BY total_sales DESC
LIMIT 5;
==================================================

Request: Compare current month sales to last month
Context: Monthly sales comparison with previous period

Generated Query:
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', o.order_date) as sale_month,
        SUM(od.quantity * od.unit_price) as total_sales
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
    GROUP BY DATE_TRUNC('month', o.order_date)
)
SELECT 
    current.total_sales as current_month_sales,
    previous.total_sales as previous_month_sales,
    ((current.total_sales - previous.total_sales) / previous.total_sales * 100) as growth_percentage
FROM monthly_sales current
JOIN monthly_sales previous 
    ON current.sale_month = previous.sale_month + INTERVAL '1 month'
WHERE current.sale_month = DATE_TRUNC('month', CURRENT_DATE);
```

## Best Practices

### 1. Template Management
For optimal results:
```python
def design_templates():
    """Best practices for template design."""
    return {
        "prefix": "Clear system role and context",
        "examples": "Progressive complexity examples",
        "suffix": "Specific task instructions",
        "variables": "Well-defined input variables"
    }
```

### 2. Memory Usage
For reliable context:
```python
def manage_memory():
    """Best practices for memory handling."""
    memory.save_context(
        {"input": "Clearly formatted input"},
        {"output": "Structured output"}
    )
    
    history = memory.load_memory_variables({})
    if len(history) > MAX_HISTORY:
        memory.clear()
```

Remember when implementing query generators:
- Design clear templates
- Provide relevant examples
- Maintain context history
- Handle memory limits
- Validate generated queries
- Include error handling
- Document query patterns
- Use consistent formatting
- Monitor memory usage
- Log generation steps

## References

### Prompt Template Documentation
- Template Concepts: [https://python.langchain.com/docs/modules/model_io/prompts/]
- Few-Shot Templates: [https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples]
- Prompt Design: [https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates]

### Memory Documentation
- Memory Concepts: [https://python.langchain.com/docs/modules/memory/]
- Buffer Memory: [https://python.langchain.com/docs/modules/memory/types/buffer]
- Memory Types: [https://python.langchain.com/docs/modules/memory/types/]

### Additional Resources
- State Management: [https://python.langchain.com/docs/modules/memory/types/buffer_window]
- Example Selection: [https://python.langchain.com/docs/modules/model_io/prompts/example_selectors]
- Best Practices: [https://python.langchain.com/docs/guides/best_practices]
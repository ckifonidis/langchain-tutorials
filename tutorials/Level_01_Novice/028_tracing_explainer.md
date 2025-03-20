# Understanding Tracing in LangChain

Welcome to this comprehensive guide on using tracing in LangChain! Tracing helps you monitor, debug, and analyze your application's execution flow. This tutorial will help you understand how to implement and use tracing effectively.

## Core Concepts

1. **What is Tracing?**
   Think of tracing as a detailed recording that:
   
   - **Monitors**: Tracks execution flow
   - **Records**: Captures timing and details
   - **Analyzes**: Helps understand behavior
   - **Debugs**: Identifies issues

2. **Key Components**
   ```python
   from langchain_core.tracers import ConsoleCallbackHandler
   from langchain.callbacks import FileCallbackHandler
   from langchain_core.tracers.context import tracing_v2_enabled
   ```

3. **Basic Structure**
   ```python
   class QueryTrace(BaseModel):
       query: str = Field(description="Input query")
       timestamp: datetime = Field(description="Processing time")
       duration: float = Field(description="Execution duration")
       response: str = Field(description="Model response")
   ```

## Implementation Breakdown

1. **Tracing Setup**
   ```python
   def setup_tracers(log_file: str = "trace.log") -> List[Any]:
       # Create handlers directory
       os.makedirs("handlers", exist_ok=True)
       
       # Initialize handlers
       console_handler = ConsoleCallbackHandler()
       file_handler = FileCallbackHandler(f"handlers/{log_file}")
       
       return [console_handler, file_handler]
   ```
   
   Features:
   - Console output
   - File logging
   - Directory management
   - Multiple handlers

2. **Tracing Context**
   ```python
   with tracing_v2_enabled(project_name="TracingExample") as session:
       start_time = datetime.now()
       response = chain.invoke(input_data)
       duration = (datetime.now() - start_time).total_seconds()
       
       trace = QueryTrace(
           query=input_data,
           timestamp=start_time,
           duration=duration,
           response=response
       )
   ```
   
   Benefits:
   - Project organization
   - Timing capture
   - Response tracking
   - Structured data

3. **Model Integration**
   ```python
   def create_chat_model(handlers: List[Any]) -> AzureChatOpenAI:
       return AzureChatOpenAI(
           # ... configuration ...
           callbacks=handlers,
           temperature=0
       )
   ```

## Best Practices

1. **Handler Management**
   ```python
   # Organize handlers by type
   handlers = [
       ConsoleCallbackHandler(),  # Real-time output
       FileCallbackHandler("handlers/trace.log")  # Persistent storage
   ]
   ```
   
   Tips:
   - Separate concerns
   - Handle errors
   - Manage resources
   - Organize output

2. **Trace Analysis**
   ```python
   def analyze_traces(traces: List[QueryTrace]):
       total_queries = len(traces)
       total_duration = sum(t.duration for t in traces)
       avg_duration = total_duration / total_queries
       
       return {
           "total_queries": total_queries,
           "total_duration": total_duration,
           "avg_duration": avg_duration
       }
   ```

3. **Project Organization**
   ```python
   # Group related traces
   with tracing_v2_enabled(project_name="UserQueries") as session:
       # Process user-related queries
       pass
   
   with tracing_v2_enabled(project_name="SystemTests") as session:
       # Run system tests
       pass
   ```

## Example Output

When running `python 028_tracing.py`, you'll see:

```
Demonstrating LangChain Tracing...

Example 1: Basic Query Tracing
--------------------------------------------------
Processing query for France...
Response: The capital of France is Paris.
Duration: 1.25 seconds
Tracing session started.

Example 3: Trace Analysis
--------------------------------------------------
Trace Analysis:
Total Queries: 3
Total Duration: 3.75 seconds
Average Duration: 1.25 seconds
```

## LangSmith Integration

LangSmith provides advanced tracing and debugging capabilities for your LangChain applications.

1. **Install Dependencies**
   ```bash
   pip install -U langchain langchain-openai
   ```

2. **Configure Environment**
   Add these variables to your environment:
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   LANGSMITH_API_KEY="<your-api-key>"
   LANGSMITH_PROJECT="<your-project-name>"
   OPENAI_API_KEY="<your-openai-api-key>"
   ```

3. **Basic Usage**
   Any LLM, Chat model, or Chain will automatically send traces to your project:
   ```python
   from langchain_openai import ChatOpenAI

   llm = ChatOpenAI()
   llm.invoke("Hello, world!")
   ```

For more information about LangSmith and its features, visit:
https://www.langchain.com/langsmith

## Common Patterns

1. **Performance Monitoring**
   ```python
   # Track execution times
   start_time = datetime.now()
   result = chain.invoke(data)
   duration = (datetime.now() - start_time).total_seconds()
   ```

2. **Error Tracking**
   ```python
   try:
       with tracing_v2_enabled(project_name="ErrorTests"):
           result = chain.invoke(data)
   except Exception as e:
       print(f"Error tracked: {str(e)}")
   ```

## Resources

1. **Official Documentation**
   - **Tracing Guide**: https://python.langchain.com/docs/concepts/tracing/
   - **Callbacks**: https://python.langchain.com/docs/how_to/#callbacks
   - **Langsmith Conceptual Guide**: https://docs.smith.langchain.com/observability/concepts

2. **Additional Resources**
   - **Debugging**: https://python.langchain.com/docs/how_to/debugging/
   - **Tracing and Monitoring**: https://python.langchain.com/docs/integrations/providers/graphsignal/#tracing-and-monitoring

## Real-World Applications

1. **Performance Analysis**
   - Response times
   - Resource usage
   - Bottleneck detection
   - Optimization opportunities

2. **Debugging**
   - Error tracking
   - Flow visualization
   - Issue diagnosis
   - Pattern detection

3. **Monitoring**
   - System health
   - Usage patterns
   - Error rates
   - Resource utilization

Remember: 
- Use project names for organization
- Enable tracing strategically
- Monitor performance
- Track errors
- Analyze patterns
- Document findings
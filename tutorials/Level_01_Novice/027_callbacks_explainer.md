# Understanding Callbacks in LangChain

Welcome to this comprehensive guide on using callbacks in LangChain! Callbacks allow you to monitor and modify the execution flow of your LangChain applications. This tutorial will help you understand how to implement and use callbacks effectively.

## Core Concepts

1. **What are Callbacks?**
   Think of callbacks as observers that can:
   
   - **Monitor**: Track operations and events
   - **Log**: Record important information
   - **Measure**: Track timing and performance
   - **Modify**: Influence execution flow

2. **Key Components**
   ```python
   from langchain_core.callbacks import (
       CallbackManager,
       BaseCallbackHandler
   )
   ```

3. **Basic Structure**
   ```python
   class CustomCallback(BaseCallbackHandler):
       """Custom callback implementation."""
       def __init__(self) -> None:
           super().__init__()
           self.logs = []
       
       def on_llm_start(self, *args, **kwargs) -> None:
           self.logs.append("LLM started")
       
       def on_llm_end(self, *args, **kwargs) -> None:
           self.logs.append("LLM completed")
   ```

## Implementation Breakdown

1. **Timing Callback**
   ```python
   class TimingCallback(BaseCallbackHandler):
       def __init__(self) -> None:
           super().__init__()
           self.start_time = None
           self.timing_logs = []
       
       def on_llm_start(self, *args, **kwargs) -> None:
           self.start_time = datetime.now()
           self.timing_logs.append(
               f"Started at {self.start_time}"
           )
       
       def on_llm_end(self, *args, **kwargs) -> None:
           end_time = datetime.now()
           duration = end_time - self.start_time
           self.timing_logs.append(
               f"Duration: {duration.total_seconds():.2f}s"
           )
   ```
   
   Features:
   - Precise timing
   - Duration tracking
   - Event logging
   - Performance monitoring

2. **Logging Callback**
   ```python
   class LoggingCallback(BaseCallbackHandler):
       def on_llm_start(
           self, serialized: Dict[str, Any], 
           prompts: List[str], **kwargs: Any
       ) -> None:
           self.logs.append(f"Prompts: {prompts}")
       
       def on_llm_error(
           self, error: Exception, **kwargs: Any
       ) -> None:
           self.logs.append(f"Error: {str(error)}")
   ```
   
   Benefits:
   - Detailed logging
   - Error tracking
   - Input monitoring
   - Debug information

3. **Callback Integration**
   ```python
   def create_chat_model(callbacks: List[BaseCallbackHandler]):
       return AzureChatOpenAI(
           # ... configuration ...
           callback_manager=CallbackManager(callbacks),
           temperature=0
       )
   ```

## Best Practices

1. **Callback Organization**
   ```python
   # Group related callbacks
   monitoring_callbacks = [
       TimingCallback(),
       LoggingCallback()
   ]
   
   # Use with model
   model = create_chat_model(monitoring_callbacks)
   ```

2. **Error Handling**
   ```python
   def on_llm_error(self, error: Exception, **kwargs):
       try:
           # Log the error
           self.errors.append(str(error))
           # Optionally notify
           self.notify_error(error)
       except Exception as e:
           print(f"Callback error: {str(e)}")
   ```

3. **Resource Management**
   ```python
   class ResourceTrackingCallback(BaseCallbackHandler):
       def on_llm_start(self, *args, **kwargs):
           self.track_resource_usage()
       
       def on_llm_end(self, *args, **kwargs):
           self.release_resources()
   ```

## Example Output

When running `python 027_callbacks.py`, you'll see:

```
Demonstrating LangChain Callbacks...

Example 1: Basic Timing and Logging
--------------------------------------------------
Processing question for France...
Response: The capital of France is Paris.

Timing Information:
LLM operation started at 2025-03-19 02:21:00
LLM operation completed at 2025-03-19 02:21:02
Duration: 2.15 seconds

Operation Logs:
Starting LLM with prompts: ['What is the capital of France?']
LLM completed with response: Paris...
```

## Common Patterns

1. **Performance Monitoring**
   ```python
   class PerformanceCallback(BaseCallbackHandler):
       def __init__(self):
           self.metrics = {
               "total_time": 0,
               "calls": 0,
               "errors": 0
           }
   ```

2. **Event Aggregation**
   ```python
   class EventAggregator(BaseCallbackHandler):
       def __init__(self):
           self.events = {
               "starts": 0,
               "completions": 0,
               "errors": 0
           }
   ```

## Resources

1. **Official Documentation**
   - **Callbacks Guide**: https://python.langchain.com/docs/concepts/callbacks/
   - **Custom Handlers**: https://python.langchain.com/docs/how_to/custom_callbacks/
   - **Callback Integrations**: https://python.langchain.com/docs/integrations/callbacks/

2. **Additional Resources**
   - **Python API Reference**: https://python.langchain.com/api_reference/
   - **Performance Monitoring**: https://medium.com/towards-agi/how-to-use-langsmith-for-monitoring-langchain-applications-ffea867515fb

## Real-World Applications

1. **Monitoring**
   - Performance tracking
   - Resource usage
   - Error detection
   - Usage analytics

2. **Debugging**
   - Event logging
   - Error tracing
   - Flow visualization
   - Problem diagnosis

3. **Integration**
   - Metrics collection
   - Alerting systems
   - Logging platforms
   - Monitoring tools

Remember: 
- Keep callbacks focused
- Handle errors gracefully
- Monitor performance impact
- Manage resources properly
- Log useful information
- Consider security implications
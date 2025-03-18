# Agents in LangChain

## Core Concepts

Agents in LangChain use language models to choose and execute actions:

1. Basic Functionality
   - Dynamic action selection
   - Sequence determination
   - Tool interaction

   ```python
   from langchain.agents import load_tools
   from langchain.agents import initialize_agent
   from langchain.agents import AgentType
   from langchain.chat_models import ChatOpenAI
   
   # Initialize language model
   llm = ChatOpenAI(temperature=0)
   
   # Load tools
   tools = load_tools(["serpapi", "llm-math"], llm=llm)
   
   # Initialize agent
   agent = initialize_agent(
       tools,
       llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   ```

2. Key Difference from Chains
   - Dynamic vs hardcoded sequences
   - Flexible decision making
   - Adaptive behavior

   ```python
   # Agent execution with dynamic decision making
   response = agent.run(
       "Who is the current president and what is their age divided by 2?"
   )
   ```

## Implementation Features

1. Action Selection
   - LLM-based decision making
   - Tool choice determination
   - Sequential planning

   ```python
   from langchain.agents import Tool
   from langchain.chains import LLMChain
   from langchain.prompts import PromptTemplate
   
   # Define custom tools
   tools = [
       Tool(
           name="Search",
           func=search.run,
           description="useful for finding information"
       ),
       Tool(
           name="Calculator",
           func=calculator.run,
           description="useful for performing calculations"
       )
   ]
   ```

2. Tool Integration
   - Search capabilities
   - Calculation functions
   - External system access

## Key Components

1. Decision Making
   - Action sequence planning
   - Tool selection logic
   - Response generation

   ```python
   from langchain.agents import create_react_agent
   from langchain.prompts import MessagesPlaceholder
   
   # Create agent with custom prompt
   prompt = PromptTemplate.from_template("""
   Answer the following questions as best you can:
   {input}
   
   Use these tools if needed: {tools}
   """)
   
   agent = create_react_agent(llm, tools, prompt)
   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   ```

2. Tool Management
   - Tool availability
   - Tool access control
   - Resource utilization

   ```python
   from langchain.tools import PythonREPLTool
   from langchain.agents import Tool
   
   # Create tool with access control
   class RestrictedPythonTool(PythonREPLTool):
       def is_allowed(self, command: str) -> bool:
           # Add security checks
           return True
           
   python_tool = RestrictedPythonTool()
   ```

## Best Practices

1. Agent Design:
   - Proper tool selection
   - Action sequence planning
   - Error handling strategy

   ```python
   def create_agent_with_error_handling(llm, tools):
       # Add error handling wrapper to tools
       wrapped_tools = [
           Tool(
               name=tool.name,
               func=handle_tool_error(tool.func),
               description=tool.description
           )
           for tool in tools
       ]
       
       return initialize_agent(
           wrapped_tools,
           llm,
           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
           handle_parsing_errors=True
       )
   ```

2. Implementation Strategy:
   - Efficient tool integration
   - Response management
   - Resource optimization

## Resources

Documentation Links:
- [Agent Building Tutorial](https://python.langchain.com/docs/tutorials/agents/)
- [Agent Concepts](https://python.langchain.com/v0.1/docs/modules/agents/concepts/)
- [Agents Documentation](https://python.langchain.com/v0.1/docs/modules/agents/)

## Implementation Considerations

1. Performance:
   - Decision efficiency
   - Tool execution speed
   - Response time

   ```python
   from langchain.callbacks import BaseCallbackHandler
   
   class PerformanceMonitor(BaseCallbackHandler):
       def on_tool_start(self, tool: str, input_str: str):
           self.start_time = time.time()
           
       def on_tool_end(self, output: str):
           duration = time.time() - self.start_time
           print(f"Tool execution time: {duration:.2f}s")
   ```

2. Flexibility:
   - Dynamic adaptation
   - Tool selection
   - Action sequencing

3. Reliability:
   - Error handling
   - Recovery strategies
   - Consistent behavior

## Common Use Cases

1. Search Operations:
   - Web queries
   - Information retrieval
   - Data gathering

   ```python
   from langchain.agents import create_sql_agent
   from langchain.agents.agent_toolkits import SQLDatabaseToolkit
   
   # Create SQL database agent
   db_toolkit = SQLDatabaseToolkit(db=db)
   sql_agent = create_sql_agent(
       llm=llm,
       toolkit=db_toolkit,
       verbose=True
   )
   ```

2. Complex Tasks:
   - Multi-step operations
   - Tool coordination
   - Sequential actions

3. Interactive Systems:
   - User queries
   - Dynamic responses
   - Adaptive behavior

## Integration Patterns

1. System Design:
   - Tool integration
   - Action flow
   - Response handling

   ```python
   from langchain.agents import AgentExecutor
   from langchain.memory import ConversationBufferMemory
   
   # Create agent with memory
   memory = ConversationBufferMemory(memory_key="chat_history")
   agent_chain = AgentExecutor.from_agent_and_tools(
       agent=agent,
       tools=tools,
       memory=memory,
       verbose=True
   )
   ```

2. Error Management:
   - Exception handling
   - Recovery procedures
   - Fallback strategies

3. Resource Control:
   - Tool access
   - Resource allocation
   - Performance monitoring

## Advanced Features

1. Custom Agents:
   - Specialized behavior
   - Custom tool sets
   - Domain adaptation

   ```python
   from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
   
   class CustomAgent(BaseSingleActionAgent):
       @property
       def input_keys(self):
           return ["input"]
           
       def plan(self, inputs: dict) -> Union[AgentAction, AgentFinish]:
           # Implement custom planning logic
           pass
   ```

2. Enhanced Capabilities:
   - Complex reasoning
   - Multi-tool coordination
   - Advanced planning

   ```python
   # Create agent with parallel tool execution
   from langchain.agents import initialize_agent
   from langchain.agents import AgentType
   
   parallel_agent = initialize_agent(
       tools,
       llm,
       agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
       handle_parsing_errors=True,
       max_iterations=5
   )
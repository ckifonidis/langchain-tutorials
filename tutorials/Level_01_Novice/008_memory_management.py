"""
LangChain Memory Management Example

This example demonstrates different types of memory management in LangChain,
showing how to maintain conversation context and state across interactions.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables from the .env file
load_dotenv()

# Check if required Azure OpenAI environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                    "Please add them to your .env file.")

def init_chat_model():
    """Initialize the Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

def demonstrate_buffer_memory():
    """Demonstrate basic conversation buffer memory."""
    print("\nDemonstrating Buffer Memory:")
    
    # Initialize chat model and memory
    chat_model = init_chat_model()
    memory = ConversationBufferMemory()
    
    try:
        # First interaction
        messages = [
            SystemMessage(content="You are a helpful travel guide."),
            HumanMessage(content="What are some must-visit places in Paris?")
        ]
        
        response = chat_model.invoke(messages)
        memory.save_context(
            {"input": messages[-1].content},
            {"output": response.content}
        )
        print("\nFirst Response:", response.content)
        
        # Second interaction using context
        follow_up = HumanMessage(
            content="How long should I plan to spend at these locations?"
        )
        
        # Get chat history and add new message
        chat_history = memory.load_memory_variables({})
        print("\nChat History:", chat_history)
        
        messages = [
            SystemMessage(content="You are a helpful travel guide."),
            *[HumanMessage(content=m) if i % 2 == 0 else AIMessage(content=m)
              for i, m in enumerate(chat_history.get("history", "").split("\n"))]
        ]
        messages.append(follow_up)
        
        response = chat_model.invoke(messages)
        memory.save_context(
            {"input": follow_up.content},
            {"output": response.content}
        )
        print("\nSecond Response:", response.content)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def demonstrate_window_memory():
    """Demonstrate window memory with limited context."""
    print("\nDemonstrating Window Memory:")
    
    # Initialize chat model and window memory (keeps last 2 interactions)
    chat_model = init_chat_model()
    memory = ConversationBufferWindowMemory(k=2)
    
    try:
        # Series of interactions
        questions = [
            "What's the basic plot of Star Wars?",
            "Who is Luke's father?",
            "What happens to Darth Vader?",
            "Tell me about the sequel trilogy."
        ]
        
        for i, question in enumerate(questions, 1):
            messages = [
                SystemMessage(content="You are a Star Wars expert.")
            ]
            
            # Add memory context if available
            chat_history = memory.load_memory_variables({})
            if "history" in chat_history:
                print(f"\nWindow Memory (Turn {i}):", chat_history["history"])
            
            messages.append(HumanMessage(content=question))
            response = chat_model.invoke(messages)
            
            memory.save_context(
                {"input": question},
                {"output": response.content}
            )
            
            print(f"\nQuestion {i}:", question)
            print(f"Response {i}:", response.content)
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def demonstrate_summary_memory():
    """Demonstrate conversation summary memory."""
    print("\nDemonstrating Summary Memory:")
    
    # Initialize chat model and summary memory
    chat_model = init_chat_model()
    memory = ConversationSummaryMemory(llm=chat_model)
    
    try:
        # Multiple interactions about a complex topic
        questions = [
            "What is quantum computing?",
            "How does quantum entanglement work?",
            "What are qubits?",
            "What are the practical applications?"
        ]
        
        for i, question in enumerate(questions, 1):
            messages = [
                SystemMessage(content="""
                    You are a quantum computing expert. Provide clear,
                    concise explanations that build upon previous context.
                """)
            ]
            
            # Get conversation summary if available
            chat_history = memory.load_memory_variables({})
            if "history" in chat_history:
                print(f"\nConversation Summary (Turn {i}):", chat_history["history"])
            
            messages.append(HumanMessage(content=question))
            response = chat_model.invoke(messages)
            
            memory.save_context(
                {"input": question},
                {"output": response.content}
            )
            
            print(f"\nQuestion {i}:", question)
            print(f"Response {i}:", response.content)
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

def main():
    print("\nDemonstrating LangChain Memory Management...")
    
    # Demonstrate different memory types
    demonstrate_buffer_memory()
    demonstrate_window_memory()
    demonstrate_summary_memory()

if __name__ == "__main__":
    main()

# Expected Output:
# Buffer Memory: Shows full conversation history
# Window Memory: Shows limited recent context
# Summary Memory: Shows summarized conversation history
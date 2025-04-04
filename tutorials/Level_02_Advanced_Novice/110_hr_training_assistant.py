#!/usr/bin/env python3
"""
LangChain HR Training Assistant (110) (LangChain v3)

This example demonstrates a banking HR training assistant using three key concepts:
1. Agents: Autonomous training interaction
2. Chat History: Contextual learning sessions
3. Few Shot Prompting: Example-based training

It provides comprehensive training support for HR teams in banking.
"""

import os
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TrainingExample(BaseModel):
    """Schema for training examples."""
    scenario: str = Field(description="Training scenario")
    question: str = Field(description="Employee question")
    response: str = Field(description="Correct response")
    explanation: str = Field(description="Response explanation")
    policy_reference: str = Field(description="Related banking policy")

class TrainingSession(BaseModel):
    """Schema for training sessions."""
    employee_id: str = Field(description="Employee identifier")
    topic: str = Field(description="Training topic")
    start_time: str = Field(description="Session start time")
    responses: List[Dict] = Field(description="Employee responses")
    score: float = Field(description="Session score")
    feedback: str = Field(description="Overall feedback")

class TrainingTool(BaseTool):
    """Tool for accessing training resources."""
    name: str = Field(default="training_resource")
    description: str = Field(default="Access banking policies and training materials")
    resources: Dict[str, str] = Field(default_factory=dict)

    def __init__(self) -> None:
        super().__init__()
        self.resources = {
            "aml": "Anti-Money Laundering (AML) Policy requires customer due diligence...",
            "kyc": "Know Your Customer (KYC) procedures require identity verification...",
            "privacy": "Data privacy policies mandate secure handling of customer data...",
            "security": "Security protocols require two-factor authentication...",
            "compliance": "Banking compliance regulations require regular training..."
        }

    def _run(self, query: str) -> str:
        """Run the tool."""
        return self.resources.get(query.lower(), "Policy not found.")

    async def _arun(self, query: str) -> str:
        """Run the tool asynchronously."""
        return self._run(query)

class HRTrainingAssistant:
    def __init__(self) -> None:
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = [TrainingTool()]
        
        # Initialize chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup few-shot examples
        self.setup_examples()
        
        # Setup agent
        self.setup_agent()

    def setup_examples(self) -> None:
        """Setup few-shot training examples."""
        example_template = """
        Scenario: {scenario}
        Question: {question}
        Response: {response}
        Explanation: {explanation}
        Policy: {policy}
        """

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=[
                {
                    "scenario": "Customer wants to open account without ID",
                    "question": "Can we proceed with account opening?",
                    "response": "No, valid ID is required per KYC policy",
                    "explanation": "KYC regulations require identity verification",
                    "policy": "KYC Policy Section 1.2"
                },
                {
                    "scenario": "Large cash deposit by new customer",
                    "question": "What checks are needed?",
                    "response": "Enhanced due diligence required per AML policy",
                    "explanation": "Transactions over threshold need verification",
                    "policy": "AML Policy Section 2.1"
                }
            ],
            example_prompt=PromptTemplate(
                input_variables=["scenario", "question", "response", "explanation", "policy"],
                template=example_template
            ),
            prefix="Here are some example banking scenarios and appropriate responses:",
            suffix="Now handle this scenario:\n{input}",
            input_variables=["input"]
        )

    def setup_agent(self) -> None:
        """Setup the training agent."""
        # System prompt
        system_prompt = """
        You are an expert banking trainer assisting employees with:
        1. Banking procedures
        2. Compliance requirements
        3. Security protocols
        4. Customer service standards

        Use examples to explain concepts and refer to relevant policies.
        Ensure all responses align with banking regulations.
        """

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    async def train_employee(self, employee_id: str, topic: str, 
                           questions: List[str]) -> TrainingSession:
        """Conduct a training session."""
        try:
            session = TrainingSession(
                employee_id=employee_id,
                topic=topic,
                start_time=datetime.now().isoformat(),
                responses=[],
                score=0.0,
                feedback=""
            )

            correct_responses = 0
            for question in questions:
                # Get example-based response
                example_prompt = self.few_shot_prompt.format(input=question)
                
                # Run agent with examples and history
                response = await self.agent_executor.ainvoke({
                    "input": example_prompt
                })

                session.responses.append({
                    "question": question,
                    "response": response["output"],
                    "timestamp": datetime.now().isoformat()
                })

                # Check response quality (simplified)
                if "policy" in response["output"].lower():
                    correct_responses += 1

            # Calculate score and generate feedback
            session.score = correct_responses / len(questions) * 100
            
            feedback_prompt = f"""
            Provide feedback for this training session:
            Topic: {topic}
            Score: {session.score}%
            Responses: {len(session.responses)}

            Give specific recommendations for improvement.
            """

            feedback_response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert banking trainer."),
                HumanMessage(content=feedback_prompt)
            ])
            
            session.feedback = feedback_response.content
            
            return session

        except Exception as e:
            print(f"Error in training session: {str(e)}")
            raise

async def demonstrate_training_assistant():
    print("\nHR Training Assistant Demo")
    print("=========================\n")

    try:
        # Initialize assistant
        assistant = HRTrainingAssistant()

        # Example training session
        questions = [
            "What steps should be taken when a customer requests a large cash withdrawal?",
            "How do we verify customer identity for online transactions?",
            "What should we do if we suspect money laundering activity?"
        ]

        print("Starting training session...")
        session = await assistant.train_employee(
            employee_id="EMP-2025-001",
            topic="Banking Security & Compliance",
            questions=questions
        )

        print("\nTraining Results:")
        print(f"Employee: {session.employee_id}")
        print(f"Topic: {session.topic}")
        print(f"Score: {session.score:.1f}%")
        
        print("\nResponses:")
        for i, response in enumerate(session.responses, 1):
            print(f"\nQuestion {i}: {response['question']}")
            print(f"Response: {response['response']}")

        print("\nFeedback:")
        print(session.feedback)
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
    
    print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_training_assistant())
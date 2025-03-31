#!/usr/bin/env python3
"""
LangChain Adaptive Learning System (LangChain v3)

This example demonstrates a dynamic learning system using three key concepts:
1. example_selectors: Smart example selection
2. lcel: LangChain Expression Language
3. prompt_templates: Dynamic prompting

It provides personalized learning with dynamic example selection and feedback.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(".env")

class DifficultyLevel(str, Enum):
    """Learning difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class TopicArea(str, Enum):
    """Learning subject areas."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    ALGORITHMS = "algorithms"

class LearningExample(BaseModel):
    """Learning example with metadata."""
    topic: TopicArea = Field(description="Subject area")
    level: DifficultyLevel = Field(description="Difficulty level")
    question: str = Field(description="Learning question")
    answer: str = Field(description="Correct answer")
    explanation: str = Field(description="Detailed explanation")
    hints: List[str] = Field(description="Progressive hints")

class LearningResponse(BaseModel):
    """Student response assessment."""
    is_correct: bool = Field(description="Whether answer is correct")
    feedback: str = Field(description="Detailed feedback")
    next_steps: List[str] = Field(description="Recommended next steps")
    relevant_examples: List[str] = Field(description="Similar examples to study")

LEARNING_EXAMPLES = {
    TopicArea.PYTHON: {
        DifficultyLevel.BEGINNER: [
            LearningExample(
                topic=TopicArea.PYTHON,
                level=DifficultyLevel.BEGINNER,
                question="What is the output of: print('Hello ' + 'World')?",
                answer="Hello World",
                explanation="String concatenation combines two strings with '+'",
                hints=["Look at string concatenation", "Think about joining texts"]
            ),
            LearningExample(
                topic=TopicArea.PYTHON,
                level=DifficultyLevel.BEGINNER,
                question="How do you create a list of numbers from 1 to 5?",
                answer="[1, 2, 3, 4, 5]",
                explanation="Lists can be created with square brackets and comma-separated values",
                hints=["Use square brackets", "Separate items with commas"]
            )
        ],
        DifficultyLevel.INTERMEDIATE: [
            LearningExample(
                topic=TopicArea.PYTHON,
                level=DifficultyLevel.INTERMEDIATE,
                question="Write a list comprehension for even numbers from 0 to 10",
                answer="[x for x in range(11) if x % 2 == 0]",
                explanation="List comprehension with conditional filtering",
                hints=["Use range()", "Filter with if condition"]
            )
        ]
    },
    TopicArea.SQL: {
        DifficultyLevel.BEGINNER: [
            LearningExample(
                topic=TopicArea.SQL,
                level=DifficultyLevel.BEGINNER,
                question="Write a query to select all columns from 'users' table",
                answer="SELECT * FROM users;",
                explanation="* selects all columns, FROM specifies the table",
                hints=["Use SELECT", "* means all columns"]
            )
        ]
    }
}

def extract_json(text: str) -> dict:
    """Extract JSON from text, handling code blocks."""
    # Try finding JSON block
    match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    
    # Clean and parse JSON
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except:
        raise ValueError(f"Invalid JSON: {json_str}")

class AdaptiveLearningSystem:
    """Dynamic learning system with example selection."""
    
    def __init__(self):
        """Initialize learning system."""
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Create example template
        example_template = """
        Question: {question}
        Answer: {answer}
        Explanation: {explanation}
        """
        example_prompt = PromptTemplate(
            input_variables=["question", "answer", "explanation"],
            template=example_template
        )
        
        # Get examples
        examples = [ex.model_dump() for examples in LEARNING_EXAMPLES.values() 
                   for level_examples in examples.values() 
                   for ex in level_examples]
        
        # Create example selector
        self.topic_selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=2  # Maximum number of examples
        )
        
        # Create few-shot prompt
        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.topic_selector,
            example_prompt=example_prompt,
            prefix="Here are some similar examples:\n\n",
            suffix="\nNow try this question: {input}",
            input_variables=["input"]
        )
        
        # Create feedback template
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
        
        self.feedback_prompt = ChatPromptTemplate.from_template(feedback_template)
        
        # Create LCEL chain
        get_examples = RunnableLambda(
            lambda x: self.few_shot_prompt.format(input=x["feedback"]["question"])
        )
        
        format_feedback = RunnableLambda(
            lambda x: {
                "question": x["feedback"]["question"],
                "correct_answer": x["feedback"]["correct_answer"],
                "student_answer": x["feedback"]["student_answer"],
                "examples": x["examples"],
                "is_correct": str(x["feedback"]["is_correct"]).lower()
            }
        )
        
        self.learning_chain = (
            RunnablePassthrough.assign(examples=get_examples)
            | format_feedback
            | self.feedback_prompt
            | self.llm
        )
    
    async def get_examples(self, topic: TopicArea, level: DifficultyLevel) -> List[LearningExample]:
        """Get relevant learning examples."""
        return LEARNING_EXAMPLES.get(topic, {}).get(level, [])
    
    async def evaluate_answer(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        topic: TopicArea,
        level: DifficultyLevel
    ) -> LearningResponse:
        """Evaluate student's answer with context."""
        try:
            # Process with chain
            result = await self.learning_chain.ainvoke({
                "topic": topic.value,
                "level": level.value,
                "input": question,
                "feedback": {
                    "question": question,
                    "correct_answer": correct_answer,
                    "student_answer": student_answer,
                    "is_correct": student_answer.strip() == correct_answer.strip()
                }
            })
            
            # Parse result
            json_data = extract_json(result.content)
            return LearningResponse.model_validate(json_data)
            
        except Exception as e:
            return LearningResponse(
                is_correct=False,
                feedback=f"Error evaluating answer: {str(e)}",
                next_steps=["Contact system administrator"],
                relevant_examples=[]
            )

async def demonstrate_system():
    """Demonstrate the learning system."""
    print("\nAdaptive Learning System Demo")
    print("============================\n")
    
    # Create system
    system = AdaptiveLearningSystem()
    
    # Test cases
    test_cases = [
        {
            "topic": TopicArea.PYTHON,
            "level": DifficultyLevel.BEGINNER,
            "question": "What is the output of: print('Hello ' + 'World')?",
            "correct_answer": "Hello World",
            "student_answer": "HelloWorld"
        },
        {
            "topic": TopicArea.SQL,
            "level": DifficultyLevel.BEGINNER,
            "question": "Write a query to select all columns from 'users' table",
            "correct_answer": "SELECT * FROM users;",
            "student_answer": "SELECT all FROM users"
        }
    ]
    
    # Process test cases
    for case in test_cases:
        print(f"\nTopic: {case['topic'].value}")
        print(f"Level: {case['level'].value}")
        print(f"Question: {case['question']}")
        print(f"Student Answer: {case['student_answer']}")
        print("-" * 40)
        
        result = await system.evaluate_answer(
            question=case["question"],
            correct_answer=case["correct_answer"],
            student_answer=case["student_answer"],
            topic=case["topic"],
            level=case["level"]
        )
        
        print(f"Correct: {result.is_correct}")
        print(f"Feedback: {result.feedback}")
        print("\nNext Steps:")
        for step in result.next_steps:
            print(f"- {step}")
        print("\nRelevant Examples:")
        for example in result.relevant_examples:
            print(f"- {example}")
        
        print("-" * 40)

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_system())

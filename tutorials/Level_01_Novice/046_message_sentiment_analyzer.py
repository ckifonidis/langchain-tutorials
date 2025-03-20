#!/usr/bin/env python3
"""
LangChain Message Sentiment Analyzer Example

This example demonstrates how to use OpenEvals capabilities along with an embeddings model
to create a system that can analyze message sentiment using semantic understanding and 
provide detailed evaluation results.

Note: For setup instructions and package requirements, please refer to USAGE_GUIDE.md in the root directory.
"""

import os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import numpy as np

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openevals.llm import create_llm_as_judge

# Load environment variables
load_dotenv()

# Check for required environment variables for chat model
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class SentimentScore(BaseModel):
    """Schema for sentiment scoring."""
    positive_score: float = Field(description="Positive sentiment score (0-1)")
    neutral_score: float = Field(description="Neutral sentiment score (0-1)")
    negative_score: float = Field(description="Negative sentiment score (0-1)")
    dominant_sentiment: str = Field(description="Dominant sentiment category")

class MessageAnalysis(BaseModel):
    """Schema for message analysis results."""
    message_id: str = Field(description="Unique message identifier")
    text: str = Field(description="Message text")
    sentiment: SentimentScore = Field(description="Sentiment analysis")
    confidence_score: float = Field(description="Analysis confidence (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisResults(BaseModel):
    """Schema for analysis result collection."""
    batch_id: str = Field(description="Unique batch identifier")
    messages: List[MessageAnalysis] = Field(description="Message analyses")
    summary: Dict[str, float] = Field(description="Summary statistics")
    timestamp: datetime = Field(default_factory=datetime.now)

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def create_embeddings() -> AzureOpenAIEmbeddings:
    """Initialize the Azure OpenAI embeddings model."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        deployment=os.getenv("AZURE_DEPLOYMENT")
    )

def create_sentiment_evaluator(llm: AzureChatOpenAI):
    """Create a sentiment evaluator using OpenEvals with a custom sentiment prompt."""
    sentiment_prompt = (
        "Analyze the sentiment of the following text and provide scores for:\n"
        "- Positive sentiment (0-1)\n"
        "- Neutral sentiment (0-1)\n"
        "- Negative sentiment (0-1)\n\n"
        "Text: {inputs}\n\n"
        "Respond with a JSON object containing the scores and feedback."
    )
    return create_llm_as_judge(
        prompt=sentiment_prompt,
        feedback_key="sentiment",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        judge=llm
    )

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def analyze_sentiment(messages: List[str], embeddings: AzureOpenAIEmbeddings) -> AnalysisResults:
    """
    Analyze sentiment of messages using embeddings.

    For each message, compute its embedding and compare it via cosine similarity to 
    precomputed embeddings of reference examples for each sentiment category. The highest 
    similarity determines the dominant sentiment. The similarity value is used as the confidence.

    Args:
        messages: List of messages to analyze.
        embeddings: Embedding model.

    Returns:
        AnalysisResults: Comprehensive sentiment analysis.
    """
    sentiment_examples = {
        "Positive": [
            "I'm really happy with this!",
            "Great job, thank you so much!",
            "This exceeds my expectations!"
        ],
        "Neutral": [
            "The product works as described.",
            "I received the delivery today.",
            "Let me know when it's ready."
        ],
        "Negative": [
            "This is not what I expected.",
            "I'm disappointed with the service.",
            "There are several issues to fix."
        ]
    }
    
    ref_embeddings = {
        category: [embeddings.embed_query(example) for example in examples]
        for category, examples in sentiment_examples.items()
    }
    
    analyses = []
    for i, message in enumerate(messages):
        try:
            msg_embedding = embeddings.embed_query(message)
            scores = {}
            for category, emb_list in ref_embeddings.items():
                sim_values = [cosine_similarity(msg_embedding, ref_emb) for ref_emb in emb_list]
                max_sim = max(sim_values) if sim_values else 0.0
                scores[f"{category.lower()}_score"] = max_sim
            dominant_key = max(scores, key=scores.get)
            dominant_category = dominant_key.replace("_score", "").capitalize()
            confidence = scores[dominant_key]
            
            analysis = MessageAnalysis(
                message_id=f"MSG{i+1:03d}",
                text=message,
                sentiment=SentimentScore(
                    positive_score=scores.get("positive_score", 0.0),
                    neutral_score=scores.get("neutral_score", 0.0),
                    negative_score=scores.get("negative_score", 0.0),
                    dominant_sentiment=dominant_category
                ),
                confidence_score=confidence
            )
            analyses.append(analysis)
        except Exception:
            analysis = MessageAnalysis(
                message_id=f"MSG{i+1:03d}",
                text=message,
                sentiment=SentimentScore(
                    positive_score=0.0,
                    neutral_score=1.0,
                    negative_score=0.0,
                    dominant_sentiment="Neutral"
                ),
                confidence_score=0.0
            )
            analyses.append(analysis)
    
    avg_positive = sum(a.sentiment.positive_score for a in analyses) / len(analyses)
    avg_neutral = sum(a.sentiment.neutral_score for a in analyses) / len(analyses)
    avg_negative = sum(a.sentiment.negative_score for a in analyses) / len(analyses)
    avg_confidence = sum(a.confidence_score for a in analyses) / len(analyses)
    
    summary = {
        "avg_positive_score": avg_positive,
        "avg_neutral_score": avg_neutral,
        "avg_negative_score": avg_negative,
        "avg_confidence": avg_confidence
    }
    
    batch_id = f"BATCH{hash(''.join(messages)) % 1000:03d}"
    return AnalysisResults(
        batch_id=batch_id,
        messages=analyses,
        summary=summary
    )

def demonstrate_sentiment_analysis():
    """Demonstrate message sentiment analysis capabilities."""
    llm = create_chat_model()
    embeddings = create_embeddings()
    _ = create_sentiment_evaluator(llm)
    
    messages = [
        "I absolutely love this product! It's exactly what I needed.",
        "The delivery arrived on time as scheduled.",
        "This is disappointing. The quality is not what I expected.",
        "The customer service team was incredibly helpful!",
        "I'm having trouble with the installation process."
    ]
    
    results = analyze_sentiment(messages, embeddings)
    
    # Display detailed results
    output = f"Batch ID: {results.batch_id}\n\n"
    for analysis in results.messages:
        output += (
            f"Message ID: {analysis.message_id}\n"
            f"Text: {analysis.text}\n"
            f"Dominant Sentiment: {analysis.sentiment.dominant_sentiment} "
            f"(Confidence: {analysis.confidence_score:.4f})\n"
            f"Scores:\n"
            f"  Positive: {analysis.sentiment.positive_score:.4f}\n"
            f"  Neutral:  {analysis.sentiment.neutral_score:.4f}\n"
            f"  Negative: {analysis.sentiment.negative_score:.4f}\n"
            f"{'-'*50}\n"
        )
    
    output += "\nBatch Summary:\n"
    for metric, value in results.summary.items():
        output += f"{metric.replace('_', ' ').capitalize()}: {value:.4f}\n"
    
    print(output)

def main():
    """Main entry point for the example."""
    demonstrate_sentiment_analysis()

if __name__ == "__main__":
    main()

"""
LangChain Price Alert System Example

This example demonstrates how to combine memory and streaming capabilities to create
a price alert system that tracks prices, manages thresholds, and provides real-time
notifications.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import time
import random
from typing import List, Dict, Any, Generator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()

class PriceAlert(BaseModel):
    """Schema for price alerts."""
    symbol: str = Field(description="Trading symbol (e.g., 'BTC-USD')")
    threshold: float = Field(description="Price threshold for alert")
    direction: str = Field(description="Alert direction ('above' or 'below')")
    created_at: datetime = Field(default_factory=datetime.now)

class PriceUpdate(BaseModel):
    """Schema for price updates."""
    symbol: str = Field(description="Trading symbol")
    price: float = Field(description="Current price")
    timestamp: datetime = Field(default_factory=datetime.now)

class AlertCallback(BaseCallbackHandler):
    """Custom callback handler for streaming alerts."""
    
    def on_llm_start(self, *args, **kwargs) -> None:
        """Handle LLM start event."""
        print("\nAnalyzing price movement...")
    
    def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """Handle new token event."""
        # Remove duplicate tokens and fix formatting
        if token.strip() and not token.isspace():
            processed_token = ' '.join(token.split())
            print(processed_token, end=' ', flush=True)
    
    def on_llm_end(self, *args, **kwargs) -> None:
        """Handle LLM end event."""
        print("\nAnalysis complete.\n")

def create_chat_model() -> AzureChatOpenAI:
    """Initialize the Azure ChatOpenAI model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        streaming=True,
        callbacks=[AlertCallback()]
    )

def simulate_price_stream() -> Generator[PriceUpdate, None, None]:
    """Simulate a stream of price updates."""
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    base_prices = {"BTC-USD": 65000, "ETH-USD": 3500, "SOL-USD": 150}
    
    while True:
        for symbol in symbols:
            base = base_prices[symbol]
            current = base * (1 + random.uniform(-0.02, 0.02))
            yield PriceUpdate(
                symbol=symbol,
                price=round(current, 2)
            )
        time.sleep(1)

class AlertManager:
    """Manages price alerts and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.llm = create_chat_model()
        self.memory = ConversationBufferMemory()
        self.alerts: List[PriceAlert] = []
    
    def add_alert(self, alert: PriceAlert) -> None:
        """Add a new price alert."""
        self.alerts.append(alert)
        self.memory.save_context(
            {"input": f"Adding alert for {alert.symbol}"},
            {"output": f"Alert set at {alert.threshold} {alert.direction}"}
        )
        print(f"Alert added: {alert.symbol} {alert.direction} {alert.threshold}")
    
    def check_price(self, update: PriceUpdate) -> None:
        """Check if price update triggers any alerts."""
        for alert in self.alerts:
            if alert.symbol == update.symbol:
                triggered = (
                    (alert.direction == "above" and update.price > alert.threshold) or
                    (alert.direction == "below" and update.price < alert.threshold)
                )
                if triggered:
                    self._handle_alert(alert, update)
    
    def _handle_alert(self, alert: PriceAlert, update: PriceUpdate) -> None:
        """Handle triggered alerts."""
        # Get historical context from memory
        history = self.memory.load_memory_variables({})
        
        # Analyze price movement with context
        messages = [
            HumanMessage(content=f"""
Analyze this price movement for {alert.symbol}:
- Alert threshold: {alert.threshold} ({alert.direction})
- Current price: {update.price}
- Historical context: {history}

Provide a brief analysis of the price movement and potential implications.
""")
        ]
        
        # Stream analysis with callbacks
        self.llm.invoke(messages)
        
        # Update memory with alert trigger
        self.memory.save_context(
            {"input": f"Alert triggered for {alert.symbol}"},
            {"output": f"Price {update.price} crossed threshold {alert.threshold}"}
        )
        
        # Remove triggered alert
        self.alerts.remove(alert)

def demonstrate_price_alerts():
    """Demonstrate the Price Alert System capabilities."""
    try:
        print("\nInitializing Price Alert System...\n")
        
        # Create alert manager
        manager = AlertManager()
        
        # Add some example alerts
        alerts = [
            PriceAlert(symbol="BTC-USD", threshold=66000, direction="above"),
            PriceAlert(symbol="ETH-USD", threshold=3400, direction="below"),
            PriceAlert(symbol="SOL-USD", threshold=155, direction="above")
        ]
        
        for alert in alerts:
            manager.add_alert(alert)
        
        print("\nMonitoring prices...")
        
        # Monitor price stream
        for update in simulate_price_stream():
            print(f"\r{update.symbol}: ${update.price:.2f}", end="", flush=True)
            manager.check_price(update)
            
            # Exit after processing a few updates for demonstration
            if not manager.alerts:
                break
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    print("\nDemonstrating LangChain Price Alert System...")
    demonstrate_price_alerts()

if __name__ == "__main__":
    main()
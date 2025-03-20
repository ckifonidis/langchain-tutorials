"""
LangChain Key-Value Stores Example

This example demonstrates how to use key-value stores in LangChain for data 
persistence and retrieval. Compatible with LangChain v0.3 and Pydantic v2.

Note: For setup instructions and package requirements, please refer to
USAGE_GUIDE.md in the root directory.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.storage import LocalFileStore

# Load environment variables
load_dotenv()

class DataEntry(BaseModel):
    """Schema for data entries in the key-value store."""
    key: str = Field(description="Unique identifier for the data")
    value: Any = Field(description="The stored data")
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now, description="When the entry was created/updated")

def create_store(path: str = "./.langchain/stores") -> LocalFileStore:
    """
    Create a local file-based key-value store.
    
    Args:
        path: Directory path for the store
        
    Returns:
        LocalFileStore: The initialized store
    """
    try:
        # Create store directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Initialize the store
        store = LocalFileStore(path)
        # Save the directory path as an attribute for later use
        store.path = path
        return store
        
    except Exception as e:
        print(f"Error creating store: {str(e)}")
        raise

def store_data(store: LocalFileStore, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> DataEntry:
    """
    Store data in the key-value store.
    
    Args:
        store: The key-value store
        key: Unique identifier
        value: Data to store
        metadata: Optional metadata
        
    Returns:
        DataEntry: The stored data entry
    """
    try:
        # Create data entry
        entry = DataEntry(
            key=key,
            value=value,
            metadata=metadata or {}
        )
        
        # Convert the entry to JSON bytes (using default=str for datetime) and store it.
        store.mset([(key, json.dumps(entry.model_dump(), default=str).encode("utf-8"))])
        
        return entry
        
    except Exception as e:
        print(f"Error storing data: {str(e)}")
        raise

def retrieve_data(store: LocalFileStore, key: str) -> Optional[DataEntry]:
    """
    Retrieve data from the key-value store.
    
    Args:
        store: The key-value store
        key: Key to retrieve
        
    Returns:
        Optional[DataEntry]: The retrieved data entry, if found
    """
    try:
        # Get data from store
        data = store.mget([key])[0]
        
        if data is None:
            return None
        
        # Convert from bytes to dict
        data_dict = json.loads(data.decode("utf-8"))
        return DataEntry(**data_dict)
        
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        raise

def list_keys(store: LocalFileStore) -> List[str]:
    """
    List all keys in the store.
    
    Args:
        store: The key-value store
        
    Returns:
        List[str]: List of keys
    """
    try:
        # Use the stored directory path to list files (keys)
        return os.listdir(store.path)
    except Exception as e:
        print(f"Error listing keys: {str(e)}")
        raise

def demonstrate_key_value_store():
    """Demonstrate key-value store capabilities."""
    try:
        print("\nDemonstrating LangChain Key-Value Stores...\n")
        
        # Example 1: Basic Storage and Retrieval
        print("Example 1: Basic Storage and Retrieval")
        print("-" * 50)
        
        # Create store
        store = create_store()
        
        # Store some data
        test_data = [
            {
                "key": "user_1",
                "value": {"name": "Alice", "age": 30},
                "metadata": {"type": "user_profile", "active": True}
            },
            {
                "key": "user_2",
                "value": {"name": "Bob", "age": 25},
                "metadata": {"type": "user_profile", "active": False}
            },
            {
                "key": "settings_1",
                "value": {"theme": "dark", "notifications": True},
                "metadata": {"type": "user_settings"}
            }
        ]
        
        for data in test_data:
            entry = store_data(store, data["key"], data["value"], data["metadata"])
            print(f"\nStored entry:")
            print(f"Key: {entry.key}")
            print(f"Value: {entry.value}")
            print(f"Metadata: {entry.metadata}")
            print(f"Timestamp: {entry.timestamp}")
        
        # Retrieve data
        print("\nRetrieving data:")
        for key in ["user_1", "settings_1"]:
            entry = retrieve_data(store, key)
            if entry:
                print(f"\nRetrieved entry for key '{key}':")
                print(f"Value: {entry.value}")
                print(f"Metadata: {entry.metadata}")
        print("=" * 50)
        
        # Example 2: Key Management
        print("\nExample 2: Key Management")
        print("-" * 50)
        
        # List all keys
        keys = list_keys(store)
        print("\nAll keys in store:")
        for key in keys:
            print(f"- {key}")
            
        # Check specific keys
        test_keys = ["user_1", "nonexistent_key"]
        print("\nChecking specific keys:")
        for key in test_keys:
            entry = retrieve_data(store, key)
            if entry:
                print(f"\nFound entry for '{key}':")
                print(f"Value: {entry.value}")
            else:
                print(f"\nNo entry found for '{key}'")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    """Main entry point for the example."""
    demonstrate_key_value_store()

if __name__ == "__main__":
    main()
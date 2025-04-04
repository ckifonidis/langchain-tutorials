#!/usr/bin/env python3
"""Test script for Banking Document Processor."""

import asyncio
import os
from datetime import datetime
from banking_document_processor import (
    BankingDocumentProcessor, 
    BankingDocument,
    DepartmentType, 
    DocumentType
)

async def run_tests():
    """Run document processor tests."""
    print("\nBanking Document Processor Tests")
    print("===============================\n")

    try:
        # Initialize processor
        processor = BankingDocumentProcessor()

        # Test documents
        test_docs = [
            BankingDocument(
                doc_id="API-2025-001",
                title="Payment API Documentation",
                department=DepartmentType.DEVELOPMENT,
                doc_type=DocumentType.API_SPEC,
                filepath="./docs/api_spec.md",
                created_at=datetime.now().isoformat(),
                metadata={"version": "2.0"}
            ),
            BankingDocument(
                doc_id="POL-2025-001",
                title="Data Privacy Policy",
                department=DepartmentType.LEGAL,
                doc_type=DocumentType.POLICY,
                filepath="./docs/policy.txt",
                created_at=datetime.now().isoformat(),
                metadata={"region": "EU"}
            ),
            BankingDocument(
                doc_id="FIN-2025-001",
                title="Q1 Financial Report",
                department=DepartmentType.FINANCE,
                doc_type=DocumentType.FINANCIAL,
                filepath="./docs/q1_report.txt",
                created_at=datetime.now().isoformat(),
                metadata={"quarter": "Q1"}
            )
        ]

        # Verify files exist
        print("Checking test files...")
        for doc in test_docs:
            if os.path.exists(doc.filepath):
                print(f"✓ Found {doc.title}")
            else:
                print(f"✗ Missing {doc.filepath}")

        print("\nProcessing documents...")
        metrics = processor.process_documents(test_docs)
        
        print("\nProcessing Metrics:")
        print(f"Total Documents: {metrics.total_docs}")
        print(f"Total Chunks: {metrics.total_chunks}")
        print(f"Average Chunk Size: {metrics.avg_chunk_size:.2f}")
        print(f"Vector Store Size: {metrics.index_size}")

        # Test queries for each department
        print("\nTesting queries...")
        queries = [
            ("Find information about API authentication", DepartmentType.DEVELOPMENT),
            ("What are the data retention periods?", DepartmentType.LEGAL),
            ("Show Q1 revenue and growth", DepartmentType.FINANCE)
        ]

        for query, dept in queries:
            print(f"\nQuery: {query}")
            print(f"Department: {dept.value}")
            results = processor.search_documents(query, department=dept)
            
            print("\nResults:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. Document: {doc.metadata.get('title', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")

    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
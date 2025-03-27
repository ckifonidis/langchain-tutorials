# Transaction Rule Engine with LangChain: Complete Guide

## Introduction

This example demonstrates a banking transaction rule engine using LangChain's chain composition (LCEL) and tool_calling capabilities. It evaluates transactions against predefined rules and executes appropriate actions based on the evaluation results.

The implementation showcases:
- Direct rule implementation in Python
- Transaction evaluation chain
- Action execution using tools
- Rich error handling

## Core LangChain Concepts

### 1. LCEL (LangChain Expression Language)

The example uses LCEL for composing the rule evaluation and action execution chain:

1. Chain Structure:
   ```python
   chain = (
       RunnablePassthrough()
       | {
           "result": eval_chain,
           "transaction": RunnablePassthrough()
       }
       | RunnableLambda(execute_actions)
   )
   ```

2. Benefits:
   - Clear data flow
   - Type safety
   - Easy composition
   - Error propagation

### 2. tool_calling

The implementation uses tool_calling for executing actions:

1. Tool Definitions:
   ```python
   @tool("approve_transaction")
   def approve_transaction(transaction_id: str, notes: str) -> Dict[str, Any]:
       """Approve a transaction with notes."""
       return {
           "status": "approved",
           "transaction_id": transaction_id,
           "notes": notes
       }
   ```

2. Tool Invocation:
   ```python
   result = approve_transaction.invoke({
       "transaction_id": tx_id,
       "notes": "Transaction approved"
   })
   ```

## Implementation Components

### 1. Transaction Rules

1. Amount-Based Rules:
   ```python
   if amount < 10000:
       result = RuleResult(
           action=RuleAction.APPROVE,
           risk_level="low"
       )
   elif amount <= 50000:
       result = RuleResult(
           action=RuleAction.FLAG,
           risk_level="medium"
       )
   else:
       result = RuleResult(
           action=RuleAction.DENY,
           risk_level="high"
       )
   ```

2. Foreign Transaction Rules:
   ```python
   if is_foreign:
       if result.risk_level == "low":
           result.risk_level = "medium"
       elif result.risk_level == "medium":
           result.risk_level = "high"
       result.required_actions.append("notify_compliance")
   ```

### 2. Chain Components

1. Evaluation Chain:
   - Direct Python rule implementation
   - Clear business logic
   - Type-safe results
   - Rich metadata

2. Action Chain:
   - Tool-based execution
   - Response collection
   - Status tracking
   - Error handling

## Advanced Features

### 1. Rule Processing

1. Transaction Evaluation:
   - Amount validation
   - Destination checking
   - Risk assessment
   - Required actions

2. Action Execution:
   - Primary actions
   - Additional notifications
   - Status tracking
   - Result collection

### 2. Error Management

1. Exception Types:
   - Validation errors
   - Tool execution errors
   - Chain errors
   - Business rule errors

2. Recovery:
   - Clear error messages
   - Status reporting
   - Chain recovery
   - Transaction tracking

## Expected Output

### 1. Standard Transaction

```text
Processing Transaction: TRX001
Type: transfer
Amount: $5,000.00 USD
From: ACCT123
To: ACCT456
----------------------------------------

Rule Evaluation Result:
Action: APPROVE
Reason: Amount $5,000.00 USD is within standard limit
Risk Level: LOW

Executed Actions:
- Approved: Amount $5,000.00 USD is within standard limit
----------------------------------------
```

### 2. High-Risk Foreign Transaction

```text
Processing Transaction: TRX002
Type: transfer
Amount: $75,000.00 USD
From: ACCT789
To: FOREIGN_ACCT
----------------------------------------

Rule Evaluation Result:
Action: DENY
Reason: Amount $75,000.00 USD exceeds maximum limit and foreign destination detected
Risk Level: HIGH

Executed Actions:
- Denied: Amount $75,000.00 USD exceeds maximum limit and foreign destination detected
- Notified: Amount $75,000.00 USD exceeds maximum limit and foreign destination detected
----------------------------------------
```

## Best Practices

### 1. Rule Implementation

1. Organization:
   - Clear rule structure
   - Readable conditions
   - Maintainable logic 
   - Easy updates

2. Error Handling:
   - Proper validation
   - Clear messages
   - Recovery options
   - Status tracking

### 2. Chain Design

1. Structure:
   - Clear components
   - Type safety
   - Error propagation
   - Easy testing

2. Tool Usage:
   - Clear interfaces
   - Error handling
   - Status tracking
   - Result collection

## References

### 1. LangChain Concepts

- [LCEL Overview](https://python.langchain.com/docs/expression_language/)
- [Tools Guide](https://python.langchain.com/docs/modules/agents/tools/)
- [Chain Composition](https://python.langchain.com/docs/expression_language/how_to/compose)
- [Error Handling](https://python.langchain.com/docs/expression_language/error_handling)

### 2. Tool Integration

- [Tool Creation](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Tool Invocation](https://python.langchain.com/docs/modules/agents/tools/how_to/invoke)
- [Tool Response Handling](https://python.langchain.com/docs/modules/agents/tools/how_to/handle_responses)

### 3. Best Practices

- [Chain Design](https://python.langchain.com/docs/expression_language/why)
- [Tool Building](https://python.langchain.com/docs/modules/agents/tools/how_to/custom_tools)
- [Error Management](https://python.langchain.com/docs/guides/errors)
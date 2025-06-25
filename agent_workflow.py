from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.trade_agent import make_portfolio_allocation
from agents.sentiment_agent import classify_sentiment
from agents.fraud_agent import predict_fraud
from agents.fact_checker_agent import fact_check
import pdb
import numpy as np

# Define state keys used in workflow
def empty_state():
    return {
        "transaction": None,
        "sentiment": None,
        "trading_action": None,
        "suspicious": None,
        "fact_check": None,
        "claim_sentiment": None,
        "final_decision": None,

    }

# New function to initialize state with parameters
def initialize_state(price_data=None, news=None, transaction=None, claim=None, additional_context=None):
    """
    Initialize the workflow state with input parameters.
    
    Args:
        price_data (dict): Historical or real-time price data for trading decisions
        news (str): Financial news headline or text for sentiment analysis
        transaction (dict): Transaction details to check for fraud
        claim (str): Statement to fact-check
        additional_context (dict): Any additional context needed for the workflow
        
    Returns:
        dict: Initial state with all required parameters
    """
    state = empty_state()
    
    # Add input parameters to state
    state["price_data"] = price_data
    state["news"] = news
    state["transaction"] = transaction
    state["claim"] = claim
    
    # Add any additional context parameters
    if additional_context:
        for key, value in additional_context.items():
            state[key] = value
            
    return state

# Step 1: TraderAgent generates action
def trader_node(state):
    action = make_portfolio_allocation(state["price_data"])
    state["trading_action"] = action
    print(f"Trading action generated: {action}")
    return state

# Step 2: SentimentAgent analyzes news
def sentiment_node(state):
    headline = state["news"]
    sentiment = classify_sentiment(headline)
    state["sentiment"] = sentiment
    print(f"Sentiment analysis result: {sentiment}")
    decision_orchestrator(state)
    return state

# Step 3: FraudAgent evaluates transaction
def fraud_node(state):
    suspicious = predict_fraud(state["transaction"], model_path="agents/models/fraud_detector.pkl")
    #if more than 1% transactions are flagged as suspicious, set suspicious to True
    if isinstance(suspicious, np.ndarray):
        suspicious = sum(suspicious) / len(suspicious) > 0.01
    else:
        suspicious = suspicious > 0.01
    #convert to boolean
    state["suspicious"] = bool(suspicious)
    print(f"Fraud detection result: {'Suspicious' if state['suspicious'] else 'Not Suspicious'}")
    decision_orchestrator(state)
    return state

# Step 4: FactChecker evaluates decision rationale if needed
def factcheck_node(state):
    pdb.set_trace()
    # Get claim from state or generate one based on available data
    if "claim" in state and state["claim"]:
        claim = state["claim"]
    else:
        # Generate a default claim based on available data
        claim = "Market conditions for trading appear favorable based on recent news"
    
    result = fact_check(claim)
    state["fact_check"] = result
    print(f"Fact check result: {result}")
    decision_orchestrator(state)
    return state

# Step 5: Orchestrator decides next step (not used in new workflow but kept for reference)
def decision_orchestrator(state):
    if state["suspicious"]:
        state["final_decision"] = "🔴 Escalate: Transaction flagged"
    elif state["sentiment"]["sentiment"] == "negative":
        state["final_decision"] = "🟡 Caution: Negative market sentiment"
    else:
        state["final_decision"] = "🟢 Proceed with trade"
    
    # The routing function will handle deciding the next node
    print(f"Final decision: {state['final_decision']}")
    return state

# Additional function to check sentiment on a claim
def claim_sentiment_node(state):
    if state["fact_check"] and "claim" in state["fact_check"]:
        claim_text = state["fact_check"]["claim"]
        sentiment = classify_sentiment(claim_text)
        state["claim_sentiment"] = sentiment
    else:
        # Default to neutral if no claim is available
        state["claim_sentiment"] = {"sentiment": "neutral", "confidence": 0.5}

    decision_orchestrator(state)

    return state

# Build LangGraph
def build_finagentx_workflow():
    # Define the state schema for the graph
    from typing import TypedDict, Optional, Dict, Any, List, Union
    
    class FinAgentState(TypedDict):
        price_data: Dict[str, Dict[str, float]]  # Stock price data
        news: str  # News headline
        transaction: Optional[Dict[str, Any]]  # Transaction details
        sentiment: Optional[Dict[str, Union[str, float]]]  # Sentiment analysis results
        trading_action: Optional[str]  # The trading decision
        suspicious: Optional[bool]  # Fraud detection results
        fact_check: Optional[Dict[str, Any]]  # Fact checking results
        claim_sentiment: Optional[Dict[str, Union[str, float]]]  # Sentiment on claim
        final_decision: Optional[str]  # Final decision outcome
    
    # Create StateGraph with the schema
    builder = StateGraph(state_schema=FinAgentState)
    
    # Use node names that don't conflict with state keys
    builder.add_node("sentiment_agent", sentiment_node)
    builder.add_node("fraud_agent", fraud_node)
    builder.add_node("factcheck_agent", factcheck_node)
    builder.add_node("trader_agent", trader_node)
    builder.add_node("claim_sentiment_agent", claim_sentiment_node)

    # Define routing functions for conditional edges
    def route_after_initial_checks(state):
        # If sentiment is negative or transaction is suspicious, end workflow (no trading)
        if (state["sentiment"] and state["sentiment"]["sentiment"] == "negative") or state["suspicious"]:
            state["final_decision"] = "❌ No Trading: Risk detected in initial checks"
            return END
        else:
            # Otherwise, proceed to fact checking
            return "factcheck_agent"
    
    def route_after_factcheck(state):
        # If fact check failed, end workflow
        if not state["fact_check"] or not state.get("fact_check", {}).get("verified", False):
            return "trader_agent"
        else:
            # If verified, analyze sentiment on the claim
            return "claim_sentiment_agent"
    
    def route_after_claim_sentiment(state):
        pdb.set_trace()
        # If claim sentiment is negative, end workflow
        if state["claim_sentiment"]["sentiment"] == "negative":
            state["final_decision"] = "❌ No Trading: Negative sentiment on verified claim"
            return END
        else:
            # Otherwise, proceed to trading
            return "trader_agent"

    # Set up the workflow
    builder.set_entry_point("sentiment_agent")
    builder.add_edge("sentiment_agent", "fraud_agent")
    
    # Conditional routing after fraud check
    builder.add_conditional_edges(
        "fraud_agent",
        route_after_initial_checks
    )
    
    # Conditional routing after fact check
    builder.add_conditional_edges(
        "factcheck_agent",
        route_after_factcheck
    )
    
    # Conditional routing after claim sentiment analysis
    builder.add_conditional_edges(
        "claim_sentiment_agent",
        route_after_claim_sentiment
    )
    
    # Final node
    builder.add_edge("trader_agent", END)


    return builder.compile()

# Add a function to create and run the workflow with inputs
def run_financial_workflow(price_data, news, transaction=None, claim=None, additional_context=None):
    """
    Create and run the financial agent workflow with specified inputs.
    
    Args:
        price_data (dict): Price data for trading decisions (required)
        news (str): News headline or text for sentiment analysis (required)
        transaction (dict): Transaction details for fraud detection (optional)
        additional_context (dict): Any additional context parameters (optional)
        
    Returns:
        dict: Final state after workflow completion
    """
    # Create the workflow
    workflow = build_finagentx_workflow()
    
    # Initialize state with input parameters
    initial_state = initialize_state(
        price_data=price_data,
        news=news,
        transaction=transaction,
        claim=claim,
        additional_context=additional_context
    )
    
    # Run the workflow with the initialized state
    final_event = None
    for event in workflow.stream(initial_state):
        # Keep track of the latest event
        final_event = event
        # You could log events here if needed
    
    pdb.set_trace()
    # Return the final state
    if isinstance(final_event, dict):
        # If the event is already the state dictionary
        return final_event
    elif hasattr(final_event, 'state'):
        # If the event has a state attribute (older versions)
        return final_event.state
    elif hasattr(final_event, 'values'):
        # If the event has values attribute
        return final_event.values
    else:
        # Fallback to the entire event
        return final_event



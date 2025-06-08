from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.trader_agent import make_trading_decision
from agents.sentiment_agent import classify_sentiment
from agents.fraud_agent import is_transaction_suspicious
from rag.fact_checker import check_claim

# Define state keys used in workflow
def empty_state():
    return {
        "transaction": None,
        "sentiment": None,
        "trading_action": None,
        "suspicious": None,
        "fact_check": None,
        "final_decision": None
    }

# Step 1: TraderAgent generates action
def trader_node(state):
    action = make_trading_decision(state["price_data"])
    state["trading_action"] = action
    return state

# Step 2: SentimentAgent analyzes news
def sentiment_node(state):
    headline = state["news"]
    sentiment = classify_sentiment(headline)
    state["sentiment"] = sentiment
    return state

# Step 3: FraudAgent evaluates transaction
def fraud_node(state):
    suspicious = is_transaction_suspicious(state["transaction"])
    state["suspicious"] = suspicious
    return state

# Step 4: FactChecker evaluates decision rationale if needed
def factcheck_node(state):
    claim = f"Proceeding with trade action: {state['trading_action']}"
    result = check_claim(claim)
    state["fact_check"] = result
    return state

# Step 5: Orchestrator decides next step
def decision_orchestrator(state):
    if state["suspicious"]:
        state["final_decision"] = "🔴 Escalate: Transaction flagged"
        return "factcheck"
    elif state["sentiment"]["sentiment"] == "negative":
        state["final_decision"] = "🟡 Caution: Negative market sentiment"
        return "factcheck"
    else:
        state["final_decision"] = "🟢 Proceed with trade"
        return END

# Build LangGraph
def build_finagentx_workflow():
    builder = StateGraph()
    builder.add_node("trader", trader_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("fraud", fraud_node)
    builder.add_node("factcheck", factcheck_node)
    builder.add_node("orchestrator", decision_orchestrator)

    builder.set_entry_point("trader")
    builder.add_edge("trader", "sentiment")
    builder.add_edge("sentiment", "fraud")
    builder.add_edge("fraud", "orchestrator")
    builder.add_conditional_edges("orchestrator", {
        "factcheck": "factcheck",
        END: END
    })
    builder.add_edge("factcheck", END)

    return builder.compile()

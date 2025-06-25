# Example price data
# want the data for at least 90 days for trading decisions in the below format
# generate synthetic data for testing for tic = ["AAPL", "MSFT", "GOOG", "AMZN"]
# capture this 90 days of data in a dataframe with columns:
# timestamp, close, high, low, open, volume, tic
from agent_workflow import run_financial_workflow

import pandas as pd
from datetime import datetime, timedelta
import pdb
import os
import json
from openai import OpenAI
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#price_df
price_df = pd.DataFrame()

def generate_synthetic_price_data(ticker, days=90):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    #Generate date in 2025-03-27 format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    # Generate a date range
    # This will create a date range from start_date to end_date with daily frequency
    # pd.date_range will generate dates in the format YYYY-MM-DD
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    data = {
        "date": dates,
        "close": [round(100 + i * 0.5 + (i % 10) * 0.1, 2) for i in range(len(dates))],
        "high": [round(100 + i * 0.6 + (i % 10) * 0.2, 2) for i in range(len(dates))],
        "low": [round(100 + i * 0.4 + (i % 10) * 0.05, 2) for i in range(len(dates))],
        "open": [round(100 + i * 0.55 + (i % 10) * 0.15, 2) for i in range(len(dates))],
        "volume": [round(1000 + i * 10 + (i % 10) * 5, 2) for i in range(len(dates))],
        "tic": ticker
    }
    return pd.DataFrame(data)

# Example transaction
#Generate transactions on MSFT, AAPL, GOOG, AMZN, META for the last 10 days in a dataframe with columns:
#txn_id,user_id,timestamp,merchant,instrument,intent,amount,order_type,channel
#instrument is one of the tickers in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
from datetime import datetime, timedelta
import random 
def generate_synthetic_transactions(tickers, days=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate a date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    transactions = []   
    for date in dates:
        for ticker in tickers:
            txn_id = f"txn_{date.strftime('%Y%m%d')}_{ticker}"
            user_id = f"user_{random.randint(1, 10)}"
            amount = round(random.uniform(1000, 10000), 2)
            intent = random.choice(["buy", "sell"])
            order_type = random.choice(["market", "limit"])
            channel = random.choice(["web", "mobile", "API"])
            merchant = random.choice(["Robinhood", "Wealthfront", "TD Ameritrade", "Fidelity"])
            transactions.append({
                "txn_id": txn_id,
                "user_id": user_id,
                "timestamp": date.strftime('%Y-%m-%d %H:%M:%S'),
                "ticker": ticker,
                "amount": amount,
                "intent": intent,
                "order_type": order_type,
                "channel": channel,
                "merchant": merchant,
                "instrument": ticker
            })
    return pd.DataFrame(transactions)

def process_final_state_with_gpt4o(final_state):
    """
    Process the final state using OpenAI's GPT-4o model to generate insights and recommendations.
    
    Args:
        final_state (dict): The final state from the workflow
        
    Returns:
        dict: GPT-4o response containing insights and recommendations
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return {"error": "OpenAI API key not found"}
    
    # Prepare a concise version of the state for the API call
    # Remove large data structures and keep only relevant information
    api_state = {
        "final_decision": final_state.get("final_decision"),
        "sentiment": final_state.get("sentiment"),
        "suspicious": final_state.get("suspicious"),
        "fact_check": final_state.get("fact_check"),
        "claim_sentiment": final_state.get("claim_sentiment"),
        "trading_action": final_state.get("trading_action"),
    }
    
    # Include summary of price data
    if "price_data" in final_state and isinstance(final_state["price_data"], pd.DataFrame):
        # Get latest price for each ticker
        latest_prices = final_state["price_data"].groupby("tic").tail(1)[["tic", "close"]].to_dict(orient="records")
        api_state["latest_prices"] = latest_prices
    
    # Add market news
    api_state["news"] = final_state.get("news")
    
    # Create a prompt for GPT-4o
    system_prompt = """You are FinGPT, an AI financial analyst assistant. 
    Analyze the results from a financial agent workflow and provide insightful recommendations.
    Your response should include:

    A Summary of sentiment analysis, fact check results, sentiment on fact check if available, fraud detection results.

    Format your response in a professional, well-structured manner suitable for a financial advisor."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze these financial workflow results and provide insights:\n{json.dumps(api_state, indent=2)}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return {
            "analysis": response.choices[0].message.content,
            "usage": response.usage.total_tokens
        }
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return {"error": str(e)}

#1. Generate synthetic price data for multiple tickers
for ticker in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]:
    price_df = pd.concat([price_df, generate_synthetic_price_data(ticker)], ignore_index=True)

price_data = price_df
pdb.set_trace()

#2. Generate synthetic transactions for the last 10 days
transaction_df = generate_synthetic_transactions(["AAPL", "MSFT", "GOOG", "AMZN", "META"])
transaction = transaction_df.to_dict(orient='records')

#3. Additional parameters
additional_context = {
    "user_risk_tolerance": "moderate",
    "portfolio_allocation": {"tech": 0.40, "finance": 0.30, "healthcare": 0.20, "other": 0.10},
    "market_conditions": "bullish"
}

#4. Example news
news = "Tech stocks rally as AI investments increase across sector"

#5. Example claim to fact-check
claim = "AAPL is planning to foray into social media business."

#6. Run the workflow with all parameters
final_event = run_financial_workflow(
    price_data=price_data,
    news=news,
    transaction=transaction,
    claim=claim,
    additional_context=additional_context
)

#7. Process results with GPT-4o
print("FinAgent Workflow Completed")
print("="*50)
final_event_key = list(final_event.keys())[0]

for key in final_event[final_event_key]:
    if key == "price_data" or key == "transaction":
        # Skip large data structures
        continue
    print(f"{key}: {final_event[final_event_key][key]}\n")

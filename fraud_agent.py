import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

fake = Faker()
np.random.seed(42)

# Step 1: Generate Synthetic Robinhood-style Transaction Data
def generate_transaction_data(start="2020-01-01", end="2021-10-01", n_per_day=10):
    tickers = ["META", "GOOG", "AAPL", "MSFT", "AMZN"]
    merchants = ["Robinhood", "Wealthfront", "TD Ameritrade", "Fidelity"]
    order_types = ["market", "limit"]
    intents = ["buy", "sell"]
    channels = ["web", "mobile", "API"]

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    transactions = []

    for date in date_range:
        for _ in range(n_per_day):
            amount = round(np.random.exponential(scale=3000), 2)
            channel = random.choice(channels)

            fraud_probability = 0.01
            if amount > 10000:
                fraud_probability += 0.03
            if channel == "API":
                fraud_probability += 0.02
            is_fraud = int(np.random.rand() < fraud_probability)

            transactions.append({
                "txn_id": fake.uuid4(),
                "user_id": fake.uuid4(),
                "timestamp": date + timedelta(minutes=random.randint(0, 1440)),
                "merchant": random.choice(merchants),
                "instrument": random.choice(tickers),
                "intent": random.choice(intents),
                "amount": amount,
                "order_type": random.choice(order_types),
                "channel": channel,
                "is_fraud": is_fraud
            })

    df = pd.DataFrame(transactions)
    df.to_csv("synthetic_transactions.csv", index=False)
    return df

# Step 2: Train a Fraud Detection Model
def train_model(df):
    df_encoded = pd.get_dummies(df[["merchant", "instrument", "intent", "order_type", "channel"]])
    features = pd.concat([df[["amount"]], df_encoded], axis=1)
    labels = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_detector.pkl")

    return model, features.columns.tolist()

# Step 3: Predict Function
def predict_fraud(txn, model_path="models/fraud_detector.pkl", feature_names=None):
    model = joblib.load(model_path)
    df = pd.DataFrame([txn])
    df_enc = pd.get_dummies(df[["merchant", "instrument", "intent", "order_type", "channel"]])
    base = pd.DataFrame([txn["amount"]], columns=["amount"])
    input_df = pd.concat([base, df_enc], axis=1).reindex(columns=feature_names, fill_value=0)
    return bool(model.predict(input_df)[0])

# Main execution
if __name__ == "__main__":
    print("Generating synthetic Robinhood-style transaction data...")
    df = generate_transaction_data()

    print("Training fraud detection model...")
    model, feature_names = train_model(df)

    # Example prediction
    new_txn = {
        "merchant": "Robinhood",
        "instrument": "GOOG",
        "intent": "buy",
        "amount": 15000,
        "order_type": "limit",
        "channel": "API"
    }
    result = predict_fraud(new_txn, feature_names=feature_names)
    print(f"\nIs the new transaction fraudulent? {'Yes' if result else 'No'}")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load FinBERT model (pretrained for financial sentiment)
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["negative", "neutral", "positive"]

def classify_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    sentiment = LABELS[np.argmax(probs)]
    return {
        "sentiment": sentiment,
        "confidence": float(np.max(probs)),
        "probabilities": dict(zip(LABELS, map(float, probs)))
    }

def batch_classify(news_list: list[str]) -> list[dict]:
    return [classify_sentiment(text) for text in news_list]

#from agents.sentiment_agent import classify_sentiment

headline = "Tesla stock drops after CEO sells $5 billion in shares"
result = classify_sentiment(headline)
print(f"Sentiment: {result['sentiment']} ({result['confidence']*100:.1f}%)")

headline = "Markets react to economic data showing unexpected growth"
result2 = classify_sentiment(headline)

print(f"Sentiment: {result2['sentiment']} ({result2['confidence']*100:.1f}%)")

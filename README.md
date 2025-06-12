# FinAgent 📈 - Financial Trading and Analysis Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FinAgent is an advanced financial trading and analysis platform that combines machine learning, reinforcement learning, and natural language processing to provide comprehensive tools for financial data analysis, automated trading strategies, and market insights.

## Features

- **Deep Reinforcement Learning Trading Strategies** - Using the FinRL library
- **Data Processing and Analysis** - for stock price data from Yahoo Finance
- **Sentiment Analysis** - Financial sentiment classification using FinBERT
- **Fact Checking** - Verification of financial claims and news
- **Fraud Detection** - Machine learning models for transaction fraud detection
- **Portfolio Allocation** - Optimize investment portfolios using RL algorithms

## Installation

1. Install FinRl:
```bash
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `/agents/` - Smart agents for various financial tasks
  - `fact_checker_agent.py` - Verifies financial claims
  - `fraud_agent.py` - Detects fraudulent transactions
  - `sentiment_agent.py` - Analyzes financial news sentiment
  - `models/` - Pre-trained ML models
  - `results/` - DRLAgent copies results
- `/data/` - Financial datasets and processed data
  - `/processed/` - Cleaned and transformed data
- `data_processor.py` - Data processing utilities for financial data
- `yfinance_data.py` - Yahoo Finance data fetching functionality

## Usage

### Data Processing

```python
from data_processor import download_stock_data

# Download stock data for specified tickers
data = download_stock_data(tickers=['SPY', 'AAPL', 'MSFT'], period='1y')
```

### Sentiment Analysis

```python
from agents.sentiment_agent import classify_sentiment

headline = "Tesla stock drops after CEO sells $5 billion in shares"
result = classify_sentiment(headline)
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2f}")
```

### Fact Checking

```python
from agents.fact_checker_agent import fetch_fact_checks

claim = "Company XYZ reported record profits last quarter"
fact_checks = fetch_fact_checks(claim)
```

## Reinforcement Learning

The project integrates FinRL, a deep reinforcement learning library for automated stock trading. See the Jupyter notebook in `/agents/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb` for examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Disclaimer: This software is for educational and research purposes only. It is not intended to be financial advice.*
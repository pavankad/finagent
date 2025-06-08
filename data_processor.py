#!/usr/bin/env python
# filepath: /Users/pavan/vscode/finAgent/data_processor.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Define constant for the output file path
PRICE_HISTORY_PATH = 'data/processed/price_history.csv'

def download_stock_data(tickers=['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL'], 
                        start_date=None, 
                        end_date=None, 
                        period='1y'):
    """
    Download stock data for the specified tickers
    
    Args:
        tickers (list): List of stock tickers to download
        start_date (str): Start date in 'YYYY-MM-DD' format (optional)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
        period (str): Period to download if start_date and end_date not provided
                     (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        DataFrame with the stock data
    """
    print(f"Downloading data for {tickers}")
    
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    else:
        data = yf.download(tickers, period=period, group_by='ticker')
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Process the data based on whether we have a single ticker or multiple tickers
    if len(tickers) == 1:
        # Single ticker case
        ticker = tickers[0]
        data.columns = [f"{col.lower()}" for col in data.columns]
        data['ticker'] = ticker
        # Rename columns to match FinRL expectations
        data.reset_index(inplace=True)  # Convert Date from index to column
        data.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        data['tic'] = ticker  # Use 'tic' instead of 'ticker' for FinRL compatibility
        
        # Save as CSV
        data.to_csv(PRICE_HISTORY_PATH, index=False)
        return data
    else:
        # Multiple tickers case
        all_data = []
        for ticker in tickers:
            ticker_data = data[ticker].copy()
            ticker_data.columns = [col.lower() for col in ticker_data.columns]
            
            # Use 'tic' instead of 'ticker' for FinRL compatibility
            ticker_data['tic'] = ticker
            
            all_data.append(ticker_data)
        
        # Combine all data
        combined_data = pd.concat(all_data)
        combined_data.reset_index(inplace=True)  # Convert Date from index to column
        combined_data.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        
        # Save as CSV
        combined_data.to_csv(PRICE_HISTORY_PATH, index=False)
        return combined_data

def add_features(df):
    """
    Add basic features to the dataset
    """
    # Make sure df is sorted by date and ticker
    if 'date' in df.columns:
        df = df.sort_values(['tic', 'date'])
    else:
        df = df.sort_index()
    
    # Calculate returns
    df['daily_return'] = df.groupby('tic')['close'].pct_change()
    
    # Calculate log returns
    df['log_return'] = np.log(df['close']/df['close'].shift(1))
    
    # Moving averages
    df['sma_10'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=10).mean())
    df['sma_30'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=30).mean())
    
    # Volatility (standard deviation over a window)
    df['volatility_10'] = df.groupby('tic')['log_return'].transform(
        lambda x: x.rolling(window=10).std()
    )
    
    return df.fillna(0)

if __name__ == "__main__":
    # Download data for some common ETFs and tech stocks
    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Download 3 years of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    df = download_stock_data(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Add features
    df = add_features(df)
    
    # Save the enhanced dataset
    df.to_csv(PRICE_HISTORY_PATH, index=False)
    
    print(f"Data downloaded and processed successfully. Shape: {df.shape}")
    print(f"Data saved to {PRICE_HISTORY_PATH}")

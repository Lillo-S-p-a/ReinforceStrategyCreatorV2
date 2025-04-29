"""
Data Fetcher Module

This module provides functionality to fetch historical financial data from Yahoo Finance.
:ComponentRole DataFetcher
:Context Data Pipeline (Req 2.1)
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Optional

# Configure logger
logger = logging.getLogger(__name__)

def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker symbol and date range.
    
    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): The start date in format 'YYYY-MM-DD'.
        end_date (str): The end date in format 'YYYY-MM-DD'.
        
    Returns:
        pd.DataFrame: DataFrame containing historical OHLCV data.
                     Returns an empty DataFrame if an error occurs.
                     
    Raises:
        No exceptions are raised as they are caught internally.
        Errors are logged as warnings instead.
    """
    try:
        # Validate inputs
        if not ticker or not isinstance(ticker, str):
            logger.warning(f"Invalid ticker format: {ticker}")
            return pd.DataFrame()
            
        if not start_date or not end_date:
            logger.warning("Start date or end date is missing")
            return pd.DataFrame()
        
        # Fetch data using yfinance
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Check if data is empty
        if data.empty:
            logger.warning(f"No data found for ticker {ticker} in the specified date range")
            return pd.DataFrame()
            
        return data
        
    except Exception as e:
        # Handle network errors or data format errors
        if "Connection" in str(e) or "Timeout" in str(e) or "Network" in str(e):
            logger.warning(f"NetworkError: Failed to fetch data for {ticker}: {str(e)}")
        else:
            logger.warning(f"DataFormatError: Error processing data for {ticker}: {str(e)}")
        
        # Return empty DataFrame on error
        return pd.DataFrame()
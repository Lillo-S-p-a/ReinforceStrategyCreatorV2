"""
Test script to verify the compatibility between backtesting/data.py and technical_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import the DataManager class from backtesting/data.py
from reinforcestrategycreator.backtesting.data import DataManager

def main():
    """
    Test the DataManager class with the TechnicalAnalyzer compatibility wrapper.
    """
    print("Testing DataManager with TechnicalAnalyzer compatibility wrapper...")
    
    # Create a sample DataFrame to simulate fetched data
    # This will be used to mock the fetch_historical_data function
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(105, 115, 100),
        'Low': np.random.uniform(95, 105, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 2000000, 100),
        'Adj Close': np.random.uniform(100, 110, 100)
    }, index=dates)
    
    # Patch the fetch_historical_data function to return our mock data
    from unittest.mock import patch
    
    with patch('reinforcestrategycreator.data_fetcher.fetch_historical_data', return_value=mock_data):
        # Create a DataManager instance
        data_manager = DataManager(
            asset="SPY",
            start_date="2023-01-01",
            end_date="2023-04-10",
            test_ratio=0.2
        )
        
        # Fetch and process data
        try:
            data = data_manager.fetch_data()
            print(f"Successfully fetched and processed data with {len(data)} rows")
            print(f"Data columns: {data.columns.tolist()}")
            
            # Check if technical indicators were added
            indicator_columns = [col for col in data.columns if col not in mock_data.columns]
            print(f"Added indicator columns: {indicator_columns}")
            
            # Get train/test data
            train_data, test_data = data_manager.get_train_test_data()
            print(f"Train data: {len(train_data)} rows, Test data: {len(test_data)} rows")
            
            print("Test completed successfully!")
            return True
        except Exception as e:
            print(f"Error during test: {e}")
            return False

if __name__ == "__main__":
    main()
"""
Data preparation module for backtesting.

This module provides functionality for fetching and preparing data
for backtesting reinforcement learning trading strategies.
"""

import logging
import pandas as pd
from typing import Tuple, Optional

from reinforcestrategycreator.data_fetcher import fetch_historical_data
from reinforcestrategycreator.technical_analyzer import TechnicalAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data fetching and preparation for backtesting.
    
    This class handles fetching historical data, adding technical indicators,
    and splitting data into training and testing sets.
    """
    
    def __init__(self, 
                 asset: str = "SPY",
                 start_date: str = "2020-01-01",
                 end_date: str = "2023-01-01",
                 test_ratio: float = 0.2) -> None:
        """
        Initialize the data manager.
        
        Args:
            asset: Asset symbol to fetch data for
            start_date: Start date for data in YYYY-MM-DD format
            end_date: End date for data in YYYY-MM-DD format
            test_ratio: Ratio of data to use for testing
        """
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.test_ratio = test_ratio
        
        # Initialize containers for data
        self.data = None
        self.train_data = None
        self.test_data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch and prepare data with technical indicators.
        
        Returns:
            DataFrame containing price data and technical indicators
        """
        logger.info(f"Fetching data for {self.asset} from {self.start_date} to {self.end_date}")
        
        try:
            # Use fetch_historical_data function to get data
            data = fetch_historical_data(
                ticker=self.asset,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Add technical indicators
            analyzer = TechnicalAnalyzer()
            data = analyzer.add_all_indicators(data)
            
            # Drop rows with NaN values (from indicator calculations)
            data = data.dropna()
            
            # Store data
            self.data = data
            
            # Split into train and test sets
            self._split_data()
            
            logger.info(f"Data fetched successfully: {len(data)} data points")
            logger.info(f"Training data: {len(self.train_data)} points, Test data: {len(self.test_data)} points")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}", exc_info=True)
            raise
    
    def _split_data(self) -> None:
        """
        Split data into training and testing sets.
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
            
        # Split into train and test sets
        split_idx = int(len(self.data) * (1 - self.test_ratio))
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
    
    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the training and testing data.
        
        Returns:
            Tuple containing (train_data, test_data)
        """
        if self.train_data is None or self.test_data is None:
            logger.warning("No data available. Fetching data first.")
            self.fetch_data()
            
        return self.train_data, self.test_data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the full dataset.
        
        Returns:
            DataFrame containing the full dataset
        """
        if self.data is None:
            logger.warning("No data available. Fetching data first.")
            self.fetch_data()
            
        return self.data
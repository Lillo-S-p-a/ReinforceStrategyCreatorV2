"""
Tests for the technical_analyzer module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from reinforcestrategycreator.technical_analyzer import calculate_indicators


class TestTechnicalAnalyzer(unittest.TestCase):
    """Test cases for the technical_analyzer module."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with sufficient data for indicator calculation
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        self.valid_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(105, 115, 50),
            'Low': np.random.uniform(95, 105, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.randint(1000000, 2000000, 50)
        }, index=dates)
        
        # Create a small DataFrame with insufficient data
        self.insufficient_data = self.valid_data.iloc[:10].copy()
        
        # Create a DataFrame without 'Close' column
        self.no_close_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(105, 115, 50),
            'Low': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(1000000, 2000000, 50)
        }, index=dates)

    def test_calculate_indicators_valid_input(self):
        """Test calculate_indicators with valid inputs."""
        # Call the function with valid data
        result = calculate_indicators(self.valid_data)
        
        # Assert that the result contains the original data
        self.assertEqual(len(result), len(self.valid_data))
        
        # Assert that the result contains the expected indicator columns
        expected_columns = [
            'RSI_14', 
            'MACD_12_26_9', 'MACD_Signal_12_26_9', 'MACD_Hist_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Expected column {col} not found in result")

    def test_calculate_indicators_empty_dataframe(self):
        """Test calculate_indicators with an empty DataFrame."""
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Call the function with empty data
        result = calculate_indicators(empty_df)
        
        # Assert that the result is the same as the input (empty DataFrame)
        self.assertTrue(result.empty)
        pd.testing.assert_frame_equal(result, empty_df)

    def test_calculate_indicators_no_close_column(self):
        """Test calculate_indicators with DataFrame missing 'Close' column."""
        # Call the function with data missing 'Close' column
        result = calculate_indicators(self.no_close_data)
        
        # Assert that the result is the same as the input
        pd.testing.assert_frame_equal(result, self.no_close_data)

    def test_calculate_indicators_insufficient_data(self):
        """Test calculate_indicators with insufficient data for calculations."""
        # Call the function with insufficient data
        result = calculate_indicators(self.insufficient_data)
        
        # Assert that the result is the same as the input
        pd.testing.assert_frame_equal(result, self.insufficient_data)

    def test_calculate_indicators_rsi_error(self):
        """Test calculate_indicators when RSI calculation fails."""
        # Directly patch the rsi method to raise an exception
        with patch('reinforcestrategycreator.technical_analyzer.RSIIndicator.rsi', 
                  side_effect=Exception("RSI calculation error")):
            # Call the function
            result = calculate_indicators(self.valid_data)
            
            # Assert that the result still contains MACD and Bollinger Bands columns
            self.assertIn('MACD_12_26_9', result.columns)
            self.assertIn('BBL_20_2.0', result.columns)
            
            # Assert that RSI column is not in the result
            self.assertNotIn('RSI_14', result.columns)

    def test_calculate_indicators_integration_signature_and_type(self):
        """Contextual Integration Test: Verify signature and return type."""
        try:
            # Call with valid parameters to check signature and type
            result = calculate_indicators(self.valid_data)
            
            # Assert the return type is a pandas DataFrame
            self.assertIsInstance(result, pd.DataFrame, "Return type should be pandas.DataFrame")
            
            # Assert that the result contains at least one indicator column
            indicator_columns = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0']
            self.assertTrue(any(col in result.columns for col in indicator_columns), 
                           "Result should contain at least one indicator column")
        except TypeError as e:
            self.fail(f"Function signature mismatch: {e}")


    def test_calculate_indicators_data_fetcher_integration(self):
        """
        Contextual Integration Test: Verify integration with data_fetcher output.
        
        This test simulates the data flow from data_fetcher to technical_analyzer
        to ensure compatibility between these components in the pipeline.
        """
        # Create a DataFrame similar to what would be returned by data_fetcher.fetch_historical_data()
        # The data_fetcher returns OHLCV data from Yahoo Finance with Date as index
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        mock_fetched_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(105, 115, 50),
            'Low': np.random.uniform(95, 105, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.randint(1000000, 2000000, 50),
            'Adj Close': np.random.uniform(100, 110, 50)  # Yahoo Finance includes this column
        }, index=dates)
        
        # Pass the mock fetched data to calculate_indicators
        result = calculate_indicators(mock_fetched_data)
        
        # Verify that the result contains all original columns
        for col in mock_fetched_data.columns:
            self.assertIn(col, result.columns, f"Original column {col} should be preserved")
        
        # Verify that the result contains the expected indicator columns
        expected_indicators = [
            'RSI_14',
            'MACD_12_26_9', 'MACD_Signal_12_26_9', 'MACD_Hist_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns, f"Expected indicator {indicator} not found")
        
        # Verify that the indicators contain valid numerical data (not all NaN)
        for indicator in expected_indicators:
            self.assertTrue(result[indicator].notna().any(), f"Indicator {indicator} should contain valid data")


if __name__ == '__main__':
    unittest.main()
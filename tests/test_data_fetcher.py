"""
Tests for the data_fetcher module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from reinforcestrategycreator.data_fetcher import fetch_historical_data


class TestDataFetcher(unittest.TestCase):
    """Test cases for the data_fetcher module."""

    def test_fetch_historical_data_valid_input(self):
        """Test fetch_historical_data with valid inputs."""
        # Create a mock DataFrame that would be returned by yf.download
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [98.0, 99.0, 100.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=[
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ])

        # Mock the yf.download function
        with patch('yfinance.download', return_value=mock_data) as mock_download:
            # Call the function
            result = fetch_historical_data('AAPL', '2023-01-01', '2023-01-03')
            
            # Assert that yf.download was called with the correct parameters
            mock_download.assert_called_once_with(
                tickers='AAPL',
                start='2023-01-01',
                end='2023-01-03',
                progress=False
            )
            
            # Assert that the result is the mock data
            pd.testing.assert_frame_equal(result, mock_data)

    def test_fetch_historical_data_empty_result(self):
        """Test fetch_historical_data when yfinance returns empty data."""
        # Mock the yf.download function to return an empty DataFrame
        with patch('yfinance.download', return_value=pd.DataFrame()) as mock_download:
            # Call the function
            result = fetch_historical_data('INVALID', '2023-01-01', '2023-01-03')
            
            # Assert that the result is an empty DataFrame
            self.assertTrue(result.empty)

    def test_fetch_historical_data_network_error(self):
        """Test fetch_historical_data when a network error occurs."""
        # Mock the yf.download function to raise a network-related exception
        with patch('yfinance.download', side_effect=Exception("Connection error")) as mock_download:
            # Call the function
            result = fetch_historical_data('AAPL', '2023-01-01', '2023-01-03')
            
            # Assert that the result is an empty DataFrame
            self.assertTrue(result.empty)

    def test_fetch_historical_data_invalid_ticker(self):
        """Test fetch_historical_data with invalid ticker."""
        # Call the function with an invalid ticker
        result = fetch_historical_data('', '2023-01-01', '2023-01-03')
        
        # Assert that the result is an empty DataFrame
        self.assertTrue(result.empty)

    def test_fetch_historical_data_invalid_dates(self):
        """Test fetch_historical_data with invalid dates."""
        # Call the function with invalid dates
        result = fetch_historical_data('AAPL', '', '2023-01-03')
        
        # Assert that the result is an empty DataFrame
        self.assertTrue(result.empty)

    def test_fetch_historical_data_integration_signature_and_type(self):
        """Contextual Integration Test: Verify signature and return type."""
        # Mock yf.download to avoid network call and ensure a return value
        mock_data = pd.DataFrame({'Close': [100.0]}) # Minimal mock data
        with patch('yfinance.download', return_value=mock_data) as mock_download:
            try:
                # Call with valid parameters to check signature and type
                result = fetch_historical_data('MSFT', '2024-01-01', '2024-01-02')
                # Assert the return type is a pandas DataFrame
                self.assertIsInstance(result, pd.DataFrame, "Return type should be pandas.DataFrame")
            except TypeError as e:
                self.fail(f"Function signature mismatch: {e}")

if __name__ == '__main__':
    unittest.main()
"""
Tests for the technical_analyzer module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pandas_ta as pta # Import pandas_ta
import numpy as np
from datetime import datetime

from reinforcestrategycreator.technical_analyzer import calculate_indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


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
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'HIST_VOL_20', # Added Historical Volatility
            'ADX_14', 'DMP_14', 'DMN_14', # Added ADX components
            'AROOND_14', 'AROONU_14', 'AROONOSC_14', # Added Aroon components
            'ATR_14' # Added ATR
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

    def test_calculate_indicators_missing_ohlc_columns(self):
        """Test calculate_indicators with DataFrame missing required OHLC columns."""
        # Test missing Close
        result = calculate_indicators(self.no_close_data)
        pd.testing.assert_frame_equal(result, self.no_close_data)

        # Test missing High
        no_high_data = self.valid_data.drop(columns=['High'])
        result_no_high = calculate_indicators(no_high_data)
        pd.testing.assert_frame_equal(result_no_high, no_high_data)

        # Test missing Low
        no_low_data = self.valid_data.drop(columns=['Low'])
        result_no_low = calculate_indicators(no_low_data)
        pd.testing.assert_frame_equal(result_no_low, no_low_data)

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
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'HIST_VOL_20', # Added Historical Volatility
            'ADX_14', # Added ADX
            'AROONOSC_14', # Added Aroon Oscillator
            'ATR_14' # Added ATR
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns, f"Expected indicator {indicator} not found")
        
        # Verify that the indicators contain valid numerical data (not all NaN)
        for indicator in expected_indicators:
            self.assertTrue(result[indicator].notna().any(), f"Indicator {indicator} should contain valid data")

    def test_calculate_indicators_numerical_accuracy(self):
        """Test the numerical accuracy of calculated indicators."""
        # Create a small DataFrame with known close prices
        # Needs enough data points for the longest window (ADX = 27)
        close_prices = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, # 10
            110, 111, 112, 113, 114, 115, 114, 113, 112, 111, # 20
            110, 109, 108, 107, 106, 105, 106, 107, 108, 109, # 30
            110, 111, 112, 113, 114, 115, 116 # 37
        ]
        # Add High and Low data for new indicators
        high_prices = [p + 2 for p in close_prices]
        low_prices = [p - 2 for p in close_prices]
        test_df = pd.DataFrame({
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices
            }, index=pd.date_range(start='2023-01-01', periods=len(close_prices), freq='D'))

        # --- Calculate indicators using the function under test ---
        result_df = calculate_indicators(test_df.copy()) # Pass a copy

        # --- Calculate expected values directly using the libraries ---
        # Legacy indicators using 'ta'
        expected_rsi = RSIIndicator(close=test_df['Close'], window=14).rsi()
        macd_indicator = MACD(close=test_df['Close'], window_slow=26, window_fast=12, window_sign=9)
        expected_macd = macd_indicator.macd()
        expected_macd_signal = macd_indicator.macd_signal()
        expected_macd_hist = macd_indicator.macd_diff()
        bb_indicator = BollingerBands(close=test_df['Close'], window=20, window_dev=2)
        expected_bbl = bb_indicator.bollinger_lband()
        expected_bbm = bb_indicator.bollinger_mavg()
        expected_bbu = bb_indicator.bollinger_hband()
        # New indicators using pandas and pandas_ta
        expected_hist_vol = test_df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        # Use pandas_ta directly for comparison
        pta_df = test_df.copy()
        pta_df.ta.adx(high='High', low='Low', close='Close', length=14, append=True)
        pta_df.ta.aroon(high='High', low='Low', length=14, append=True)
        pta_df.ta.atr(high='High', low='Low', close='Close', length=14, append=True)
        expected_adx = pta_df['ADX_14']
        expected_aroonosc = pta_df['AROONOSC_14']
        expected_atr = pta_df['ATRr_14']

        # --- Compare results (allow for small floating point differences) ---
        # Check names=False because the function adds suffixes like _14, _12_26_9 etc.
        # Legacy indicators
        pd.testing.assert_series_equal(result_df['RSI_14'], expected_rsi, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['MACD_12_26_9'], expected_macd, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['MACDs_12_26_9'], expected_macd_signal, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['MACDh_12_26_9'], expected_macd_hist, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['BBL_20_2.0'], expected_bbl, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['BBM_20_2.0'], expected_bbm, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['BBU_20_2.0'], expected_bbu, check_names=False, rtol=1e-5, atol=1e-8)
        # New indicators
        pd.testing.assert_series_equal(result_df['HIST_VOL_20'], expected_hist_vol, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['ADX_14'], expected_adx, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['AROONOSC_14'], expected_aroonosc, check_names=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_series_equal(result_df['ATR_14'], expected_atr, check_names=False, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
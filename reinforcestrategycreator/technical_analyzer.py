"""
Technical Analyzer Module

This module provides functionality to calculate technical indicators on financial data.
:ComponentRole TechnicalAnalyzer
:Context TA Layer (Req 2.2)
"""

import logging
import pandas as pd
import pandas_ta as pta # Import pandas_ta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from typing import Optional
import numpy as np # For historical volatility calculation

# Configure logger
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Backward-compatibility wrapper class for technical indicator calculations.
    
    This class provides a class-based interface to the function-based technical
    indicator calculations for backward compatibility with existing code.
    """
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators on the input DataFrame.
        
        This is a wrapper method that calls the calculate_indicators function.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                               Must have 'high', 'low', 'close' columns (case-insensitive).
        
        Returns:
            pd.DataFrame: Original DataFrame with added indicator columns.
        """
        return calculate_indicators(data)


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators on the input DataFrame.

    Uses ta library for legacy indicators and pandas_ta for newer ones:
    - RSI(14)
    - MACD(12,26,9)
    - Bollinger Bands(20,2)
    - Historical Volatility (rolling std dev of returns, window=20)
    - ADX(14)
    - Aroon(14)
    - ATR(14)

    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data.
                             Must have 'high', 'low', 'close' columns (case-insensitive).

    Returns:
        pd.DataFrame: Original DataFrame with added indicator columns.
                     If calculation fails, returns the original DataFrame.

    Note:
        Added columns include:
        - RSI_14: Relative Strength Index with period 14
        - MACD_12_26_9: MACD line (12,26)
        - MACDs_12_26_9: MACD signal line (9) (Renamed from ta output)
        - MACDh_12_26_9: MACD histogram (Renamed from ta output)
        - BBL_20_2.0: Bollinger Band Lower
        - BBM_20_2.0: Bollinger Band Middle
        - BBU_20_2.0: Bollinger Band Upper
        - HIST_VOL_20: Historical Volatility (20-day rolling std dev of daily returns, annualized)
        - ADX_14: Average Directional Index (14)
        - DMP_14: Directional Movement Plus (from ADX calc)
        - DMN_14: Directional Movement Minus (from ADX calc)
        - AROOND_14: Aroon Down (14)
        - AROONU_14: Aroon Up (14)
        - AROONOSC_14: Aroon Oscillator (14)
        - ATR_14: Average True Range (14)
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = data.copy()
    
    try:
        # Find the 'close', 'high', 'low' columns case-insensitively
        close_col = next((col for col in result_df.columns if col.lower() == 'close'), None)
        high_col = next((col for col in result_df.columns if col.lower() == 'high'), None)
        low_col = next((col for col in result_df.columns if col.lower() == 'low'), None)

        # Check if DataFrame is empty or doesn't have the required columns
        if result_df.empty or close_col is None or high_col is None or low_col is None:
            logger.warning("CalculationError: Input DataFrame is empty or missing 'high', 'low', or 'close' columns (case-insensitive)")
            return data # Return original data if missing required columns

        # Check if there's enough data for the calculations
        # ADX(14) needs 2*14 - 1 = 27 points minimum. MACD needs 26. BB needs 20. Aroon needs 14. ATR needs 14. HistVol needs 21 (20 window + 1 for pct_change).
        min_required_length = 27
        if len(result_df) < min_required_length:
            logger.warning(f"CalculationError: Insufficient data for indicator calculation (minimum {min_required_length} points required)")
            return data # Return original data if insufficient length
        
        # --- Calculate Indicators ---
        # Use the found close_col name
        
        # Calculate RSI using ta
        try:
            rsi_indicator = RSIIndicator(close=result_df[close_col], window=14)
            result_df['RSI_14'] = rsi_indicator.rsi()
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate RSI: {str(e)}")
            # Optionally return data here if RSI is critical and fails
            # return data
        
        # Calculate MACD using ta
        try:
            macd_indicator = MACD(
                close=result_df[close_col],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            result_df['MACD_12_26_9'] = macd_indicator.macd()
            result_df['MACD_Signal_12_26_9'] = macd_indicator.macd_signal()
            result_df['MACD_Hist_12_26_9'] = macd_indicator.macd_diff()
            # Rename columns to match expected test/pandas-ta names
            result_df.rename(columns={
                'MACD_Signal_12_26_9': 'MACDs_12_26_9',
                'MACD_Hist_12_26_9': 'MACDh_12_26_9'
            }, inplace=True)
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate MACD: {str(e)}")
            # Optionally return data here if MACD is critical and fails
            # return data
        
        # Calculate Bollinger Bands using ta
        try:
            bb = BollingerBands(close=result_df[close_col], window=20, window_dev=2)
            result_df['BBL_20_2.0'] = bb.bollinger_lband()
            result_df['BBM_20_2.0'] = bb.bollinger_mavg()
            result_df['BBU_20_2.0'] = bb.bollinger_hband()
            # Note: Removed bollinger_bandwidth and bollinger_pband methods as they don't exist in the current version
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Bollinger Bands: {str(e)}")
        
        # --- Calculate New Indicators using pandas and pandas_ta ---

        # Calculate Historical Volatility (e.g., 20-day rolling std dev of daily returns)
        try:
            # Calculate daily returns
            result_df['daily_return'] = result_df[close_col].pct_change()
            # Calculate rolling standard deviation of returns
            hist_vol_window = 20
            # Annualize by multiplying by sqrt(252) - assuming daily data
            result_df[f'HIST_VOL_{hist_vol_window}'] = result_df['daily_return'].rolling(window=hist_vol_window).std() * np.sqrt(252)
            result_df.drop(columns=['daily_return'], inplace=True) # Drop intermediate column
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Historical Volatility: {str(e)}")

        # Calculate ADX, Aroon, ATR using pandas_ta
        # Ensure columns match expected names (lowercase) for pandas_ta if necessary
        # pandas_ta can often handle case-insensitivity or allows mapping.
        try:
            # Calculate ADX
            # Note: pandas_ta might require lowercase column names depending on version/config.
            # If issues arise, consider renaming columns temporarily:
            # temp_df = result_df.rename(columns={high_col: 'high', low_col: 'low', close_col: 'close'})
            # temp_df.ta.adx(length=14, append=True)
            # result_df = result_df.join(temp_df[['ADX_14', 'DMP_14', 'DMN_14']]) # Join results back
            result_df.ta.adx(high=result_df[high_col], low=result_df[low_col], close=result_df[close_col], length=14, append=True)
            # Expected columns: ADX_14, DMP_14, DMN_14
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate ADX: {str(e)}")

        try:
            # Calculate Aroon
            result_df.ta.aroon(high=result_df[high_col], low=result_df[low_col], length=14, append=True)
            # Expected columns: AROOND_14, AROONU_14, AROONOSC_14
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Aroon: {str(e)}")

        try:
            # Calculate ATR
            result_df.ta.atr(high=result_df[high_col], low=result_df[low_col], close=result_df[close_col], length=14, append=True)
            # Expected column: ATR_14
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate ATR: {str(e)}")

        return result_df

    except Exception as e:
        # Catch any unexpected errors
        logger.warning(f"CalculationError: Unexpected error in calculate_indicators: {str(e)}")
        return data
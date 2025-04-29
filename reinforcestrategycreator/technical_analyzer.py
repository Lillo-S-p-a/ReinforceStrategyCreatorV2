"""
Technical Analyzer Module

This module provides functionality to calculate technical indicators on financial data.
:ComponentRole TechnicalAnalyzer
:Context TA Layer (Req 2.2)
"""

import logging
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from typing import Optional

# Configure logger
logger = logging.getLogger(__name__)

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators on the input DataFrame.
    
    Uses ta library to calculate:
    - RSI(14)
    - MACD(12,26,9)
    - Bollinger Bands(20,2)
    
    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data.
                            Must have a 'Close' column.
        
    Returns:
        pd.DataFrame: Original DataFrame with added indicator columns.
                     If calculation fails, returns the original DataFrame.
                     
    Note:
        Added columns include:
        - RSI_14: Relative Strength Index with period 14
        - MACD_12_26_9: MACD line (12,26)
        - MACD_Signal_12_26_9: MACD signal line (9)
        - MACD_Hist_12_26_9: MACD histogram
        - BBL_20_2.0: Bollinger Band Lower
        - BBM_20_2.0: Bollinger Band Middle
        - BBU_20_2.0: Bollinger Band Upper
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = data.copy()
    
    try:
        # Check if DataFrame is empty or doesn't have required columns
        if result_df.empty or 'Close' not in result_df.columns:
            logger.warning("CalculationError: Input DataFrame is empty or missing 'Close' column")
            return data
        
        # Check if there's enough data for the calculations
        if len(result_df) < 26:  # 26 is the largest window size needed (for MACD)
            logger.warning("CalculationError: Insufficient data for indicator calculation (minimum 26 points required)")
            return data
        
        # Calculate RSI using ta
        try:
            rsi_indicator = RSIIndicator(close=result_df['Close'], window=14)
            result_df['RSI_14'] = rsi_indicator.rsi()
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate RSI: {str(e)}")
        
        # Calculate MACD using ta
        try:
            macd_indicator = MACD(
                close=result_df['Close'], 
                window_slow=26, 
                window_fast=12, 
                window_sign=9
            )
            result_df['MACD_12_26_9'] = macd_indicator.macd()
            result_df['MACD_Signal_12_26_9'] = macd_indicator.macd_signal()
            result_df['MACD_Hist_12_26_9'] = macd_indicator.macd_diff()
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate MACD: {str(e)}")
        
        # Calculate Bollinger Bands using ta
        try:
            bb = BollingerBands(close=result_df['Close'], window=20, window_dev=2)
            result_df['BBL_20_2.0'] = bb.bollinger_lband()
            result_df['BBM_20_2.0'] = bb.bollinger_mavg()
            result_df['BBU_20_2.0'] = bb.bollinger_hband()
            # Note: Removed bollinger_bandwidth and bollinger_pband methods as they don't exist in the current version
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Bollinger Bands: {str(e)}")
        
        return result_df
        
    except Exception as e:
        # Catch any unexpected errors
        logger.warning(f"CalculationError: Unexpected error in calculate_indicators: {str(e)}")
        return data
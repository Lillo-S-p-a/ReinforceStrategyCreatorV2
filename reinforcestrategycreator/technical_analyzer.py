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
        # Handle simple or MultiIndex columns
        def find_col(target_name: str, df_columns):
            for col_identifier in df_columns:
                if isinstance(col_identifier, tuple): # MultiIndex column
                    if col_identifier[0].lower() == target_name.lower():
                        return col_identifier
                elif isinstance(col_identifier, str): # Simple Index column
                    if col_identifier.lower() == target_name.lower():
                        return col_identifier
            return None

        close_col = find_col('Close', result_df.columns)
        high_col = find_col('High', result_df.columns)
        low_col = find_col('Low', result_df.columns)

        # Check if DataFrame is empty or doesn't have the required columns
        if result_df.empty or close_col is None or high_col is None or low_col is None:
            missing_cols = []
            if close_col is None: missing_cols.append('Close')
            if high_col is None: missing_cols.append('High')
            if low_col is None: missing_cols.append('Low')
            logger.warning(f"CalculationError: Input DataFrame is empty or missing required columns (case-insensitive): {', '.join(missing_cols)}. Columns found: {result_df.columns.tolist()}")
            return data # Return original data if missing required columns

        # Check if there's enough data for the calculations
        min_required_length = 27 # Based on ADX(14)
        if len(result_df) < min_required_length:
            logger.warning(f"CalculationError: Insufficient data for indicator calculation (minimum {min_required_length} points required, got {len(result_df)})")
            return data 
        
        # --- Calculate Indicators ---
        
        # Calculate RSI using ta
        try:
            rsi_indicator = RSIIndicator(close=result_df[close_col], window=14)
            result_df['RSI_14'] = rsi_indicator.rsi()
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate RSI: {str(e)}")
        
        # Calculate MACD using ta
        try:
            macd_indicator = MACD(
                close=result_df[close_col],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            result_df['MACD_12_26_9'] = macd_indicator.macd()
            result_df['MACDs_12_26_9'] = macd_indicator.macd_signal() # Will be renamed
            result_df['MACDh_12_26_9'] = macd_indicator.macd_diff()   # Will be renamed
            # Rename columns to match expected test/pandas-ta names
            result_df.rename(columns={
                'MACDs_12_26_9': 'MACDs_12_26_9', # Corrected original typo in comment, was MACD_Signal_...
                'MACDh_12_26_9': 'MACDh_12_26_9'  # Corrected original typo in comment, was MACD_Hist_...
            }, inplace=True)
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate MACD: {str(e)}")
        
        # Calculate Bollinger Bands using ta
        try:
            bb = BollingerBands(close=result_df[close_col], window=20, window_dev=2)
            result_df['BBL_20_2.0'] = bb.bollinger_lband()
            result_df['BBM_20_2.0'] = bb.bollinger_mavg()
            result_df['BBU_20_2.0'] = bb.bollinger_hband()
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Bollinger Bands: {str(e)}")
        
        # Calculate Historical Volatility
        try:
            result_df['daily_return'] = result_df[close_col].pct_change()
            hist_vol_window = 20
            result_df[f'HIST_VOL_{hist_vol_window}'] = result_df['daily_return'].rolling(window=hist_vol_window).std() * np.sqrt(252)
            result_df.drop(columns=['daily_return'], inplace=True) 
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Historical Volatility: {str(e)}")

        # Calculate ADX, Aroon, ATR using pandas_ta
        try:
            result_df.ta.adx(high=result_df[high_col], low=result_df[low_col], close=result_df[close_col], length=14, append=True)
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate ADX: {str(e)}")

        try:
            result_df.ta.aroon(high=result_df[high_col], low=result_df[low_col], length=14, append=True)
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate Aroon: {str(e)}")

        try:
            result_df.ta.atr(high=result_df[high_col], low=result_df[low_col], close=result_df[close_col], length=14, append=True)
        except Exception as e:
            logger.warning(f"CalculationError: Failed to calculate ATR: {str(e)}")

        return result_df

    except Exception as e:
        logger.warning(f"CalculationError: Unexpected error in calculate_indicators: {str(e)}", exc_info=True)
        return data
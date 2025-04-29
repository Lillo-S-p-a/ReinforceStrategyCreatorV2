"""
Tests for the reset method of the TradingEnv class.

This module contains targeted tests for the reset method of the TradingEnv class,
focusing on both core logic and contextual integration aspects.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from reinforcestrategycreator.trading_environment import TradingEnv


@pytest.fixture
def basic_df():
    """Create a basic DataFrame with price data and no NaN values."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'rsi': [60.0, 61.0, 62.0, 63.0, 64.0],
        'macd': [0.5, 0.6, 0.7, 0.8, 0.9],
        'bb_upper': [110.0, 111.0, 112.0, 113.0, 114.0],
        'bb_lower': [90.0, 91.0, 92.0, 93.0, 94.0]
    })


@pytest.fixture
def df_with_initial_nans():
    """Create a DataFrame with NaN values in the first few rows for indicators."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'rsi': [np.nan, np.nan, 62.0, 63.0, 64.0],
        'macd': [np.nan, np.nan, 0.7, 0.8, 0.9],
        'bb_upper': [np.nan, np.nan, 112.0, 113.0, 114.0],
        'bb_lower': [np.nan, np.nan, 92.0, 93.0, 94.0]
    })


@pytest.fixture
def df_with_all_nan_indicators():
    """Create a DataFrame with all NaN values for indicators but valid price data."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'rsi': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'macd': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'bb_upper': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'bb_lower': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()


def test_reset_portfolio_initialization(basic_df):
    """Test that reset correctly initializes portfolio state."""
    # Create environment with a specific initial balance
    initial_balance = 10000.0
    env = TradingEnv(basic_df, initial_balance=initial_balance)
    
    # Reset the environment
    observation, info = env.reset()
    
    # Verify portfolio state initialization
    assert env.balance == initial_balance
    assert env.shares_held == 0
    assert env.portfolio_value == initial_balance
    assert env.last_portfolio_value == initial_balance
    assert env.current_price == basic_df.iloc[env.current_step]['close']


def test_reset_finds_first_non_nan_row(df_with_initial_nans):
    """Test that reset sets current_step to the first row where all indicators are non-NaN."""
    env = TradingEnv(df_with_initial_nans)
    
    # Reset the environment
    observation, info = env.reset()
    
    # The first two rows have NaN values, so current_step should be 2
    assert env.current_step == 2
    assert info['starting_step'] == 2
    
    # Verify the current price is from the correct row
    assert env.current_price == df_with_initial_nans.iloc[2]['close']


def test_reset_fallback_to_price_data(df_with_all_nan_indicators):
    """Test that reset falls back to the first row with valid price data when all indicators are NaN."""
    env = TradingEnv(df_with_all_nan_indicators)
    
    # Reset the environment
    observation, info = env.reset()
    
    # Since all indicators are NaN but price data is available, it should use the first row
    assert env.current_step == 0
    assert info['starting_step'] == 0
    
    # Verify the current price is from the first row
    assert env.current_price == df_with_all_nan_indicators.iloc[0]['close']


def test_reset_empty_dataframe(empty_df):
    """Test that reset raises a RuntimeError when the DataFrame is empty."""
    env = TradingEnv(empty_df)
    
    # Reset should raise a RuntimeError
    with pytest.raises(RuntimeError):
        env.reset()


def test_reset_observation_corresponds_to_current_step(df_with_initial_nans):
    """Test that the returned observation corresponds to the determined current_step."""
    env = TradingEnv(df_with_initial_nans)
    
    # Reset the environment
    observation, info = env.reset()
    
    # The current_step should be 2 (first non-NaN row)
    assert env.current_step == 2
    
    # Get the observation directly for comparison
    direct_observation = env._get_observation()
    
    # The observation returned by reset should match the one we get directly
    np.testing.assert_array_equal(observation, direct_observation)


def test_reset_info_dictionary_content(basic_df):
    """Test that the info dictionary returned by reset contains the correct information."""
    env = TradingEnv(basic_df)
    
    # Reset the environment
    observation, info = env.reset()
    
    # Verify info dictionary contains all required keys with correct values
    assert 'balance' in info and info['balance'] == env.balance
    assert 'shares_held' in info and info['shares_held'] == env.shares_held
    assert 'portfolio_value' in info and info['portfolio_value'] == env.portfolio_value
    assert 'starting_step' in info and info['starting_step'] == env.current_step
    assert 'current_price' in info and info['current_price'] == env.current_price


def test_reset_return_types(basic_df):
    """Test that reset returns the correct types according to the Gymnasium API spec."""
    env = TradingEnv(basic_df)
    
    # Reset the environment
    result = env.reset()
    
    # Verify the result is a tuple with two elements
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # Unpack the tuple
    observation, info = result
    
    # Verify observation is a numpy array with the correct shape and dtype
    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    
    # Verify info is a dictionary
    assert isinstance(info, dict)


def test_reset_with_seed(basic_df):
    """Test that reset accepts and uses a seed parameter."""
    env = TradingEnv(basic_df)
    
    # Reset with a specific seed
    seed_value = 42
    observation1, info1 = env.reset(seed=seed_value)
    
    # Reset again with the same seed
    env = TradingEnv(basic_df)
    observation2, info2 = env.reset(seed=seed_value)
    
    # The observations should be the same when using the same seed
    np.testing.assert_array_equal(observation1, observation2)
    
    # The info dictionaries should have the same values
    assert info1['starting_step'] == info2['starting_step']
    assert info1['current_price'] == info2['current_price']


def test_reset_with_options(basic_df):
    """Test that reset accepts an options parameter (even if not used)."""
    env = TradingEnv(basic_df)
    
    # Reset with options (even though they might not be used in the current implementation)
    options = {'test_option': True}
    observation, info = env.reset(options=options)
    
    # Verify the reset still works correctly
    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert isinstance(info, dict)
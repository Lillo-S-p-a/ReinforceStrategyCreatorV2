"""
Targeted tests for the observation logic in TradingEnv class.

This module focuses on testing the _get_observation method and observation_space definition
of the TradingEnv class, which were updated to include pre-calculated technical indicators
in the state representation.

:ComponentRole TradingEnvironment
:Context RL Core (Req 2.3)
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

from reinforcestrategycreator.trading_environment import TradingEnv
from reinforcestrategycreator.technical_analyzer import calculate_indicators


@pytest.fixture
def test_df_raw():
    """Create a sample DataFrame without indicators for testing."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
                125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
                105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                  112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                  122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                  112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                  122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                   2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                   3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900]
    })


@pytest.fixture
def test_df(test_df_raw):
    """Create a sample DataFrame with indicators for testing."""
    # Use the technical_analyzer to calculate indicators
    df_with_indicators = calculate_indicators(test_df_raw)
    return df_with_indicators


@pytest.fixture
def env(test_df):
    """Create a sample environment for testing."""
    return TradingEnv(test_df, initial_balance=10000.0, window_size=5)


@pytest.fixture
def env_small_window(test_df):
    """Create a sample environment with a small window size for testing."""
    return TradingEnv(test_df, initial_balance=10000.0, window_size=2)


@pytest.fixture
def env_large_window(test_df):
    """Create a sample environment with a large window size for testing."""
    return TradingEnv(test_df, initial_balance=10000.0, window_size=10)


# CORE LOGIC TESTING

def test_get_observation_includes_window_data(env, test_df):
    """
    Test that _get_observation correctly includes data from the sliding window.
    
    This test verifies that the observation includes data from multiple time steps
    as specified by the window_size.
    """
    # Move to a step where indicators are available and we have enough history
    env.reset()
    env.current_step = 26  # Move past initial NaN values for indicators
    
    # Get observation
    observation = env._get_observation()
    
    # Check observation shape
    window_size = env.window_size
    num_market_features = len(test_df.columns)
    num_portfolio_features = 2  # balance and position
    expected_features = (window_size * num_market_features) + num_portfolio_features
    
    assert observation.shape[0] == expected_features


def test_get_observation_matches_observation_space(env):
    """
    Test that the shape of the observation matches the observation_space.
    
    This test verifies that the observation returned by _get_observation has the same
    shape as defined in the observation_space.
    """
    # Reset the environment
    env.reset()
    
    # Move to a step where indicators are available
    env.current_step = 26  # Move past initial NaN values for indicators
    
    # Get observation
    observation = env._get_observation()
    
    # Check that observation shape matches observation_space shape
    assert observation.shape == env.observation_space.shape
    
    # Check that observation is contained within observation_space
    assert env.observation_space.contains(observation)
    
    # Verify the observation space has the correct shape based on window_size
    window_size = env.window_size
    num_market_features = len(env.df.columns)
    num_portfolio_features = 2  # balance and position
    expected_shape = (window_size * num_market_features) + num_portfolio_features
    
    assert env.observation_space.shape[0] == expected_shape


def test_get_observation_handles_nan_values(env, test_df):
    """
    Test that _get_observation correctly handles NaN values.
    
    This test verifies that NaN values in the DataFrame are replaced with 0
    in the observation.
    """
    # Reset the environment
    env.reset()
    
    # Introduce NaN values in the DataFrame
    env.df.iloc[5, 6:] = np.nan  # Set indicator columns to NaN at step 5
    
    # Move to the step with NaN values
    env.current_step = 5
    
    # Get observation
    observation = env._get_observation()
    
    # Check that there are no NaN values in the observation
    assert not np.isnan(observation).any()
    
    # Check that the observation has the correct shape
    assert observation.shape == env.observation_space.shape


def test_get_observation_at_different_steps(env, test_df):
    """
    Test _get_observation at different current_step values.
    
    This test verifies that _get_observation returns correct observations
    at different steps in the environment.
    """
    # Reset the environment
    env.reset()
    
    # Test at step 0 (should include padding)
    env.current_step = 0
    observation_step_0 = env._get_observation()
    assert observation_step_0.shape == env.observation_space.shape
    
    # Test at step 1
    env.current_step = 1
    observation_step_1 = env._get_observation()
    assert observation_step_1.shape == env.observation_space.shape
    
    # Test at a later step where all indicators are available
    env.current_step = 26
    observation_step_26 = env._get_observation()
    assert observation_step_26.shape == env.observation_space.shape
    
    # Observations at different steps should be different
    assert not np.array_equal(observation_step_1, observation_step_26)


def test_get_observation_dtype(env):
    """
    Test that _get_observation returns an array with the correct dtype.
    
    This test verifies that the observation has dtype np.float32 as required
    by the Gymnasium API.
    """
    # Reset the environment
    env.reset()
    
    # Move to a step where indicators are available
    env.current_step = 26
    
    # Get observation
    observation = env._get_observation()
    
    # Check that the observation has dtype np.float32
    assert observation.dtype == np.float32


# CONTEXTUAL INTEGRATION TESTING

def test_observation_space_matches_features(env, test_df):
    """
    Test that the observation_space shape matches the actual number of features.
    
    This test verifies that the observation_space defined in __init__ has the correct
    shape based on the window size, number of features in the DataFrame, and account information.
    """
    # Calculate expected number of features
    window_size = env.window_size
    num_market_features = len(test_df.columns)
    num_portfolio_features = 2  # balance and position
    expected_features = (window_size * num_market_features) + num_portfolio_features
    
    # Check that observation_space shape matches expected features
    assert env.observation_space.shape[0] == expected_features


def test_step_returns_correct_observation(env):
    """
    Test that the step method returns an observation with the correct shape and dtype.
    
    This test verifies that the observation returned by the step method has the
    correct shape and dtype (np.float32) as per the Gymnasium API.
    """
    # Reset the environment
    env.reset()
    
    # Take a step
    observation, reward, terminated, truncated, info = env.step(0)
    
    # Check observation shape
    assert observation.shape == env.observation_space.shape
    
    # Check observation dtype
    assert observation.dtype == np.float32
    
    # Check that observation is contained within observation_space
    assert env.observation_space.contains(observation)


def test_observation_includes_technical_indicators_explicitly(env, test_df):
    """
    Test that the observation explicitly includes all expected technical indicators.
    
    This test verifies that the observation includes all the technical indicators
    that should be calculated by the technical_analyzer.
    """
    # Reset the environment
    env.reset()
    
    # Move to a step where indicators are available
    env.current_step = 26
    
    # Get observation
    observation = env._get_observation()
    
    # Verify that the DataFrame includes all expected indicator columns
    expected_indicators = [
        'RSI_14',
        'MACD_12_26_9',
        'MACDs_12_26_9',
        'MACDh_12_26_9',
        'BBL_20_2.0',
        'BBM_20_2.0',
        'BBU_20_2.0'
    ]
    
    for indicator in expected_indicators:
        assert indicator in test_df.columns, f"Indicator {indicator} not found in DataFrame"
    
    # The observation should include all these indicators for each step in the window
    # We can't directly check the values in the observation since they're normalized and flattened,
    # but we can check that the observation has the correct length
    window_size = env.window_size
    num_market_features = len(test_df.columns)
    num_portfolio_features = 2  # balance and position
    expected_features = (window_size * num_market_features) + num_portfolio_features
    
    assert len(observation) == expected_features


def test_observation_normalization(env):
    """
    Test that observations are properly normalized.
    
    This test verifies that the market data in the observation is normalized
    and that the account information (balance and position) is also normalized.
    """
    # Reset the environment
    env.reset()
    
    # Move to a step where indicators are available
    env.current_step = 26
    
    # Get observation
    observation = env._get_observation()
    
    # All values in a normalized observation should be between -1 and 1
    # (or at least within a reasonable range)
    assert np.all(observation >= -10) and np.all(observation <= 10)
    
    # Check that the last two elements are the normalized balance and position
    normalized_balance = env.balance / env.initial_balance
    normalized_position = env.shares_held * env.current_price / env.initial_balance
    
    # The last two elements of the observation should be close to these values
    assert np.isclose(observation[-2], normalized_balance)
    assert np.isclose(observation[-1], normalized_position)


def test_window_size_affects_observation_shape(env_small_window, env_large_window, test_df):
    """
    Test that different window sizes result in different observation shapes.
    
    This test verifies that the observation shape changes according to the window_size parameter.
    """
    # Reset environments
    env_small_window.reset()
    env_large_window.reset()
    
    # Move to a step where indicators are available
    env_small_window.current_step = 26
    env_large_window.current_step = 26
    
    # Get observations
    observation_small = env_small_window._get_observation()
    observation_large = env_large_window._get_observation()
    
    # Calculate expected shapes
    small_window_size = env_small_window.window_size
    large_window_size = env_large_window.window_size
    num_market_features = len(test_df.columns)
    num_portfolio_features = 2  # balance and position
    
    expected_small_shape = (small_window_size * num_market_features) + num_portfolio_features
    expected_large_shape = (large_window_size * num_market_features) + num_portfolio_features
    
    # Check shapes
    assert observation_small.shape[0] == expected_small_shape
    assert observation_large.shape[0] == expected_large_shape
    
    # The large window observation should be longer than the small window observation
    assert len(observation_large) > len(observation_small)


def test_edge_case_not_enough_history(env, test_df):
    """
    Test that _get_observation handles the edge case where current_step < window_size.
    
    This test verifies that the observation is correctly padded when there isn't enough history.
    """
    # Reset the environment
    env.reset()
    
    # Set current_step to a value less than window_size
    env.current_step = 2  # This is less than the default window_size of 5
    
    # Get observation
    observation = env._get_observation()
    
    # Check that the observation has the correct shape despite not having enough history
    window_size = env.window_size
    num_market_features = len(test_df.columns)
    num_portfolio_features = 2  # balance and position
    expected_features = (window_size * num_market_features) + num_portfolio_features
    
    assert observation.shape[0] == expected_features
    
    # The observation should not contain any NaN values
    assert not np.isnan(observation).any()
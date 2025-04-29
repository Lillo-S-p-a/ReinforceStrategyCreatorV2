"""
Targeted tests for the Sharpe Ratio reward calculation in TradingEnv class.

This module focuses on testing the Sharpe Ratio reward calculation in the TradingEnv class,
which has been reimplemented to provide a risk-adjusted return metric for the RL agent.

:ComponentRole TradingEnvironment
:Context RL Core
"""

import pytest
import numpy as np
import pandas as pd
from collections import deque

from reinforcestrategycreator.trading_environment import TradingEnv


@pytest.fixture
def test_df():
    """Create a sample DataFrame with price data and indicators for testing."""
    df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
                125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
               105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
               115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                  3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900]
    })
    
    # Add mock indicator columns
    df['RSI_14'] = [50.0] * 30
    df['MACD_12_26_9'] = [0.5] * 30
    df['MACD_Signal_12_26_9'] = [0.3] * 30
    df['MACD_Hist_12_26_9'] = [0.2] * 30
    df['BBL_20_2.0'] = [95.0] * 30
    df['BBM_20_2.0'] = [100.0] * 30
    df['BBU_20_2.0'] = [105.0] * 30
    
    return df


@pytest.fixture
def env(test_df):
    """Create a sample environment with default sharpe_window_size for testing."""
    initial_balance = 10000.0
    transaction_fee_percent = 0.1
    sharpe_window_size = 20
    return TradingEnv(test_df, initial_balance, transaction_fee_percent, sharpe_window_size=sharpe_window_size)


@pytest.fixture
def env_small_window(test_df):
    """Create a sample environment with small sharpe_window_size for testing."""
    initial_balance = 10000.0
    transaction_fee_percent = 0.1
    sharpe_window_size = 5
    return TradingEnv(test_df, initial_balance, transaction_fee_percent, sharpe_window_size=sharpe_window_size)


# CORE LOGIC TESTS

def test_portfolio_value_history_initialization(env):
    """Test that _portfolio_value_history is initialized correctly in __init__."""
    # Check that _portfolio_value_history is a deque
    assert isinstance(env._portfolio_value_history, deque)
    
    # Check that _portfolio_value_history has the correct maxlen
    assert env._portfolio_value_history.maxlen == env.sharpe_window_size
    
    # Check that _portfolio_value_history is initially empty
    assert len(env._portfolio_value_history) == 0


def test_portfolio_value_history_reset(env):
    """Test that _portfolio_value_history is cleared in reset."""
    # First, manually add some values to _portfolio_value_history
    for i in range(5):
        env._portfolio_value_history.append(10000.0 + i * 100)
    
    # Check that _portfolio_value_history has values
    assert len(env._portfolio_value_history) == 5
    
    # Reset the environment
    env.reset()
    
    # Check that _portfolio_value_history is cleared
    assert len(env._portfolio_value_history) == 0


def test_portfolio_value_history_update_in_step(env):
    """Test that _portfolio_value_history is updated in step."""
    # Reset the environment
    env.reset()
    
    # Check that _portfolio_value_history is empty
    assert len(env._portfolio_value_history) == 0
    
    # Take a step
    env.step(0)  # Flat action
    
    # Check that _portfolio_value_history has one value
    assert len(env._portfolio_value_history) == 1
    
    # Check that the value is the current portfolio value
    assert env._portfolio_value_history[0] == env.portfolio_value
    
    # Take more steps
    for _ in range(5):
        env.step(0)  # Flat action
    
    # Check that _portfolio_value_history has more values
    assert len(env._portfolio_value_history) == 6  # 1 + 5 = 6


def test_calculate_reward_data_availability(env):
    """Test reward calculation behavior based on last_portfolio_value availability."""
    # Reset the environment - last_portfolio_value should be None
    env.reset()
    # After reset, last_portfolio_value should be the initial balance
    assert pytest.approx(env.last_portfolio_value) == env.initial_balance

    # Calculate reward when last_portfolio_value is None
    reward = env._calculate_reward()
    # Check that reward is 0.0
    assert reward == 0.0

    # Manually set last_portfolio_value to 0
    env.last_portfolio_value = 0.0
    env.portfolio_value = 10000.0 # Set a current value
    # Calculate reward when last_portfolio_value is 0
    reward = env._calculate_reward()
    # Check that reward is 0.0 (division by zero protection)
    assert reward == 0.0

    # Manually set valid last_portfolio_value and portfolio_value
    env.last_portfolio_value = 10000.0
    env.portfolio_value = 10100.0
    # Calculate reward with valid values
    reward = env._calculate_reward()
    # Check that reward is calculated correctly (0.01 for 1% increase)
    assert pytest.approx(reward) == 0.01


def test_calculate_reward_numerical_stability(env):
    """Test reward calculation with very small non-zero last_portfolio_value."""
    # Reset the environment
    env.reset()

    # Set very small last_portfolio_value and a slightly larger portfolio_value
    env.last_portfolio_value = 1e-9
    env.portfolio_value = 2e-9

    # Calculate reward
    reward = env._calculate_reward()

    # Calculate expected reward: (2e-9 - 1e-9) / 1e-9 = 1.0
    expected_reward = 1.0

    # Check that reward is calculated correctly even with small numbers
    assert pytest.approx(reward) == expected_reward

    # Test with negative small numbers
    env.last_portfolio_value = -1e-9
    env.portfolio_value = -2e-9
    reward = env._calculate_reward()
    # Expected: (-2e-9 - (-1e-9)) / -1e-9 = -1e-9 / -1e-9 = 1.0
    expected_reward = 1.0
    assert pytest.approx(reward) == expected_reward

    # Test with small positive change from small negative
    env.last_portfolio_value = -2e-9
    env.portfolio_value = -1e-9
    reward = env._calculate_reward()
    # Expected: (-1e-9 - (-2e-9)) / -2e-9 = 1e-9 / -2e-9 = -0.5
    expected_reward = -0.5
    assert pytest.approx(reward) == expected_reward


def test_calculate_reward_positive_change(env):
    """Test _calculate_reward for a positive portfolio value change."""
    # Reset the environment
    env.reset()

    # Set previous and current portfolio values for a positive change
    env.last_portfolio_value = 10000.0
    env.portfolio_value = 10100.0  # 1% increase

    # Calculate reward
    reward = env._calculate_reward()

    # Calculate expected reward (percentage change)
    expected_reward = (10100.0 - 10000.0) / 10000.0  # Should be 0.01

    # Check that reward is positive and matches the expected percentage change
    assert reward > 0.0
    assert pytest.approx(reward) == expected_reward


def test_calculate_reward_negative_change(env):
    """Test _calculate_reward for a negative portfolio value change."""
    # Reset the environment
    env.reset()

    # Set previous and current portfolio values for a negative change
    env.last_portfolio_value = 10000.0
    env.portfolio_value = 9800.0  # 2% decrease

    # Calculate reward
    reward = env._calculate_reward()

    # Calculate expected reward (percentage change)
    expected_reward = (9800.0 - 10000.0) / 10000.0  # Should be -0.02

    # Check that reward is negative and matches the expected percentage change
    assert reward < 0.0
    assert pytest.approx(reward) == expected_reward


def test_calculate_reward_no_change(env):
    """Test _calculate_reward when the portfolio value does not change."""
    # Reset the environment
    env.reset()

    # Set previous and current portfolio values to be the same
    env.last_portfolio_value = 10000.0
    env.portfolio_value = 10000.0

    # Calculate reward
    reward = env._calculate_reward()

    # Calculate expected reward (percentage change)
    expected_reward = (10000.0 - 10000.0) / 10000.0  # Should be 0.0

    # Check that reward is zero
    assert pytest.approx(reward) == expected_reward


# CONTEXTUAL INTEGRATION TESTS

def test_step_returns_percentage_change_reward(env):
    """Test that the reward returned by step is the calculated percentage change."""
    # Reset the environment
    obs1, info1 = env.reset()
    initial_value = env.portfolio_value # Should be initial_balance
    # After reset, last_portfolio_value should be the initial balance
    assert pytest.approx(env.last_portfolio_value) == env.initial_balance

    # Take step 1 (e.g., flat action, portfolio value might change slightly due to time passing if fees applied differently, but assume no change for simplicity here)
    # In a real scenario, price changes would affect value. We use flat action for minimal change.
    obs2, reward1, term1, trunc1, info2 = env.step(0)
    value_after_step1 = env.portfolio_value
    last_value_after_step1 = env.last_portfolio_value # Should be initial_value

    # Reward for the first step should be 0.0 as last_portfolio_value was None initially
    assert reward1 == 0.0
    assert last_value_after_step1 == initial_value

    # Take step 2
    obs3, reward2, term2, trunc2, info3 = env.step(0)
    value_after_step2 = env.portfolio_value
    last_value_after_step2 = env.last_portfolio_value # Should be value_after_step1

    # Reward for the second step should be the percentage change from initial_value to value_after_step1
    expected_reward2 = 0.0
    if initial_value != 0: # Avoid division by zero
        expected_reward2 = (value_after_step1 - initial_value) / initial_value

    # Check that the reward is a float
    assert isinstance(reward2, float)

    # Check that the reward matches the expected percentage change
    assert pytest.approx(reward2) == expected_reward2
    assert last_value_after_step2 == value_after_step1


def test_basic_run_through_steps(env):
    """Test a basic run through several steps to ensure the environment doesn't crash."""
    # Reset the environment
    observation, info = env.reset()
    
    # Take several steps with different actions
    actions = [0, 1, 0, 2, 0]  # Flat, Long, Flat, Short, Flat
    
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Check that the returned values have the correct types
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check that the reward is finite
        assert np.isfinite(reward)
        
        # If the episode terminated, break the loop
        if terminated:
            break
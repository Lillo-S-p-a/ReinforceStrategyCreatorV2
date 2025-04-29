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
    """Test that _calculate_reward returns 0.0 when not enough data is available."""
    # Reset the environment
    env.reset()
    
    # Check that _portfolio_value_history is empty
    assert len(env._portfolio_value_history) == 0
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # Check that reward is 0.0 when _portfolio_value_history is empty
    assert reward == 0.0
    
    # Add some values to _portfolio_value_history, but less than sharpe_window_size
    for i in range(env.sharpe_window_size - 1):
        env._portfolio_value_history.append(10000.0 + i * 100)
    
    # Check that _portfolio_value_history has less than sharpe_window_size values
    assert len(env._portfolio_value_history) < env.sharpe_window_size
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # Check that reward is 0.0 when _portfolio_value_history has less than sharpe_window_size values
    assert reward == 0.0


def test_calculate_reward_numerical_stability(env):
    """Test that _calculate_reward handles numerical instability correctly."""
    # Reset the environment
    env.reset()
    
    # Add constant values to _portfolio_value_history (will result in zero standard deviation)
    for _ in range(env.sharpe_window_size):
        env._portfolio_value_history.append(10000.0)
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # Check that reward is 0.0 when standard deviation is zero
    assert reward == 0.0
    
    # Add values with very small changes to _portfolio_value_history
    env._portfolio_value_history.clear()
    for i in range(env.sharpe_window_size):
        env._portfolio_value_history.append(10000.0 + i * 1e-10)
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # Check that reward is 0.0 when standard deviation is near-zero
    assert reward == 0.0


def test_calculate_reward_linear_increase(env_small_window):
    """Test that _calculate_reward calculates Sharpe Ratio correctly for linearly increasing portfolio values."""
    env = env_small_window
    # Reset the environment
    env.reset()
    
    # Add linearly increasing values to _portfolio_value_history
    for i in range(env.sharpe_window_size):
        env._portfolio_value_history.append(10000.0 + i * 100)
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # For linearly increasing values, the Sharpe ratio should be positive
    assert reward > 0.0
    
    # Calculate expected Sharpe ratio manually
    returns = []
    values = list(env._portfolio_value_history)
    for i in range(1, len(values)):
        prev_value = values[i-1]
        curr_value = values[i]
        returns.append((curr_value - prev_value) / prev_value)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    expected_sharpe = mean_return / std_return
    
    # Check that the calculated reward matches the expected Sharpe ratio
    assert pytest.approx(reward, abs=1e-5) == expected_sharpe


def test_calculate_reward_linear_decrease(env_small_window):
    """Test that _calculate_reward calculates Sharpe Ratio correctly for linearly decreasing portfolio values."""
    env = env_small_window
    # Reset the environment
    env.reset()
    
    # Add linearly decreasing values to _portfolio_value_history
    for i in range(env.sharpe_window_size):
        env._portfolio_value_history.append(10000.0 - i * 100)
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # For linearly decreasing values, the Sharpe ratio should be negative
    assert reward < 0.0
    
    # Calculate expected Sharpe ratio manually
    returns = []
    values = list(env._portfolio_value_history)
    for i in range(1, len(values)):
        prev_value = values[i-1]
        curr_value = values[i]
        returns.append((curr_value - prev_value) / prev_value)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    expected_sharpe = mean_return / std_return
    
    # Check that the calculated reward matches the expected Sharpe ratio
    assert pytest.approx(reward, abs=1e-5) == expected_sharpe


def test_calculate_reward_volatile(env_small_window):
    """Test that _calculate_reward calculates Sharpe Ratio correctly for volatile portfolio values."""
    env = env_small_window
    # Reset the environment
    env.reset()
    
    # Add volatile values to _portfolio_value_history
    values = [10000.0, 10200.0, 10100.0, 10300.0, 10050.0]
    for value in values:
        env._portfolio_value_history.append(value)
    
    # Calculate reward
    reward = env._calculate_reward()
    
    # Calculate expected Sharpe ratio manually
    returns = []
    for i in range(1, len(values)):
        prev_value = values[i-1]
        curr_value = values[i]
        returns.append((curr_value - prev_value) / prev_value)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    expected_sharpe = mean_return / std_return
    
    # Check that the calculated reward matches the expected Sharpe ratio
    assert pytest.approx(reward, abs=1e-5) == expected_sharpe


# CONTEXTUAL INTEGRATION TESTS

def test_step_returns_sharpe_ratio_as_reward(env_small_window):
    """Test that the reward returned by step is the calculated Sharpe Ratio."""
    env = env_small_window
    # Reset the environment
    env.reset()
    
    # Take steps to fill the _portfolio_value_history
    for _ in range(env.sharpe_window_size):
        _, reward, _, _, _ = env.step(0)  # Flat action
    
    # At this point, _portfolio_value_history should be full
    assert len(env._portfolio_value_history) == env.sharpe_window_size
    
    # Calculate the expected Sharpe ratio
    returns = []
    values = list(env._portfolio_value_history)
    for i in range(1, len(values)):
        prev_value = values[i-1]
        curr_value = values[i]
        returns.append((curr_value - prev_value) / prev_value)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Handle the case where std_return is near-zero
    if std_return < 1e-8:
        expected_sharpe = 0.0
    else:
        expected_sharpe = mean_return / std_return
    
    # Take one more step and check that the reward matches the expected Sharpe ratio
    _, reward, _, _, _ = env.step(0)  # Flat action
    
    # Check that the reward is a float
    assert isinstance(reward, float)
    
    # Check that the reward matches the expected Sharpe ratio
    assert pytest.approx(reward, abs=1e-5) == expected_sharpe


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
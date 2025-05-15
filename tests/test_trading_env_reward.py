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
    """Create a sample environment with default parameters for testing."""
    initial_balance = 10000.0
    transaction_fee_percent = 0.1
    sharpe_window_size = 20
    use_sharpe_ratio = True
    trading_frequency_penalty = 0.01
    drawdown_penalty = 0.1
    risk_free_rate = 0.0
    return TradingEnv(test_df, initial_balance, transaction_fee_percent,
                      sharpe_window_size=sharpe_window_size, use_sharpe_ratio=use_sharpe_ratio,
                      trading_frequency_penalty=trading_frequency_penalty, drawdown_penalty=drawdown_penalty,
                      risk_free_rate=risk_free_rate)


@pytest.fixture
def env_small_window(test_df):
    """Create a sample environment with small sharpe_window_size and default penalties for testing."""
    initial_balance = 10000.0
    transaction_fee_percent = 0.1
    sharpe_window_size = 5
    use_sharpe_ratio = True
    trading_frequency_penalty = 0.01
    drawdown_penalty = 0.1
    risk_free_rate = 0.0
    return TradingEnv(test_df, initial_balance, transaction_fee_percent,
                      sharpe_window_size=sharpe_window_size, use_sharpe_ratio=use_sharpe_ratio,
                      trading_frequency_penalty=trading_frequency_penalty, drawdown_penalty=drawdown_penalty,
                      risk_free_rate=risk_free_rate)


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
    # Reset the environment
    env.reset()
    # After reset, last_portfolio_value should be the initial balance
    assert pytest.approx(env.last_portfolio_value) == env.initial_balance

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
    # Check that reward is calculated correctly (includes penalties)
    # Expected: risk_adj_return (0.01) - trading_penalty (0) - drawdown_penalty (0) = 0.01
    assert pytest.approx(reward) == 0.01


def test_calculate_reward_numerical_stability(env):
    """Test reward calculation with very small non-zero last_portfolio_value."""
    # Reset the environment
    env.reset()
    env.use_sharpe_ratio = False # Focus on percentage change for this test

    # Set very small last_portfolio_value and a slightly larger portfolio_value
    env.last_portfolio_value = 1e-9
    env.portfolio_value = 2e-9
    # Manually reset max_portfolio_value for this specific test to avoid drawdown penalty
    env.max_portfolio_value = env.last_portfolio_value

    # Calculate reward
    reward = env._calculate_reward()

    # Calculate expected reward: risk_adj_return (1.0) - penalties (0) = 1.0
    expected_reward = 1.0

    # Check that reward is calculated correctly even with small numbers
    assert pytest.approx(reward) == expected_reward

    # Test with negative small numbers
    env.last_portfolio_value = -1e-9
    env.portfolio_value = -2e-9
    env.max_portfolio_value = env.last_portfolio_value # Reset max value
    reward = env._calculate_reward()
    # Expected: risk_adj_return (1.0) - penalties (0) = 1.0
    expected_reward = 1.0
    assert pytest.approx(reward) == expected_reward

    # Test with small positive change from small negative
    env.last_portfolio_value = -2e-9
    env.portfolio_value = -1e-9
    env.max_portfolio_value = env.last_portfolio_value # Reset max value
    reward = env._calculate_reward()
    # Expected: risk_adj_return (-0.5) - penalties (0) = -0.5
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

    # Calculate expected reward (percentage change - penalties)
    expected_reward = (10100.0 - 10000.0) / 10000.0  # Should be 0.01 (no penalties yet)

    # Check that reward is positive and matches the expected value
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

    # Calculate expected reward (percentage change - penalties)
    # Drawdown = (10000 - 9800) / 10000 = 0.02
    # Drawdown penalty = 0.1 * 0.02 = 0.002
    # Expected reward = -0.02 - 0.002 = -0.022
    expected_reward = (9800.0 - 10000.0) / 10000.0 - env.drawdown_penalty * ((10000.0 - 9800.0) / 10000.0)

    # Check that reward is negative and matches the expected value
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

    # Calculate expected reward (percentage change - penalties)
    expected_reward = (10000.0 - 10000.0) / 10000.0  # Should be 0.0 (no penalties)

    # Check that reward is zero
    assert pytest.approx(reward) == expected_reward


# CONTEXTUAL INTEGRATION TESTS

def test_step_returns_calculated_reward(env):
    """Test that the reward returned by step is the calculated reward including penalties."""
    # Reset the environment
    env.reset()
    env.use_sharpe_ratio = False # Simplify reward for this integration test
    initial_value = env.portfolio_value # Should be initial_balance
    assert pytest.approx(env.last_portfolio_value) == env.initial_balance

    # Take step 1 (Flat action)
    obs2, reward1, term1, trunc1, info2 = env.step(0)
    value_after_step1 = env.portfolio_value
    last_value_after_step1 = env.last_portfolio_value # Should be initial_value

    # Reward for the first step (no history for Sharpe, no trades, no drawdown)
    # Expected: percentage_change (0) - trading_penalty (0) - drawdown_penalty (0) = 0.0
    assert pytest.approx(reward1) == 0.0
    assert last_value_after_step1 == initial_value

    # Take step 2 (Long action - triggers trade)
    obs3, reward2, term2, trunc2, info3 = env.step(1)
    value_after_step2 = env.portfolio_value
    last_value_after_step2 = env.last_portfolio_value # Should be value_after_step1

    # Reward for the second step
    # Assume value_after_step1 = initial_value = 10000
    # Assume value_after_step2 = 10050 (after buying shares and price moving)
    # Percentage change = (10050 - 10000) / 10000 = 0.005
    # Sharpe ratio not applicable yet (only 1 return)
    # Trade count = 1, Trading penalty = 0.01 * 1 = 0.01
    # Max value = 10000, Current value = 10050, Drawdown = 0, Drawdown penalty = 0
    # Expected reward = 0.005 - 0.01 - 0 = -0.005
    
    # Manually calculate expected reward based on the state *after* step 2
    # Use the actual values from the environment after the step
    percentage_change_step2 = (value_after_step2 - last_value_after_step2) / last_value_after_step2 if last_value_after_step2 != 0 else 0.0
    # Since use_sharpe_ratio is False, risk_adj_return is just percentage_change
    risk_adj_return2 = percentage_change_step2
    trading_penalty2 = env.trading_frequency_penalty * env._trade_count # trade_count is 1
    # Need to update max_portfolio_value based on step 1's value before calculating drawdown
    env.max_portfolio_value = max(env.max_portfolio_value, last_value_after_step2)
    drawdown2 = max(0, (env.max_portfolio_value - value_after_step2) / env.max_portfolio_value) if env.max_portfolio_value > 0 else 0
    drawdown_penalty2 = env.drawdown_penalty * drawdown2
    expected_reward2 = risk_adj_return2 - trading_penalty2 - drawdown_penalty2

    assert isinstance(reward2, float)
    assert pytest.approx(reward2) == expected_reward2
    assert last_value_after_step2 == value_after_step1
    assert env._trade_count == 1 # Check trade count updated

    # Take step 3 (Flat action - triggers trade)
    obs4, reward3, term3, trunc3, info4 = env.step(0)
    value_after_step3 = env.portfolio_value
    last_value_after_step3 = env.last_portfolio_value # Should be value_after_step2

    # Manually calculate expected reward based on the state *after* step 3
    percentage_change_step3 = (value_after_step3 - last_value_after_step3) / last_value_after_step3 if last_value_after_step3 != 0 else 0.0
    # Since use_sharpe_ratio is False, risk_adj_return is just percentage_change
    risk_adj_return3 = percentage_change_step3
    trading_penalty3 = env.trading_frequency_penalty * env._trade_count # trade_count is 2
    # Need to update max_portfolio_value based on step 2's value before calculating drawdown
    env.max_portfolio_value = max(env.max_portfolio_value, last_value_after_step3)
    drawdown3 = max(0, (env.max_portfolio_value - value_after_step3) / env.max_portfolio_value) if env.max_portfolio_value > 0 else 0
    drawdown_penalty3 = env.drawdown_penalty * drawdown3
    expected_reward3 = risk_adj_return3 - trading_penalty3 - drawdown_penalty3
    
    assert isinstance(reward3, float)
    assert pytest.approx(reward3) == expected_reward3
    assert last_value_after_step3 == value_after_step2
    assert env._trade_count == 2 # Check trade count updated


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

# --- New Tests for Enhanced Reward Components ---

def test_sharpe_ratio_calculation(env_small_window):
    """Test the Sharpe ratio calculation within the reward function."""
    env = env_small_window
    env.use_sharpe_ratio = True
    env.risk_free_rate = 0.0
    env.reset()

    # Simulate some steps with varying returns by setting portfolio values
    # _calculate_reward will handle appending returns
    
    # Step 1
    env.last_portfolio_value = 10000
    env.portfolio_value = 10100 # Return = 0.01
    reward1 = env._calculate_reward() # Appends 0.01. Sharpe not used. Reward = 0.01
    
    # Step 2
    env.last_portfolio_value = 10100
    env.portfolio_value = 10050 # Return = -0.004950495...
    reward2 = env._calculate_reward() # Appends -0.00495... Sharpe uses [0.01, -0.00495...]
    
    # Step 3
    env.last_portfolio_value = 10050
    env.portfolio_value = 10150 # Return = 0.009950248...
    reward3 = env._calculate_reward() # Appends 0.00995... Sharpe uses [0.01, -0.00495..., 0.00995...]

    # Step 4 - Calculate reward and check against manual Sharpe calculation
    env.last_portfolio_value = 10150
    env.portfolio_value = 10150 # Return = 0.0
    reward4 = env._calculate_reward() # Appends 0.0. Sharpe uses [0.01, -0.00495..., 0.00995..., 0.0]

    # Manually calculate expected Sharpe-based return for step 4
    # Get the returns history as calculated by the environment
    returns_hist = list(env._portfolio_returns)
    assert len(returns_hist) == 4 # Should have 4 returns now
    
    returns_array = np.array(returns_hist)
    returns_mean = np.mean(returns_array)
    returns_std = np.std(returns_array)
    
    if returns_std == 0:
        expected_risk_adj_return = returns_mean
    else:
        expected_sharpe = (returns_mean - env.risk_free_rate) / returns_std
        expected_risk_adj_return = expected_sharpe * 0.01 # Apply scaling factor

    # Calculate expected total reward for step 4 (assuming no penalties for simplicity)
    env._trade_count = 0
    env.max_portfolio_value = 10150 # Assume no drawdown
    expected_reward4 = expected_risk_adj_return - 0 - 0
    
    assert pytest.approx(reward4) == expected_reward4


def test_trading_frequency_penalty(env):
    """Test the trading frequency penalty component."""
    env.reset()
    env.use_sharpe_ratio = False # Focus on penalties
    env.trading_frequency_penalty = 0.05 # Use a distinct value

    # Step 1: No trade yet
    env.last_portfolio_value = 10000
    env.portfolio_value = 10100
    reward1 = env._calculate_reward()
    expected_reward1 = 0.01 # Only percentage change
    assert pytest.approx(reward1) == expected_reward1
    assert env._trade_count == 0

    # Step 2: Simulate a trade
    env._trade_count = 1
    env.last_portfolio_value = 10100
    env.portfolio_value = 10200
    reward2 = env._calculate_reward()
    percentage_change2 = (10200 - 10100) / 10100 # ~0.0099
    expected_penalty2 = 0.05 * 1 # Penalty for 1 trade
    expected_reward2 = percentage_change2 - expected_penalty2
    assert pytest.approx(reward2) == expected_reward2

    # Step 3: Simulate another trade
    env._trade_count = 2
    env.last_portfolio_value = 10200
    env.portfolio_value = 10300
    reward3 = env._calculate_reward()
    percentage_change3 = (10300 - 10200) / 10200 # ~0.0098
    expected_penalty3 = 0.05 * 2 # Penalty for 2 trades
    expected_reward3 = percentage_change3 - expected_penalty3
    assert pytest.approx(reward3) == expected_reward3


def test_drawdown_penalty(env):
    """Test the drawdown penalty component."""
    env.reset()
    env.use_sharpe_ratio = False # Focus on penalties
    env.drawdown_penalty = 0.2 # Use a distinct value

    # Step 1: Increase value, no drawdown
    env.last_portfolio_value = 10000
    env.portfolio_value = 10500
    env.max_portfolio_value = 10500 # Update max value
    reward1 = env._calculate_reward()
    expected_reward1 = 0.05 # Only percentage change
    assert pytest.approx(reward1) == expected_reward1

    # Step 2: Decrease value, trigger drawdown
    env.last_portfolio_value = 10500
    env.portfolio_value = 10200 # Value drops
    # Max value is still 10500
    reward2 = env._calculate_reward()
    percentage_change2 = (10200 - 10500) / 10500 # ~ -0.02857
    current_drawdown = (10500 - 10200) / 10500 # ~ 0.02857
    expected_penalty2 = 0.2 * current_drawdown
    expected_reward2 = percentage_change2 - expected_penalty2
    assert pytest.approx(reward2) == expected_reward2

    # Step 3: Value recovers slightly, but still in drawdown
    env.last_portfolio_value = 10200
    env.portfolio_value = 10300
    # Max value is still 10500
    reward3 = env._calculate_reward()
    percentage_change3 = (10300 - 10200) / 10200 # ~ 0.0098
    current_drawdown3 = (10500 - 10300) / 10500 # ~ 0.01905
    expected_penalty3 = 0.2 * current_drawdown3
    expected_reward3 = percentage_change3 - expected_penalty3
    assert pytest.approx(reward3) == expected_reward3

    # Step 4: Value reaches new peak, drawdown resets
    env.last_portfolio_value = 10300
    env.portfolio_value = 10600
    env.max_portfolio_value = 10600 # New peak
    reward4 = env._calculate_reward()
    percentage_change4 = (10600 - 10300) / 10300 # ~ 0.0291
    current_drawdown4 = 0 # No drawdown from new peak
    expected_penalty4 = 0.2 * current_drawdown4
    expected_reward4 = percentage_change4 - expected_penalty4
    assert pytest.approx(reward4) == expected_reward4
"""
Tests for the trading_environment module.
"""

import pytest
import numpy as np
import pandas as pd

from reinforcestrategycreator.trading_environment import TradingEnv


@pytest.fixture
def test_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


@pytest.fixture
def initial_balance():
    """Set initial balance for testing."""
    return 10000.0


@pytest.fixture
def env(test_df, initial_balance):
    """Create a sample environment for testing."""
    return TradingEnv(test_df, initial_balance)


def test_environment_initialization(env, test_df, initial_balance):
    """Test that the environment initializes correctly."""
    # Assert that the environment has the correct attributes
    assert env.df is not None
    assert len(env.df) == len(test_df)
    assert env.initial_balance == initial_balance
    assert env.current_step == 0
    
    # Check action and observation spaces
    assert env.action_space.n == 3  # 3 actions: hold, buy, sell
    assert env.observation_space.shape[0] == len(test_df.columns) + 2  # features + balance + position


def test_reset(env):
    """Test that reset returns a valid initial observation."""
    # Reset the environment
    observation, info = env.reset()
    
    # Assert that the observation has the correct shape
    assert observation.shape == env.observation_space.shape
    
    # Assert that info is a dictionary
    assert isinstance(info, dict)
    
    # Assert that current_step is reset to 0
    assert env.current_step == 0


def test_step(env):
    """Test that step returns valid values."""
    # Reset the environment
    env.reset()
    
    # Take a step with action 0 (hold)
    observation, reward, terminated, truncated, info = env.step(0)
    
    # Assert that the returned values have the correct types
    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Assert that current_step is incremented
    assert env.current_step == 1
    
    # Assert that info contains the expected keys
    assert 'balance' in info
    assert 'shares_held' in info
    assert 'portfolio_value' in info
    assert 'current_price' in info


def test_step_hold_action(env):
    """Test the hold action (0)."""
    # Reset the environment
    env.reset()
    
    # Get initial balance and shares
    initial_balance = env.balance
    initial_shares = env.shares_held
    
    # Take a step with action 0 (hold)
    _, _, _, _, info = env.step(0)
    
    # Assert that balance and shares remain unchanged
    assert env.balance == initial_balance
    assert env.shares_held == initial_shares


def test_step_buy_action(env):
    """Test the buy action (1)."""
    # Reset the environment
    env.reset()
    
    # Get initial balance and shares
    initial_balance = env.balance
    initial_shares = env.shares_held
    
    # Take a step with action 1 (buy)
    _, _, _, _, info = env.step(1)
    
    # Assert that balance decreased and shares increased
    assert env.balance < initial_balance
    assert env.shares_held > initial_shares


def test_step_sell_action_with_shares(env):
    """Test the sell action (2) when shares are held."""
    # Reset the environment
    env.reset()
    
    # First buy some shares
    env.step(1)
    
    # Get balance and shares after buying
    balance_after_buy = env.balance
    shares_after_buy = env.shares_held
    
    # Make sure we have shares to sell
    assert shares_after_buy > 0
    
    # Take a step with action 2 (sell)
    _, _, _, _, info = env.step(2)
    
    # Assert that balance increased and shares decreased to zero
    assert env.balance > balance_after_buy
    assert env.shares_held == 0


def test_step_sell_action_without_shares(env):
    """Test the sell action (2) when no shares are held."""
    # Reset the environment
    env.reset()
    
    # Ensure we have no shares
    env.shares_held = 0
    
    # Get initial balance
    initial_balance = env.balance
    
    # Take a step with action 2 (sell)
    _, _, _, _, info = env.step(2)
    
    # Assert that balance remains unchanged
    assert env.balance == initial_balance
    assert env.shares_held == 0


def test_reward_calculation(env):
    """Test that rewards are calculated correctly."""
    # Reset the environment
    env.reset()
    
    # Take a step and get the reward
    _, reward1, _, _, _ = env.step(1)  # Buy
    
    # The reward should be related to the change in portfolio value
    # For the first step after buying, it might be slightly negative due to transaction fees
    
    # Take another step
    _, reward2, _, _, _ = env.step(2)  # Sell
    
    # The reward should reflect the change in portfolio value
    # It could be positive or negative depending on price movement
    
    # Both rewards should be finite numbers
    assert np.isfinite(reward1)
    assert np.isfinite(reward2)


def test_portfolio_value_calculation(env):
    """Test that portfolio value is calculated correctly."""
    # Reset the environment
    env.reset()
    
    # Take a step to buy shares
    _, _, _, _, info1 = env.step(1)
    
    # Calculate expected portfolio value
    expected_value = env.balance + (env.shares_held * env.current_price)
    
    # Assert that portfolio value is calculated correctly
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_value
    assert pytest.approx(info1['portfolio_value'], abs=1e-5) == expected_value


def test_episode_termination(env, test_df):
    """Test that the episode terminates when reaching the end of data."""
    # Reset the environment
    env.reset()
    
    # Run until one step before the end
    for _ in range(len(test_df) - 2):
        _, _, terminated, _, _ = env.step(0)
        assert not terminated
    
    # Take the final step
    _, _, terminated, _, _ = env.step(0)
    
    # Assert that the episode is terminated
    assert terminated


def test_render(env):
    """Test that render returns the expected output."""
    # Reset the environment
    env.reset()
    
    # Test human mode
    human_render = env.render(mode='human')
    assert human_render is None
    
    # Test rgb_array mode
    rgb_render = env.render(mode='rgb_array')
    assert isinstance(rgb_render, np.ndarray)
    assert rgb_render.shape == (100, 100, 3)


def test_close(env):
    """Test that close method runs without errors."""
    # This should not raise an exception
    env.close()
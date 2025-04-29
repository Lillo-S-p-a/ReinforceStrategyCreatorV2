"""
Targeted tests for the step method in TradingEnv class.

This module focuses on testing the step method of the TradingEnv class,
which is central to the RL Core, processing agent actions, updating portfolio state,
calculating rewards, and returning the next state according to the Gymnasium API.

:ComponentRole TradingEnvironment
:Context RL Core
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from reinforcestrategycreator.trading_environment import TradingEnv
from reinforcestrategycreator.rl_agent import StrategyAgent


@pytest.fixture
def test_df():
    """Create a sample DataFrame with price data for testing."""
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


@pytest.fixture
def env(test_df):
    """Create a sample environment for testing."""
    initial_balance = 10000.0
    transaction_fee_percent = 0.1
    return TradingEnv(test_df, initial_balance, transaction_fee_percent)


@pytest.fixture
def agent(env):
    """Create a sample agent for integration testing."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return StrategyAgent(state_size, action_size)


# CORE LOGIC TESTS

def test_step_hold_action_detailed(env):
    """Test the hold action (0) in detail."""
    # Reset the environment
    observation, info = env.reset()
    
    # Get initial state
    initial_balance = env.balance
    initial_shares = env.shares_held
    initial_portfolio_value = env.portfolio_value
    
    # Take a step with action 0 (hold)
    next_observation, reward, terminated, truncated, info = env.step(0)
    
    # Assert that balance and shares remain unchanged
    assert env.balance == initial_balance
    assert env.shares_held == initial_shares
    
    # Portfolio value should change only due to price movement, not due to transactions
    # Since we're holding, any change in portfolio value is due to price changes
    expected_portfolio_value = initial_balance + (initial_shares * env.current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
    
    # Check that current_step was incremented
    assert env.current_step == 1
    
    # Check that the reward reflects the change in portfolio value
    expected_reward = ((env.portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_buy_action_detailed(env):
    """Test the buy action (1) in detail."""
    # Reset the environment
    observation, info = env.reset()
    
    # Get initial state
    initial_balance = env.balance
    initial_shares = env.shares_held
    initial_portfolio_value = env.portfolio_value
    
    # Take a step with action 1 (buy)
    next_observation, reward, terminated, truncated, info = env.step(1)
    
    # Calculate expected values
    # Expected shares to buy (based on the implementation in _execute_trade_action)
    expected_shares_to_buy = int(initial_balance / (env.current_price * (1 + env.transaction_fee_percent / 100)))
    
    # Expected cost including fees
    expected_cost = expected_shares_to_buy * env.current_price
    expected_fee = expected_cost * (env.transaction_fee_percent / 100)
    expected_total_cost = expected_cost + expected_fee
    
    # Expected balance after buying
    expected_balance = initial_balance - expected_total_cost
    
    # Assert that balance decreased correctly
    assert pytest.approx(env.balance, abs=1e-5) == expected_balance
    
    # Assert that shares increased correctly
    assert env.shares_held == expected_shares_to_buy
    
    # Check portfolio value calculation
    expected_portfolio_value = env.balance + (env.shares_held * env.current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
    
    # Check that the reward reflects the change in portfolio value
    expected_reward = ((env.portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_sell_action_detailed(env):
    """Test the sell action (2) in detail."""
    # Reset the environment
    observation, info = env.reset()
    
    # First buy some shares
    env.step(1)
    
    # Get state after buying
    balance_after_buy = env.balance
    shares_after_buy = env.shares_held
    portfolio_value_after_buy = env.portfolio_value
    
    # Take a step with action 2 (sell)
    next_observation, reward, terminated, truncated, info = env.step(2)
    
    # Calculate expected values
    # Expected revenue including fees
    expected_revenue = shares_after_buy * env.current_price
    expected_fee = expected_revenue * (env.transaction_fee_percent / 100)
    expected_total_revenue = expected_revenue - expected_fee
    
    # Expected balance after selling
    expected_balance = balance_after_buy + expected_total_revenue
    
    # Assert that balance increased correctly
    assert pytest.approx(env.balance, abs=1e-5) == expected_balance
    
    # Assert that shares decreased to zero
    assert env.shares_held == 0
    
    # Check portfolio value calculation
    expected_portfolio_value = env.balance + (env.shares_held * env.current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
    
    # Check that the reward reflects the change in portfolio value
    expected_reward = ((env.portfolio_value - portfolio_value_after_buy) / portfolio_value_after_buy) * 100
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_sell_with_zero_shares(env):
    """Test the sell action (2) when no shares are held (edge case)."""
    # Reset the environment
    observation, info = env.reset()
    
    # Ensure we have no shares
    env.shares_held = 0
    
    # Get initial state
    initial_balance = env.balance
    initial_portfolio_value = env.portfolio_value
    
    # Take a step with action 2 (sell)
    next_observation, reward, terminated, truncated, info = env.step(2)
    
    # Assert that balance remains unchanged
    assert env.balance == initial_balance
    
    # Assert that shares remain at zero
    assert env.shares_held == 0
    
    # Check portfolio value calculation (should only change due to time step increment)
    expected_portfolio_value = env.balance + (env.shares_held * env.current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
    
    # Check that the reward reflects the change in portfolio value
    expected_reward = ((env.portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_buy_with_insufficient_balance(env):
    """Test the buy action (1) with insufficient balance (edge case)."""
    # Reset the environment
    observation, info = env.reset()
    
    # Set a very low balance
    env.balance = 10.0  # Not enough to buy even one share
    initial_shares = env.shares_held
    initial_portfolio_value = env.portfolio_value
    
    # Take a step with action 1 (buy)
    next_observation, reward, terminated, truncated, info = env.step(1)
    
    # Assert that shares remain unchanged (couldn't buy any)
    assert env.shares_held == initial_shares
    
    # Check portfolio value calculation
    expected_portfolio_value = env.balance + (env.shares_held * env.current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
    
    # Check that the reward reflects the change in portfolio value
    expected_reward = ((env.portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_observation_generation(env):
    """Test that the step method generates correct observations."""
    # Reset the environment
    observation, info = env.reset()
    
    # Take a step
    next_observation, reward, terminated, truncated, info = env.step(1)
    
    # Check observation shape
    assert next_observation.shape == env.observation_space.shape
    
    # Check observation type
    assert isinstance(next_observation, np.ndarray)
    assert next_observation.dtype == np.float32
    
    # Check that observation contains normalized market data and account information
    # The last two elements should be normalized balance and position
    normalized_balance = env.balance / env.initial_balance
    normalized_position = env.shares_held * env.current_price / env.initial_balance
    
    # Get the market data from the current step
    market_data = env.df.iloc[env.current_step].values
    market_data_normalized = market_data / (np.max(np.abs(market_data)) + 1e-10)
    
    # Construct expected observation
    expected_observation = np.append(market_data_normalized, [normalized_balance, normalized_position])
    expected_observation = expected_observation.astype(np.float32)
    
    # Check that the observation matches the expected values
    assert np.allclose(next_observation, expected_observation, rtol=1e-5)


def test_step_reward_calculation(env):
    """Test that the step method calculates rewards correctly."""
    # Reset the environment
    observation, info = env.reset()
    
    # Take a step with action 1 (buy)
    next_observation, reward, terminated, truncated, info = env.step(1)
    
    # Calculate expected reward
    portfolio_change = env.portfolio_value - env.initial_balance
    expected_reward = (portfolio_change / env.initial_balance) * 100
    
    # Check that the reward matches the expected value
    assert pytest.approx(reward, abs=1e-5) == expected_reward
    
    # Take another step with action 2 (sell)
    portfolio_value_before_sell = env.portfolio_value
    next_observation, reward, terminated, truncated, info = env.step(2)
    
    # Calculate expected reward
    portfolio_change = env.portfolio_value - portfolio_value_before_sell
    expected_reward = (portfolio_change / portfolio_value_before_sell) * 100
    
    # Check that the reward matches the expected value
    assert pytest.approx(reward, abs=1e-5) == expected_reward


def test_step_termination(env):
    """Test that the step method correctly determines when an episode is terminated."""
    # Reset the environment
    observation, info = env.reset()
    
    # Run until one step before the end
    for _ in range(len(env.df) - 2):
        _, _, terminated, _, _ = env.step(0)
        assert not terminated
    
    # Take the final step
    _, _, terminated, _, _ = env.step(0)
    
    # Assert that the episode is terminated
    assert terminated


# CONTEXTUAL INTEGRATION TESTS

def test_step_interaction_with_agent(env, agent):
    """Test the interaction between the environment's step method and an agent."""
    # Reset the environment
    observation, info = env.reset()
    
    # Use the agent to select an action
    action = agent.select_action(observation)
    
    # Ensure the action is valid
    assert 0 <= action < env.action_space.n
    
    # Take a step with the selected action
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    # Check that the returned values have the correct types
    assert isinstance(next_observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check that the agent can process the next observation
    next_action = agent.select_action(next_observation)
    assert 0 <= next_action < env.action_space.n
    
    # Check that the agent can learn from the transition
    agent.learn(observation, action, reward, next_observation, terminated)
    
    # This test passes if no exceptions are raised during the interaction


def test_step_gymnasium_api_compatibility(env):
    """Test that the step method returns values matching the expected Gymnasium API types and structure."""
    # Reset the environment
    observation, info = env.reset()
    
    # Take a step
    next_observation, reward, terminated, truncated, info = env.step(0)
    
    # Check return types according to Gymnasium API
    assert isinstance(next_observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check observation space compliance
    assert env.observation_space.contains(next_observation)
    
    # Check info dictionary contains expected keys
    assert 'balance' in info
    assert 'shares_held' in info
    assert 'current_price' in info
    assert 'portfolio_value' in info
    assert 'step' in info
    
    # Check that the values in info have the correct types
    assert isinstance(info['balance'], float)
    assert isinstance(info['shares_held'], int)
    assert isinstance(info['current_price'], float)
    assert isinstance(info['portfolio_value'], float)
    assert isinstance(info['step'], int)


def test_step_multiple_actions_sequence(env):
    """Test a sequence of actions to ensure consistent behavior over multiple steps."""
    # Reset the environment
    observation, info = env.reset()
    
    # Define a sequence of actions
    actions = [1, 0, 2, 0, 1]  # Buy, Hold, Sell, Hold, Buy
    
    # Execute the sequence
    for action in actions:
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Check that the returned values have the correct types
        assert isinstance(next_observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check that portfolio value is calculated correctly
        expected_portfolio_value = env.balance + (env.shares_held * env.current_price)
        assert pytest.approx(env.portfolio_value, abs=1e-5) == expected_portfolio_value
        
        # If the episode terminated, break the loop
        if terminated:
            break
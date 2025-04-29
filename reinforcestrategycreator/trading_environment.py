"""
Trading Environment Module

This module provides a reinforcement learning environment for trading simulation.
:ComponentRole TradingEnvironment
:Context RL Core (Req 3.2)
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Trading Environment for reinforcement learning.
    
    This class implements a trading environment that simulates market interactions
    for a reinforcement learning agent. It inherits from gymnasium.Env and implements
    the required methods.
    
    Attributes:
        df (pd.DataFrame): Historical market data.
        initial_balance (float): Initial account balance.
        action_space (gym.spaces.Space): The action space of the environment.
        observation_space (gym.spaces.Space): The observation space of the environment.
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee_percent: float = 0.1):
        """
        Initialize the trading environment.
        
        Args:
            df (pd.DataFrame): Historical market data for simulation.
            initial_balance (float): Initial account balance.
            transaction_fee_percent (float): Fee percentage for each transaction (default: 0.1%).
        """
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Placeholder for current step in the environment
        self.current_step = 0
        
        # Portfolio state
        self.balance = initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.portfolio_value = initial_balance
        self.last_portfolio_value = initial_balance
        
        # Define action and observation spaces
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market data features + account information
        # Features include price data, indicators, balance, and position
        num_features = len(df.columns) + 2  # +2 for balance and position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(df)} data points and initial balance {initial_balance}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options for reset.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset environment state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        
        # Get the initial observation
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value
        }
        
        logger.info("Environment reset")
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the given action.
        
        Args:
            action (int): The action to take (0 = hold, 1 = buy, 2 = sell).
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                - Observation: The new state after taking the action.
                - Reward: The reward received for taking the action.
                - Terminated: Whether the episode has terminated.
                - Truncated: Whether the episode was truncated.
                - Info: Additional information about the step.
        """
        # Save the last portfolio value for reward calculation
        self.last_portfolio_value = self.portfolio_value
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if episode is done (reached the end of data)
        done = self.current_step >= len(self.df) - 1
        
        # Get the current price from the dataframe
        self.current_price = self.df.iloc[self.current_step]['close']
        
        # Execute the trading action
        self._execute_trade_action(action)
        
        # Calculate the current portfolio value
        self.portfolio_value = self.balance + self.shares_held * self.current_price
        
        # Calculate reward as the change in portfolio value
        reward = self._calculate_reward()
        
        # Get the new observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'step': self.current_step
        }
        
        terminated = done
        truncated = False
        
        logger.debug(f"Step {self.current_step}: action={action}, reward={reward}, done={done}")
        return observation, reward, terminated, truncated, info
    
    def _execute_trade_action(self, action: int) -> None:
        """
        Execute the specified trading action.
        
        Args:
            action (int): The action to take (0 = hold, 1 = buy, 2 = sell).
        """
        if action == 0:  # Hold
            return
        
        elif action == 1:  # Buy
            # Calculate maximum shares that can be bought
            max_shares_possible = self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100))
            
            # Buy all possible shares (simplified approach)
            shares_to_buy = int(max_shares_possible)
            
            # Check if we can buy at least 1 share
            if shares_to_buy > 0:
                # Calculate the cost including fees
                cost = shares_to_buy * self.current_price
                transaction_fee = cost * (self.transaction_fee_percent / 100)
                total_cost = cost + transaction_fee
                
                # Update balance and shares
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                
                logger.debug(f"Bought {shares_to_buy} shares at {self.current_price} each, "
                           f"total cost: {total_cost} (including fee: {transaction_fee})")
            
        elif action == 2:  # Sell
            # Check if we have shares to sell
            if self.shares_held > 0:
                # Sell all shares (simplified approach)
                # Calculate the revenue including fees
                revenue = self.shares_held * self.current_price
                transaction_fee = revenue * (self.transaction_fee_percent / 100)
                total_revenue = revenue - transaction_fee
                
                # Update balance and shares
                self.balance += total_revenue
                
                logger.debug(f"Sold {self.shares_held} shares at {self.current_price} each, "
                           f"total revenue: {total_revenue} (after fee: {transaction_fee})")
                
                self.shares_held = 0
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the change in portfolio value.
        
        Returns:
            float: The calculated reward.
        """
        # Calculate the change in portfolio value
        portfolio_change = self.portfolio_value - self.last_portfolio_value
        
        # Calculate percentage change for scaling
        percentage_change = portfolio_change / self.last_portfolio_value if self.last_portfolio_value > 0 else 0
        
        # Scale the reward (multiplying by 100 to make it more meaningful)
        reward = percentage_change * 100
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state representation).
        
        Returns:
            np.ndarray: The current state observation.
        """
        # If we're at the beginning, we can't look back
        if self.current_step == 0:
            # Return zeros for the first observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get the current row of market data
        market_data = self.df.iloc[self.current_step].values
        
        # Normalize the market data (simple min-max normalization)
        # This is a simplified approach - in a real implementation, you might want to use
        # a more sophisticated normalization method or use a rolling window
        market_data_normalized = market_data / (np.max(np.abs(market_data)) + 1e-10)
        
        # Add account information (balance and shares held)
        # Normalize these values as well
        normalized_balance = self.balance / self.initial_balance
        normalized_position = self.shares_held * self.current_price / self.initial_balance
        
        # Combine market data and account information
        observation = np.append(market_data_normalized, [normalized_balance, normalized_position])
        
        return observation.astype(np.float32)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode (str): The rendering mode.
            
        Returns:
            Optional[np.ndarray]: Rendered image if mode is 'rgb_array', None otherwise.
        """
        # Placeholder implementation
        # Will be expanded in future tasks
        if mode == 'human':
            # Print current state information
            logger.info(f"Current step: {self.current_step}/{len(self.df) - 1}")
            return None
        elif mode == 'rgb_array':
            # Return an empty image as placeholder
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        # Placeholder implementation
        # Will be expanded if needed in future tasks
        logger.info("Environment closed")
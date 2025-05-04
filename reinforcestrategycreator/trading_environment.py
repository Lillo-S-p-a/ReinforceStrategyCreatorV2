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
from typing import Tuple, Dict, Any, Optional, Union, List
from collections import deque

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
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee_percent: float = 0.1, window_size: int = 5, sharpe_window_size: int = 20):
        """
        Initialize the trading environment.
        
        Args:
            df (pd.DataFrame): Historical market data for simulation, including pre-calculated technical indicators.
                               Assumes the DataFrame already contains indicator columns calculated by
                               technical_analyzer.calculate_indicators().
            initial_balance (float): Initial account balance.
            transaction_fee_percent (float): Fee percentage for each transaction (default: 0.1%).
            window_size (int): Number of time steps to include in each observation (default: 5).
            sharpe_window_size (int): Number of time steps to use for Sharpe ratio calculation (default: 20).
        """
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.sharpe_window_size = sharpe_window_size
        
        # Placeholder for current step in the environment
        self.current_step = 0
        
        # Portfolio state
        self.balance = initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.portfolio_value = initial_balance
        self.last_portfolio_value = initial_balance
        
        # Portfolio value history for Sharpe ratio calculation
        self._portfolio_value_history = deque(maxlen=sharpe_window_size)
        
        # Current position state: 0 = Flat, 1 = Long, -1 = Short
        self.current_position = 0
        
        # Define action and observation spaces
        # Action space: 0 = Flat, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market data features + technical indicators + account information
        # Features include price data, indicators, balance, and position
        # For sliding window, we include window_size steps of market data plus current account information
        num_market_features = len(df.columns)
        num_portfolio_features = 2  # balance and position
        total_features = (window_size * num_market_features) + num_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(df)} data points (including indicators) and initial balance {initial_balance}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Finds the first valid step index where indicators are available (non-NaN) and sets
        the current_step to this index. This ensures that the environment starts
        at a point where technical indicators are properly calculated.
        
        If no step with all non-NaN values is found, it will use the first step where
        at least the price data is available.
        
        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options for reset.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
            
        Raises:
            RuntimeError: If no valid starting step is found (e.g., DataFrame is empty).
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset environment state
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        self.current_position = 0  # Reset to Flat position
        
        # Clear portfolio value history
        self._portfolio_value_history.clear()
        
        if len(self.df) == 0:
            error_msg = "DataFrame is empty. Cannot reset environment."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # First try: Find the first step where all indicators are non-NaN
        valid_start_step = None
        for i in range(len(self.df)):
            if not self.df.iloc[i].isna().any():
                valid_start_step = i
                logger.info(f"Found step {i} with all indicators non-NaN")
                break
        
        # Second try: If no step with all non-NaN values is found,
        # find the first step where at least the price data is available
        if valid_start_step is None:
            for i in range(len(self.df)):
                # Check if 'close' price is available (not NaN)
                if not pd.isna(self.df.iloc[i]['close']):
                    valid_start_step = i
                    logger.info(f"Found step {i} with valid price data (some indicators may be NaN)")
                    break
        
        # If still no valid starting step is found, use the first step but log a warning
        if valid_start_step is None:
            valid_start_step = 0
            logger.warning("No step with valid price data found. Using first step (index 0) as fallback.")
        
        # Set the current step to the first valid step
        # Ensure we have enough data for a full window if possible
        # If valid_start_step is already at least window_size, we can use it
        # Otherwise, we need to find a balance between having valid data and having enough history
        if valid_start_step >= self.window_size - 1:
            self.current_step = valid_start_step
        else:
            # If we don't have enough history before valid_start_step, use valid_start_step
            # This means we'll need to pad the observation in _get_observation
            self.current_step = valid_start_step
            logger.info(f"Starting at step {valid_start_step} with less than window_size history")
        
        # Get the current price from the dataframe
        self.current_price = self.df.iloc[self.current_step]['close']
        
        # Get the initial observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value,
            'starting_step': self.current_step,
            'current_price': self.current_price
        }
        
        logger.info(f"Environment reset - Starting at step {self.current_step}")
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the given action.
        
        Args:
            action (int): The action to take (0 = Flat, 1 = Long, 2 = Short).
            
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
        
        # Add current portfolio value to history for Sharpe ratio calculation
        self._portfolio_value_history.append(self.portfolio_value)
        
        # Calculate reward as the Sharpe ratio
        reward = self._calculate_reward()
        
        # Get the new observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'current_position': self.current_position,
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
            action (int): The action to take (0 = Flat, 1 = Long, 2 = Short).
        """
        # Get the current position before executing the action
        prev_position = self.current_position
        
        if action == 0:  # Flat
            if prev_position == 1:  # Long -> Flat (Sell all shares)
                if self.shares_held > 0:
                    # Calculate the revenue including fees
                    revenue = self.shares_held * self.current_price
                    transaction_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - transaction_fee
                    
                    # Update balance and shares
                    self.balance += total_revenue
                    
                    logger.debug(f"Long -> Flat: Sold {self.shares_held} shares at {self.current_price} each, "
                               f"total revenue: {total_revenue} (after fee: {transaction_fee})")
                    
                    self.shares_held = 0
                    self.current_position = 0
            
            elif prev_position == -1:  # Short -> Flat (Cover short position)
                if self.shares_held < 0:
                    # Calculate the cost to buy back shares including fees
                    cost = abs(self.shares_held) * self.current_price
                    transaction_fee = cost * (self.transaction_fee_percent / 100)
                    total_cost = cost + transaction_fee
                    
                    # Check if we have enough balance to cover
                    if self.balance >= total_cost:
                        # Update balance and shares
                        self.balance -= total_cost
                        
                        logger.debug(f"Short -> Flat: Covered {abs(self.shares_held)} shares at {self.current_price} each, "
                                   f"total cost: {total_cost} (including fee: {transaction_fee})")
                        
                        self.shares_held = 0
                        self.current_position = 0
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cost}, Available: {self.balance}")
        
        elif action == 1:  # Long
            if prev_position == 0:  # Flat -> Long (Buy shares)
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
                    self.current_position = 1
                    
                    logger.debug(f"Flat -> Long: Bought {shares_to_buy} shares at {self.current_price} each, "
                               f"total cost: {total_cost} (including fee: {transaction_fee})")
            
            elif prev_position == -1:  # Short -> Long (Cover short and then buy)
                # First, cover the short position
                if self.shares_held < 0:
                    # Calculate the cost to buy back shares including fees
                    cover_cost = abs(self.shares_held) * self.current_price
                    cover_fee = cover_cost * (self.transaction_fee_percent / 100)
                    total_cover_cost = cover_cost + cover_fee
                    
                    # Check if we have enough balance to cover
                    if self.balance >= total_cover_cost:
                        # Update balance after covering
                        self.balance -= total_cover_cost
                        
                        logger.debug(f"Short -> Long (Step 1): Covered {abs(self.shares_held)} shares at {self.current_price} each, "
                                   f"total cost: {total_cover_cost} (including fee: {cover_fee})")
                        
                        self.shares_held = 0
                        
                        # Then, buy shares with remaining balance
                        max_shares_possible = self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100))
                        shares_to_buy = int(max_shares_possible)
                        
                        if shares_to_buy > 0:
                            # Calculate the cost including fees
                            buy_cost = shares_to_buy * self.current_price
                            buy_fee = buy_cost * (self.transaction_fee_percent / 100)
                            total_buy_cost = buy_cost + buy_fee
                            
                            # Update balance and shares
                            self.balance -= total_buy_cost
                            self.shares_held += shares_to_buy
                            self.current_position = 1
                            
                            logger.debug(f"Short -> Long (Step 2): Bought {shares_to_buy} shares at {self.current_price} each, "
                                       f"total cost: {total_buy_cost} (including fee: {buy_fee})")
                        else:
                            # If we can't buy any shares after covering, we're just flat
                            self.current_position = 0
                            logger.debug("Short -> Long: Covered short position but insufficient funds to go long")
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cover_cost}, Available: {self.balance}")
        
        elif action == 2:  # Short
            if prev_position == 0:  # Flat -> Short (Sell short)
                # Calculate number of shares to short (similar to buying, but we're selling shares we don't own)
                # Use a similar approach to buying, but for shorting
                # Calculate shares based on what could be bought (mirroring long logic for simplicity)
                buyable_shares = int(self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100)))
                shares_to_short = buyable_shares # Short the same amount we could buy
                
                if shares_to_short > 0:
                    # Calculate the proceeds from shorting
                    proceeds = shares_to_short * self.current_price
                    transaction_fee = proceeds * (self.transaction_fee_percent / 100)
                    total_proceeds = proceeds - transaction_fee
                    
                    # Update balance and shares (negative shares for short)
                    self.balance += total_proceeds
                    self.shares_held = -shares_to_short
                    self.current_position = -1
                    
                    logger.debug(f"Flat -> Short: Sold short {shares_to_short} shares at {self.current_price} each, "
                               f"total proceeds: {total_proceeds} (after fee: {transaction_fee})")
            
            elif prev_position == 1:  # Long -> Short (Sell all shares and then short)
                # First, sell all long shares
                if self.shares_held > 0:
                    # Calculate the revenue including fees
                    revenue = self.shares_held * self.current_price
                    sell_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - sell_fee
                    
                    # Update balance after selling
                    self.balance += total_revenue
                    
                    logger.debug(f"Long -> Short (Step 1): Sold {self.shares_held} shares at {self.current_price} each, "
                               f"total revenue: {total_revenue} (after fee: {sell_fee})")
                    
                    self.shares_held = 0
                    
                    # Then, short shares
                    # Calculate shares based on what could be bought (mirroring long logic for simplicity)
                    buyable_shares = int(self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100)))
                    shares_to_short = buyable_shares # Short the same amount we could buy
                    
                    if shares_to_short > 0:
                        # Calculate the proceeds from shorting
                        proceeds = shares_to_short * self.current_price
                        short_fee = proceeds * (self.transaction_fee_percent / 100)
                        total_proceeds = proceeds - short_fee
                        
                        # Update balance and shares
                        self.balance += total_proceeds
                        self.shares_held = -shares_to_short
                        self.current_position = -1
                        
                        logger.debug(f"Long -> Short (Step 2): Sold short {shares_to_short} shares at {self.current_price} each, "
                                   f"total proceeds: {total_proceeds} (after fee: {short_fee})")
                    else:
                        # If we can't short any shares after selling, we're just flat
                        self.current_position = 0
                        logger.debug("Long -> Short: Sold long position but couldn't establish short position")
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the percentage change in portfolio value from the last step.

        Returns:
            float: The percentage change in portfolio value for the step.
                   Returns 0 if the last portfolio value was 0 to avoid division by zero.
        """
        if self.last_portfolio_value == 0:
            logger.warning("Last portfolio value was 0, returning 0 reward to avoid division by zero.")
            return 0.0  # Avoid division by zero

        # Calculate reward as the percentage change in portfolio value
        reward = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state representation) using a sliding window approach.
        
        Returns:
            np.ndarray: The current state observation including windowed price data, technical indicators,
                       and account information.
        """
        # Calculate the start index for the window
        window_start = max(0, self.current_step - self.window_size + 1)
        
        # Initialize an empty list to store the windowed market data
        windowed_data = []
        
        # If we're at the beginning and don't have enough history, we need to pad
        padding_needed = max(0, self.window_size - (self.current_step + 1))
        
        # Add padding if needed (repeat the earliest available data)
        if padding_needed > 0:
            # Get the earliest available data
            earliest_data = self.df.iloc[0].values
            earliest_data = np.nan_to_num(earliest_data, nan=0.0)
            
            # Normalize the earliest data
            max_abs_val = np.max(np.abs(earliest_data)) + 1e-10
            earliest_data_normalized = earliest_data / max_abs_val
            
            # Add padding by repeating the earliest data
            for _ in range(padding_needed):
                windowed_data.append(earliest_data_normalized)
        
        # Add actual historical data
        for i in range(window_start, self.current_step + 1):
            # Get market data for this step
            market_data = self.df.iloc[i].values
            
            # Handle NaN values
            market_data = np.nan_to_num(market_data, nan=0.0)
            
            # Normalize the market data
            max_abs_val = np.max(np.abs(market_data)) + 1e-10
            market_data_normalized = market_data / max_abs_val
            
            # Add to windowed data
            windowed_data.append(market_data_normalized)
        
        # Flatten the windowed data
        flattened_market_data = np.concatenate(windowed_data)
        
        # Add account information (balance and shares held)
        # Normalize these values as well
        normalized_balance = self.balance / self.initial_balance
        # Use current_position instead of just shares_held for position representation
        normalized_position = self.shares_held * self.current_price / self.initial_balance
        
        # Combine windowed market data and account information
        observation = np.append(flattened_market_data, [normalized_balance, normalized_position])
        
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

    def get_completed_trades(self) -> List[Dict[str, Any]]:
        """
        Returns the list of completed trades for the current episode.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a completed trade.
        """
        # Return a copy to prevent external modification
        return list(self._completed_trades)

    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        # Placeholder implementation
        # Will be expanded if needed in future tasks
        logger.info("Environment closed")
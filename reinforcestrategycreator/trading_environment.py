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
from reinforcestrategycreator.db_models import OperationType # Added
import ray # Added for RLlib integration

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
    
    # Class-level attribute for system-wide graceful shutdown signaling
    _system_wide_graceful_shutdown_active = False
        
    def __init__(self, env_config: dict):
        """
        Initialize the trading environment.

        Args:
            env_config (dict): Configuration dictionary for the environment. Expected keys:
                df (ray.ObjectRef): Ray object reference to the historical market data DataFrame.
                initial_balance (float): Initial account balance.
                transaction_fee_percent (float): Fee percentage for each transaction.
                window_size (int): Number of time steps for observation.
                sharpe_window_size (int): Window for Sharpe ratio calculation.
                use_sharpe_ratio (bool): Whether to use Sharpe ratio for reward.
                trading_frequency_penalty (float): Penalty for frequent trading.
                drawdown_penalty (float): Penalty for drawdowns.
                risk_free_rate (float): Risk-free rate for Sharpe ratio.
                stop_loss_pct (Optional[float]): Stop-loss percentage.
                take_profit_pct (Optional[float]): Take-profit percentage.
                position_sizing_method (str): 'fixed_fractional' or 'all_in'.
                risk_fraction (float): Fraction of balance to risk.
                normalization_window_size (int): Window for rolling normalization.
        """
        super(TradingEnv, self).__init__()

        # Extract parameters from env_config
        # Retrieve the DataFrame from the Ray object store
        data_ref = env_config["df"]
        self.df = ray.get(data_ref)
        logger.info(f"TradingEnv instance retrieved DataFrame from Ray object store.")

        self.initial_balance = env_config.get("initial_balance", 10000.0)
        self.transaction_fee_percent = env_config.get("transaction_fee_percent", 0.1)
        self.window_size = env_config.get("window_size", 5)
        self.sharpe_window_size = env_config.get("sharpe_window_size", 20)
        self.use_sharpe_ratio = env_config.get("use_sharpe_ratio", True)
        self.trading_frequency_penalty = env_config.get("trading_frequency_penalty", 0.01)
        self.drawdown_penalty = env_config.get("drawdown_penalty", 0.1)
        self.risk_free_rate = env_config.get("risk_free_rate", 0.0)
        self.stop_loss_pct = env_config.get("stop_loss_pct", None)
        self.take_profit_pct = env_config.get("take_profit_pct", None)
        self.position_sizing_method = env_config.get("position_sizing_method", "fixed_fractional")
        self.risk_fraction = env_config.get("risk_fraction", 0.1)
        self.normalization_window_size = env_config.get("normalization_window_size", 20)



        # Placeholder for current step in the environment
        self.current_step = 0
        
        # Portfolio state
        self.balance = self.initial_balance # Corrected: use self.initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.portfolio_value = self.initial_balance # Corrected: use self.initial_balance
        self.last_portfolio_value = self.initial_balance # Corrected: use self.initial_balance
        self.max_portfolio_value = self.initial_balance  # Corrected: use self.initial_balance
        
        # List to store details of completed trades
        self._completed_trades = [] # Fix: Initialize the missing attribute
        self._entry_price = 0.0     # Price at which the current position was entered
        self._entry_step = 0        # Step at which the current position was entered
        self._trade_count = 0       # Count of trades in the current episode
        
        # Portfolio value history for Sharpe ratio calculation
        self._portfolio_value_history = deque(maxlen=self.sharpe_window_size) # Corrected
        self._portfolio_returns = deque(maxlen=self.sharpe_window_size)  # Corrected: Store returns for Sharpe ratio
        self._episode_portfolio_returns = [] # For calculating episode-level Sharpe ratio
        self.episode_max_drawdown = 0.0 # Initialize episode max drawdown
        self._episode_total_reward = 0.0 # Accumulator for episode total reward
        self._episode_steps = 0 # Accumulator for episode steps
        
        # Current position state: 0 = Flat, 1 = Long, -1 = Short
        self.current_position = 0
        
        # Define action and observation spaces
        # Action space: 0 = Flat, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market data features + technical indicators + account information
        # Features include price data, indicators, balance, and position
        # For sliding window, we include window_size steps of market data plus current account information
        num_market_features = len(self.df.columns) # Corrected
        num_portfolio_features = 2  # balance and position
        total_features = (self.window_size * num_market_features) + num_portfolio_features # Corrected
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(self.df)} data points (including indicators) and initial balance {self.initial_balance}") # Corrected
 
        self.graceful_shutdown_signaled = False # Added for graceful shutdown
        self.cached_final_info_for_callback = None # Cache for the last info dict of a completed episode
    
    def signal_graceful_shutdown(self):
        """Sets a flag to indicate that the environment should try to terminate the episode."""
        logger.info(f"Graceful shutdown signaled for environment instance at step {self.current_step}. Activating system-wide flag.")
        self.graceful_shutdown_signaled = True
        TradingEnv._system_wide_graceful_shutdown_active = True


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
        self.max_portfolio_value = self.initial_balance
        self.current_position = 0  # Reset to Flat position
        self._entry_price = 0.0
        self._entry_step = 0
        self._trade_count = 0
        
        # Clear trade and portfolio history
        self._completed_trades.clear()
        self._portfolio_value_history.clear()
        self._portfolio_returns.clear()
        self._episode_portfolio_returns.clear() # Reset for new episode
        self.episode_max_drawdown = 0.0 # Reset for new episode
        self._episode_total_reward = 0.0 # Reset episode total reward
        self._episode_steps = 0 # Reset episode steps
        self.graceful_shutdown_signaled = False # Reset instance flag on environment reset
        self.cached_final_info_for_callback = None # Clear cached info on reset
 
        # Check if system-wide shutdown is active and re-apply signal if needed
        if TradingEnv._system_wide_graceful_shutdown_active:
            self.graceful_shutdown_signaled = True
            logger.info(f"Environment reset: System-wide graceful shutdown is active. Re-signaling this new episode instance (current_step: {self.current_step}).")
        
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

        # Determine if the episode is done
        natural_end_of_data = self.current_step >= len(self.df) - 1

        # Episode is terminated if it's a natural end
        terminated = natural_end_of_data
        truncated = False
        
        # Get the current price from the dataframe
        # Ensure current_step is within bounds if it's the very last step
        actual_current_step_for_price = min(self.current_step, len(self.df) - 1)
        self.current_price = self.df.iloc[actual_current_step_for_price]['close']
        
        # --- Risk Management Checks (SL/TP) ---
        original_action = action
        sl_triggered = False
        tp_triggered = False

        if self.current_position != 0 and self._entry_price > 0: # Check if in a position and entry price is valid
            if self.current_position == 1: # Long position checks
                # Stop-Loss Check
                if self.stop_loss_pct is not None:
                    sl_price = self._entry_price * (1 - self.stop_loss_pct / 100)
                    if self.current_price <= sl_price:
                        action = 0 # Force Flat action
                        sl_triggered = True
                        logger.info(f"Step {self.current_step}: Stop-Loss triggered for LONG position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, SL: {sl_price:.2f})")

                # Take-Profit Check (only if SL wasn't triggered)
                if not sl_triggered and self.take_profit_pct is not None:
                    tp_price = self._entry_price * (1 + self.take_profit_pct / 100)
                    if self.current_price >= tp_price:
                        action = 0 # Force Flat action
                        tp_triggered = True
                        logger.info(f"Step {self.current_step}: Take-Profit triggered for LONG position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, TP: {tp_price:.2f})")

            elif self.current_position == -1: # Short position checks
                # Stop-Loss Check
                if self.stop_loss_pct is not None:
                    sl_price = self._entry_price * (1 + self.stop_loss_pct / 100)
                    if self.current_price >= sl_price:
                        action = 0 # Force Flat action
                        sl_triggered = True
                        logger.info(f"Step {self.current_step}: Stop-Loss triggered for SHORT position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, SL: {sl_price:.2f})")

                # Take-Profit Check (only if SL wasn't triggered)
                if not sl_triggered and self.take_profit_pct is not None:
                    tp_price = self._entry_price * (1 - self.take_profit_pct / 100)
                    if self.current_price <= tp_price:
                        action = 0 # Force Flat action
                        tp_triggered = True
                        logger.info(f"Step {self.current_step}: Take-Profit triggered for SHORT position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, TP: {tp_price:.2f})")

        # --- Execute Trade Action (potentially overridden by SL/TP) and get operation details ---
        # _execute_trade_action will now return details about the operation performed
        operation_details_for_log = self._execute_trade_action(action)
        # operation_details_for_log should be a dict like:
        # {'operation_type_for_log': OperationType.ENTRY_LONG, 'shares_transacted_this_step': X, 'execution_price_this_step': Y}
        # or {'operation_type_for_log': OperationType.HOLD, 'shares_transacted_this_step': 0, 'execution_price_this_step': self.current_price}

        # Calculate the current portfolio value
        self.portfolio_value = self.balance + self.shares_held * self.current_price

        # Add current portfolio value to history for Sharpe ratio calculation
        self._portfolio_value_history.append(self.portfolio_value)

        # Calculate reward
        step_reward = self._calculate_reward() # Renamed to step_reward for clarity
        reward = step_reward # The reward returned by step() is this step_reward

        # Accumulate episode total reward and increment steps
        self._episode_total_reward += reward
        self._episode_steps += 1
 
        # Get the new observation
        observation = self._get_observation()
 
        # Ensure numeric values that might be NaN are handled before putting into info
        safe_current_price = np.nan_to_num(self.current_price, nan=0.0)
        safe_portfolio_value = np.nan_to_num(self.portfolio_value, nan=0.0)
        safe_step_reward = np.nan_to_num(reward, nan=0.0) # 'reward' is the final step reward variable here

        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': safe_current_price,
            'portfolio_value': safe_portfolio_value, # This is the current (potentially final) portfolio value
            'current_position': self.current_position,
            'step': self.current_step,
            'action_taken': action, # Actual action taken (could be overridden)
            'original_action': original_action, # Agent's intended action
            'sl_triggered': sl_triggered,
            'tp_triggered': tp_triggered,
            'step_reward': safe_step_reward, # Use the safe version of the step's reward
            'operation_type_for_log': operation_details_for_log.get('operation_type_for_log', OperationType.HOLD),
            'shares_transacted_this_step': operation_details_for_log.get('shares_transacted_this_step', 0),
            'execution_price_this_step': operation_details_for_log.get('execution_price_this_step', safe_current_price) # Use safe_current_price here too
        }
 
        # `terminated` and `truncated` flags are already set based on natural_end_of_data or force_terminate_due_to_shutdown
        
 
        if terminated: # This now includes the check above
            # Add all required metrics to info dict upon episode termination
            info['initial_portfolio_value'] = self.initial_balance # Use initial_balance from reset
            info['final_portfolio_value'] = self.portfolio_value # Current portfolio_value is the final one
            info['pnl'] = self.portfolio_value - self.initial_balance
            info['completed_trades'] = list(self._completed_trades) # Make a copy
            
            if self._completed_trades:
                winning_trades = sum(1 for trade in self._completed_trades if trade.get('pnl', 0) > 0)
                info['win_rate'] = winning_trades / len(self._completed_trades) if len(self._completed_trades) > 0 else 0.0
            else:
                info['win_rate'] = 0.0
            
            info['max_drawdown'] = self.episode_max_drawdown
            
            # Calculate episode Sharpe Ratio
            if len(self._episode_portfolio_returns) >= 2:
                episode_returns_array = np.array(self._episode_portfolio_returns)
                episode_returns_mean = np.mean(episode_returns_array)
                episode_returns_std = np.std(episode_returns_array)
                if episode_returns_std > 1e-8: # Avoid division by zero or very small std
                    # Non-annualized Sharpe based on episode returns
                    info['sharpe_ratio'] = (episode_returns_mean - self.risk_free_rate) / episode_returns_std
                else:
                    info['sharpe_ratio'] = 0.0
            else:
                info['sharpe_ratio'] = 0.0
            
            info['total_reward'] = self._episode_total_reward # Accumulated total reward for the episode
            info['total_steps'] = self._episode_steps # Accumulated total steps for the episode
            
            logger.info(
                f"Episode ended (Terminated: {terminated}, Truncated: {truncated}, NaturalEnd: {natural_end_of_data}, GracefulShutdown: {self.graceful_shutdown_signaled}). "
                f"Final info: initial_pf={info.get('initial_portfolio_value')}, "
                f"final_pf={info.get('final_portfolio_value')}, pnl={info.get('pnl')}, "
                f"sharpe={info.get('sharpe_ratio')}, mdd={info.get('max_drawdown')}, "
                f"total_reward={info.get('total_reward')}, total_steps={info.get('total_steps')}, "
                f"win_rate={info.get('win_rate')}, trades_count={len(info.get('completed_trades', []))}"
            )
            # Cache the final info dictionary for the callback
            self.cached_final_info_for_callback = info.copy()
 
        logger.debug(f"Step {self.current_step}: action={action}, reward={reward:.4f}, terminated={terminated}, truncated={truncated}, SL={sl_triggered}, TP={tp_triggered}")
        return observation, reward, terminated, truncated, info
    
    def _execute_trade_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the specified trading action.
        
        Args:
            action (int): The action to take (0 = Flat, 1 = Long, 2 = Short).
        
        Returns:
            Dict[str, Any]: Details of the operation performed for logging.
                            Keys: 'operation_type_for_log', 'shares_transacted_this_step', 'execution_price_this_step'.
        """
        # Track if a trade occurs in this step
        trade_occurred = False
        # Get the current position before executing the action
        prev_position = self.current_position
        
        # Initialize operation details for logging
        operation_log_details = {'operation_type_for_log': OperationType.HOLD, 'shares_transacted_this_step': 0, 'execution_price_this_step': self.current_price}
        
        if action == 0:  # Flat
            if prev_position == 1:  # Long -> Flat (Sell all shares)
                if self.shares_held > 0:
                    # Calculate the revenue including fees
                    revenue = self.shares_held * self.current_price
                    transaction_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - transaction_fee
                    
                    # Calculate Profit/Loss for the closed long position
                    shares_sold = self.shares_held
                    pnl = (self.current_price - self._entry_price) * shares_sold - transaction_fee
                    
                    # Record completed trade
                    entry_time = self.df.index[self._entry_step]
                    exit_time = self.df.index[self.current_step]
                    trade_details = {
                        'entry_step': self._entry_step,
                        'exit_step': self.current_step,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': self._entry_price,
                        'exit_price': self.current_price,
                        'quantity': shares_sold,         # Use 'quantity'
                        'direction': 'long',             # Use 'direction'
                        'pnl': pnl,                      # Use 'pnl'
                        'costs': transaction_fee         # Use 'costs'
                    }
                    self._completed_trades.append(trade_details)
                    
                    # Update balance and shares
                    self.balance += total_revenue
                    
                    logger.debug(f"Long -> Flat: Sold {shares_sold} shares at {self.current_price} each, "
                               f"total revenue: {total_revenue} (after fee: {transaction_fee}), PnL: {pnl}")
                    
                    self.shares_held = 0
                    self.current_position = 0
                    self._entry_price = 0.0 # Reset entry price
                    self._entry_step = 0   # Reset entry step
                    trade_occurred = True
                    self._trade_count += 1
                    operation_log_details = {
                        'operation_type_for_log': OperationType.EXIT_LONG,
                        'shares_transacted_this_step': shares_sold, # Positive for sell to close long
                        'execution_price_this_step': self.current_price
                    }
            
            elif prev_position == -1:  # Short -> Flat (Cover short position)
                if self.shares_held < 0:
                    # Calculate the cost to buy back shares including fees
                    cost = abs(self.shares_held) * self.current_price
                    transaction_fee = cost * (self.transaction_fee_percent / 100)
                    total_cost = cost + transaction_fee
                    
                    # Check if we have enough balance to cover
                    if self.balance >= total_cost:
                        # Calculate Profit/Loss for the closed short position
                        shares_covered = abs(self.shares_held)
                        pnl = (self._entry_price - self.current_price) * shares_covered - transaction_fee
                        
                        # Record completed trade
                        entry_time = self.df.index[self._entry_step]
                        exit_time = self.df.index[self.current_step]
                        trade_details = {
                            'entry_step': self._entry_step,
                            'exit_step': self.current_step,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': self._entry_price,
                            'exit_price': self.current_price,
                            'quantity': shares_covered,      # Use 'quantity'
                            'direction': 'short',            # Use 'direction'
                            'pnl': pnl,                      # Use 'pnl'
                            'costs': transaction_fee         # Use 'costs'
                        }
                        self._completed_trades.append(trade_details)
                        
                        # Update balance and shares
                        self.balance -= total_cost
                        
                        logger.debug(f"Short -> Flat: Covered {shares_covered} shares at {self.current_price} each, "
                                   f"total cost: {total_cost} (including fee: {transaction_fee}), PnL: {pnl}")
                        
                        self.shares_held = 0
                        self.current_position = 0
                        self._entry_price = 0.0 # Reset entry price
                        self._entry_step = 0   # Reset entry step
                        trade_occurred = True
                        self._trade_count += 1
                        operation_log_details = {
                            'operation_type_for_log': OperationType.EXIT_SHORT,
                            'shares_transacted_this_step': shares_covered, # Positive for buy to close short
                            'execution_price_this_step': self.current_price
                        }
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cost}, Available: {self.balance}")
        
        elif action == 1:  # Long
            if prev_position == 0:  # Flat -> Long (Buy shares)
                # --- Position Sizing ---
                if self.position_sizing_method == "fixed_fractional":
                    target_position_value = self.balance * self.risk_fraction
                    shares_to_buy = int(target_position_value / self.current_price)
                    logger.debug(f"Position Sizing (Fixed Fractional): Target Value={target_position_value:.2f}, Shares={shares_to_buy}")
                else: # Default to 'all_in' or handle other methods
                    # Calculate maximum shares that can be bought (original 'all_in' logic)
                    max_shares_possible = self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100))
                    shares_to_buy = int(max_shares_possible)
                    logger.debug(f"Position Sizing (All In): Max Shares={shares_to_buy}")

                # Ensure we don't try to buy more than we can afford after fees
                cost_per_share_with_fee = self.current_price * (1 + self.transaction_fee_percent / 100)
                affordable_shares = int(self.balance / cost_per_share_with_fee)
                shares_to_buy = min(shares_to_buy, affordable_shares) # Take the minimum

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
                    self._entry_price = self.current_price # Record entry price
                    self._entry_step = self.current_step   # Record entry step
                    trade_occurred = True
                    self._trade_count += 1
                    operation_log_details = {
                        'operation_type_for_log': OperationType.ENTRY_LONG,
                        'shares_transacted_this_step': shares_to_buy,
                        'execution_price_this_step': self.current_price
                    }
 
                    logger.debug(f"Flat -> Long: Bought {shares_to_buy} shares at {self.current_price:.2f} each, "
                                f"total cost: {total_cost:.2f} (including fee: {transaction_fee:.2f})")
                else:
                    logger.debug("Flat -> Long: Cannot buy shares (insufficient funds or zero shares calculated).")

            elif prev_position == -1:  # Short -> Long (Cover short and then buy)
                # First, cover the short position
                shares_covered_in_reversal = 0
                if self.shares_held < 0:
                    # Calculate the cost to buy back shares including fees
                    cover_cost = abs(self.shares_held) * self.current_price
                    cover_fee = cover_cost * (self.transaction_fee_percent / 100)
                    total_cover_cost = cover_cost + cover_fee
                    
                    # Check if we have enough balance to cover
                    if self.balance >= total_cover_cost:
                        # Calculate Profit/Loss for the closed short position
                        shares_covered_in_reversal = abs(self.shares_held)
                        pnl = (self._entry_price - self.current_price) * shares_covered_in_reversal - cover_fee
                        
                        # Record completed trade
                        entry_time = self.df.index[self._entry_step]
                        exit_time = self.df.index[self.current_step]
                        trade_details = {
                            'entry_step': self._entry_step,
                            'exit_step': self.current_step,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': self._entry_price,
                            'exit_price': self.current_price,
                            'quantity': shares_covered_in_reversal,      # Use 'quantity'
                            'direction': 'short',            # Use 'direction'
                            'pnl': pnl,                      # Use 'pnl'
                            'costs': cover_fee               # Use 'costs'
                        }
                        self._completed_trades.append(trade_details)
                        
                        # Update balance after covering
                        self.balance -= total_cover_cost
                        
                        logger.debug(f"Short -> Long (Step 1): Covered {shares_covered_in_reversal} shares at {self.current_price} each, "
                                   f"total cost: {total_cover_cost} (including fee: {cover_fee}), PnL: {pnl}")
                        
                        self.shares_held = 0
                        self._entry_price = 0.0 # Reset entry price/step after closing short
                        self._entry_step = 0
                        
                        # Then, buy shares with remaining balance using position sizing
                        if self.position_sizing_method == "fixed_fractional":
                            target_position_value = self.balance * self.risk_fraction
                            shares_to_buy = int(target_position_value / self.current_price)
                            logger.debug(f"Position Sizing (Fixed Fractional): Target Value={target_position_value:.2f}, Shares={shares_to_buy}")
                        else: # Default to 'all_in'
                            max_shares_possible = self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100))
                            shares_to_buy = int(max_shares_possible)
                            logger.debug(f"Position Sizing (All In): Max Shares={shares_to_buy}")

                        # Ensure affordability
                        cost_per_share_with_fee = self.current_price * (1 + self.transaction_fee_percent / 100)
                        affordable_shares = int(self.balance / cost_per_share_with_fee)
                        shares_to_buy = min(shares_to_buy, affordable_shares)

                        if shares_to_buy > 0:
                            # Calculate the cost including fees
                            buy_cost = shares_to_buy * self.current_price
                            buy_fee = buy_cost * (self.transaction_fee_percent / 100)
                            total_buy_cost = buy_cost + buy_fee

                            # Update balance and shares
                            self.balance -= total_buy_cost
                            self.shares_held += shares_to_buy
                            self.current_position = 1
                            self._entry_price = self.current_price # Record entry price for new long position
                            self._entry_step = self.current_step   # Record entry step for new long position
                            trade_occurred = True
                            self._trade_count += 1
                            # Log the entry part of the reversal
                            operation_log_details = {
                                'operation_type_for_log': OperationType.ENTRY_LONG,
                                'shares_transacted_this_step': shares_to_buy,
                                'execution_price_this_step': self.current_price
                            }
 
                            logger.debug(f"Short -> Long (Step 2): Bought {shares_to_buy} shares at {self.current_price:.2f} each, "
                                       f"total cost: {total_buy_cost:.2f} (including fee: {buy_fee:.2f})")
                        else:
                            # If we can't buy any shares after covering, we're just flat
                            self.current_position = 0
                            logger.debug("Short -> Long: Covered short position but insufficient funds/shares to go long.")
                            # Log the exit part of the reversal if entry didn't happen
                            operation_log_details = {
                                'operation_type_for_log': OperationType.EXIT_SHORT,
                                'shares_transacted_this_step': shares_covered_in_reversal,
                                'execution_price_this_step': self.current_price
                            }
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cover_cost:.2f}, Available: {self.balance:.2f}")
        
        elif action == 2:  # Short
            if prev_position == 0:  # Flat -> Short (Sell short)
                # --- Position Sizing ---
                if self.position_sizing_method == "fixed_fractional":
                    # For shorting, the 'value' is based on the potential proceeds
                    target_position_value = self.balance * self.risk_fraction # Use balance as proxy for capital base
                    shares_to_short = int(target_position_value / self.current_price)
                    logger.debug(f"Position Sizing (Fixed Fractional): Target Value={target_position_value:.2f}, Shares={shares_to_short}")
                else: # Default to 'all_in'
                    # Mirroring long logic for simplicity: short the amount we *could* buy
                    buyable_shares = int(self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100)))
                    shares_to_short = buyable_shares
                    logger.debug(f"Position Sizing (All In): Max Shares={shares_to_short}")

                # Basic check: Ensure we have some balance to act as margin (simplistic)
                if shares_to_short > 0 and self.balance > 0:
                    # Calculate the proceeds from shorting
                    proceeds = shares_to_short * self.current_price
                    transaction_fee = proceeds * (self.transaction_fee_percent / 100)
                    total_proceeds = proceeds - transaction_fee

                    # Update balance and shares (negative shares for short)
                    self.balance += total_proceeds # Add proceeds (margin increases)
                    self.shares_held = -shares_to_short
                    self.current_position = -1
                    self._entry_price = self.current_price # Record entry price
                    self._entry_step = self.current_step   # Record entry step
                    trade_occurred = True
                    self._trade_count += 1
                    operation_log_details = {
                        'operation_type_for_log': OperationType.ENTRY_SHORT,
                        'shares_transacted_this_step': -shares_to_short, # Negative for short entry
                        'execution_price_this_step': self.current_price
                    }
 
                    logger.debug(f"Flat -> Short: Sold short {shares_to_short} shares at {self.current_price:.2f} each, "
                                f"total proceeds: {total_proceeds:.2f} (after fee: {transaction_fee:.2f})")
                else:
                     logger.debug("Flat -> Short: Cannot short shares (zero shares calculated or zero balance).")

            elif prev_position == 1:  # Long -> Short (Sell all shares and then short)
                # First, sell all long shares
                shares_sold_in_reversal = 0
                if self.shares_held > 0:
                    # Calculate the revenue including fees
                    revenue = self.shares_held * self.current_price
                    sell_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - sell_fee
                    
                    # Calculate Profit/Loss for the closed long position
                    shares_sold_in_reversal = self.shares_held
                    pnl = (self.current_price - self._entry_price) * shares_sold_in_reversal - sell_fee
                    
                    # Record completed trade
                    entry_time = self.df.index[self._entry_step]
                    exit_time = self.df.index[self.current_step]
                    trade_details = {
                        'entry_step': self._entry_step,
                        'exit_step': self.current_step,
                        'entry_time': entry_time,         # Add entry timestamp
                        'exit_time': exit_time,           # Add exit timestamp
                        'entry_price': self._entry_price,
                        'exit_price': self.current_price,
                        'quantity': shares_sold_in_reversal,         # Use 'quantity'
                        'direction': 'long',             # Use 'direction'
                        'pnl': pnl,                      # Use 'pnl'
                        'costs': sell_fee                # Use 'costs'
                    }
                    self._completed_trades.append(trade_details)
                    
                    # Update balance after selling
                    self.balance += total_revenue
                    
                    logger.debug(f"Long -> Short (Step 1): Sold {shares_sold_in_reversal} shares at {self.current_price} each, "
                                f"total revenue: {total_revenue} (after fee: {sell_fee}), PnL: {pnl}")
                    
                    self.shares_held = 0
                    self._entry_price = 0.0 # Reset entry price/step after closing long
                    self._entry_step = 0
                    
                    # Then, short shares using position sizing
                    if self.position_sizing_method == "fixed_fractional":
                        target_position_value = self.balance * self.risk_fraction
                        shares_to_short = int(target_position_value / self.current_price)
                        logger.debug(f"Position Sizing (Fixed Fractional): Target Value={target_position_value:.2f}, Shares={shares_to_short}")
                    else: # Default to 'all_in'
                        buyable_shares = int(self.balance / (self.current_price * (1 + self.transaction_fee_percent / 100)))
                        shares_to_short = buyable_shares
                        logger.debug(f"Position Sizing (All In): Max Shares={shares_to_short}")

                    if shares_to_short > 0 and self.balance > 0:
                        # Calculate the proceeds from shorting
                        proceeds = shares_to_short * self.current_price
                        short_fee = proceeds * (self.transaction_fee_percent / 100)
                        total_proceeds = proceeds - short_fee

                        # Update balance and shares
                        self.balance += total_proceeds
                        self.shares_held = -shares_to_short
                        self.current_position = -1
                        self._entry_price = self.current_price # Record entry price for new short position
                        self._entry_step = self.current_step   # Record entry step for new short position
                        trade_occurred = True # Mark trade occurred for short entry
                        self._trade_count += 1 # Increment trade count for short entry
                        # Log the entry part of the reversal
                        operation_log_details = {
                            'operation_type_for_log': OperationType.ENTRY_SHORT,
                            'shares_transacted_this_step': -shares_to_short, # Negative for short entry
                            'execution_price_this_step': self.current_price
                        }
 
                        logger.debug(f"Long -> Short (Step 2): Sold short {shares_to_short} shares at {self.current_price:.2f} each, "
                                    f"total proceeds: {total_proceeds:.2f} (after fee: {short_fee:.2f})")
                    else:
                        # If we can't short any shares after selling, we're just flat
                        self.current_position = 0
                        logger.debug("Long -> Short: Sold long position but couldn't establish short position (zero shares or balance).")
                        # Log the exit part of the reversal if entry didn't happen
                        operation_log_details = {
                            'operation_type_for_log': OperationType.EXIT_LONG,
                            'shares_transacted_this_step': shares_sold_in_reversal,
                            'execution_price_this_step': self.current_price
                        }
        
        return operation_log_details

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on risk-adjusted returns, trading frequency, and drawdowns.
        
        The reward consists of three components:
        1. Risk-adjusted return (Sharpe ratio or simple percentage change)
        2. Trading frequency penalty (to discourage excessive trading)
        3. Drawdown penalty (to encourage capital preservation)
        
        Returns:
            float: The calculated reward value.
        """
        # Update max portfolio value for drawdown calculation
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Calculate percentage change in portfolio value
        if self.last_portfolio_value == 0:
            logger.warning("Last portfolio value was 0, returning 0 reward to avoid division by zero.")
            return 0.0  # Avoid division by zero
        
        percentage_change = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        
        # Add the return to history for Sharpe ratio calculation
        self._portfolio_returns.append(percentage_change)
        self._episode_portfolio_returns.append(percentage_change) # Also add to episode-level returns
        
        # Component 1: Risk-adjusted return
        if self.use_sharpe_ratio and len(self._portfolio_returns) >= 2:
            # Calculate Sharpe ratio using the portfolio returns history
            returns_array = np.array(self._portfolio_returns)
            returns_mean = np.mean(returns_array)
            returns_std = np.std(returns_array)
            
            # Avoid division by zero
            if returns_std == 0:
                risk_adjusted_return = returns_mean  # If no volatility, just use the mean return
            else:
                # Sharpe ratio = (Mean Return - Risk Free Rate) / Standard Deviation
                risk_adjusted_return = (returns_mean - self.risk_free_rate) / returns_std
                
                # Scale the Sharpe ratio to be comparable to percentage returns
                # Typical Sharpe values might be between -3 and +3, while returns are often smaller
                # REMOVED: risk_adjusted_return *= 0.01  # Scaling factor removed as it drastically reduced reward signal
        else:
            # If not using Sharpe ratio or not enough history, use percentage change
            risk_adjusted_return = percentage_change
        
        # Component 2: Trading frequency penalty
        # Penalize based on the number of trades in this episode
        trading_penalty = self.trading_frequency_penalty * self._trade_count
        
        # Component 3: Drawdown penalty
        # Calculate current drawdown as percentage from peak
        if self.max_portfolio_value > 0:
            current_drawdown = max(0, (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value)
            self.episode_max_drawdown = max(self.episode_max_drawdown, current_drawdown) # Track max drawdown for episode
            drawdown_penalty = self.drawdown_penalty * current_drawdown
        else:
            drawdown_penalty = 0
        
        # Combine all components into final reward
        reward = risk_adjusted_return - trading_penalty - drawdown_penalty
        
        logger.debug(f"Reward components: risk_adjusted={risk_adjusted_return:.6f}, "
                   f"trading_penalty={trading_penalty:.6f}, drawdown_penalty={drawdown_penalty:.6f}, "
                   f"final_reward={reward:.6f}")
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state representation) using a sliding window approach
        with rolling z-score normalization for market features.

        Returns:
            np.ndarray: The current state observation including normalized windowed price data,
                       technical indicators, and normalized account information.
        """
        # Calculate the start index for the observation window
        window_start = max(0, self.current_step - self.window_size + 1)

        # Initialize an empty list to store the normalized windowed market data
        windowed_data = []

        # --- Calculate Rolling Stats for Normalization ---
        # Use data up to the current step for calculating normalization stats
        # Ensure we only select numeric columns if df might contain non-numeric ones
        numeric_df = self.df.select_dtypes(include=np.number)
        history_df = numeric_df.iloc[:self.current_step + 1]

        # Use expanding window until normalization_window_size is reached
        rolling_mean = history_df.rolling(window=self.normalization_window_size, min_periods=1).mean()
        rolling_std = history_df.rolling(window=self.normalization_window_size, min_periods=1).std()

        # Get mean and std for the current step (last row of rolling calculation)
        current_mean = rolling_mean.iloc[-1].fillna(0) # Fill potential NaNs in mean
        # Add epsilon to std to prevent division by zero, fill NaN std (e.g., first step)
        current_std = rolling_std.iloc[-1].fillna(1e-8) + 1e-8
        
        # Get mean and std for the earliest step (for padding normalization)
        earliest_mean = rolling_mean.iloc[0].fillna(0) # Fill potential NaNs in mean
        earliest_std = rolling_std.iloc[0].fillna(1e-8) + 1e-8
        # -------------------------------------------------

        # Determine padding needed for the observation window
        padding_needed = max(0, self.window_size - (self.current_step + 1))

        # Add padding if needed (repeat the earliest available data, normalized)
        if padding_needed > 0:
            # Get the earliest available raw data (numeric columns only)
            earliest_data = numeric_df.iloc[0].values
            earliest_data = np.nan_to_num(earliest_data, nan=0.0) # Handle NaNs

            # Normalize the earliest data using its corresponding rolling stats
            earliest_data_normalized = (earliest_data - earliest_mean.values) / earliest_std.values

            # Add padding by repeating the normalized earliest data
            for _ in range(padding_needed):
                windowed_data.append(earliest_data_normalized)

        # Add actual historical data (normalized) from the observation window
        for i in range(window_start, self.current_step + 1):
            # Get market data for this step (numeric columns only)
            market_data = numeric_df.iloc[i].values

            # Handle NaN values before normalization
            market_data = np.nan_to_num(market_data, nan=0.0)

            # Normalize the market data using the *current* step's rolling stats
            # (Applying current stats to past data in the window)
            market_data_normalized = (market_data - current_mean.values) / current_std.values

            # Add to windowed data
            windowed_data.append(market_data_normalized)

        # Flatten the windowed data
        # Ensure windowed_data is not empty before concatenating
        if not windowed_data:
             # Should not happen if window_size > 0, but handle defensively
             num_market_features = len(numeric_df.columns)
             flattened_market_data = np.zeros(self.window_size * num_market_features)
        else:
            flattened_market_data = np.concatenate(windowed_data)

        # Add account information (balance and position value relative to initial balance)
        # Normalize these values relative to the initial balance
        normalized_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        # Ensure current_price is updated before calling _get_observation (should be done in step)
        # Use nan_to_num for current_price robustness
        safe_current_price = np.nan_to_num(self.current_price, nan=0.0)
        normalized_position_value = self.shares_held * safe_current_price / self.initial_balance if self.initial_balance > 0 else 0
        
        # Combine normalized windowed market data and normalized account information
        observation = np.append(flattened_market_data, [normalized_balance, normalized_position_value])

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
def register_rllib_env():
    """Registers the TradingEnv with RLlib."""
    from ray.tune.registry import register_env
    register_env("TradingEnv-v0", lambda config: TradingEnv(config))
    logger.info("TradingEnv-v0 registered with RLlib.")
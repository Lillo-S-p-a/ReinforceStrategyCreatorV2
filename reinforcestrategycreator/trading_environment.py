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
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee_percent: float = 0.1,
                 window_size: int = 5, sharpe_window_size: int = 20, use_sharpe_ratio: bool = True,
                 trading_frequency_penalty: float = 0.01, drawdown_penalty: float = 0.1,
                 risk_free_rate: float = 0.0,
                 stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None,
                 position_sizing_method: str = "fixed_fractional", risk_fraction: float = 0.1,
                 normalization_window_size: int = 20): # Added normalization window size
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
            use_sharpe_ratio (bool): Whether to use Sharpe ratio for reward calculation (default: True).
            trading_frequency_penalty (float): Weight for the trading frequency penalty (default: 0.01).
            drawdown_penalty (float): Weight for the drawdown penalty (default: 0.1).
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation (default: 0.0).
            stop_loss_pct (Optional[float]): Percentage below entry price to trigger stop-loss for long positions
                                             (or above for short positions). None to disable. (default: None).
            take_profit_pct (Optional[float]): Percentage above entry price to trigger take-profit for long positions
                                               (or below for short positions). None to disable. (default: None).
            position_sizing_method (str): Method for calculating position size ('fixed_fractional' or 'all_in').
                                          (default: "fixed_fractional").
            risk_fraction (float): Fraction of balance to risk per trade when using 'fixed_fractional' sizing.
                               (default: 0.1, meaning 10%).
       normalization_window_size (int): Window size for rolling normalization (e.g., z-score). (default: 20).
        """
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.sharpe_window_size = sharpe_window_size
        self.use_sharpe_ratio = use_sharpe_ratio
        self.trading_frequency_penalty = trading_frequency_penalty
        self.drawdown_penalty = drawdown_penalty
        self.risk_free_rate = risk_free_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_sizing_method = position_sizing_method
        self.risk_fraction = risk_fraction
        self.normalization_window_size = normalization_window_size 

        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.current_price = 0.0 # Initialize as float
        self.portfolio_value = initial_balance
        self.last_portfolio_value = initial_balance
        self.max_portfolio_value = initial_balance
        
        self._completed_trades = [] 
        self._entry_price = 0.0     
        self._entry_step = 0        
        self._trade_count = 0       
        
        self._portfolio_value_history = deque(maxlen=sharpe_window_size)
        self._portfolio_returns = deque(maxlen=sharpe_window_size) 
        
        self.current_position = 0
        
        self.action_space = spaces.Discrete(3)
        
        num_market_features = len(df.columns)
        num_portfolio_features = 2 
        total_features = (window_size * num_market_features) + num_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        logger.info(f"TradingEnv initialized with {len(df)} data points and initial balance {initial_balance}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.current_position = 0
        self._entry_price = 0.0
        self._entry_step = 0
        self._trade_count = 0
        
        self._completed_trades.clear()
        self._portfolio_value_history.clear()
        self._portfolio_returns.clear()
        
        if len(self.df) == 0:
            error_msg = "DataFrame is empty. Cannot reset environment."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        valid_start_step = None
        for i in range(len(self.df)):
            if not self.df.iloc[i].isna().any():
                valid_start_step = i
                logger.info(f"Found step {i} with all indicators non-NaN")
                break
        
        if valid_start_step is None:
            for i in range(len(self.df)):
                if not pd.isna(self.df.iloc[i]['close']): # Assuming 'close' is the primary price column
                    valid_start_step = i
                    logger.info(f"Found step {i} with valid price data (some indicators may be NaN)")
                    break
        
        if valid_start_step is None:
            valid_start_step = 0
            logger.warning("No step with valid price data found. Using first step (index 0) as fallback.")
        
        if valid_start_step >= self.window_size - 1:
            self.current_step = valid_start_step
        else:
            self.current_step = valid_start_step
            logger.info(f"Starting at step {valid_start_step} with less than window_size history")
        
        # Ensure current_price is a scalar float
        current_price_series = self.df.iloc[self.current_step]['close']
        self.current_price = float(current_price_series.item() if isinstance(current_price_series, pd.Series) else current_price_series)

        observation = self._get_observation()
        
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
        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_price_series = self.df.iloc[self.current_step]['close']
        self.current_price = float(current_price_series.item() if isinstance(current_price_series, pd.Series) else current_price_series)

        original_action = action
        sl_triggered = False
        tp_triggered = False

        if self.current_position != 0 and self._entry_price > 0:
            if self.current_position == 1: 
                if self.stop_loss_pct is not None:
                    sl_price = self._entry_price * (1 - self.stop_loss_pct / 100)
                    if self.current_price <= sl_price:
                        action = 0 
                        sl_triggered = True
                        logger.info(f"Step {self.current_step}: Stop-Loss triggered for LONG position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, SL: {sl_price:.2f})")

                if not sl_triggered and self.take_profit_pct is not None:
                    tp_price = self._entry_price * (1 + self.take_profit_pct / 100)
                    if self.current_price >= tp_price:
                        action = 0 
                        tp_triggered = True
                        logger.info(f"Step {self.current_step}: Take-Profit triggered for LONG position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, TP: {tp_price:.2f})")

            elif self.current_position == -1: 
                if self.stop_loss_pct is not None:
                    sl_price = self._entry_price * (1 + self.stop_loss_pct / 100)
                    if self.current_price >= sl_price:
                        action = 0 
                        sl_triggered = True
                        logger.info(f"Step {self.current_step}: Stop-Loss triggered for SHORT position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, SL: {sl_price:.2f})")

                if not sl_triggered and self.take_profit_pct is not None:
                    tp_price = self._entry_price * (1 - self.take_profit_pct / 100)
                    if self.current_price <= tp_price:
                        action = 0 
                        tp_triggered = True
                        logger.info(f"Step {self.current_step}: Take-Profit triggered for SHORT position at price {self.current_price:.2f} (Entry: {self._entry_price:.2f}, TP: {tp_price:.2f})")

        self._execute_trade_action(action)
        self.portfolio_value = self.balance + self.shares_held * self.current_price
        self._portfolio_value_history.append(self.portfolio_value)
        reward = self._calculate_reward()
        observation = self._get_observation()

        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'current_position': self.current_position,
            'step': self.current_step,
            'action_taken': action,
            'original_action': original_action,
            'sl_triggered': sl_triggered,
            'tp_triggered': tp_triggered
        }
        terminated = done
        truncated = False
        logger.debug(f"Step {self.current_step}: action={action}, reward={reward:.4f}, done={done}, SL={sl_triggered}, TP={tp_triggered}")
        return observation, reward, terminated, truncated, info

    def _execute_trade_action(self, action: int) -> None:
        trade_occurred = False
        prev_position = self.current_position
        current_price_scalar = self.current_price # Already a float

        if action == 0:  # Flat
            if prev_position == 1:  # Long -> Flat
                if self.shares_held > 0:
                    revenue = self.shares_held * current_price_scalar
                    transaction_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - transaction_fee
                    pnl = (current_price_scalar - self._entry_price) * self.shares_held - transaction_fee
                    
                    entry_time = self.df.index[self._entry_step]
                    exit_time = self.df.index[self.current_step]
                    self._completed_trades.append({
                        'entry_step': self._entry_step, 'exit_step': self.current_step,
                        'entry_time': entry_time, 'exit_time': exit_time,
                        'entry_price': self._entry_price, 'exit_price': current_price_scalar,
                        'quantity': self.shares_held, 'direction': 'long', 'pnl': pnl, 'costs': transaction_fee
                    })
                    self.balance += total_revenue
                    logger.debug(f"Long -> Flat: Sold {self.shares_held} shares at {current_price_scalar:.2f} each, "
                               f"total revenue: {total_revenue:.2f} (after fee: {transaction_fee:.2f}), PnL: {pnl:.2f}")
                    self.shares_held = 0
                    self.current_position = 0
                    self._entry_price = 0.0
                    self._entry_step = 0
                    trade_occurred = True
                    self._trade_count += 1
            
            elif prev_position == -1:  # Short -> Flat
                if self.shares_held < 0:
                    cost = abs(self.shares_held) * current_price_scalar
                    transaction_fee = cost * (self.transaction_fee_percent / 100)
                    total_cost = cost + transaction_fee
                    if self.balance >= total_cost:
                        shares_covered = abs(self.shares_held)
                        pnl = (self._entry_price - current_price_scalar) * shares_covered - transaction_fee
                        entry_time = self.df.index[self._entry_step]
                        exit_time = self.df.index[self.current_step]
                        self._completed_trades.append({
                            'entry_step': self._entry_step, 'exit_step': self.current_step,
                            'entry_time': entry_time, 'exit_time': exit_time,
                            'entry_price': self._entry_price, 'exit_price': current_price_scalar,
                            'quantity': shares_covered, 'direction': 'short', 'pnl': pnl, 'costs': transaction_fee
                        })
                        self.balance -= total_cost
                        logger.debug(f"Short -> Flat: Covered {shares_covered} shares at {current_price_scalar:.2f} each, "
                                   f"total cost: {total_cost:.2f} (including fee: {transaction_fee:.2f}), PnL: {pnl:.2f}")
                        self.shares_held = 0
                        self.current_position = 0
                        self._entry_price = 0.0
                        self._entry_step = 0
                        trade_occurred = True
                        self._trade_count += 1
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cost:.2f}, Available: {self.balance:.2f}")
        
        elif action == 1:  # Long
            if prev_position == 0:  # Flat -> Long
                if self.position_sizing_method == "fixed_fractional":
                    target_position_value = self.balance * self.risk_fraction
                    shares_to_buy = int(target_position_value / current_price_scalar)
                else: 
                    max_shares_possible = self.balance / (current_price_scalar * (1 + self.transaction_fee_percent / 100))
                    shares_to_buy = int(max_shares_possible)
                
                cost_per_share_with_fee = current_price_scalar * (1 + self.transaction_fee_percent / 100)
                affordable_shares = int(self.balance / cost_per_share_with_fee)
                shares_to_buy = min(shares_to_buy, affordable_shares)

                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price_scalar
                    transaction_fee = cost * (self.transaction_fee_percent / 100)
                    total_cost = cost + transaction_fee
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    self.current_position = 1
                    self._entry_price = current_price_scalar
                    self._entry_step = self.current_step
                    trade_occurred = True
                    self._trade_count += 1
                    logger.debug(f"Flat -> Long: Bought {shares_to_buy} shares at {current_price_scalar:.2f} each, "
                               f"total cost: {total_cost:.2f} (including fee: {transaction_fee:.2f})")
                else:
                    logger.debug("Flat -> Long: Cannot buy shares (insufficient funds or zero shares calculated).")

            elif prev_position == -1:  # Short -> Long
                if self.shares_held < 0:
                    cover_cost = abs(self.shares_held) * current_price_scalar
                    cover_fee = cover_cost * (self.transaction_fee_percent / 100)
                    total_cover_cost = cover_cost + cover_fee
                    if self.balance >= total_cover_cost:
                        shares_covered = abs(self.shares_held)
                        pnl = (self._entry_price - current_price_scalar) * shares_covered - cover_fee
                        entry_time = self.df.index[self._entry_step]
                        exit_time = self.df.index[self.current_step]
                        self._completed_trades.append({
                            'entry_step': self._entry_step, 'exit_step': self.current_step,
                            'entry_time': entry_time, 'exit_time': exit_time,
                            'entry_price': self._entry_price, 'exit_price': current_price_scalar,
                            'quantity': shares_covered, 'direction': 'short', 'pnl': pnl, 'costs': cover_fee
                        })
                        self.balance -= total_cover_cost
                        logger.debug(f"Short -> Long (Step 1): Covered {shares_covered} shares at {current_price_scalar:.2f} each, "
                                   f"total cost: {total_cover_cost:.2f} (including fee: {cover_fee:.2f}), PnL: {pnl:.2f}")
                        self.shares_held = 0
                        self._entry_price = 0.0 
                        self._entry_step = 0
                        
                        if self.position_sizing_method == "fixed_fractional":
                            target_position_value = self.balance * self.risk_fraction
                            shares_to_buy = int(target_position_value / current_price_scalar)
                        else: 
                            max_shares_possible = self.balance / (current_price_scalar * (1 + self.transaction_fee_percent / 100))
                            shares_to_buy = int(max_shares_possible)
                        
                        cost_per_share_with_fee = current_price_scalar * (1 + self.transaction_fee_percent / 100)
                        affordable_shares = int(self.balance / cost_per_share_with_fee)
                        shares_to_buy = min(shares_to_buy, affordable_shares)

                        if shares_to_buy > 0:
                            buy_cost = shares_to_buy * current_price_scalar
                            buy_fee = buy_cost * (self.transaction_fee_percent / 100)
                            total_buy_cost = buy_cost + buy_fee
                            self.balance -= total_buy_cost
                            self.shares_held += shares_to_buy
                            self.current_position = 1
                            self._entry_price = current_price_scalar
                            self._entry_step = self.current_step
                            trade_occurred = True
                            self._trade_count += 1
                            logger.debug(f"Short -> Long (Step 2): Bought {shares_to_buy} shares at {current_price_scalar:.2f} each, "
                                       f"total cost: {total_buy_cost:.2f} (including fee: {buy_fee:.2f})")
                        else:
                            self.current_position = 0
                            logger.debug("Short -> Long: Covered short position but insufficient funds/shares to go long.")
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Required: {total_cover_cost:.2f}, Available: {self.balance:.2f}")

        elif action == 2:  # Short
            if prev_position == 0:  # Flat -> Short
                if self.position_sizing_method == "fixed_fractional":
                    target_position_value = self.balance * self.risk_fraction
                    shares_to_short = int(target_position_value / current_price_scalar)
                else: 
                    buyable_shares = int(self.balance / (current_price_scalar * (1 + self.transaction_fee_percent / 100)))
                    shares_to_short = buyable_shares
                
                if shares_to_short > 0 and self.balance > 0:
                    proceeds = shares_to_short * current_price_scalar
                    transaction_fee = proceeds * (self.transaction_fee_percent / 100)
                    total_proceeds = proceeds - transaction_fee
                    self.balance += total_proceeds
                    self.shares_held = -shares_to_short
                    self.current_position = -1
                    self._entry_price = current_price_scalar
                    self._entry_step = self.current_step
                    trade_occurred = True
                    self._trade_count += 1
                    logger.debug(f"Flat -> Short: Sold short {shares_to_short} shares at {current_price_scalar:.2f} each, "
                               f"total proceeds: {total_proceeds:.2f} (after fee: {transaction_fee:.2f})")
                else:
                     logger.debug("Flat -> Short: Cannot short shares (zero shares calculated or zero balance).")

            elif prev_position == 1:  # Long -> Short
                if self.shares_held > 0:
                    revenue = self.shares_held * current_price_scalar
                    sell_fee = revenue * (self.transaction_fee_percent / 100)
                    total_revenue = revenue - sell_fee
                    shares_sold = self.shares_held
                    pnl = (current_price_scalar - self._entry_price) * shares_sold - sell_fee
                    entry_time = self.df.index[self._entry_step]
                    exit_time = self.df.index[self.current_step]
                    self._completed_trades.append({
                        'entry_step': self._entry_step, 'exit_step': self.current_step,
                        'entry_time': entry_time, 'exit_time': exit_time,
                        'entry_price': self._entry_price, 'exit_price': current_price_scalar,
                        'quantity': shares_sold, 'direction': 'long', 'pnl': pnl, 'costs': sell_fee
                    })
                    self.balance += total_revenue
                    logger.debug(f"Long -> Short (Step 1): Sold {shares_sold} shares at {current_price_scalar:.2f} each, "
                               f"total revenue: {total_revenue:.2f} (after fee: {sell_fee:.2f}), PnL: {pnl:.2f}")
                    self.shares_held = 0
                    self._entry_price = 0.0 
                    self._entry_step = 0
                    
                    if self.position_sizing_method == "fixed_fractional":
                        target_position_value = self.balance * self.risk_fraction
                        shares_to_short = int(target_position_value / current_price_scalar)
                    else: 
                        buyable_shares = int(self.balance / (current_price_scalar * (1 + self.transaction_fee_percent / 100)))
                        shares_to_short = buyable_shares
                    
                    if shares_to_short > 0 and self.balance > 0:
                        proceeds = shares_to_short * current_price_scalar
                        short_fee = proceeds * (self.transaction_fee_percent / 100)
                        total_proceeds = proceeds - short_fee
                        self.balance += total_proceeds
                        self.shares_held = -shares_to_short
                        self.current_position = -1
                        self._entry_price = current_price_scalar
                        self._entry_step = self.current_step
                        trade_occurred = True 
                        self._trade_count += 1 
                        logger.debug(f"Long -> Short (Step 2): Sold short {shares_to_short} shares at {current_price_scalar:.2f} each, "
                                   f"total proceeds: {total_proceeds:.2f} (after fee: {short_fee:.2f})")
                    else:
                        self.current_position = 0
                        logger.debug("Long -> Short: Sold long position but couldn't establish short position (zero shares or balance).")

    def _calculate_reward(self) -> float:
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        if self.last_portfolio_value == 0:
            logger.warning("Last portfolio value was 0, returning 0 reward to avoid division by zero.")
            return 0.0
        
        percentage_change = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self._portfolio_returns.append(percentage_change)
        
        if self.use_sharpe_ratio and len(self._portfolio_returns) >= 2:
            returns_array = np.array(self._portfolio_returns)
            returns_mean = np.mean(returns_array)
            returns_std = np.std(returns_array)
            if returns_std == 0:
                risk_adjusted_return = returns_mean 
            else:
                risk_adjusted_return = (returns_mean - self.risk_free_rate) / returns_std
        else:
            risk_adjusted_return = percentage_change
        
        trading_penalty = self.trading_frequency_penalty * self._trade_count
        
        if self.max_portfolio_value > 0:
            current_drawdown = max(0, (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value)
            drawdown_penalty = self.drawdown_penalty * current_drawdown
        else:
            drawdown_penalty = 0
        
        reward = risk_adjusted_return - trading_penalty - drawdown_penalty
        logger.debug(f"Reward components: risk_adjusted={risk_adjusted_return:.6f}, "
                   f"trading_penalty={trading_penalty:.6f}, drawdown_penalty={drawdown_penalty:.6f}, "
                   f"final_reward={reward:.6f}")
        return reward
    
    def _get_observation(self) -> np.ndarray:
        window_start = max(0, self.current_step - self.window_size + 1)
        windowed_data = []
        numeric_df = self.df.select_dtypes(include=np.number)
        history_df = numeric_df.iloc[:self.current_step + 1]

        rolling_mean = history_df.rolling(window=self.normalization_window_size, min_periods=1).mean()
        rolling_std = history_df.rolling(window=self.normalization_window_size, min_periods=1).std()

        current_mean = rolling_mean.iloc[-1].fillna(0) 
        current_std = rolling_std.iloc[-1].fillna(1e-8) + 1e-8
        
        earliest_mean = rolling_mean.iloc[0].fillna(0) 
        earliest_std = rolling_std.iloc[0].fillna(1e-8) + 1e-8

        padding_needed = max(0, self.window_size - (self.current_step + 1))

        if padding_needed > 0:
            earliest_data = numeric_df.iloc[0].values
            earliest_data = np.nan_to_num(earliest_data, nan=0.0) 
            earliest_data_normalized = (earliest_data - earliest_mean.values) / earliest_std.values
            for _ in range(padding_needed):
                windowed_data.append(earliest_data_normalized)

        for i in range(window_start, self.current_step + 1):
            market_data = numeric_df.iloc[i].values
            market_data = np.nan_to_num(market_data, nan=0.0)
            market_data_normalized = (market_data - current_mean.values) / current_std.values
            windowed_data.append(market_data_normalized)

        if not windowed_data:
             num_market_features = len(numeric_df.columns)
             flattened_market_data = np.zeros(self.window_size * num_market_features)
        else:
            flattened_market_data = np.concatenate(windowed_data)

        normalized_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        safe_current_price = float(np.nan_to_num(self.current_price, nan=0.0)) # self.current_price is now float
        normalized_position_value = self.shares_held * safe_current_price / self.initial_balance if self.initial_balance > 0 else 0
        
        logger.debug(f"Type of normalized_balance: {type(normalized_balance)}, value: {normalized_balance}")
        logger.debug(f"Type of normalized_position_value: {type(normalized_position_value)}, value: {normalized_position_value}")
        logger.debug(f"Type of flattened_market_data: {type(flattened_market_data)}, shape: {getattr(flattened_market_data, 'shape', 'N/A')}, dtype: {getattr(flattened_market_data, 'dtype', 'N/A')}")
        account_info = np.array([normalized_balance, normalized_position_value], dtype=np.float32)
        observation = np.concatenate((flattened_market_data.astype(np.float32), account_info))

        return observation.astype(np.float32)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode == 'human':
            logger.info(f"Current step: {self.current_step}/{len(self.df) - 1}")
            return None
        elif mode == 'rgb_array':
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def get_completed_trades(self) -> List[Dict[str, Any]]:
        return list(self._completed_trades)

    def close(self) -> None:
        logger.info("Environment closed")
"""
Benchmarks module for backtesting.

This module provides implementations of benchmark trading strategies
for comparison with reinforcement learning strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class BenchmarkStrategy:
    """
    Base class for benchmark trading strategies.
    
    This class defines the interface for benchmark strategies and
    provides common functionality.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001) -> None:
        """
        Initialize the benchmark strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
        """
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.name = "Base Strategy"
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the benchmark strategy on the provided data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics and trade history
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def calculate_metrics(self, 
                         portfolio_values: List[float],
                         trades: int,
                         profitable_trades: int) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: List of portfolio values over time
            trades: Number of trades executed
            profitable_trades: Number of profitable trades
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        pnl = final_value - initial_value
        pnl_percentage = (pnl / initial_value) * 100
        
        # Calculate Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        if trades > 0:
            win_rate = profitable_trades / trades
        else:
            win_rate = 0
            
        return {
            "pnl": pnl,
            "pnl_percentage": pnl_percentage,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades": trades
        }


class BuyAndHoldStrategy(BenchmarkStrategy):
    """
    Simple buy and hold strategy.
    
    This strategy buys the asset at the beginning and holds until the end.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001) -> None:
        """
        Initialize the buy and hold strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
        """
        super().__init__(initial_balance, transaction_fee)
        self.name = "Buy and Hold"
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the buy and hold strategy on the provided data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics and trade history
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) == 0:
            logger.warning("Empty data provided")
            return {
                "pnl": 0,
                "pnl_percentage": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }
        
        try:
            # Get price data - Yahoo Finance returns capitalized column names
            # Try different possible column names
            if 'close' in data.columns:
                prices = data['close'].values
            elif 'Close' in data.columns:
                prices = data['Close'].values
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close'].values
            else:
                # If none of the expected columns are found, log available columns and return default
                logger.error(f"Cannot find price data in columns: {data.columns}")
                return {
                    "pnl": 0,
                    "pnl_percentage": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "trades": 0
                }
            
            # Calculate number of shares to buy
            initial_price = prices[0]
            shares = self.initial_balance / initial_price
            shares = shares * (1 - self.transaction_fee)  # Account for transaction fee
            
            # Calculate portfolio value over time
            portfolio_values = [self.initial_balance]  # Initial balance
            for price in prices[1:]:
                portfolio_value = shares * price
                portfolio_values.append(portfolio_value)
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                portfolio_values=portfolio_values,
                trades=1,  # Buy once
                profitable_trades=1 if portfolio_values[-1] > portfolio_values[0] else 0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "error": str(e),
                "pnl": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }


class SMAStrategy(BenchmarkStrategy):
    """
    Simple Moving Average (SMA) crossover strategy.
    
    This strategy generates buy/sell signals based on the crossover
    of short-term and long-term moving averages.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001,
                 short_window: int = 20,
                 long_window: int = 50) -> None:
        """
        Initialize the SMA strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        super().__init__(initial_balance, transaction_fee)
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"SMA({short_window},{long_window})"
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the SMA strategy on the provided data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics and trade history
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) < self.long_window:
            logger.warning(f"Insufficient data for {self.name} strategy")
            return {
                "pnl": 0,
                "pnl_percentage": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }
        
        try:
            # Get price data - Yahoo Finance returns capitalized column names
            # Try different possible column names
            if 'close' in data.columns:
                prices = data['close'].values
            elif 'Close' in data.columns:
                prices = data['Close'].values
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close'].values
            else:
                # If none of the expected columns are found, log available columns and return default
                logger.error(f"Cannot find price data in columns: {data.columns}")
                return {
                    "pnl": 0,
                    "pnl_percentage": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "trades": 0
                }
            
            # Calculate moving averages
            short_ma = np.convolve(prices, np.ones(self.short_window)/self.short_window, mode='valid')
            long_ma = np.convolve(prices, np.ones(self.long_window)/self.long_window, mode='valid')
            
            # Align moving averages (they have different lengths)
            diff = self.long_window - self.short_window
            if diff > 0:
                short_ma = short_ma[diff:]
            
            # Generate signals
            signals = np.zeros(len(long_ma))
            signals[short_ma > long_ma] = 1  # Buy signal
            signals[short_ma < long_ma] = -1  # Sell signal
            
            # Pad signals to match original data length
            pad_length = len(prices) - len(signals)
            signals = np.pad(signals, (pad_length, 0), 'constant', constant_values=0)
            
            # Initialize portfolio
            cash = self.initial_balance
            shares = 0
            portfolio_values = [cash]
            position = 0  # 0: no position, 1: long
            
            trades = 0
            profitable_trades = 0
            entry_price = 0
            
            # Simulate trading
            for i in range(1, len(prices)):
                signal = signals[i]
                price = prices[i]
                
                # Check for position change
                if signal == 1 and position == 0:  # Buy
                    shares = cash / price
                    shares = shares * (1 - self.transaction_fee)  # Account for transaction fee
                    cash = 0
                    position = 1
                    trades += 1
                    entry_price = price
                elif signal == -1 and position == 1:  # Sell
                    cash = shares * price * (1 - self.transaction_fee)  # Account for transaction fee
                    if price > entry_price:
                        profitable_trades += 1
                    shares = 0
                    position = 0
                
                # Calculate portfolio value
                portfolio_value = cash + (shares * price)
                portfolio_values.append(portfolio_value)
            
            # Liquidate final position if any
            if position == 1:
                cash = shares * prices[-1] * (1 - self.transaction_fee)
                if prices[-1] > entry_price:
                    profitable_trades += 1
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                portfolio_values=portfolio_values,
                trades=trades,
                profitable_trades=profitable_trades
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "error": str(e),
                "pnl": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }


class RandomStrategy(BenchmarkStrategy):
    """
    Random trading strategy.
    
    This strategy generates random buy/sell signals for benchmarking.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001,
                 trade_probability: float = 0.05,
                 random_seed: Optional[int] = None) -> None:
        """
        Initialize the random strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
            trade_probability: Probability of making a trade on each day
            random_seed: Random seed for reproducibility
        """
        super().__init__(initial_balance, transaction_fee)
        self.trade_probability = trade_probability
        self.random_seed = random_seed
        self.name = "Random"
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the random strategy on the provided data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics and trade history
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) == 0:
            logger.warning("Empty data provided")
            return {
                "pnl": 0,
                "pnl_percentage": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }
        
        try:
            # Get price data - Yahoo Finance returns capitalized column names
            # Try different possible column names
            if 'close' in data.columns:
                prices = data['close'].values
            elif 'Close' in data.columns:
                prices = data['Close'].values
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close'].values
            else:
                # If none of the expected columns are found, log available columns and return default
                logger.error(f"Cannot find price data in columns: {data.columns}")
                return {
                    "pnl": 0,
                    "pnl_percentage": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "trades": 0
                }
            
            # Generate random signals
            signals = np.random.random(len(prices)) < self.trade_probability
            
            # Initialize portfolio
            cash = self.initial_balance
            shares = 0
            portfolio_values = [cash]
            position = 0  # 0: no position, 1: long
            
            trades = 0
            profitable_trades = 0
            entry_price = 0
            
            # Simulate trading
            for i in range(1, len(prices)):
                price = prices[i]
                
                # Check for trade signal
                if signals[i]:
                    if position == 0:  # Buy
                        shares = cash / price
                        shares = shares * (1 - self.transaction_fee)  # Account for transaction fee
                        cash = 0
                        position = 1
                        trades += 1
                        entry_price = price
                    else:  # Sell
                        cash = shares * price * (1 - self.transaction_fee)  # Account for transaction fee
                        if price > entry_price:
                            profitable_trades += 1
                        shares = 0
                        position = 0
                
                # Calculate portfolio value
                portfolio_value = cash + (shares * price)
                portfolio_values.append(portfolio_value)
            
            # Liquidate final position if any
            if position == 1:
                cash = shares * prices[-1] * (1 - self.transaction_fee)
                if prices[-1] > entry_price:
                    profitable_trades += 1
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                portfolio_values=portfolio_values,
                trades=trades,
                profitable_trades=profitable_trades
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "error": str(e),
                "pnl": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades": 0
            }
"""Benchmark strategies for model evaluation.

This module provides benchmark trading strategies for comparing
reinforcement learning models against traditional approaches.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class BenchmarkStrategy:
    """Base class for benchmark trading strategies."""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001,
        name: str = "Base Strategy"
    ):
        """Initialize the benchmark strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
            name: Name of the strategy
        """
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.name = name
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the strategy on the provided data.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def _get_price_column(self, data: pd.DataFrame) -> np.ndarray:
        """Extract price data from DataFrame.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Numpy array of prices
            
        Raises:
            ValueError: If no suitable price column is found
        """
        # Try different possible column names
        price_columns = ['close', 'Close', 'Adj Close', 'price', 'Price']
        
        for col in price_columns:
            if col in data.columns:
                return np.array(data[col].values, dtype=float).flatten()
        
        # If no standard column found, try to use the first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.warning(f"Using first numeric column '{numeric_cols[0]}' as price data")
            return np.array(data[numeric_cols[0]].values, dtype=float).flatten()
        
        raise ValueError(f"Cannot find price data in columns: {list(data.columns)}")
    
    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        trades_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance metrics using MetricsCalculator.
        
        Args:
            portfolio_values: List of portfolio values over time
            trades_info: Dictionary with trade information
            
        Returns:
            Dictionary of performance metrics
        """
        calculator = MetricsCalculator()
        
        # Convert to numpy array
        portfolio_values = np.array(portfolio_values, dtype=float)
        
        # Calculate returns
        returns = None
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        metrics = calculator.calculate_all_metrics(
            portfolio_values=portfolio_values,
            returns=returns,
            trades_count=trades_info.get('trades_count', 0),
            win_rate=trades_info.get('win_rate', 0.0),
            requested_metrics=[
                "pnl", "pnl_percentage", "sharpe_ratio", 
                "max_drawdown", "win_rate"
            ]
        )
        
        # Add trades count
        metrics['trades'] = trades_info.get('trades_count', 0)
        
        return metrics


class BuyAndHoldStrategy(BenchmarkStrategy):
    """Buy and hold benchmark strategy."""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001
    ):
        """Initialize the buy and hold strategy."""
        super().__init__(initial_balance, transaction_fee, "Buy and Hold")
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the buy and hold strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) == 0:
            logger.warning("Empty data provided")
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }
        
        try:
            # Get price data
            prices = self._get_price_column(data)
            
            # Calculate number of shares to buy
            initial_price = float(prices[0])
            shares = self.initial_balance / initial_price
            shares = shares * (1 - self.transaction_fee)  # Account for transaction fee
            
            # Calculate portfolio value over time
            portfolio_values = [float(self.initial_balance)]  # Initial balance
            for price in prices[1:]:
                portfolio_value = float(shares * price)
                portfolio_values.append(portfolio_value)
            
            # Determine if the single trade was profitable
            final_value = portfolio_values[-1]
            win_rate = 1.0 if final_value > self.initial_balance else 0.0
            
            # Calculate metrics
            trades_info = {
                'trades_count': 1,
                'win_rate': win_rate
            }
            
            metrics = self._calculate_metrics(portfolio_values, trades_info)
            
            logger.debug(f"{self.name} completed - PnL: {metrics['pnl']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }


class SimpleMovingAverageStrategy(BenchmarkStrategy):
    """Simple Moving Average (SMA) crossover strategy."""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001,
        short_window: int = 20,
        long_window: int = 50
    ):
        """Initialize the SMA strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        super().__init__(
            initial_balance, 
            transaction_fee, 
            f"SMA({short_window},{long_window})"
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the SMA crossover strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) < self.long_window:
            logger.warning(f"Insufficient data for {self.name} strategy")
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }
        
        try:
            # Get price data
            prices = self._get_price_column(data)
            
            # Calculate moving averages using pandas for efficiency
            price_series = pd.Series(prices)
            short_ma = price_series.rolling(window=self.short_window).mean()
            long_ma = price_series.rolling(window=self.long_window).mean()
            
            # Generate signals
            signals = pd.Series(index=price_series.index, dtype=float)
            signals[short_ma > long_ma] = 1.0  # Buy signal
            signals[short_ma <= long_ma] = 0.0  # Sell/no position signal
            
            # Forward fill NaN values at the beginning
            signals = signals.fillna(0)
            
            # Detect position changes
            positions = signals.diff()
            
            # Initialize portfolio tracking
            cash = self.initial_balance
            shares = 0
            portfolio_values = []
            
            trades_count = 0
            profitable_trades = 0
            entry_price = 0
            
            # Simulate trading
            for i in range(len(prices)):
                price = prices[i]
                
                # Check for position change
                if i > 0 and positions.iloc[i] != 0:
                    if positions.iloc[i] > 0:  # Buy signal
                        if cash > 0:
                            # Buy shares
                            shares = cash / price
                            shares = shares * (1 - self.transaction_fee)
                            cash = 0
                            entry_price = price
                            trades_count += 1
                    else:  # Sell signal
                        if shares > 0:
                            # Sell shares
                            cash = shares * price * (1 - self.transaction_fee)
                            if price > entry_price:
                                profitable_trades += 1
                            shares = 0
                
                # Calculate portfolio value
                portfolio_value = cash + (shares * price)
                portfolio_values.append(portfolio_value)
            
            # Close any open position at the end
            if shares > 0:
                final_price = prices[-1]
                if final_price > entry_price:
                    profitable_trades += 1
            
            # Calculate win rate
            win_rate = profitable_trades / trades_count if trades_count > 0 else 0.0
            
            # Calculate metrics
            trades_info = {
                'trades_count': trades_count,
                'win_rate': win_rate
            }
            
            metrics = self._calculate_metrics(portfolio_values, trades_info)
            
            logger.debug(f"{self.name} completed - Trades: {trades_count}, PnL: {metrics['pnl']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }


class RandomStrategy(BenchmarkStrategy):
    """Random trading strategy for baseline comparison."""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001,
        trade_probability: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """Initialize the random strategy.
        
        Args:
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction
            trade_probability: Probability of trading on each step
            random_seed: Random seed for reproducibility
        """
        super().__init__(initial_balance, transaction_fee, "Random")
        self.trade_probability = trade_probability
        self.random_seed = random_seed
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the random trading strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Running {self.name} strategy")
        
        if len(data) == 0:
            logger.warning("Empty data provided")
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        try:
            # Get price data
            prices = self._get_price_column(data)
            
            # Generate random signals
            signals = np.random.random(len(prices)) < self.trade_probability
            
            # Initialize portfolio tracking
            cash = self.initial_balance
            shares = 0
            portfolio_values = []
            position = 0  # 0: no position, 1: long
            
            trades_count = 0
            profitable_trades = 0
            entry_price = 0
            
            # Simulate trading
            for i in range(len(prices)):
                price = prices[i]
                
                # Check for trade signal
                if signals[i] and i > 0:  # Don't trade on first step
                    if position == 0:  # Buy
                        if cash > 0:
                            shares = cash / price
                            shares = shares * (1 - self.transaction_fee)
                            cash = 0
                            position = 1
                            trades_count += 1
                            entry_price = price
                    else:  # Sell
                        if shares > 0:
                            cash = shares * price * (1 - self.transaction_fee)
                            if price > entry_price:
                                profitable_trades += 1
                            shares = 0
                            position = 0
                
                # Calculate portfolio value
                portfolio_value = cash + (shares * price)
                portfolio_values.append(portfolio_value)
            
            # Close any open position at the end
            if position == 1 and shares > 0:
                final_price = prices[-1]
                if final_price > entry_price:
                    profitable_trades += 1
            
            # Calculate win rate
            win_rate = profitable_trades / trades_count if trades_count > 0 else 0.0
            
            # Calculate metrics
            trades_info = {
                'trades_count': trades_count,
                'win_rate': win_rate
            }
            
            metrics = self._calculate_metrics(portfolio_values, trades_info)
            
            logger.debug(f"{self.name} completed - Trades: {trades_count}, PnL: {metrics['pnl']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.name} strategy: {e}", exc_info=True)
            return {
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades": 0
            }


class BenchmarkEvaluator:
    """Evaluator for running and comparing benchmark strategies."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """Initialize the benchmark evaluator.
        
        Args:
            config: Configuration dictionary
            metrics_calculator: Optional metrics calculator instance
        """
        self.config = config
        self.metrics_calculator = metrics_calculator or MetricsCalculator(config)
        
        # Initialize benchmark strategies
        self.strategies = {
            "buy_and_hold": BuyAndHoldStrategy(
                initial_balance=config.get("initial_balance", 10000),
                transaction_fee=config.get("transaction_fee", 0.001)
            ),
            "simple_moving_average": SimpleMovingAverageStrategy(
                initial_balance=config.get("initial_balance", 10000),
                transaction_fee=config.get("transaction_fee", 0.001),
                short_window=config.get("sma_short_window", 20),
                long_window=config.get("sma_long_window", 50)
            ),
            "random": RandomStrategy(
                initial_balance=config.get("initial_balance", 10000),
                transaction_fee=config.get("transaction_fee", 0.001),
                trade_probability=config.get("random_trade_probability", 0.05),
                random_seed=config.get("random_seed", 42)
            )
        }
    
    def compare_with_benchmarks(
        self,
        test_data: pd.DataFrame,
        model_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare model performance with benchmark strategies.
        
        Args:
            test_data: DataFrame containing test data
            model_metrics: Dictionary of model performance metrics
            
        Returns:
            Dictionary containing benchmark results and comparisons
        """
        logger.info("Running benchmark strategies for comparison")
        
        benchmarks = {}
        
        # Run each benchmark strategy
        for name, strategy in self.strategies.items():
            logger.info(f"Running {name} benchmark")
            try:
                benchmarks[name] = strategy.run(test_data)
            except Exception as e:
                logger.error(f"Error running {name} benchmark: {e}")
                benchmarks[name] = {
                    "pnl": 0.0,
                    "pnl_percentage": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "trades": 0,
                    "error": str(e)
                }
        
        # Calculate relative performance
        relative_performance = {}
        
        for name, bench_metrics in benchmarks.items():
            if "error" in bench_metrics:
                continue
                
            model_pnl = model_metrics.get("pnl", 0)
            bench_pnl = bench_metrics.get("pnl", 0)
            
            relative_performance[name] = {
                "absolute_difference": model_pnl - bench_pnl,
                "percentage_difference": (
                    ((model_pnl / bench_pnl) - 1) * 100 
                    if bench_pnl != 0 else float('inf')
                ),
                "sharpe_ratio_difference": (
                    model_metrics.get("sharpe_ratio", 0) - 
                    bench_metrics.get("sharpe_ratio", 0)
                )
            }
        
        return {
            "benchmarks": benchmarks,
            "relative_performance": relative_performance
        }
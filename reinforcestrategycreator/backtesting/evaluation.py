"""
Evaluation module for backtesting.

This module provides functionality for evaluating trading strategies,
calculating performance metrics, and comparing with benchmarks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from reinforcestrategycreator.trading_environment import TradingEnvironment

# Configure logging
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates performance metrics for trading strategies.
    
    This class provides methods for calculating key performance metrics
    such as PnL, Sharpe ratio, max drawdown, and win rate.
    """
    
    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        pass
    
    def get_episode_metrics(self, env: TradingEnvironment) -> Dict[str, float]:
        """
        Calculate key episode metrics.
        
        Args:
            env: Trading environment after episode completion
            
        Returns:
            Dictionary of metrics
        """
        # Extract portfolio history
        portfolio_values = env.portfolio_value_history
        
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
        trades = env.trades
        if trades > 0:
            win_rate = env.profitable_trades / trades
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


class BenchmarkEvaluator:
    """
    Evaluates and compares trading strategies against benchmarks.
    
    This class provides methods for running benchmark strategies and
    comparing model performance against these benchmarks.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 metrics_calculator: MetricsCalculator = None) -> None:
        """
        Initialize the benchmark evaluator.
        
        Args:
            config: Dictionary containing configuration parameters
            metrics_calculator: MetricsCalculator instance (optional)
        """
        self.config = config
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        
        # Import benchmark strategies here to avoid circular imports
        from reinforcestrategycreator.backtesting.benchmarks import (
            BuyAndHoldStrategy,
            SMAStrategy,
            RandomStrategy
        )
        
        # Initialize benchmark strategies
        self.strategies = {
            "buy_and_hold": BuyAndHoldStrategy(
                initial_balance=config.get("initial_balance", 10000),
                transaction_fee=config.get("transaction_fee", 0.001)
            ),
            "simple_moving_average": SMAStrategy(
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
    
    def compare_with_benchmarks(self, test_data: pd.DataFrame, model_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance with benchmark strategies.
        
        Args:
            test_data: DataFrame containing the test data
            model_metrics: Dictionary of model performance metrics
            
        Returns:
            Dictionary of benchmark metrics and relative performance
        """
        logger.info("Comparing model performance with benchmarks")
        
        # Run benchmark strategies
        benchmarks = {}
        for name, strategy in self.strategies.items():
            logger.info(f"Running benchmark strategy: {name}")
            benchmarks[name] = strategy.run(test_data)
        
        # Calculate relative performance
        model_pnl = model_metrics["pnl"]
        relative_performance = {}
        
        for name, bench_metrics in benchmarks.items():
            bench_pnl = bench_metrics["pnl"]
            relative_performance[name] = {
                "absolute_difference": model_pnl - bench_pnl,
                "percentage_difference": ((model_pnl / bench_pnl) - 1) * 100 if bench_pnl != 0 else float('inf'),
                "sharpe_ratio_difference": model_metrics["sharpe_ratio"] - bench_metrics["sharpe_ratio"]
            }
        
        return {
            "benchmarks": benchmarks,
            "relative_performance": relative_performance
        }
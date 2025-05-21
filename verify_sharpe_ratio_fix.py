#!/usr/bin/env python3
"""
Script to verify the Sharpe Ratio calculation fix.

This script creates a simple test environment and calculates the Sharpe Ratio
using different window sizes to demonstrate the effect of the fix.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque

from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator
from reinforcestrategycreator.trading_environment import TradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_environment(returns, initial_balance=10000):
    """Create a mock environment with predefined returns."""
    # Create a simple environment with portfolio values based on returns
    env = type('MockEnv', (), {})()
    
    # Calculate portfolio values from returns
    portfolio_values = [initial_balance]
    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    # Set portfolio value history
    env._portfolio_value_history = portfolio_values
    
    # Set other required attributes
    env._completed_trades = []
    env.cached_final_info_for_callback = None
    env.initial_balance = initial_balance
    
    return env

def main():
    """Run the verification test."""
    logger.info("Starting Sharpe Ratio calculation verification")
    
    # Create synthetic returns data (daily returns for 1 year)
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0005, 0.01, 252)  # Mean: 0.05% daily, Std: 1%
    
    # Create mock environments
    env = create_mock_environment(daily_returns)
    
    # Test with different window sizes
    window_sizes = [None, 20, 50, 100, 252]
    
    logger.info("Testing Sharpe Ratio calculation with different window sizes:")
    for window_size in window_sizes:
        # Create metrics calculator with specific window size
        metrics_calc = MetricsCalculator(sharpe_window_size=window_size)
        
        # Calculate metrics
        metrics = metrics_calc.get_episode_metrics(env)
        
        # Display results
        window_desc = "all returns" if window_size is None else f"last {window_size} returns"
        logger.info(f"Window Size: {window_desc}, Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    # Calculate theoretical annualized Sharpe ratio for comparison
    annual_mean = np.mean(daily_returns) * 252
    annual_std = np.std(daily_returns) * np.sqrt(252)
    theoretical_sharpe = annual_mean / annual_std
    
    logger.info(f"Theoretical Annualized Sharpe Ratio: {theoretical_sharpe:.4f}")
    logger.info("Verification complete!")

if __name__ == "__main__":
    main()
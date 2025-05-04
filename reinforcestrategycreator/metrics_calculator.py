"""
Metrics Calculator Module

Provides functions to calculate various performance metrics for trading strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Sequence

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the Sharpe Ratio for a given series of returns.

    Args:
        returns (pd.Series): A pandas Series of periodic returns (e.g., daily, step-based).
        risk_free_rate (float): The risk-free rate for the same period as the returns. Defaults to 0.0.

    Returns:
        float: The calculated Sharpe Ratio. Returns 0.0 if standard deviation is zero or NaN if input is invalid.
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()

    # Optional: Annualize if returns are daily (adjust sqrt value for other frequencies)
    # Assuming daily returns for annualization example:
    # sharpe_ratio *= np.sqrt(252) # 252 trading days in a year

    return float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0

def calculate_max_drawdown(portfolio_values: Sequence[float]) -> float:
    """
    Calculates the Maximum Drawdown (MDD) from a sequence of portfolio values.

    Args:
        portfolio_values (Sequence[float]): A sequence (list, numpy array, pandas Series)
                                             of portfolio values over time.

    Returns:
        float: The Maximum Drawdown as a positive percentage (e.g., 0.1 for 10% drawdown).
               Returns 0.0 if the input sequence is empty or has fewer than 2 values.
    """
    if len(portfolio_values) < 2:
        return 0.0

    values = pd.Series(portfolio_values)
    cumulative_max = values.cummax()
    drawdown = (cumulative_max - values) / cumulative_max
    max_drawdown = drawdown.max()

    # Handle potential NaN if initial value is 0 or negative
    return float(max_drawdown) if pd.notna(max_drawdown) and max_drawdown > 0 else 0.0


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculates the Win Rate from a list of completed trades.

    Args:
        trades (List[Dict[str, Any]]): A list of trade dictionaries, where each dictionary
                                       must have a 'pnl' key.

    Returns:
        float: The Win Rate as a percentage (0.0 to 100.0). Returns 0.0 if no trades are provided.
    """
    if not trades:
        return 0.0

    profitable_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    total_trades = len(trades)

    win_rate = (profitable_trades / total_trades) * 100.0
    return win_rate

# Note: Trade Frequency and Success Rate are simple enough to calculate directly in train.py
# Trade Frequency = len(trades)
# Success Rate = 1 if episode_pnl > 0 else 0
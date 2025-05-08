"""
Metrics Calculator Module

Provides functions to calculate various performance metrics for trading strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Sequence

def calculate_sharpe_ratio(daily_returns: pd.Series, annual_risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculates the Annualized Sharpe Ratio from daily returns.

    Args:
        daily_returns (pd.Series): A pandas Series of daily returns.
        annual_risk_free_rate (float): The annualized risk-free rate. Defaults to 0.0.
        annualization_factor (int): Number of trading periods in a year (e.g., 252 for daily). Defaults to 252.

    Returns:
        float: The calculated Annualized Sharpe Ratio. Returns 0.0 if standard deviation is zero or NaN.
    """
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0

    # Convert annual risk-free rate to daily
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/annualization_factor) - 1
    
    excess_returns = daily_returns - daily_risk_free_rate
    
    # Calculate Sharpe ratio using daily excess returns
    # Check for zero standard deviation of excess returns
    if excess_returns.std() == 0:
        # If mean excess return is also zero, Sharpe is 0. If positive, could be inf. For safety, return 0.
        return 0.0 if excess_returns.mean() == 0 else np.inf if excess_returns.mean() > 0 else -np.inf

    sharpe_ratio_daily = excess_returns.mean() / excess_returns.std()
    
    # Annualize the Sharpe ratio
    annualized_sharpe_ratio = sharpe_ratio_daily * np.sqrt(annualization_factor)

    return float(annualized_sharpe_ratio) if not np.isnan(annualized_sharpe_ratio) else 0.0

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

def calculate_sortino_ratio(daily_returns: pd.Series, annual_risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculates the Annualized Sortino Ratio from daily returns.

    Args:
        daily_returns (pd.Series): A pandas Series of daily returns.
        annual_risk_free_rate (float): The annualized risk-free rate. Defaults to 0.0.
        annualization_factor (int): Factor to annualize the ratio (e.g., 252 for daily returns). Defaults to 252.

    Returns:
        float: The calculated annualized Sortino Ratio. Returns 0.0 if downside deviation is zero or NaN.
    """
    if daily_returns.empty:
        return 0.0

    # Convert annual risk-free rate to daily
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/annualization_factor) - 1
    
    excess_returns = daily_returns - daily_risk_free_rate
    # Target downside returns are those falling below the (daily) risk-free rate (or MAR, Minimum Acceptable Return)
    # For Sortino, the target is often the risk-free rate itself.
    # So, we consider returns < daily_risk_free_rate, or excess_returns < 0.
    downside_excess_returns = excess_returns[excess_returns < 0]
    
    if downside_excess_returns.empty:
        # If no returns are below the target, downside deviation is 0.
        # If mean excess return is positive, Sortino could be inf. If zero or negative, Sortino is 0 or negative.
        # Conventionally, return 0 or handle as np.inf if mean_excess_return > 0.
        return 0.0 if excess_returns.mean() <=0 else np.inf

    # Calculate downside deviation using only returns below the target (daily_risk_free_rate)
    # Squaring the negative excess returns (which are already < 0)
    downside_deviation = np.sqrt(np.mean(downside_excess_returns**2))

    if downside_deviation == 0:
        # This case should ideally be caught by downside_excess_returns.empty() if MAR is used correctly.
        # If it somehow occurs, and mean_excess_return > 0, Sortino is inf.
        return 0.0 if excess_returns.mean() <=0 else np.inf

    mean_excess_return = excess_returns.mean()
    sortino_ratio_daily = mean_excess_return / downside_deviation
    
    annualized_sortino_ratio = sortino_ratio_daily * np.sqrt(annualization_factor)

    return float(annualized_sortino_ratio) if not np.isnan(annualized_sortino_ratio) else 0.0

def calculate_annualized_volatility(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    Calculates the Annualized Volatility (standard deviation of returns).

    Args:
        daily_returns (pd.Series): A pandas Series of periodic returns (e.g., daily).
        annualization_factor (int): Factor to annualize the volatility (e.g., 252 for daily returns). Defaults to 252.

    Returns:
        float: The calculated Annualized Volatility. Returns 0.0 if daily_returns are empty or NaN.
    """
    if daily_returns.empty:
        return 0.0
    
    volatility = daily_returns.std() * np.sqrt(annualization_factor)
    return float(volatility) if not np.isnan(volatility) else 0.0

def calculate_beta(agent_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Beta of the agent's returns relative to benchmark returns.

    Args:
        agent_returns (pd.Series): A pandas Series of the agent's periodic returns.
        benchmark_returns (pd.Series): A pandas Series of the benchmark's periodic returns.
                                      Should be of the same length and frequency as agent_returns.

    Returns:
        float: The calculated Beta. Returns 0.0 if inputs are invalid or variance of benchmark is zero.
    """
    if agent_returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align returns by index and drop NaNs that might result from different lengths or missing dates
    combined_returns = pd.DataFrame({'agent': agent_returns, 'benchmark': benchmark_returns}).dropna()
    
    if combined_returns.empty or len(combined_returns) < 2: # Need at least 2 data points for variance/covariance
        return 0.0

    aligned_agent_returns = combined_returns['agent']
    aligned_benchmark_returns = combined_returns['benchmark']

    # Calculate covariance between aligned agent returns and aligned benchmark returns
    # np.cov returns a 2x2 matrix: [[var(agent), cov(agent, bench)], [cov(bench, agent), var(bench)]]
    covariance_matrix = np.cov(aligned_agent_returns, aligned_benchmark_returns)
    
    if covariance_matrix.shape != (2, 2): # Should be a 2x2 matrix
        return 0.0 # Should not happen if inputs are valid Series
        
    covariance = covariance_matrix[0, 1] # Covariance is at [0,1] or [1,0]
    
    # Calculate variance of aligned benchmark returns
    benchmark_variance = aligned_benchmark_returns.var()

    if benchmark_variance == 0:
        # If benchmark variance is zero (e.g., benchmark never changed), beta is undefined or can be taken as 0 or NaN.
        # Returning 0 for simplicity, but context might require NaN or specific handling.
        return 0.0
        
    beta = covariance / benchmark_variance
    return float(beta) if not np.isnan(beta) else 0.0

def calculate_alpha(annualized_agent_return: float, annualized_benchmark_return: float,
                    annual_risk_free_rate: float, beta: float) -> float:
    """
    Calculates Alpha (Jensen's Alpha) using annualized figures.

    Args:
        annualized_agent_return (float): The annualized return of the agent.
        annualized_benchmark_return (float): The annualized return of the benchmark.
        annual_risk_free_rate (float): The annualized risk-free rate.
        beta (float): The beta of the agent relative to the benchmark.

    Returns:
        float: The calculated Alpha.
    """
    # Alpha = Ra - (Rf + Beta * (Rm - Rf))
    # Ra = annualized_agent_return
    # Rm = annualized_benchmark_return
    # Rf = annual_risk_free_rate
    alpha = annualized_agent_return - (annual_risk_free_rate + beta * (annualized_benchmark_return - annual_risk_free_rate))
    return float(alpha) if not np.isnan(alpha) else 0.0

# Note: Trade Frequency and Average Annual Trades will be calculated in evaluate_strategy.py
# as they depend on the list of trades and the test period duration.
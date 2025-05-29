"""Metrics calculation module for model evaluation.

This module provides comprehensive metrics calculation for trading strategies,
including financial metrics, risk metrics, and trading performance indicators.
"""

import logging
from typing import Dict, List, Optional, Union, Sequence
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for comprehensive trading and RL metrics.
    
    Provides methods to calculate various performance metrics including:
    - Financial metrics (PnL, returns, etc.)
    - Risk metrics (Sharpe ratio, Sortino ratio, max drawdown, etc.)
    - Trading metrics (win rate, profit factor, etc.)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        
        # Risk-free rate for Sharpe/Sortino calculations (annualized)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        
        # Trading days per year for annualization
        self.trading_days_per_year = self.config.get("trading_days_per_year", 252)
        
        # Sharpe ratio calculation window
        self.sharpe_window_size = self.config.get("sharpe_window_size", None)
        
        # Available metrics
        self.available_metrics = {
            "pnl": self.calculate_pnl,
            "pnl_percentage": self.calculate_pnl_percentage,
            "total_return": self.calculate_total_return,
            "sharpe_ratio": self.calculate_sharpe_ratio,
            "sortino_ratio": self.calculate_sortino_ratio,
            "max_drawdown": self.calculate_max_drawdown,
            "calmar_ratio": self.calculate_calmar_ratio,
            "win_rate": self.calculate_win_rate,
            "profit_factor": self.calculate_profit_factor,
            "average_win": self.calculate_average_win,
            "average_loss": self.calculate_average_loss,
            "expectancy": self.calculate_expectancy,
            "volatility": self.calculate_volatility,
            "downside_deviation": self.calculate_downside_deviation,
            "value_at_risk": self.calculate_value_at_risk,
            "conditional_value_at_risk": self.calculate_conditional_value_at_risk,
        }
    
    def calculate_all_metrics(
        self,
        portfolio_values: Sequence[float],
        returns: Optional[Sequence[float]] = None,
        trades: Optional[List[Dict]] = None,
        trades_count: Optional[int] = None,
        win_rate: Optional[float] = None,
        requested_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate all requested metrics.
        
        Args:
            portfolio_values: Sequence of portfolio values over time
            returns: Optional pre-calculated returns
            trades: Optional list of trade dictionaries with 'pnl' key
            trades_count: Optional number of trades (used if trades list not provided)
            win_rate: Optional pre-calculated win rate
            requested_metrics: List of specific metrics to calculate (all if None)
            
        Returns:
            Dictionary of calculated metrics
        """
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values, dtype=float)
        
        # Calculate returns if not provided
        if returns is None and len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        elif returns is not None:
            returns = np.array(returns, dtype=float)
        else:
            returns = np.array([])
        
        # Determine which metrics to calculate
        if requested_metrics is None:
            metrics_to_calculate = list(self.available_metrics.keys())
        else:
            metrics_to_calculate = [m for m in requested_metrics if m in self.available_metrics]
        
        # Calculate metrics
        results = {}
        
        for metric_name in metrics_to_calculate:
            try:
                if metric_name in ["pnl", "pnl_percentage", "total_return"]:
                    value = self.available_metrics[metric_name](portfolio_values)
                elif metric_name in ["sharpe_ratio", "sortino_ratio", "volatility", 
                                   "downside_deviation", "value_at_risk", 
                                   "conditional_value_at_risk"]:
                    value = self.available_metrics[metric_name](returns)
                elif metric_name in ["max_drawdown", "calmar_ratio"]:
                    value = self.available_metrics[metric_name](portfolio_values, returns)
                elif metric_name == "win_rate":
                    if win_rate is not None:
                        value = win_rate
                    elif trades:
                        value = self.calculate_win_rate(trades)
                    else:
                        value = 0.0
                elif metric_name in ["profit_factor", "average_win", "average_loss", "expectancy"]:
                    if trades:
                        value = self.available_metrics[metric_name](trades)
                    else:
                        value = 0.0
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue
                
                results[metric_name] = value
                
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {str(e)}")
                results[metric_name] = 0.0
        
        # Add trade count if provided
        if trades_count is not None:
            results["trades_count"] = trades_count
        elif trades:
            results["trades_count"] = len(trades)
        
        return results
    
    def calculate_pnl(self, portfolio_values: Sequence[float]) -> float:
        """Calculate total profit and loss.
        
        Args:
            portfolio_values: Sequence of portfolio values
            
        Returns:
            Total PnL
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        return float(portfolio_values[-1] - portfolio_values[0])
    
    def calculate_pnl_percentage(self, portfolio_values: Sequence[float]) -> float:
        """Calculate PnL as percentage of initial value.
        
        Args:
            portfolio_values: Sequence of portfolio values
            
        Returns:
            PnL percentage
        """
        if len(portfolio_values) < 2 or portfolio_values[0] == 0:
            return 0.0
        
        pnl = self.calculate_pnl(portfolio_values)
        return float((pnl / portfolio_values[0]) * 100)
    
    def calculate_total_return(self, portfolio_values: Sequence[float]) -> float:
        """Calculate total return (same as PnL percentage but as decimal).
        
        Args:
            portfolio_values: Sequence of portfolio values
            
        Returns:
            Total return as decimal
        """
        return self.calculate_pnl_percentage(portfolio_values) / 100
    
    def calculate_sharpe_ratio(self, returns: Sequence[float]) -> float:
        """Calculate annualized Sharpe ratio.
        
        Args:
            returns: Sequence of returns
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        
        # Use window if specified
        if self.sharpe_window_size and len(returns) > self.sharpe_window_size:
            returns = returns[-self.sharpe_window_size:]
        
        # Calculate excess returns
        daily_risk_free = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - daily_risk_free
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize
        annualized_sharpe = sharpe * np.sqrt(self.trading_days_per_year)
        
        return float(annualized_sharpe)
    
    def calculate_sortino_ratio(self, returns: Sequence[float]) -> float:
        """Calculate annualized Sortino ratio.
        
        Args:
            returns: Sequence of returns
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        
        # Calculate excess returns
        daily_risk_free = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - daily_risk_free
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std
        
        # Annualize
        annualized_sortino = sortino * np.sqrt(self.trading_days_per_year)
        
        return float(annualized_sortino)
    
    def calculate_max_drawdown(
        self, 
        portfolio_values: Sequence[float],
        returns: Optional[Sequence[float]] = None
    ) -> float:
        """Calculate maximum drawdown.
        
        Args:
            portfolio_values: Sequence of portfolio values
            returns: Optional returns (not used, kept for interface consistency)
            
        Returns:
            Maximum drawdown as decimal (0.1 = 10% drawdown)
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (cumulative_max - values) / cumulative_max
        
        max_dd = np.max(drawdowns)
        
        return float(max_dd) if not np.isnan(max_dd) else 0.0
    
    def calculate_calmar_ratio(
        self,
        portfolio_values: Sequence[float],
        returns: Sequence[float]
    ) -> float:
        """Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            portfolio_values: Sequence of portfolio values
            returns: Sequence of returns
            
        Returns:
            Calmar ratio
        """
        max_dd = self.calculate_max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return 0.0
        
        # Calculate annualized return
        if len(returns) == 0:
            return 0.0
        
        total_return = self.calculate_total_return(portfolio_values)
        years = len(returns) / self.trading_days_per_year
        
        if years == 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return float(annualized_return / max_dd)
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Win rate as decimal (0.5 = 50%)
        """
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        
        return float(winning_trades / len(trades))
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Profit factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def calculate_average_win(self, trades: List[Dict]) -> float:
        """Calculate average winning trade.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Average win amount
        """
        if not trades:
            return 0.0
        
        winning_trades = [trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]
        
        if not winning_trades:
            return 0.0
        
        return float(np.mean(winning_trades))
    
    def calculate_average_loss(self, trades: List[Dict]) -> float:
        """Calculate average losing trade.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Average loss amount (positive value)
        """
        if not trades:
            return 0.0
        
        losing_trades = [abs(trade['pnl']) for trade in trades if trade.get('pnl', 0) < 0]
        
        if not losing_trades:
            return 0.0
        
        return float(np.mean(losing_trades))
    
    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """Calculate trade expectancy.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Expected value per trade
        """
        if not trades:
            return 0.0
        
        win_rate = self.calculate_win_rate(trades)
        avg_win = self.calculate_average_win(trades)
        avg_loss = self.calculate_average_loss(trades)
        
        return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))
    
    def calculate_volatility(self, returns: Sequence[float]) -> float:
        """Calculate annualized volatility.
        
        Args:
            returns: Sequence of returns
            
        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
        
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(self.trading_days_per_year)
        
        return float(annualized_vol)
    
    def calculate_downside_deviation(self, returns: Sequence[float]) -> float:
        """Calculate downside deviation.
        
        Args:
            returns: Sequence of returns
            
        Returns:
            Downside deviation
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return 0.0
        
        return float(np.std(negative_returns))
    
    def calculate_value_at_risk(
        self,
        returns: Sequence[float],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Sequence of returns
            confidence_level: Confidence level (default 95%)
            
        Returns:
            VaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return float(abs(var))
    
    def calculate_conditional_value_at_risk(
        self,
        returns: Sequence[float],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Sequence of returns
            confidence_level: Confidence level (default 95%)
            
        Returns:
            CVaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        var = self.calculate_value_at_risk(returns, confidence_level)
        
        # Get returns worse than VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = np.mean(tail_returns)
        
        return float(abs(cvar))
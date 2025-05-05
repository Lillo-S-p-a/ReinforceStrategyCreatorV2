import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def format_metric(value, metric_type):
    """Format metrics correctly based on their type"""
    if value is None:
        return "N/A"
    
    if metric_type == "pnl":
        return f"${value:,.2f}"
    elif metric_type == "percentage":
        # Ensure win_rate is between 0-100%
        if value > 1 and value <= 100:
            # Already in percentage format (0-100)
            return f"{value:.2f}%"
        elif value > 100:
            # Likely a calculation error, cap at 100%
            return "100.00% (capped)"
        elif value <= 1:
            # Likely in decimal format (0-1), convert to percentage
            return f"{value * 100:.2f}%"
    elif metric_type == "ratio":
        return f"{value:.3f}"
    else:
        return f"{value:,}"

def calculate_additional_metrics(steps_df: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional performance metrics not already provided by the API."""
    metrics = {}
    
    # Calculate position holding stats
    if trades:
        trades_df = pd.DataFrame(trades)
        if 'entry_time' in trades_df and 'exit_time' in trades_df and not trades_df.empty:
            trades_df['holding_time'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
            
            # Overall stats
            metrics['avg_trade_duration'] = trades_df['holding_time'].mean()
            
            # Stats by direction
            if 'direction' in trades_df and 'pnl' in trades_df:
                metrics['long_trades_count'] = len(trades_df[trades_df['direction'] == 'long'])
                metrics['short_trades_count'] = len(trades_df[trades_df['direction'] == 'short'])
                
                if metrics['long_trades_count'] > 0:
                    metrics['long_win_rate'] = 100 * len(trades_df[(trades_df['direction'] == 'long') & (trades_df['pnl'] > 0)]) / metrics['long_trades_count']
                else:
                    metrics['long_win_rate'] = 0
                
                if metrics['short_trades_count'] > 0:
                    metrics['short_win_rate'] = 100 * len(trades_df[(trades_df['direction'] == 'short') & (trades_df['pnl'] > 0)]) / metrics['short_trades_count']
                else:
                    metrics['short_win_rate'] = 0
    
    # Calculate portfolio stats from steps data
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        # Ensure portfolio_value is numeric
        steps_df['portfolio_value'] = pd.to_numeric(steps_df['portfolio_value'], errors='coerce')
        
        portfolio_values = steps_df['portfolio_value'].values
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics['volatility'] = np.std(returns) * 100  # as percentage
        
        # Sortino ratio (downside risk only)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
            metrics['sortino_ratio'] = np.mean(returns) / downside_deviation if downside_deviation != 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf')  # No negative returns
        
        # Calmar ratio (return / max drawdown)
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        if max_dd > 0:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            metrics['calmar_ratio'] = total_return / max_dd
        else:
            metrics['calmar_ratio'] = float('inf')  # No drawdown
    
    return metrics

# Constants
ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
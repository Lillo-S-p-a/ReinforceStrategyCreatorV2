import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_decision_making(steps_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze decision making patterns in the episode.
    
    Looks for:
    1. Action consistency
    2. Responsiveness to price movements
    3. Ability to capture trends
    4. Decision timing
    """
    # Check for required columns, including the new asset_price
    required_cols = ['action', 'portfolio_value', 'asset_price', 'reward']
    if steps_df.empty or not all(col in steps_df.columns for col in required_cols):
        logging.warning(f"Missing required columns for decision analysis. Found: {steps_df.columns.tolist()}")
        return {}
    
    # Check how many rows have NaN asset_price
    nan_count = steps_df['asset_price'].isna().sum()
    total_count = len(steps_df)
    
    if nan_count > 0:
        logging.info(f"Found {nan_count}/{total_count} rows with NaN asset_price values")
        
        if nan_count < total_count:
            # If we have some valid values, drop only the NaN rows
            steps_df_filtered = steps_df.dropna(subset=['asset_price'])
            logging.info(f"Dropped {nan_count} rows with NaN asset_price, {len(steps_df_filtered)} rows remaining")
            steps_df = steps_df_filtered
        else:
            # All asset_price values are NaN, log a warning but try to continue with other metrics
            logging.warning("All asset_price values are NaN. Some metrics will be unavailable.")
            # We'll continue with the full dataset and skip asset_price-dependent calculations
    
    if steps_df.empty:
        logging.warning("No valid steps data after filtering.")
        return {}

    analysis = {}
    
    # Action codes: 0=flat, 1=long, 2=short
    steps_df['action_num'] = pd.to_numeric(steps_df['action'], errors='coerce') # Already mapped in api.py, but coerce just in case

    # 1. Action consistency - how often does the model change actions?
    steps_df['action_change'] = steps_df['action_num'].diff().fillna(0).abs() > 0 # fillna(0) for first row
    action_change_rate = steps_df['action_change'].mean() * 100
    analysis['action_change_rate'] = action_change_rate
    
    # 2. Responsiveness to ASSET price movements
    # Use asset_price for more meaningful analysis, but check if we have valid data first
    if not steps_df['asset_price'].isna().all():
        # Calculate price changes only on valid asset_price values
        valid_price_df = steps_df.dropna(subset=['asset_price'])
        asset_price_changes = valid_price_df['asset_price'].pct_change()
        
        # Add the changes back to the main DataFrame, matching on index
        steps_df['asset_price_change'] = np.nan  # Initialize with NaN
        steps_df.loc[valid_price_df.index, 'asset_price_change'] = asset_price_changes
        
        # When asset price goes up, how often does agent buy (action 1)?
        valid_up_df = steps_df.dropna(subset=['asset_price_change'])
        buy_on_up = valid_up_df[(valid_up_df['asset_price_change'] > 0.0005) & (valid_up_df['action_num'] == 1)].shape[0]
        total_price_up = valid_up_df[valid_up_df['asset_price_change'] > 0.0005].shape[0]
        analysis['buy_on_asset_price_up_rate'] = (buy_on_up / total_price_up * 100) if total_price_up > 0 else 0
        
        # When asset price goes down, how often does agent sell (action 2)?
        sell_on_down = valid_up_df[(valid_up_df['asset_price_change'] < -0.0005) & (valid_up_df['action_num'] == 2)].shape[0]
        total_price_down = valid_up_df[valid_up_df['asset_price_change'] < -0.0005].shape[0]
        analysis['sell_on_asset_price_down_rate'] = (sell_on_down / total_price_down * 100) if total_price_down > 0 else 0
    else:
        logging.warning("Skipping asset price movement analysis due to missing asset_price data")
        analysis['buy_on_asset_price_up_rate'] = None
        analysis['sell_on_asset_price_down_rate'] = None
    
    # 3. Long-term trend alignment (using asset price)
    # Calculate if agent goes long (1) primarily in uptrends, short (2) in downtrends
    # Use a simple 20-period moving average on asset_price
    trend_window = 20
    
    # Check if we have enough data points and valid asset_price values
    if len(steps_df) >= trend_window and not steps_df['asset_price'].isna().all():
        # Use only rows with valid asset_price for trend calculation
        valid_price_df = steps_df.dropna(subset=['asset_price'])
        
        if len(valid_price_df) >= trend_window:
            # Calculate moving average on valid data
            valid_price_df['asset_ma'] = valid_price_df['asset_price'].rolling(window=trend_window).mean()
            valid_price_df['asset_uptrend'] = valid_price_df['asset_price'] > valid_price_df['asset_ma']
            
            # Add the trend indicators back to the main DataFrame, matching on index
            steps_df['asset_ma'] = np.nan  # Initialize with NaN
            steps_df['asset_uptrend'] = np.nan  # Initialize with NaN
            steps_df.loc[valid_price_df.index, 'asset_ma'] = valid_price_df['asset_ma']
            steps_df.loc[valid_price_df.index, 'asset_uptrend'] = valid_price_df['asset_uptrend']
            
            # Drop NaN values for the trend analysis
            trend_df = steps_df.dropna(subset=['asset_uptrend'])
            
            if not trend_df.empty:
                # In uptrends, agent buys (action 1)
                buy_in_uptrend = trend_df[(trend_df['asset_uptrend']) & (trend_df['action_num'] == 1)].shape[0]
                total_uptrend = trend_df[trend_df['asset_uptrend']].shape[0]
                analysis['buy_in_asset_uptrend_rate'] = (buy_in_uptrend / total_uptrend * 100) if total_uptrend > 0 else 0
                
                # In downtrends, agent sells (action 2)
                sell_in_downtrend = trend_df[(~trend_df['asset_uptrend']) & (trend_df['action_num'] == 2)].shape[0]
                total_downtrend = trend_df[~trend_df['asset_uptrend']].shape[0]
                analysis['sell_in_asset_downtrend_rate'] = (sell_in_downtrend / total_downtrend * 100) if total_downtrend > 0 else 0
            else:
                logging.warning("No valid trend data after filtering NaN values")
                analysis['buy_in_asset_uptrend_rate'] = None
                analysis['sell_in_asset_downtrend_rate'] = None
        else:
            logging.warning(f"Not enough valid asset_price data points for trend analysis (need {trend_window}, have {len(valid_price_df)})")
            analysis['buy_in_asset_uptrend_rate'] = None
            analysis['sell_in_asset_downtrend_rate'] = None
    else:
        if len(steps_df) < trend_window:
            logging.warning(f"Not enough data points for trend analysis (need {trend_window}, have {len(steps_df)})")
        else:
            logging.warning("No valid asset_price data for trend analysis")
        analysis['buy_in_asset_uptrend_rate'] = None
        analysis['sell_in_asset_downtrend_rate'] = None

    # 4. Decision timing (using portfolio value for future return)
    # Correlate actions with future PORTFOLIO returns to assess if decisions were profitable
    future_return_period = 5
    if len(steps_df) > future_return_period:
        steps_df['future_portfolio_return'] = steps_df['portfolio_value'].pct_change(periods=future_return_period).shift(-future_return_period)
        
        # Correlation between action (0, 1, 2) and future portfolio return
        # Ensure future_portfolio_return is numeric and handle NaNs
        valid_corr_data = steps_df[['action_num', 'future_portfolio_return']].dropna()
        if not valid_corr_data.empty and len(valid_corr_data) > 1:
             action_future_corr = valid_corr_data['action_num'].corr(valid_corr_data['future_portfolio_return'])
             analysis['action_future_return_correlation'] = action_future_corr
        else:
             analysis['action_future_return_correlation'] = None
    else:
        analysis['action_future_return_correlation'] = None

    # 5. Clustering decision contexts (using asset price features)
    cluster_features = ['asset_price', 'asset_price_change', 'action_num']
    if len(steps_df) >= 50 and all(col in steps_df.columns for col in cluster_features):
        try:
            # Create features for clustering, handle NaNs
            features_df = steps_df[cluster_features].copy()
            features_df['asset_price_change'] = features_df['asset_price_change'].fillna(0) # Fill NaN price change (e.g., first step)
            features_df = features_df.dropna() # Drop rows if asset_price or action is NaN

            if not features_df.empty:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_df)
                
                # Cluster into decision groups (e.g., 3 clusters)
                n_clusters = 3
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
                clusters = kmeans.fit_predict(features_scaled)
                
                # Add cluster labels back to the original DataFrame (handle index alignment)
                features_df['cluster'] = clusters
                steps_df['cluster'] = features_df['cluster'] # Assign based on index

                # Analyze clusters (using steps_df which includes reward)
                cluster_stats = steps_df.dropna(subset=['cluster']).groupby('cluster').agg(
                    avg_asset_price=('asset_price', 'mean'),
                    avg_asset_price_change=('asset_price_change', 'mean'),
                    typical_action=('action_num', lambda x: x.mode()[0] if not x.mode().empty else -1), # Use action_num
                    avg_reward=('reward', 'mean'),
                    count=('cluster', 'size') # Add count for context
                ).reset_index()

                # Map typical_action back to string for readability if needed (0:flat, 1:long, 2:short)
                action_map_rev = {0: 'flat', 1: 'long', 2: 'short', -1: 'unknown'}
                cluster_stats['typical_action_str'] = cluster_stats['typical_action'].map(action_map_rev)

                if not cluster_stats.empty:
                    # Find best and worst clusters based on reward
                    best_cluster = cluster_stats.loc[cluster_stats['avg_reward'].idxmax()]
                    worst_cluster = cluster_stats.loc[cluster_stats['avg_reward'].idxmin()]
                    
                    analysis['best_decision_context'] = {
                        'cluster_id': int(best_cluster['cluster']),
                        'avg_asset_price': float(best_cluster['avg_asset_price']),
                        'avg_asset_price_change': float(best_cluster['avg_asset_price_change']),
                        'typical_action': str(best_cluster['typical_action_str']), # Use mapped string
                        'avg_reward': float(best_cluster['avg_reward']),
                        'count': int(best_cluster['count'])
                    }
                    
                    analysis['worst_decision_context'] = {
                        'cluster_id': int(worst_cluster['cluster']),
                        'avg_asset_price': float(worst_cluster['avg_asset_price']),
                        'avg_asset_price_change': float(worst_cluster['avg_asset_price_change']),
                        'typical_action': str(worst_cluster['typical_action_str']), # Use mapped string
                        'avg_reward': float(worst_cluster['avg_reward']),
                        'count': int(worst_cluster['count'])
                    }
                    
                    # Calculate cluster distributions
                    cluster_counts = steps_df['cluster'].value_counts(normalize=True) * 100
                    analysis['decision_context_distribution'] = {
                        f"cluster_{i}": float(cluster_counts.get(i, 0)) for i in range(n_clusters)
                    }
                else:
                     analysis['best_decision_context'] = None
                     analysis['worst_decision_context'] = None
                     analysis['decision_context_distribution'] = None

            else: # features_df was empty after dropna
                analysis['best_decision_context'] = None
                analysis['worst_decision_context'] = None
                analysis['decision_context_distribution'] = None

        except Exception as e:
            logging.error(f"Error in clustering analysis: {e}", exc_info=True)
            analysis['best_decision_context'] = None
            analysis['worst_decision_context'] = None
            analysis['decision_context_distribution'] = None
            pass # Skip clustering if it fails

    # 6. New Metrics (Proposed)
    # Reward Volatility
    if 'reward' in steps_df.columns and not steps_df['reward'].isnull().all():
         analysis['reward_volatility'] = steps_df['reward'].std()
    else:
         analysis['reward_volatility'] = None

    # Consecutive Action Analysis
    if 'action_num' in steps_df.columns:
        steps_df['action_block'] = (steps_df['action_num'].diff() != 0).cumsum()
        action_streaks = steps_df.groupby('action_block')['action_num'].agg(['first', 'size'])
        analysis['avg_consecutive_action_duration'] = action_streaks['size'].mean()
        # Avg duration per action type
        avg_duration_per_action = action_streaks.groupby('first')['size'].mean().to_dict()
        analysis['avg_consecutive_duration_per_action'] = {
            action_map_rev.get(k, 'unknown'): v for k, v in avg_duration_per_action.items()
        }

    # Reward per Action Type
    if 'reward' in steps_df.columns and 'action_num' in steps_df.columns:
         reward_by_action = steps_df.groupby('action_num')['reward'].mean().to_dict()
         analysis['avg_reward_per_action'] = {
             action_map_rev.get(k, 'unknown'): v for k, v in reward_by_action.items()
         }

    return analysis

def analyze_why_episode_performed(
    episode_data: Dict[str, Any], 
    steps_df: pd.DataFrame, 
    trades: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze why an episode performed well or poorly.
    
    Returns a dictionary with findings and a summary.
    """
    if steps_df.empty:
        return {"summary": "Insufficient data to analyze episode performance."}
    
    findings = {}
    
    # 1. Overall performance metrics
    pnl = episode_data.get('pnl')
    win_rate = episode_data.get('win_rate')
    max_drawdown = episode_data.get('max_drawdown')
    sharpe_ratio = episode_data.get('sharpe_ratio')
    
    findings["performance_metrics"] = {
        "pnl": pnl,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio
    }
    
    # 2. Trade analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Profitable vs non-profitable trades
        if 'pnl' in trades_df.columns:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            unprofitable_trades = trades_df[trades_df['pnl'] <= 0]
            
            findings["trade_analysis"] = {
                "profitable_trade_count": len(profitable_trades),
                "unprofitable_trade_count": len(unprofitable_trades),
                "avg_profit_per_winning_trade": profitable_trades['pnl'].mean() if not profitable_trades.empty else 0,
                "avg_loss_per_losing_trade": unprofitable_trades['pnl'].mean() if not unprofitable_trades.empty else 0,
                "max_profit_trade": profitable_trades['pnl'].max() if not profitable_trades.empty else 0,
                "max_loss_trade": unprofitable_trades['pnl'].min() if not unprofitable_trades.empty else 0,
                "profit_factor": abs(profitable_trades['pnl'].sum() / unprofitable_trades['pnl'].sum()) 
                                if not unprofitable_trades.empty and unprofitable_trades['pnl'].sum() != 0 else float('inf')
            }
            
            # Direction analysis - performance in long vs short trades
            if 'direction' in trades_df.columns:
                long_trades = trades_df[trades_df['direction'] == 'long']
                short_trades = trades_df[trades_df['direction'] == 'short']
                
                findings["trade_analysis"]["long_trades_pnl"] = long_trades['pnl'].sum() if not long_trades.empty else 0
                findings["trade_analysis"]["short_trades_pnl"] = short_trades['pnl'].sum() if not short_trades.empty else 0
                findings["trade_analysis"]["long_win_rate"] = (
                    100 * len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) 
                    if not long_trades.empty else 0
                )
                findings["trade_analysis"]["short_win_rate"] = (
                    100 * len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) 
                    if not short_trades.empty else 0
                )
    
    # 3. Decision making analysis
    decision_analysis = analyze_decision_making(steps_df)
    findings["decision_making"] = decision_analysis
    
    # 4. Market adaptation
    # Check if agent adapted to changing market conditions
    if len(steps_df) > 50:
        # Divide episode into 3 equal parts
        part_size = len(steps_df) // 3
        early_steps = steps_df.iloc[:part_size]
        mid_steps = steps_df.iloc[part_size:2*part_size]
        late_steps = steps_df.iloc[2*part_size:]
        
        # Calculate action distributions in each part
        early_actions = early_steps['action'].value_counts(normalize=True)
        mid_actions = mid_steps['action'].value_counts(normalize=True)
        late_actions = late_steps['action'].value_counts(normalize=True)
        
        # Calculate Jensen-Shannon divergence between distributions (simplified)
        action_adaptation = sum(abs(late_actions.get(a, 0) - early_actions.get(a, 0)) for a in set(early_actions.index) | set(late_actions.index)) / 2
        
        findings["market_adaptation"] = {
            "action_distribution_change": action_adaptation,
            "early_reward_mean": early_steps['reward'].mean(),
            "mid_reward_mean": mid_steps['reward'].mean(), 
            "late_reward_mean": late_steps['reward'].mean(),
            "reward_improvement": late_steps['reward'].mean() - early_steps['reward'].mean()
        }
    
    # 5. Generate performance summary
    summary = generate_episode_summary(findings)
    findings["summary"] = summary
    
    return findings

def generate_episode_summary(findings: Dict[str, Any]) -> str:
    """Generate a human-readable summary of why an episode performed as it did."""
    performance = findings.get("performance_metrics", {})
    trade_analysis = findings.get("trade_analysis", {})
    decision_making = findings.get("decision_making", {})
    market_adaptation = findings.get("market_adaptation", {})
    
    pnl = performance.get("pnl")
    win_rate = performance.get("win_rate")
    sharpe = performance.get("sharpe_ratio")
    
    summary_points = []
    
    # Overall performance assessment
    if pnl is not None:
        if pnl > 0:
            summary_points.append(f"This episode was profitable with a PnL of ${pnl:.2f}.")
        else:
            summary_points.append(f"This episode was unprofitable with a PnL of ${pnl:.2f}.")
    
    # Trade analysis insights
    if trade_analysis:
        profit_factor = trade_analysis.get("profit_factor")
        if profit_factor and profit_factor != float('inf'):
            if profit_factor > 2:
                summary_points.append(f"Strong profit factor of {profit_factor:.2f} (ratio of gains to losses).")
            elif profit_factor > 1:
                summary_points.append(f"Modest profit factor of {profit_factor:.2f}.")
            else:
                summary_points.append(f"Poor profit factor of {profit_factor:.2f}, losses exceeded gains.")
        
        # Direction strengths
        long_win = trade_analysis.get("long_win_rate")
        short_win = trade_analysis.get("short_win_rate")
        if long_win is not None and short_win is not None:
            if long_win > short_win + 10:
                summary_points.append(f"Performed significantly better in long trades ({long_win:.1f}% win rate) than short trades ({short_win:.1f}% win rate).")
            elif short_win > long_win + 10:
                summary_points.append(f"Performed significantly better in short trades ({short_win:.1f}% win rate) than long trades ({long_win:.1f}% win rate).")
    
    # Decision making insights
    if decision_making:
        # Action timing correlation
        corr = decision_making.get("action_future_return_correlation")
        if corr is not None:
            if corr > 0.2:
                summary_points.append(f"Showed excellent decision timing, with actions strongly correlated with future returns (correlation: {corr:.2f}).")
            elif corr > 0.05:
                summary_points.append(f"Showed some foresight in decision timing (correlation with future returns: {corr:.2f}).")
            elif corr < -0.1:
                summary_points.append(f"Poor decision timing, with actions negatively correlated with future returns (correlation: {corr:.2f}).")
        
        # Trend alignment
        buy_uptrend = decision_making.get("buy_in_uptrend_rate")
        sell_downtrend = decision_making.get("sell_in_downtrend_rate")
        if buy_uptrend is not None and sell_downtrend is not None:
            if buy_uptrend > 60 and sell_downtrend > 60:
                summary_points.append("Excellent trend alignment - buying in uptrends and selling in downtrends.")
            elif buy_uptrend < 40 and sell_downtrend < 40:
                summary_points.append("Poor trend alignment - frequently buying in downtrends and selling in uptrends.")
    
    # Market adaptation insights
    if market_adaptation:
        reward_improvement = market_adaptation.get("reward_improvement")
        if reward_improvement is not None:
            if reward_improvement > 0.1:
                summary_points.append(f"Showed strong adaptation during the episode, with improving rewards over time.")
            elif reward_improvement < -0.1:
                summary_points.append(f"Deteriorating performance during the episode, with decreasing rewards over time.")
    
    # Best and worst decision contexts
    best_context = decision_making.get("best_decision_context", {})
    worst_context = decision_making.get("worst_decision_context", {})
    
    if best_context and worst_context:
        best_action = best_context.get("typical_action")
        best_price_change = best_context.get("avg_price_change")
        worst_action = worst_context.get("typical_action")
        worst_price_change = worst_context.get("avg_price_change")
        
        if best_action and best_price_change is not None and worst_action and worst_price_change is not None:
            best_state = "rising" if best_price_change > 0 else "falling"
            worst_state = "rising" if worst_price_change > 0 else "falling"
            
            summary_points.append(f"Performed best when taking action '{best_action}' during {best_state} prices.")
            summary_points.append(f"Performed worst when taking action '{worst_action}' during {worst_state} prices.")
    
    # Overall conclusion
    if sharpe is not None and win_rate is not None:
        if sharpe > 1.0 and win_rate > 50:
            summary_points.append("Overall, this was a successful episode with good risk-adjusted returns.")
        elif sharpe < 0 and win_rate < 50:
            summary_points.append("Overall, this was an unsuccessful episode with poor trading decisions.")
        else:
            summary_points.append("This episode showed mixed results with room for improvement.")
    
    return " ".join(summary_points)

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
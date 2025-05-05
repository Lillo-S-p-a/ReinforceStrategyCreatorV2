import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
import datetime
from typing import List, Dict, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8001/api/v1"
API_KEY = "test-key-123"
API_HEADERS = {"X-API-Key": API_KEY}
ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
MODELS_DIR = "models"
PRODUCTION_MODELS_DIR = "production_models"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRODUCTION_MODELS_DIR, exist_ok=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Helper Functions ---
@st.cache_data(ttl=60)
def fetch_api_data(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Generic function to fetch data from the API."""
    url = f"{API_BASE_URL}{endpoint}"
    log_headers = API_HEADERS.copy()
    if "X-API-Key" in log_headers:
        log_headers["X-API-Key"] = "****"

    logging.info(f"Attempting to fetch data from API endpoint: {url}")
    logging.info(f"  Headers: {log_headers}")
    logging.info(f"  Params: {params}")
    try:
        response = requests.get(url, headers=API_HEADERS, params=params, timeout=10)
        logging.info(f"  API Response Status Code: {response.status_code}")
        response_snippet = response.text[:200] + "..." if len(response.text) > 200 else response.text
        logging.info(f"  API Response Snippet: {response_snippet}")

        response.raise_for_status()
        logging.info(f"  Successfully fetched and parsed JSON for {url}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred fetching {url}: {http_err}")
        logging.error(f"  Response Body: {response.text}")
        st.error(f"API Request Error (HTTP {response.status_code}) fetching {endpoint}: Check API server logs.")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred fetching {url}: {conn_err}")
        st.error(f"API Connection Error fetching {endpoint}: Is the API server running at {API_BASE_URL}?")
        return None
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred fetching {url}: {timeout_err}")
        st.error(f"API Timeout Error fetching {endpoint}.")
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Ambiguous request error occurred fetching {url}: {req_err}")
        st.error(f"API Request Error fetching {endpoint}: {req_err}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred processing API response from {url}: {e}", exc_info=True)
        st.error(f"Error processing API response from {endpoint}: {e}")
        return None

def fetch_latest_run() -> Optional[Dict[str, Any]]:
    """Fetches the most recent training run."""
    data = fetch_api_data("/runs/", params={"page": 1, "page_size": 1})
    if data and data.get("items"):
        return data["items"][0]
    st.warning("Could not fetch latest training run.")
    return None

def fetch_run_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the episode summary for a given run."""
    return fetch_api_data(f"/runs/{run_id}/episodes/summary/")

def fetch_run_episodes(run_id: str) -> List[Dict[str, Any]]:
    """Fetches all episodes for a given run (handles pagination)."""
    episodes = []
    page = 1
    while True:
        data = fetch_api_data(f"/runs/{run_id}/episodes/", params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            break
        episodes.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    if not episodes:
        st.warning(f"No episodes found for run {run_id}.")
    return episodes

def fetch_episode_steps(episode_id: int) -> pd.DataFrame:
    """Fetches all steps for a given episode and returns a DataFrame."""
    steps_list = []
    page = 1
    while True:
        data = fetch_api_data(f"/episodes/{episode_id}/steps/", params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            if page == 1:
                 st.warning(f"No steps found for episode {episode_id}.")
            break
        steps_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1

    if not steps_list:
        return pd.DataFrame()

    df = pd.DataFrame(steps_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
    return df.sort_index()

@st.cache_data(ttl=60)
def fetch_episode_trades(episode_id: int) -> List[Dict[str, Any]]:
    """Fetches all trades for a given episode."""
    trades_list = []
    page = 1
    while True:
        endpoint = f"/episodes/{episode_id}/trades/"
        data = fetch_api_data(endpoint, params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            if page == 1:
                 logging.warning(f"No trades found in API response for episode {episode_id} on page 1.")
            break
        trades_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logging.info(f"Fetched {len(trades_list)} trades for episode {episode_id}.")
    for trade in trades_list:
        trade['entry_time'] = pd.to_datetime(trade['entry_time'])
        if trade.get('exit_time'):
            trade['exit_time'] = pd.to_datetime(trade['exit_time'])
    return sorted(trades_list, key=lambda x: x['entry_time'])

@st.cache_data(ttl=60)
def fetch_episode_operations(episode_id: int) -> List[Dict[str, Any]]:
    """Fetches all trading operations for a given episode."""
    operations_list = []
    page = 1
    logging.info(f"Fetching operations for episode {episode_id}")
    while True:
        endpoint = f"/episodes/{episode_id}/operations/"
        data = fetch_api_data(endpoint, params={"page": page, "page_size": 100})
        logging.debug(f"Raw operations data page {page} for episode {episode_id}: {data}")
        if not data or not data.get("items"):
            if page == 1:
                 logging.warning(f"No operations items found in API response for episode {episode_id} on page 1.")
            break
        operations_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logging.info(f"Fetched {len(operations_list)} operations for episode {episode_id}.")
    for op in operations_list:
        op['timestamp'] = pd.to_datetime(op['timestamp'])
    return sorted(operations_list, key=lambda x: x['timestamp'])

@st.cache_data(ttl=60)
def fetch_episode_model(episode_id: int) -> Optional[Dict[str, Any]]:
    """Fetches model parameters for a given episode."""
    # This is a placeholder - in a real system, this would fetch actual model parameters from the API
    # For now, we'll simulate model parameters
    endpoint = f"/episodes/{episode_id}/model/"
    data = fetch_api_data(endpoint)
    if not data:
        # Generate mock model data for demonstration
        return {
            "episode_id": episode_id,
            "learning_rate": 0.001 + (episode_id % 10) * 0.0001,
            "gamma": 0.95 + (episode_id % 5) * 0.01,
            "epsilon": max(0.1, 1.0 - (episode_id * 0.02)),
            "batch_size": 32 + (episode_id % 3) * 16,
            "hidden_layer_size": 64 + (episode_id % 4) * 16,
            "memory_size": 10000 + (episode_id % 5) * 1000,
            "architecture": "DQN" if episode_id % 2 == 0 else "DDQN",
            "optimizer": "Adam" if episode_id % 3 != 0 else "RMSprop",
            "loss_function": "MSE" if episode_id % 2 == 0 else "Huber",
            "exploration_strategy": "epsilon-greedy",
            "feature_extractors": ["price", "volume", "macd", "rsi"]
        }
    return data

def save_model_to_production(episode_id: int, run_id: str, model_data: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Save a model to production directory.
    Returns the path where the model was saved.
    """
    # Create a unique model filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_ep{episode_id}_run{run_id}_{timestamp}.json"
    model_path = os.path.join(PRODUCTION_MODELS_DIR, model_filename)
    
    # Combine model data with metrics
    model_info = {
        "model_data": model_data,
        "metrics": metrics,
        "saved_at": timestamp,
        "episode_id": episode_id,
        "run_id": run_id
    }
    
    # Save the model info as JSON
    with open(model_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    return model_path

def get_saved_production_models() -> List[Dict[str, Any]]:
    """Get list of saved production models with their metrics"""
    models = []
    try:
        for filename in os.listdir(PRODUCTION_MODELS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(PRODUCTION_MODELS_DIR, filename)
                with open(file_path, 'r') as f:
                    model_info = json.load(f)
                    model_info['filename'] = filename
                    models.append(model_info)
    except Exception as e:
        logging.error(f"Error reading production models: {e}")
        
    # Sort by saved timestamp, newest first
    return sorted(models, key=lambda x: x.get('saved_at', ''), reverse=True)

# --- Analysis Functions ---

def analyze_decision_making(steps_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze decision making patterns in the episode.
    
    Looks for:
    1. Action consistency
    2. Responsiveness to price movements
    3. Ability to capture trends
    4. Decision timing
    """
    if steps_df.empty or 'action' not in steps_df.columns or 'portfolio_value' not in steps_df.columns:
        return {}
    
    analysis = {}
    
    # 1. Action consistency - how often does the model change actions?
    steps_df['action_change'] = steps_df['action'].diff().abs() > 0
    action_change_rate = steps_df['action_change'].mean() * 100
    analysis['action_change_rate'] = action_change_rate
    
    # 2. Responsiveness to price movements
    price_changes = steps_df['portfolio_value'].pct_change()
    steps_df['price_change'] = price_changes
    
    # When price goes up, how often does agent buy?
    buy_on_up = steps_df[(steps_df['price_change'] > 0.0005) & (steps_df['action'] == '1')].shape[0]
    total_price_up = steps_df[steps_df['price_change'] > 0.0005].shape[0]
    analysis['buy_on_price_up_rate'] = (buy_on_up / total_price_up * 100) if total_price_up > 0 else 0
    
    # When price goes down, how often does agent sell?
    sell_on_down = steps_df[(steps_df['price_change'] < -0.0005) & (steps_df['action'] == '2')].shape[0]
    total_price_down = steps_df[steps_df['price_change'] < -0.0005].shape[0]
    analysis['sell_on_price_down_rate'] = (sell_on_down / total_price_down * 100) if total_price_down > 0 else 0
    
    # 3. Long-term trend alignment
    # Calculate if agent goes long primarily in uptrends
    # Use a simple 20-period moving average to determine trend
    if len(steps_df) >= 20:
        steps_df['ma20'] = steps_df['portfolio_value'].rolling(window=20).mean()
        steps_df['uptrend'] = steps_df['portfolio_value'] > steps_df['ma20']
        
        # In uptrends, agent buys
        buy_in_uptrend = steps_df[(steps_df['uptrend']) & (steps_df['action'] == '1')].shape[0]
        total_uptrend = steps_df[steps_df['uptrend']].shape[0]
        analysis['buy_in_uptrend_rate'] = (buy_in_uptrend / total_uptrend * 100) if total_uptrend > 0 else 0
        
        # In downtrends, agent sells
        sell_in_downtrend = steps_df[(~steps_df['uptrend']) & (steps_df['action'] == '2')].shape[0]
        total_downtrend = steps_df[~steps_df['uptrend']].shape[0]
        analysis['sell_in_downtrend_rate'] = (sell_in_downtrend / total_downtrend * 100) if total_downtrend > 0 else 0
    
    # 4. Decision timing
    # Correlate actions with future returns to assess if decisions were timely
    if len(steps_df) > 5:
        # Map string action values to integers for correlation
        steps_df['action_num'] = pd.to_numeric(steps_df['action'], errors='coerce')
        
        # Calculate future returns (5 steps ahead)
        steps_df['future_return'] = steps_df['portfolio_value'].pct_change(periods=5).shift(-5)
        
        # Correlation between action and future return
        action_future_corr = steps_df['action_num'].corr(steps_df['future_return'])
        analysis['action_future_return_correlation'] = action_future_corr
    
    # 5. Clustering decision contexts (advanced)
    if len(steps_df) >= 50 and 'portfolio_value' in steps_df.columns and 'action' in steps_df.columns:
        try:
            # Create features for clustering
            features = pd.DataFrame({
                'price': steps_df['portfolio_value'],
                'price_change': steps_df['price_change'].fillna(0),
                'action': pd.to_numeric(steps_df['action'], errors='coerce')
            })
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features.fillna(0))
            
            # Cluster into decision groups (3 clusters)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Analyze clusters
            steps_df['cluster'] = clusters
            cluster_stats = steps_df.groupby('cluster').agg({
                'portfolio_value': 'mean',
                'price_change': 'mean', 
                'action': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
                'reward': 'mean'
            }).reset_index()
            
            # Find best and worst clusters based on reward
            best_cluster = cluster_stats.loc[cluster_stats['reward'].idxmax()]
            worst_cluster = cluster_stats.loc[cluster_stats['reward'].idxmin()]
            
            analysis['best_decision_context'] = {
                'cluster_id': int(best_cluster['cluster']),
                'avg_price': float(best_cluster['portfolio_value']),
                'avg_price_change': float(best_cluster['price_change']),
                'typical_action': str(best_cluster['action']),
                'avg_reward': float(best_cluster['reward'])
            }
            
            analysis['worst_decision_context'] = {
                'cluster_id': int(worst_cluster['cluster']),
                'avg_price': float(worst_cluster['portfolio_value']),
                'avg_price_change': float(worst_cluster['price_change']),
                'typical_action': str(worst_cluster['action']),
                'avg_reward': float(worst_cluster['reward'])
            }
            
            # Calculate cluster distributions
            cluster_counts = steps_df['cluster'].value_counts(normalize=True) * 100
            analysis['decision_context_distribution'] = {
                f"cluster_{i}": float(cluster_counts.get(i, 0)) for i in range(3)
            }
            
        except Exception as e:
            logging.error(f"Error in clustering analysis: {e}")
            # Skip clustering if it fails
            pass
    
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

# --- Plotting Functions ---

def create_price_operations_chart(steps_df: pd.DataFrame, operations: List[Dict[str, Any]], template="plotly_dark") -> tuple[go.Figure, bool]:
    """Creates a Plotly chart showing portfolio value and trading operations."""
    fig = go.Figure()

    # 1. Add Portfolio Value Line
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        fig.add_trace(go.Scatter(
            x=steps_df.index,
            y=steps_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue')
        ))
    else:
        logging.warning("Steps DataFrame is empty or missing 'portfolio_value', cannot plot main line.")

    # 2. Prepare data for markers
    marker_data = {
        'ENTRY_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'triangle-up', 'size': 10},
        'EXIT_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'x', 'size': 12},
        'ENTRY_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'triangle-down', 'size': 10},
        'EXIT_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'x', 'size': 12},
    }

    # Create a dictionary to quickly look up portfolio values by timestamp
    portfolio_values = {}
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        for timestamp, row in steps_df.iterrows():
            portfolio_values[timestamp] = row['portfolio_value']

    for op in operations:
        op_type = op.get('operation_type')
        if op_type in marker_data:
            timestamp = op.get('timestamp')
            price = op.get('price')
            size = op.get('size')
            
            if timestamp is not None and price is not None:
                # Find the closest timestamp in steps_df to get the portfolio value
                portfolio_value = None
                
                # Exact match
                if timestamp in portfolio_values:
                    portfolio_value = portfolio_values[timestamp]
                else:
                    # Find the closest timestamp
                    closest_timestamp = min(portfolio_values.keys(), key=lambda x: abs(x - timestamp), default=None)
                    if closest_timestamp is not None:
                        portfolio_value = portfolio_values[closest_timestamp]
                
                # Use portfolio value for y-coordinate if available, otherwise use price
                y_value = portfolio_value if portfolio_value is not None else price
                
                marker_data[op_type]['x'].append(timestamp)
                marker_data[op_type]['y'].append(y_value)
                hover_text = f"Type: {op_type}<br>Price: {price:.2f}<br>Size: {size or 'N/A'}<br>Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                marker_data[op_type]['text'].append(hover_text)

    # 3. Add Marker Traces
    markers_plotted = False
    for op_type, data in marker_data.items():
        if data['x']:
            markers_plotted = True
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(
                    color=data['color'],
                    symbol=data['symbol'],
                    size=data['size']
                ),
                name=op_type,
                text=data['text'],
                hoverinfo='text'
            ))

    # 4. Customize Layout
    fig.update_layout(
        title=f"Portfolio Value and Trading Operations",
        xaxis_title="Time",
        yaxis_title="Value / Price",
        hovermode="x unified",
        legend_title="Trace Type",
        height=500,
        template=template
    )

    return fig, markers_plotted

def create_drawdown_chart(steps_df: pd.DataFrame, template="plotly_dark") -> go.Figure:
    """Creates a Drawdown chart based on portfolio values."""
    if steps_df.empty or 'portfolio_value' not in steps_df.columns:
        return None
    
    # Calculate drawdowns
    portfolio_values = steps_df['portfolio_value'].values
    peak = portfolio_values[0]
    drawdowns = []
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown_pct = (peak - value) / peak * 100 if peak > 0 else 0
        drawdowns.append(drawdown_pct)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps_df.index,
        y=drawdowns,
        fill='tozeroy',
        mode='lines',
        line=dict(color='red'),
        name='Drawdown (%)'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        yaxis=dict(tickformat='.2f', ticksuffix='%'),
        height=400,
        template=template
    )
    
    return fig

def create_trade_analysis_charts(trades: List[Dict[str, Any]], template="plotly_dark") -> tuple[go.Figure, go.Figure, go.Figure]:
    """Creates trade analysis charts: PnL distribution, holding time, and cumulative PnL."""
    
    if not trades:
        return None, None, None
    
    # Convert to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate holding time in minutes
    if 'entry_time' in trades_df and 'exit_time' in trades_df:
        trades_df['holding_time'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
    
    # 1. PnL Distribution Chart
    fig_pnl_dist = px.histogram(
        trades_df, 
        x="pnl", 
        title="Trade PnL Distribution",
        labels={"pnl": "Profit/Loss ($)", "count": "Number of Trades"},
        color_discrete_sequence=['teal'],
        template=template
    )
    fig_pnl_dist.update_layout(height=350)
    
    # 2. Holding Time vs PnL Chart
    fig_holding_pnl = px.scatter(
        trades_df,
        x="holding_time",
        y="pnl",
        color="direction",
        title="Holding Time vs PnL",
        labels={"holding_time": "Holding Time (minutes)", "pnl": "Profit/Loss ($)", "direction": "Trade Direction"},
        template=template
    )
    fig_holding_pnl.update_layout(height=350)
    
    # 3. Cumulative PnL Chart
    trades_df = trades_df.sort_values('exit_time')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    fig_cum_pnl = px.line(
        trades_df,
        x="exit_time",
        y="cumulative_pnl",
        title="Cumulative PnL Over Time",
        labels={"exit_time": "Time", "cumulative_pnl": "Cumulative P&L ($)"},
        template=template
    )
    fig_cum_pnl.update_layout(height=350)
    
    return fig_pnl_dist, fig_holding_pnl, fig_cum_pnl

def create_action_analysis(steps_df: pd.DataFrame, template="plotly_dark") -> tuple[go.Figure, go.Figure]:
    """Create action distribution and action transition charts."""
    
    if steps_df.empty or 'action' not in steps_df.columns:
        return None, None
    
    # Action distribution
    action_counts = steps_df['action'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']
    
    fig_action_dist = px.pie(
        action_counts, 
        names='action', 
        values='count',
        title='Action Distribution',
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        template=template
    )
    fig_action_dist.update_layout(height=350)
    
    # Action transitions (from one action to the next)
    steps_df['next_action'] = steps_df['action'].shift(-1)
    transitions = steps_df.dropna(subset=['next_action']).groupby(['action', 'next_action']).size().reset_index()
    transitions.columns = ['From', 'To', 'Count']
    
    # Converting to strings for better display
    transitions['From'] = transitions['From'].astype(str)
    transitions['To'] = transitions['To'].astype(str)
    
    # Create Sankey diagram for transitions
    fig_transitions = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(set(transitions['From'].tolist() + transitions['To'].tolist())),
            color="blue"
        ),
        link=dict(
            source=[list(set(transitions['From'].tolist() + transitions['To'].tolist())).index(x) for x in transitions['From']],
            target=[list(set(transitions['From'].tolist() + transitions['To'].tolist())).index(x) for x in transitions['To']],
            value=transitions['Count'],
        )
    )])
    
    fig_transitions.update_layout(
        title_text="Action Transitions",
        height=350,
        template=template
    )
    
    return fig_action_dist, fig_transitions

def create_reward_analysis(steps_df: pd.DataFrame, template="plotly_dark") -> tuple[go.Figure, go.Figure]:
    """Create reward distribution and running avg reward charts."""
    
    if steps_df.empty or 'reward' not in steps_df.columns:
        return None, None
    
    # Reward distribution histogram
    fig_reward_dist = px.histogram(
        steps_df, 
        x="reward", 
        title="Reward Distribution",
        labels={"reward": "Reward", "count": "Frequency"},
        color_discrete_sequence=['goldenrod'],
        template=template
    )
    fig_reward_dist.update_layout(height=350)
    
    # Running average reward
    steps_df['reward_cumsum'] = steps_df['reward'].cumsum()
    steps_df['step_number'] = range(1, len(steps_df) + 1)
    steps_df['running_avg_reward'] = steps_df['reward_cumsum'] / steps_df['step_number']
    
    fig_running_reward = px.line(
        steps_df,
        x=steps_df.index,
        y="running_avg_reward",
        title="Running Average Reward",
        labels={"index": "Time", "running_avg_reward": "Running Avg Reward"},
        template=template
    )
    fig_running_reward.update_layout(height=350)
    
    return fig_reward_dist, fig_running_reward

def create_model_parameter_radar(model_data: Dict[str, Any], template="plotly_dark") -> go.Figure:
    """Create a radar chart visualizing model parameters."""
    
    # Extract numeric parameters
    params = {}
    max_values = {
        "learning_rate": 0.01,
        "gamma": 1.0,
        "epsilon": 1.0,
        "batch_size": 128,
        "hidden_layer_size": 256,
        "memory_size": 20000
    }
    
    for param, max_val in max_values.items():
        if param in model_data:
            # Normalize to 0-1 scale
            params[param] = min(model_data[param] / max_val, 1.0)
    
    # Create radar chart data
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[params.get(p, 0) for p in max_values.keys()],
        theta=list(max_values.keys()),
        fill='toself',
        name='Model Parameters',
        line_color='#3366cc'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Parameter Profile",
        height=400,
        template=template
    )
    
    return fig

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

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Enhanced RL Trading Dashboard v2", page_icon="📈")
st.title("📈 Enhanced RL Trading Agent Performance Dashboard")

# Dashboard sidebar for config options
st.sidebar.title("Configuration")
theme = st.sidebar.selectbox("Dashboard Theme", ["Dark", "Light"], index=0)
plot_template = "plotly_dark" if theme == "Dark" else "plotly_white"

# Fetch Latest Run Info
latest_run = fetch_latest_run()

if latest_run:
    run_id = latest_run.get("run_id")
    st.sidebar.info(f"Displaying data for Run ID: {run_id}")
    with st.sidebar.expander("Run Parameters", expanded=False):
        st.json(latest_run.get("parameters", {}))

    # --- Production Model Management ---
    with st.sidebar.expander("Production Model Management", expanded=True):
        production_models = get_saved_production_models()
        st.write(f"**{len(production_models)} models saved for production**")
        
        if production_models:
            # Display a table of top 3 saved models
            model_table = []
            for i, model in enumerate(production_models[:3]):
                metrics = model.get('metrics', {})
                model_table.append({
                    "Filename": model.get('filename', 'Unknown'),
                    "Episode": model.get('episode_id', 'Unknown'),
                    "PnL": metrics.get('pnl', 'N/A'),
                    "Win Rate": metrics.get('win_rate', 'N/A'),
                    "Saved At": model.get('saved_at', 'Unknown')
                })
            
            st.dataframe(pd.DataFrame(model_table))

    # --- Overall Summary ---
    st.header("📊 Overall Run Summary")
    run_summary = fetch_run_summary(run_id)

    if run_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Episodes", format_metric(run_summary.get('total_episodes'), "count"))
        col2.metric("Avg PnL", format_metric(run_summary.get('average_pnl'), "pnl"))
        col3.metric("Avg Sharpe Ratio", format_metric(run_summary.get('average_sharpe_ratio'), "ratio"))
        col4.metric("Avg Win Rate", format_metric(run_summary.get('average_win_rate'), "percentage"))
        
        # Add explanation for metrics
        with st.expander("📊 Metrics Explanation"):
            st.markdown("""
            ### Metrics Explanation
            
            - **Avg PnL**: Average Profit and Loss across all episodes. This is calculated as the average difference between final and initial portfolio values.
            - **Avg Sharpe Ratio**: Average risk-adjusted return measure. Higher values indicate better risk-adjusted performance.
            - **Avg Win Rate**: Average percentage of profitable trades across all episodes. A value between 0-100%.
            """)
    else:
        st.warning("Could not fetch run summary data.")

    # --- Fetch Episodes for Slider and Overall Plots ---
    episodes = fetch_run_episodes(run_id)

    if episodes:
        # Create a DataFrame for easier handling
        episodes_df = pd.DataFrame(episodes)
        episodes_df['episode_id'] = pd.to_numeric(episodes_df['episode_id'])
        episodes_df = episodes_df.sort_values('episode_id').set_index('episode_id')

        # --- Episode Selector ---
        st.header("🔬 Episode Details")
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            min_ep = episodes_df.index.min()
            max_ep = episodes_df.index.max()
            selected_episode_id = st.slider(
                "Select Episode ID:",
                min_value=min_ep,
                max_value=max_ep,
                value=max_ep # Default to last episode
            )
        
        with col2:
            show_advanced = st.checkbox("Show Advanced Analysis", value=True)
        
        with col3:
            save_model_button = st.button("🔄 Save Model to Production")

        # Fetch and display data for selected episode
        episode_steps_df = fetch_episode_steps(selected_episode_id)
        episode_trades = fetch_episode_trades(selected_episode_id)
        episode_operations = fetch_episode_operations(selected_episode_id)
        episode_model = fetch_episode_model(selected_episode_id)

        if not episode_steps_df.empty:
            # Get the selected episode data
            selected_episode = episodes_df.loc[selected_episode_id]
            
            # Handle model saving
            if save_model_button:
                metrics = {
                    "pnl": selected_episode.get('pnl'),
                    "win_rate": selected_episode.get('win_rate'),
                    "sharpe_ratio": selected_episode.get('sharpe_ratio'),
                    "max_drawdown": selected_episode.get('max_drawdown')
                }
                
                saved_path = save_model_to_production(selected_episode_id, run_id, episode_model, metrics)
                st.success(f"✅ Model for episode {selected_episode_id} saved to {saved_path}!")
            
            # Display episode metrics
            st.subheader(f"Performance for Episode {selected_episode_id}")
            
            # Calculate additional metrics
            additional_metrics = calculate_additional_metrics(episode_steps_df, episode_trades)
            
            # Create two rows of metrics for better organization
            ep_row1_col1, ep_row1_col2, ep_row1_col3 = st.columns(3)
            ep_row2_col1, ep_row2_col2, ep_row2_col3 = st.columns(3)
            
            # Format episode metrics
            initial_value = selected_episode.get('initial_portfolio_value')
            final_value = selected_episode.get('final_portfolio_value')
            pnl = selected_episode.get('pnl')
            sharpe_ratio = selected_episode.get('sharpe_ratio')
            max_drawdown = selected_episode.get('max_drawdown')
            win_rate = selected_episode.get('win_rate')
            
            # First row of metrics
            ep_row1_col1.metric("Initial Value", format_metric(initial_value, "pnl"))
            ep_row1_col2.metric("Final Value", format_metric(final_value, "pnl"))
            ep_row1_col3.metric("PnL", format_metric(pnl, "pnl"))
            
            # Second row of metrics
            ep_row2_col1.metric("Sharpe Ratio", format_metric(sharpe_ratio, "ratio"))
            ep_row2_col2.metric("Max Drawdown", format_metric(max_drawdown, "percentage"))
            ep_row2_col3.metric("Win Rate", format_metric(win_rate, "percentage"))

            # --- Performance Analysis: WHY did it perform well/poorly ---
            st.subheader("🔎 Performance Analysis")
            performance_analysis = analyze_why_episode_performed(selected_episode, episode_steps_df, episode_trades)
            
            # Display the performance summary
            st.info(f"**Analysis Summary:** {performance_analysis.get('summary', 'Analysis not available.')}")
            
            # Detailed analysis in expandable sections
            with st.expander("📈 Detailed Decision Analysis", expanded=False):
                decision_making = performance_analysis.get("decision_making", {})
                if decision_making:
                    # Display key decision metrics in columns
                    dm_col1, dm_col2, dm_col3 = st.columns(3)
                    
                    # Action change rate - how often the agent changes its mind
                    change_rate = decision_making.get("action_change_rate")
                    if change_rate is not None:
                        dm_col1.metric("Action Change Rate", f"{change_rate:.1f}%")
                        if change_rate > 50:
                            dm_col1.caption("⚠️ High change rate may indicate indecision")
                        else:
                            dm_col1.caption("✅ Lower change rate suggests consistent strategy")
                    
                    # Timing correlation - do actions correlate with future returns?
                    timing_corr = decision_making.get("action_future_return_correlation")
                    if timing_corr is not None:
                        dm_col2.metric("Action-Future Return Correlation", f"{timing_corr:.3f}")
                        if timing_corr > 0.1:
                            dm_col2.caption("✅ Positive correlation suggests good timing")
                        elif timing_corr < -0.1:
                            dm_col2.caption("❌ Negative correlation suggests poor timing")
                        else:
                            dm_col2.caption("⚠️ Near-zero correlation suggests random timing")
                    
                    # Trend alignment - buying in uptrends, selling in downtrends
                    buy_uptrend = decision_making.get("buy_in_uptrend_rate")
                    sell_downtrend = decision_making.get("sell_in_downtrend_rate")
                    if buy_uptrend is not None and sell_downtrend is not None:
                        avg_trend_alignment = (buy_uptrend + sell_downtrend) / 2
                        dm_col3.metric("Trend Alignment", f"{avg_trend_alignment:.1f}%")
                        if avg_trend_alignment > 60:
                            dm_col3.caption("✅ Good trend alignment")
                        elif avg_trend_alignment < 40:
                            dm_col3.caption("❌ Poor trend alignment")
                        else:
                            dm_col3.caption("⚠️ Mixed trend alignment")
                    
                    # Decision contexts - best and worst situations for the agent
                    st.subheader("Decision Contexts")
                    context_col1, context_col2 = st.columns(2)
                    
                    best_context = decision_making.get("best_decision_context")
                    if best_context:
                        with context_col1:
                            st.write("**Best Performing Context:**")
                            st.write(f"- Action: {best_context.get('typical_action', 'Unknown')}")
                            st.write(f"- Avg Price Change: {best_context.get('avg_price_change', 0)*100:.2f}%")
                            st.write(f"- Avg Reward: {best_context.get('avg_reward', 0):.3f}")
                    
                    worst_context = decision_making.get("worst_decision_context")
                    if worst_context:
                        with context_col2:
                            st.write("**Worst Performing Context:**")
                            st.write(f"- Action: {worst_context.get('typical_action', 'Unknown')}")
                            st.write(f"- Avg Price Change: {worst_context.get('avg_price_change', 0)*100:.2f}%")
                            st.write(f"- Avg Reward: {worst_context.get('avg_reward', 0):.3f}")
                else:
                    st.write("No decision analysis available.")
            
            with st.expander("📊 Trade Analysis", expanded=False):
                trade_analysis = performance_analysis.get("trade_analysis", {})
                if trade_analysis:
                    # Display key trade metrics
                    st.write("**Trade Performance:**")
                    trade_col1, trade_col2, trade_col3 = st.columns(3)
                    
                    # Win/loss counts
                    win_count = trade_analysis.get("profitable_trade_count", 0)
                    loss_count = trade_analysis.get("unprofitable_trade_count", 0)
                    total_trades = win_count + loss_count
                    
                    trade_col1.metric("Win/Loss Ratio", f"{win_count}/{loss_count}")
                    trade_col1.caption(f"Total trades: {total_trades}")
                    
                    # Average profit and loss
                    avg_profit = trade_analysis.get("avg_profit_per_winning_trade", 0)
                    avg_loss = trade_analysis.get("avg_loss_per_losing_trade", 0)
                    
                    trade_col2.metric("Avg Profit/Trade", f"${avg_profit:.2f}")
                    trade_col2.caption(f"Avg Loss/Trade: ${avg_loss:.2f}")
                    
                    # Profit factor
                    profit_factor = trade_analysis.get("profit_factor", 0)
                    
                    trade_col3.metric("Profit Factor", f"{profit_factor:.2f}")
                    if profit_factor > 2:
                        trade_col3.caption("✅ Excellent profit factor (>2.0)")
                    elif profit_factor > 1.5:
                        trade_col3.caption("✓ Good profit factor (>1.5)")
                    elif profit_factor > 1.0:
                        trade_col3.caption("⚠️ Marginal profit factor")
                    else:
                        trade_col3.caption("❌ Poor profit factor (<1.0)")
                    
                    # Direction performance
                    st.write("**Performance by Direction:**")
                    dir_col1, dir_col2 = st.columns(2)
                    
                    long_pnl = trade_analysis.get("long_trades_pnl", 0)
                    long_win = trade_analysis.get("long_win_rate", 0)
                    
                    dir_col1.metric("Long Trades PnL", f"${long_pnl:.2f}")
                    dir_col1.caption(f"Win Rate: {long_win:.1f}%")
                    
                    short_pnl = trade_analysis.get("short_trades_pnl", 0)
                    short_win = trade_analysis.get("short_win_rate", 0)
                    
                    dir_col2.metric("Short Trades PnL", f"${short_pnl:.2f}")
                    dir_col2.caption(f"Win Rate: {short_win:.1f}%")
                else:
                    st.write("No trade analysis available.")
            
            with st.expander("🧠 Model Analysis", expanded=False):
                if episode_model:
                    model_col1, model_col2 = st.columns(2)
                    
                    with model_col1:
                        st.write("**Model Parameters:**")
                        for param, value in episode_model.items():
                            if param != 'episode_id' and param != 'feature_extractors':
                                st.write(f"- **{param}:** {value}")
                        
                        if 'feature_extractors' in episode_model:
                            st.write("**Feature Extractors:**")
                            for feature in episode_model['feature_extractors']:
                                st.write(f"- {feature}")
                    
                    with model_col2:
                        radar_chart = create_model_parameter_radar(episode_model, template=plot_template)
                        if radar_chart:
                            st.plotly_chart(radar_chart, use_container_width=True)
                    
                    # Performance recommendations
                    st.write("**Model Recommendations:**")
                    
                    # Generate recommendations based on performance
                    recommendations = []
                    
                    if selected_episode.get('sharpe_ratio', 0) < 0.5:
                        if episode_model.get('gamma', 0) < 0.9:
                            recommendations.append("Consider increasing gamma (discount factor) to prioritize long-term rewards.")
                        
                        if episode_model.get('learning_rate', 0) > 0.005:
                            recommendations.append("Consider decreasing learning rate for more stable learning.")
                    
                    if selected_episode.get('max_drawdown', 0) > 20:
                        recommendations.append("Drawdown is high. Consider adding risk management features or implementing early stopping.")
                    
                    if trade_analysis := performance_analysis.get("trade_analysis", {}):
                        if trade_analysis.get("profit_factor", 0) < 1.2:
                            recommendations.append("Poor profit factor. Consider adjusting reward function to emphasize winning trades.")
                    
                    if decision_making := performance_analysis.get("decision_making", {}):
                        if decision_making.get("action_change_rate", 0) > 50:
                            recommendations.append("High action change rate. Consider increasing batch size or memory size for more consistent decisions.")
                    
                    if not recommendations:
                        recommendations.append("Model is performing well. Consider increasing batch size or reducing learning rate further to improve stability.")
                    
                    for i, recommendation in enumerate(recommendations, 1):
                        st.write(f"{i}. {recommendation}")
                else:
                    st.write("No model data available.")
            
            with st.expander("🔄 Market Adaptation", expanded=False):
                market_adaptation = performance_analysis.get("market_adaptation", {})
                if market_adaptation:
                    # Action distribution change
                    adapt_col1, adapt_col2 = st.columns(2)
                    
                    action_change = market_adaptation.get("action_distribution_change", 0)
                    
                    adapt_col1.metric("Action Distribution Change", f"{action_change:.2f}")
                    if action_change > 0.4:
                        adapt_col1.caption("✅ High adaptation to market conditions")
                    elif action_change < 0.1:
                        adapt_col1.caption("⚠️ Low adaptation to changing conditions")
                    
                    # Reward improvement
                    reward_imp = market_adaptation.get("reward_improvement", 0)
                    
                    adapt_col2.metric("Reward Improvement", f"{reward_imp:.3f}")
                    if reward_imp > 0.1:
                        adapt_col2.caption("✅ Improved rewards over time")
                    elif reward_imp < -0.1:
                        adapt_col2.caption("❌ Degrading rewards over time")
                    
                    # Reward over time
                    reward_data = {
                        "Episode Section": ["Early", "Middle", "Late"],
                        "Average Reward": [
                            market_adaptation.get("early_reward_mean", 0),
                            market_adaptation.get("mid_reward_mean", 0),
                            market_adaptation.get("late_reward_mean", 0)
                        ]
                    }
                    
                    fig_reward_progress = px.bar(
                        reward_data,
                        x="Episode Section",
                        y="Average Reward",
                        title="Reward Progression",
                        template=plot_template
                    )
                    
                    st.plotly_chart(fig_reward_progress, use_container_width=True)
                else:
                    st.write("No market adaptation analysis available.")

            # --- Advanced Metrics ---
            if show_advanced and additional_metrics:
                st.subheader("Advanced Performance Metrics")
                adv_row1_col1, adv_row1_col2, adv_row1_col3, adv_row1_col4 = st.columns(4)
                
                # Format additional metrics
                sortino = additional_metrics.get('sortino_ratio')
                calmar = additional_metrics.get('calmar_ratio')
                volatility = additional_metrics.get('volatility')
                avg_trade_dur = additional_metrics.get('avg_trade_duration')
                
                adv_row1_col1.metric("Sortino Ratio", format_metric(sortino, "ratio"))
                adv_row1_col2.metric("Calmar Ratio", format_metric(calmar, "ratio"))
                adv_row1_col3.metric("Volatility", format_metric(volatility, "percentage"))
                adv_row1_col4.metric("Avg Trade Duration", f"{avg_trade_dur:.1f} min" if avg_trade_dur else "N/A")
                
                adv_row2_col1, adv_row2_col2, adv_row2_col3, adv_row2_col4 = st.columns(4)
                
                long_trades = additional_metrics.get('long_trades_count', 0)
                short_trades = additional_metrics.get('short_trades_count', 0)
                long_win_rate = additional_metrics.get('long_win_rate', 0)
                short_win_rate = additional_metrics.get('short_win_rate', 0)
                
                adv_row2_col1.metric("Long Trades", str(long_trades))
                adv_row2_col2.metric("Short Trades", str(short_trades))
                adv_row2_col3.metric("Long Win Rate", format_metric(long_win_rate, "percentage"))
                adv_row2_col4.metric("Short Win Rate", format_metric(short_win_rate, "percentage"))

            # --- Episode-Specific Plots ---
            st.subheader("Portfolio Performance")
            
            # Portfolio Value Chart with Trading Operations
            price_ops_fig, markers_were_plotted = create_price_operations_chart(episode_steps_df, episode_operations, template=plot_template)
            st.plotly_chart(price_ops_fig, use_container_width=True)
            
            # Add a note based on why markers might be missing
            if not episode_operations:
                st.caption(f"Note: No trading operations were found/returned by the API for episode {selected_episode_id}.")
            elif not markers_were_plotted:
                st.caption(f"Note: Trading operations data was found, but markers could not be plotted for episode {selected_episode_id} (check data format or logs).")
            
            # Drawdown Chart
            drawdown_fig = create_drawdown_chart(episode_steps_df, template=plot_template)
            if drawdown_fig:
                st.plotly_chart(drawdown_fig, use_container_width=True)

            # --- Trade Analysis ---
            if episode_trades:
                st.subheader("Trade Analysis")
                
                fig_pnl_dist, fig_holding_pnl, fig_cum_pnl = create_trade_analysis_charts(episode_trades, template=plot_template)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pnl_dist, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_cum_pnl, use_container_width=True)
                
                st.plotly_chart(fig_holding_pnl, use_container_width=True)
            
            # --- Action Analysis ---
            st.subheader("Action Analysis")
            
            # Actions over time
            st.plotly_chart(px.line(episode_steps_df, y='action', title="Actions Over Time", 
                                   labels={"action": "Action", "index": "Time"},
                                   template=plot_template), use_container_width=True)
            
            # Action distribution and transitions
            fig_action_dist, fig_transitions = create_action_analysis(episode_steps_df, template=plot_template)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_action_dist, use_container_width=True)
            with col2:
                st.plotly_chart(fig_transitions, use_container_width=True)
            
            # --- Reward Analysis ---
            st.subheader("Reward Analysis")
            
            # Rewards over time
            st.plotly_chart(px.line(episode_steps_df, y='reward', title="Rewards Over Time",
                                   labels={"reward": "Reward", "index": "Time"},
                                   template=plot_template), use_container_width=True)
            
            # Reward distribution and running average
            fig_reward_dist, fig_running_reward = create_reward_analysis(episode_steps_df, template=plot_template)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_reward_dist, use_container_width=True)
            with col2:
                st.plotly_chart(fig_running_reward, use_container_width=True)
        
        else:
            st.warning(f"No step data found for episode {selected_episode_id}.")

        # --- Overall Trends ---
        st.header("📈 Overall Trends Across Episodes")
        
        # PnL and Sharpe Trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(px.line(episodes_df, y='final_portfolio_value', title="Final Portfolio Value by Episode",
                                   labels={"final_portfolio_value": "Final Portfolio Value ($)", "episode_id": "Episode ID"},
                                   template=plot_template), use_container_width=True)
        
        with col2:
            st.plotly_chart(px.line(episodes_df, y='sharpe_ratio', title="Sharpe Ratio by Episode",
                                   labels={"sharpe_ratio": "Sharpe Ratio", "episode_id": "Episode ID"},
                                   template=plot_template), use_container_width=True)
        
        # Win Rate and Max Drawdown Trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(px.line(episodes_df, y='win_rate', title="Win Rate by Episode",
                                   labels={"win_rate": "Win Rate (%)", "episode_id": "Episode ID"},
                                   template=plot_template), use_container_width=True)
        
        with col2:
            st.plotly_chart(px.line(episodes_df, y='max_drawdown', title="Max Drawdown by Episode",
                                   labels={"max_drawdown": "Max Drawdown (%)", "episode_id": "Episode ID"},
                                   template=plot_template), use_container_width=True)

    else:
        st.warning("Could not load episode data for this run.")

else:
    st.error("Failed to fetch initial run data. Cannot display dashboard.")

# --- Footer ---
st.markdown("---")
st.caption("Enhanced RL Trading Performance Dashboard v2 | Includes Performance Analysis, Model Explanations and Production Model Management")
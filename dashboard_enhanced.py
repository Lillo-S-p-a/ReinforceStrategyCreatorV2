import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from typing import List, Dict, Optional, Any

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8001/api/v1"
API_KEY = "test-key-123"
API_HEADERS = {"X-API-Key": API_KEY}
ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

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

# --- Plotting Functions ---
def create_price_operations_chart(steps_df: pd.DataFrame, operations: List[Dict[str, Any]]) -> tuple[go.Figure, bool]:
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
        template="plotly_dark"
    )

    return fig, markers_plotted

def create_drawdown_chart(steps_df: pd.DataFrame) -> go.Figure:
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
        template="plotly_dark"
    )
    
    return fig

def create_trade_analysis_charts(trades: List[Dict[str, Any]]) -> tuple[go.Figure, go.Figure, go.Figure]:
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
        template="plotly_dark"
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
        template="plotly_dark"
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
        template="plotly_dark"
    )
    fig_cum_pnl.update_layout(height=350)
    
    return fig_pnl_dist, fig_holding_pnl, fig_cum_pnl

def create_action_analysis(steps_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
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
        template="plotly_dark"
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
        template="plotly_dark"
    )
    
    return fig_action_dist, fig_transitions

def create_reward_analysis(steps_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
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
        template="plotly_dark"
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
        template="plotly_dark"
    )
    fig_running_reward.update_layout(height=350)
    
    return fig_reward_dist, fig_running_reward

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
st.set_page_config(layout="wide", page_title="Enhanced RL Trading Dashboard", page_icon="📈")
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
    st.sidebar.write("Run Parameters:")
    st.sidebar.json(latest_run.get("parameters", {}), expanded=False)

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
            
            **Note**: The portfolio value chart for a specific episode may show growth, while the average PnL across all episodes could be negative. This indicates that the selected episode performed better than the average.
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
        col1, col2 = st.columns([3, 1])
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

        # Fetch and display data for selected episode
        episode_steps_df = fetch_episode_steps(selected_episode_id)
        episode_trades = fetch_episode_trades(selected_episode_id)
        episode_operations = fetch_episode_operations(selected_episode_id)

        if not episode_steps_df.empty:
            # Get the selected episode data
            selected_episode = episodes_df.loc[selected_episode_id]
            
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
            price_ops_fig, markers_were_plotted = create_price_operations_chart(episode_steps_df, episode_operations)
            st.plotly_chart(price_ops_fig, use_container_width=True)
            
            # Add a note based on why markers might be missing
            if not episode_operations:
                st.caption(f"Note: No trading operations were found/returned by the API for episode {selected_episode_id}.")
            elif not markers_were_plotted:
                st.caption(f"Note: Trading operations data was found, but markers could not be plotted for episode {selected_episode_id} (check data format or logs).")
            
            # Drawdown Chart
            drawdown_fig = create_drawdown_chart(episode_steps_df)
            if drawdown_fig:
                st.plotly_chart(drawdown_fig, use_container_width=True)

            # --- Trade Analysis ---
            if episode_trades:
                st.subheader("Trade Analysis")
                
                fig_pnl_dist, fig_holding_pnl, fig_cum_pnl = create_trade_analysis_charts(episode_trades)
                
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
            fig_action_dist, fig_transitions = create_action_analysis(episode_steps_df)
            
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
            fig_reward_dist, fig_running_reward = create_reward_analysis(episode_steps_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_reward_dist, use_container_width=True)
            with col2:
                st.plotly_chart(fig_running_reward, use_container_width=True)
        
        else:
            st.warning(f"No step data found for episode {selected_episode_id}.")

        # --- Overall Trends ---
        st.header("📉 Overall Trends Across Episodes")
        
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
st.caption("Enhanced RL Trading Performance Dashboard | Built with Streamlit and Plotly")
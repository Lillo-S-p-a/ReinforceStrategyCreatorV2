import streamlit as st
import pandas as pd
import numpy as np
import logging # Added
import plotly.graph_objects as go
# import os # No longer needed
import requests # Added
from typing import List, Dict, Optional, Any # Added

# --- Configuration ---
# Remove LOG_FILE
# LOG_FILE = 'training_log.csv'
API_BASE_URL = "http://127.0.0.1:8001/api/v1"
# WARNING: Hardcoding keys is insecure. Use Streamlit secrets or env vars in production.
API_KEY = "test-key-123"
API_HEADERS = {"X-API-Key": API_KEY}
ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'} # Keep for potential future use or reference

# Configure basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Helper Functions ---

@st.cache_data(ttl=60) # Cache API calls for 1 minute
def fetch_api_data(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Generic function to fetch data from the API."""
    url = f"{API_BASE_URL}{endpoint}"
    # Mask API key for logging if needed, though it's hardcoded here
    log_headers = API_HEADERS.copy()
    if "X-API-Key" in log_headers:
        log_headers["X-API-Key"] = "****" # Mask sensitive key in logs

    logging.info(f"Attempting to fetch data from API endpoint: {url}")
    logging.info(f"  Headers: {log_headers}")
    logging.info(f"  Params: {params}")
    try:
        response = requests.get(url, headers=API_HEADERS, params=params, timeout=10) # Added timeout
        logging.info(f"  API Response Status Code: {response.status_code}")
        # Log a snippet of the response for debugging, be careful with large responses
        response_snippet = response.text[:200] + "..." if len(response.text) > 200 else response.text
        logging.info(f"  API Response Snippet: {response_snippet}")

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logging.info(f"  Successfully fetched and parsed JSON for {url}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred fetching {url}: {http_err}")
        logging.error(f"  Response Body: {response.text}") # Log full body on HTTP error
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
    except requests.exceptions.RequestException as req_err: # Catch more specific exceptions first
        logging.error(f"Ambiguous request error occurred fetching {url}: {req_err}")
        st.error(f"API Request Error fetching {endpoint}: {req_err}")
        return None
    except Exception as e:
        # Catch JSONDecodeError specifically if possible, or other parsing errors
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
        data = fetch_api_data(f"/runs/{run_id}/episodes/", params={"page": page, "page_size": 100}) # Use max page size
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
    """Fetches all steps for a given episode (handles pagination) and returns a DataFrame."""
    steps_list = []
    page = 1
    while True:
        data = fetch_api_data(f"/episodes/{episode_id}/steps/", params={"page": page, "page_size": 100}) # Use max page size
        if not data or not data.get("items"):
            if page == 1: # Only warn if no steps found at all
                 st.warning(f"No steps found for episode {episode_id}.")
            break
        steps_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1

    if not steps_list:
        return pd.DataFrame() # Return empty DataFrame if no steps

    df = pd.DataFrame(steps_list)
    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Set timestamp as index for easier plotting
    df = df.set_index('timestamp')
    # Ensure numeric types
    df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
    # Keep action as string as returned by API
    return df.sort_index() # Ensure chronological order
# NOTE: This function currently fetches portfolio_value, reward, action etc.
    # It does NOT fetch the raw asset price (e.g., 'close_price') needed for
    # accurately plotting the price line chart requested in TASK-FE-LEAD-20250505-122900.
    # The chart will use portfolio_value as a proxy line for now.

@st.cache_data(ttl=60)
def fetch_episode_operations(episode_id: int) -> List[Dict[str, Any]]:
    """Fetches all trading operations for a given episode (handles pagination)."""
    operations_list = []
    page = 1
    logging.info(f"Fetching operations for episode {episode_id}")
    while True:
        endpoint = f"/episodes/{episode_id}/operations/"
        data = fetch_api_data(endpoint, params={"page": page, "page_size": 100}) # Use max page size
        logging.debug(f"Raw operations data page {page} for episode {episode_id}: {data}") # DEBUG: Log raw data
        if not data or not data.get("items"):
            if page == 1: # Only warn if no operations found at all
                 logging.warning(f"No operations items found in API response for episode {episode_id} on page 1.")
                 # st.warning(f"No trading operations found for episode {episode_id}.") # Avoid cluttering UI? Log is enough.
            break
        operations_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logging.info(f"Fetched {len(operations_list)} operations for episode {episode_id}.")
    # Convert timestamp string to datetime objects for easier plotting
    for op in operations_list:
        op['timestamp'] = pd.to_datetime(op['timestamp']) # Correctly indented
    return sorted(operations_list, key=lambda x: x['timestamp']) # Correctly indented

# --- Plotting Functions ---

def create_price_operations_chart(steps_df: pd.DataFrame, operations: List[Dict[str, Any]]) -> tuple[go.Figure, bool]:
    """Creates a Plotly chart showing portfolio value and trading operations. Returns figure and a boolean indicating if markers were plotted."""
    fig = go.Figure()
    markers_plotted = False # Initialize the flag

    # 1. Add Portfolio Value Line (Using as proxy for price)
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        fig.add_trace(go.Scatter(
            x=steps_df.index,
            y=steps_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value (Proxy)', # Indicate it's a proxy
            line=dict(color='blue')
        ))
    else:
        logging.warning("Steps DataFrame is empty or missing 'portfolio_value', cannot plot main line.")
        # Optionally add a placeholder or return an empty figure?
        # For now, continue to plot markers if available.

    # 2. Prepare data for markers
    marker_data = {
        'ENTRY_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'triangle-up', 'size': 10},
        'EXIT_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'triangle-down', 'size': 10},
        'ENTRY_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'triangle-down', 'size': 10},
        'EXIT_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'triangle-up', 'size': 10},
    }

    for op in operations:
        op_type = op.get('operation_type')
        if op_type in marker_data:
            timestamp = op.get('timestamp')
            price = op.get('price')
            size = op.get('size')
            if timestamp is not None and price is not None: # Ensure data exists
                marker_data[op_type]['x'].append(timestamp)
                marker_data[op_type]['y'].append(price)
                hover_text = f"Type: {op_type}<br>Price: {price:.2f}<br>Size: {size or 'N/A'}<br>Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                marker_data[op_type]['text'].append(hover_text)

    # 3. Add Marker Traces
    for op_type, data in marker_data.items():
        if data['x']: # Only add trace if there are markers of this type
            markers_plotted = True # Set flag if we add any marker trace
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
                hoverinfo='text' # Show only the custom text on hover
            ))

    # 4. Customize Layout
    fig.update_layout(
        title=f"Portfolio Value and Trading Operations",
        xaxis_title="Time",
        yaxis_title="Value / Price",
        hovermode="x unified", # Show hover info for all traces at a given x
        legend_title="Trace Type",
        height=500 # Adjust height as needed
    )

    return fig, markers_plotted # Return the flag as well
    # --- End of create_price_operations_chart ---

# --- Helper Functions (Old - To be commented out) ---
# --- Helper Functions (Old - To be commented out) ---
# def load_data(file_path):
#     """Loads the training log data."""
#     if not os.path.exists(file_path):
#         st.error(f"Error: Training log file '{file_path}' not found.")
#         st.info("Please run the training script (`train.py`) first to generate the log file.")
#         return None
#     try:
#         df = pd.read_csv(file_path)
#         # Ensure correct data types
#         df['episode'] = df['episode'].astype(int)
#         df['step'] = df['step'].astype(int)
#         df['action'] = df['action'].astype(int)
#         df['reward'] = df['reward'].astype(float)
#         # df['total_reward'] = df['total_reward'].astype(float) # Column not present in log
#         df['portfolio_value'] = df['portfolio_value'].astype(float)
#         df['balance'] = df['balance'].astype(float)
#         df['shares_held'] = df['shares_held'].astype(float)
#         # Epsilon might not always be present or consistent
#         if 'epsilon' in df.columns:
#             df['epsilon'] = df['epsilon'].astype(float)
#         return df
#     except Exception as e:
#         st.error(f"Error loading or processing data from '{file_path}': {e}")
#         return None
#
# def calculate_overall_metrics(df):
#     """Calculates overall performance metrics."""
#     if df is None or df.empty:
#         return {}
#
#     metrics = {}
#     last_episode = df['episode'].max()
#     metrics['total_episodes'] = last_episode + 1 # Episodes are 0-indexed
#
#     # Final portfolio value of the last episode
#     last_episode_df = df[df['episode'] == last_episode]
#     if not last_episode_df.empty:
#         metrics['final_portfolio_value'] = last_episode_df['portfolio_value'].iloc[-1]
#     else:
#         metrics['final_portfolio_value'] = np.nan
#
#     # Overall win rate (episodes with positive SUM of rewards)
#     # Calculate sum of rewards per episode as 'total_reward' is not logged directly
#     episode_sum_rewards = df.groupby('episode')['reward'].sum()
#     metrics['win_rate'] = (episode_sum_rewards > 0).mean() * 100 if not episode_sum_rewards.empty else 0
#
#     # Overall action distribution
#     action_counts = df['action'].map(ACTION_MAP).value_counts(normalize=True) * 100
#     metrics['action_distribution'] = action_counts.to_dict()
#
#     return metrics

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📈 RL Trading Agent Performance Dashboard (API Driven)")

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
        col1.metric("Total Episodes", f"{run_summary.get('total_episodes', 'N/A'):,}")
        col2.metric("Avg PnL", f"${run_summary.get('average_pnl', 'N/A'):,.2f}" if run_summary.get('average_pnl') is not None else "N/A")
        col3.metric("Avg Sharpe Ratio", f"{run_summary.get('average_sharpe_ratio', 'N/A'):.3f}" if run_summary.get('average_sharpe_ratio') is not None else "N/A")
        col4.metric("Avg Win Rate", f"{run_summary.get('average_win_rate', 'N/A')*100:.2f}%" if run_summary.get('average_win_rate') is not None else "N/A")
        # Removed Overall Action Distribution - requires fetching all steps for all episodes
        # st.subheader("Overall Action Distribution")
        # ...
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
        min_ep = episodes_df.index.min()
        max_ep = episodes_df.index.max()
        selected_episode_id = st.slider(
            "Select Episode ID:",
            min_value=min_ep,
            max_value=max_ep,
            value=max_ep # Default to last episode
        )

        # Fetch and display data for selected episode
        episode_steps_df = fetch_episode_steps(selected_episode_id)

        if not episode_steps_df.empty:
            st.subheader(f"Performance for Episode {selected_episode_id}")

            # --- Episode-Specific Plots ---
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                st.line_chart(episode_steps_df['portfolio_value'], use_container_width=True)
                st.caption("Portfolio Value Over Time")

                # Removed Balance vs Shares plot - data not in Step schema
                # st.line_chart(episode_data[['balance', 'shares_held']], use_container_width=True)
                # st.caption("Cash Balance vs. Shares Held Over Steps")

            with plot_col2:
                # Plot actions - Use strings directly
                st.line_chart(episode_steps_df['action'], use_container_width=True) # Or st.scatter_chart
                st.caption("Actions Taken Over Time")

                st.line_chart(episode_steps_df['reward'], use_container_width=True)
                st.caption("Reward Per Step")

# --- Price/Operations Chart (Plotly) ---
            st.subheader("Portfolio Value & Trading Operations")
            episode_operations = fetch_episode_operations(selected_episode_id)

            # Create and display the chart
            price_ops_fig, markers_were_plotted = create_price_operations_chart(episode_steps_df, episode_operations)
            st.plotly_chart(price_ops_fig, use_container_width=True)

            # Add a note based on why markers might be missing
            if not episode_steps_df.empty: # Only show note if price line is present
                if not episode_operations:
                    st.caption(f"Note: No trading operations were found/returned by the API for episode {selected_episode_id}.")
                elif not markers_were_plotted:
                    st.caption(f"Note: Trading operations data was found, but markers could not be plotted for episode {selected_episode_id} (check data format or logs).")
        else:
            st.warning(f"No step data found for episode {selected_episode_id}.")

        # --- Overall Plots ---
        st.header("📉 Overall Trends")
        overall_plot_col1, overall_plot_col2 = st.columns(2)

        with overall_plot_col1:
            # Portfolio Value per Episode (End Value)
            st.line_chart(episodes_df['final_portfolio_value'], use_container_width=True)
            st.caption("Final Portfolio Value per Episode")

        with overall_plot_col2:
            # Sharpe Ratio per Episode
            st.line_chart(episodes_df['sharpe_ratio'], use_container_width=True)
            st.caption("Sharpe Ratio per Episode")
            # Removed Epsilon plot - data not available via API

    else:
        st.warning("Could not load episode data for this run.")

else:
    st.error("Failed to fetch initial run data. Cannot display dashboard.")

# Removed sidebar info about log file
# st.sidebar.info(f"Data loaded from: `{LOG_FILE}`")
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Configuration ---
LOG_FILE = 'training_log.csv'
ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# --- Helper Functions ---
def load_data(file_path):
    """Loads the training log data."""
    if not os.path.exists(file_path):
        st.error(f"Error: Training log file '{file_path}' not found.")
        st.info("Please run the training script (`train.py`) first to generate the log file.")
        return None
    try:
        df = pd.read_csv(file_path)
        # Ensure correct data types
        df['episode'] = df['episode'].astype(int)
        df['step'] = df['step'].astype(int)
        df['action'] = df['action'].astype(int)
        df['reward'] = df['reward'].astype(float)
        # df['total_reward'] = df['total_reward'].astype(float) # Column not present in log
        df['portfolio_value'] = df['portfolio_value'].astype(float)
        df['balance'] = df['balance'].astype(float)
        df['shares_held'] = df['shares_held'].astype(float)
        # Epsilon might not always be present or consistent
        if 'epsilon' in df.columns:
            df['epsilon'] = df['epsilon'].astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading or processing data from '{file_path}': {e}")
        return None

def calculate_overall_metrics(df):
    """Calculates overall performance metrics."""
    if df is None or df.empty:
        return {}

    metrics = {}
    last_episode = df['episode'].max()
    metrics['total_episodes'] = last_episode + 1 # Episodes are 0-indexed

    # Final portfolio value of the last episode
    last_episode_df = df[df['episode'] == last_episode]
    if not last_episode_df.empty:
        metrics['final_portfolio_value'] = last_episode_df['portfolio_value'].iloc[-1]
    else:
        metrics['final_portfolio_value'] = np.nan

    # Overall win rate (episodes with positive SUM of rewards)
    # Calculate sum of rewards per episode as 'total_reward' is not logged directly
    episode_sum_rewards = df.groupby('episode')['reward'].sum()
    metrics['win_rate'] = (episode_sum_rewards > 0).mean() * 100 if not episode_sum_rewards.empty else 0

    # Overall action distribution
    action_counts = df['action'].map(ACTION_MAP).value_counts(normalize=True) * 100
    metrics['action_distribution'] = action_counts.to_dict()

    return metrics

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📈 RL Trading Agent Performance Dashboard")

# Load Data
data = load_data(LOG_FILE)

if data is not None:
    # --- Overall Summary ---
    st.header("📊 Overall Summary")
    metrics = calculate_overall_metrics(data)

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Episodes Trained", f"{metrics.get('total_episodes', 'N/A'):,}")
        col2.metric("Final Portfolio Value (Last Ep.)", f"${metrics.get('final_portfolio_value', 'N/A'):,.2f}")
        col3.metric("Overall Win Rate", f"{metrics.get('win_rate', 'N/A'):.2f}%")

        st.subheader("Overall Action Distribution")
        action_dist = metrics.get('action_distribution', {})
        # Display as columns or a chart
        action_cols = st.columns(len(action_dist))
        for i, (action, percentage) in enumerate(action_dist.items()):
             action_cols[i].metric(action, f"{percentage:.2f}%")
        # Alternative: Bar chart
        # st.bar_chart(pd.Series(action_dist))

    # --- Episode Selector ---
    st.header("🔬 Episode Details")
    total_episodes = metrics.get('total_episodes', 1)
    selected_episode = st.slider(
        "Select Episode:",
        min_value=0,
        max_value=total_episodes - 1,
        value=max(0, total_episodes - 1) # Default to last episode
    )

    # Filter data for selected episode
    episode_data = data[data['episode'] == selected_episode].set_index('step')

    if not episode_data.empty:
        st.subheader(f"Performance for Episode {selected_episode}")

        # --- Episode-Specific Plots ---
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.line_chart(episode_data['portfolio_value'], use_container_width=True)
            st.caption("Portfolio Value Over Steps")

            st.line_chart(episode_data[['balance', 'shares_held']], use_container_width=True)
            st.caption("Cash Balance vs. Shares Held Over Steps")

        with plot_col2:
            # Map actions for better visualization if needed, or plot raw numbers
            st.line_chart(episode_data['action'], use_container_width=True)
            st.caption(f"Actions Taken Over Steps ({', '.join([f'{v}={k}' for k,v in ACTION_MAP.items()])})")

            st.line_chart(episode_data['reward'], use_container_width=True)
            st.caption("Reward Per Step")


    else:
        st.warning(f"No data found for episode {selected_episode}.")

    # --- Overall Plots ---
    st.header("📉 Overall Trends")
    overall_plot_col1, overall_plot_col2 = st.columns(2)

    with overall_plot_col1:
        # Portfolio Value per Episode (End Value)
        episode_portfolio = data.groupby('episode')['portfolio_value'].last()
        st.line_chart(episode_portfolio, use_container_width=True)
        st.caption("Final Portfolio Value per Episode")

    with overall_plot_col2:
        # Epsilon Decay (Optional)
        if 'epsilon' in data.columns:
            # Assuming epsilon is logged once per episode or consistently
            epsilon_trend = data.groupby('episode')['epsilon'].first() # Or last() or mean() depending on logging
            if not epsilon_trend.isnull().all():
                st.line_chart(epsilon_trend, use_container_width=True)
                st.caption("Epsilon Decay Over Episodes")
            else:
                st.info("Epsilon data not available or consistently logged for plotting.")
        else:
            st.info("Epsilon data not logged.")

else:
    st.warning("Could not load data to display the dashboard.")

st.sidebar.info(f"Data loaded from: `{LOG_FILE}`")
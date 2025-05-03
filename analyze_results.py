import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

# --- Configuration ---
LOG_FILENAME = "training_log.csv"
PLOTS_DIR = "results_plots"
ACTION_MAP = {0: 'Flat/Hold', 1: 'Long/Buy', 2: 'Short/Sell'} # Map numeric actions to labels

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filename: str) -> pd.DataFrame:
    """Loads the training log CSV file."""
    if not os.path.exists(filename):
        logging.error(f"Error: Log file '{filename}' not found.")
        return pd.DataFrame() # Return empty DataFrame
    try:
        df = pd.read_csv(filename)
        logging.info(f"Successfully loaded '{filename}' with {len(df)} records.")
        # Convert action to categorical type using the map
        if 'action' in df.columns:
            df['action_label'] = df['action'].map(ACTION_MAP)
            df['action_label'] = pd.Categorical(df['action_label'], categories=ACTION_MAP.values(), ordered=False)
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file '{filename}': {e}")
        return pd.DataFrame()

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculates key performance metrics from the log data."""
    if df.empty:
        logging.warning("DataFrame is empty, cannot calculate metrics.")
        return {}

    metrics = {}

    # 1. Total rewards per episode
    rewards_per_episode = df.groupby('episode')['reward'].sum()
    metrics['rewards_per_episode'] = rewards_per_episode

    # 2. Action distributions
    if 'action_label' in df.columns:
        action_counts = df['action_label'].value_counts(normalize=True) * 100 # Percentage
        metrics['action_distribution'] = action_counts
    else:
        logging.warning("Column 'action_label' not found, cannot calculate action distribution.")
        metrics['action_distribution'] = pd.Series(dtype=float)


    # 3. Portfolio value over time (extract for plotting later)
    # We'll group by episode and step for potential plotting
    metrics['portfolio_value_data'] = df[['episode', 'step', 'portfolio_value']].copy()

    # 4. Win/loss ratio (based on episodes with positive total reward)
    positive_reward_episodes = (rewards_per_episode > 0).sum()
    total_episodes = rewards_per_episode.count()
    win_rate = (positive_reward_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    metrics['episode_win_rate_percent'] = win_rate

    logging.info("Metrics calculated successfully.")
    logging.info(f"  - Episode Win Rate: {win_rate:.2f}%")
    if 'action_distribution' in metrics and not metrics['action_distribution'].empty:
         logging.info(f"  - Action Distribution:\n{metrics['action_distribution']}")

    return metrics

def generate_visualizations(df: pd.DataFrame, metrics: dict, plots_dir: str):
    """Generates and saves visualizations for the calculated metrics."""
    if df.empty or not metrics:
        logging.warning("DataFrame or metrics are empty, skipping visualization generation.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Saving plots to directory: '{plots_dir}'")

    # 1. Plot: Total rewards per episode
    if 'rewards_per_episode' in metrics and not metrics['rewards_per_episode'].empty:
        plt.figure(figsize=(10, 5))
        metrics['rewards_per_episode'].plot(kind='line', marker='o')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'rewards_per_episode.png'))
        plt.close()
        logging.info("  - Saved 'rewards_per_episode.png'")
    else:
        logging.warning("Skipping rewards per episode plot (no data).")


    # 2. Plot: Portfolio value over time (for the last episode)
    if 'portfolio_value_data' in metrics and not metrics['portfolio_value_data'].empty:
        last_episode = df['episode'].max()
        last_episode_data = df[df['episode'] == last_episode]

        if not last_episode_data.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(last_episode_data['step'], last_episode_data['portfolio_value'])
            plt.title(f'Portfolio Value Over Time (Episode {last_episode})')
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'portfolio_value_episode_{last_episode}.png'))
            plt.close()
            logging.info(f"  - Saved 'portfolio_value_episode_{last_episode}.png'")
        else:
             logging.warning(f"Skipping portfolio value plot (no data for last episode {last_episode}).")
    else:
        logging.warning("Skipping portfolio value plot (no data).")

    # 3. Plot: Action distribution (Pie Chart)
    if 'action_distribution' in metrics and not metrics['action_distribution'].empty:
        plt.figure(figsize=(7, 7))
        metrics['action_distribution'].plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False)
        plt.title('Action Distribution (Overall)')
        plt.ylabel('') # Hide the y-label which defaults to the series name
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'action_distribution_pie.png'))
        plt.close()
        logging.info("  - Saved 'action_distribution_pie.png'")

        # Optional: Bar Chart for Action Distribution
        plt.figure(figsize=(8, 5))
        metrics['action_distribution'].plot(kind='bar')
        plt.title('Action Distribution (Overall %)')
        plt.xlabel('Action')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'action_distribution_bar.png'))
        plt.close()
        logging.info("  - Saved 'action_distribution_bar.png'")
    else:
        logging.warning("Skipping action distribution plots (no data).")


def main():
    """Main function to run the analysis."""
    logging.info("Starting training results analysis...")
    df_results = load_data(LOG_FILENAME)

    if df_results.empty:
        logging.error("Failed to load data. Exiting analysis.")
        return

    metrics = calculate_metrics(df_results)

    if not metrics:
        logging.error("Failed to calculate metrics. Exiting analysis.")
        return

    generate_visualizations(df_results, metrics, PLOTS_DIR)

    logging.info("Analysis complete.")

if __name__ == "__main__":
    main()
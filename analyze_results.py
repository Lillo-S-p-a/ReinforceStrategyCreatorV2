import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import argparse # To accept run_id as argument
from sqlalchemy import select, func
from sqlalchemy.orm import Session

# --- Local Imports ---
# Assuming db_utils and db_models are accessible from this script's location
# Adjust the import path if necessary based on your project structure
try:
    from reinforcestrategycreator.db_utils import get_db_session
    from reinforcestrategycreator.db_models import TrainingRun, Episode, Step, Trade
except ImportError:
    logging.error("Could not import database modules. Make sure PYTHONPATH is set correctly or run from project root.")
    exit(1)

# --- Configuration ---
# LOG_FILENAME = "training_log.csv" # Removed CSV dependency
PLOTS_DIR = "results_plots"
ACTION_MAP = {'flat': 'Flat/Hold', 'long': 'Long/Buy', 'short': 'Short/Sell'} # Map string actions

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_run_id(db: Session) -> str | None:
    """Fetches the ID of the most recent completed training run."""
    try:
        latest_run = db.query(TrainingRun.run_id)\
                       .filter(TrainingRun.status == 'completed')\
                       .order_by(TrainingRun.start_time.desc())\
                       .first()
        return latest_run.run_id if latest_run else None
    except Exception as e:
        logging.error(f"Error fetching latest run ID: {e}")
        return None

def load_data_from_db(db: Session, run_id: str) -> dict[str, pd.DataFrame]:
    """Loads episode, step, and trade data for a specific run_id from the database."""
    data = {}
    try:
        # Load Episodes
        episodes_query = select(Episode).where(Episode.run_id == run_id).order_by(Episode.episode_id)
        data['episodes'] = pd.read_sql(episodes_query, db.bind)
        if data['episodes'].empty:
            logging.warning(f"No episode data found for run_id: {run_id}")
            return {} # Return empty if no episodes

        # Load Steps
        steps_query = select(Step)\
                      .join(Episode, Step.episode_id == Episode.episode_id)\
                      .where(Episode.run_id == run_id)\
                      .order_by(Step.episode_id, Step.timestamp)
        data['steps'] = pd.read_sql(steps_query, db.bind)
        if 'action' in data['steps'].columns:
             data['steps']['action_label'] = data['steps']['action'].map(ACTION_MAP)
             # Ensure consistent categories even if some actions didn't occur
             all_actions = list(ACTION_MAP.values())
             data['steps']['action_label'] = pd.Categorical(data['steps']['action_label'], categories=all_actions, ordered=False)


        # Load Trades (Optional, if needed for specific analysis not covered by episode metrics)
        trades_query = select(Trade)\
                       .join(Episode, Trade.episode_id == Episode.episode_id)\
                       .where(Episode.run_id == run_id)\
                       .order_by(Trade.episode_id, Trade.entry_time)
        data['trades'] = pd.read_sql(trades_query, db.bind)

        logging.info(f"Successfully loaded data for run_id '{run_id}' from DB.")
        logging.info(f"  - Episodes: {len(data.get('episodes', pd.DataFrame()))}")
        logging.info(f"  - Steps: {len(data.get('steps', pd.DataFrame()))}")
        logging.info(f"  - Trades: {len(data.get('trades', pd.DataFrame()))}")
        return data

    except Exception as e:
        logging.error(f"Error loading data from database for run_id '{run_id}': {e}")
        return {}

def calculate_metrics_from_db(db_data: dict[str, pd.DataFrame]) -> dict:
    """Calculates key performance metrics from the database query results."""
    if not db_data or 'episodes' not in db_data or db_data['episodes'].empty:
        logging.warning("Episode data is empty, cannot calculate metrics.")
        return {}

    metrics = {}
    episodes_df = db_data['episodes']
    steps_df = db_data.get('steps', pd.DataFrame())
    # trades_df = db_data.get('trades', pd.DataFrame()) # Use if needed

    # 1. Use metrics directly from Episode table
    metrics['rewards_per_episode'] = episodes_df.set_index('episode_id')['total_reward']
    metrics['pnl_per_episode'] = episodes_df.set_index('episode_id')['pnl']
    metrics['sharpe_per_episode'] = episodes_df.set_index('episode_id')['sharpe_ratio']
    metrics['mdd_per_episode'] = episodes_df.set_index('episode_id')['max_drawdown']
    metrics['win_rate_per_episode'] = episodes_df.set_index('episode_id')['win_rate']
    # Add others if they were added to the model (e.g., trade_frequency, success_rate)

    # 2. Action distributions (from steps)
    if not steps_df.empty and 'action_label' in steps_df.columns:
        # Ensure NaN categories are handled if present
        action_counts = steps_df['action_label'].value_counts(normalize=True, dropna=False) * 100 # Percentage
        metrics['action_distribution'] = action_counts.fillna(0) # Replace NaN counts with 0
    else:
        logging.warning("Steps data or 'action_label' column not found, cannot calculate action distribution.")
        metrics['action_distribution'] = pd.Series(dtype=float)

    # 3. Portfolio value over time (from steps)
    if not steps_df.empty:
        metrics['portfolio_value_data'] = steps_df[['episode_id', 'timestamp', 'portfolio_value']].copy()
        # Ensure timestamp is datetime for plotting
        metrics['portfolio_value_data']['timestamp'] = pd.to_datetime(metrics['portfolio_value_data']['timestamp'])
    else:
         metrics['portfolio_value_data'] = pd.DataFrame(columns=['episode_id', 'timestamp', 'portfolio_value'])


    # 4. Overall Episode Win Rate (based on PnL > 0)
    positive_pnl_episodes = (episodes_df['pnl'] > 0).sum()
    total_episodes = len(episodes_df)
    win_rate = (positive_pnl_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    metrics['overall_episode_win_rate_percent'] = win_rate

    # 5. Overall Total Return %
    # Assumes episodes_df is sorted by episode_id (done in load_data_from_db)
    overall_total_return_pct = None # Initialize
    if not episodes_df.empty:
        first_initial_value = episodes_df.iloc[0]['initial_portfolio_value']
        last_final_value = episodes_df.iloc[-1]['final_portfolio_value']
        if first_initial_value is not None and first_initial_value != 0 and last_final_value is not None:
            overall_total_return_pct = ((last_final_value / first_initial_value) - 1) * 100
        else:
            logging.warning("Could not calculate Overall Total Return %: Initial or Final portfolio value is missing or initial is zero.")
    metrics['overall_total_return_percent'] = overall_total_return_pct

    # 6. Average Sharpe Ratio
    # .mean() automatically handles NaNs by default (skipna=True)
    average_sharpe_ratio = episodes_df['sharpe_ratio'].mean()
    metrics['average_sharpe_ratio'] = average_sharpe_ratio if pd.notna(average_sharpe_ratio) else None # Store None if result is NaN

    logging.info("Metrics calculated successfully from DB data.")
    logging.info(f"  - Overall Episode Win Rate: {metrics.get('overall_episode_win_rate_percent', 0):.2f}%")
    # Add logging for new metrics
    logging.info(f"  - Overall Total Return: {metrics.get('overall_total_return_percent', 'N/A')}%" if metrics.get('overall_total_return_percent') is not None else "  - Overall Total Return: N/A")
    logging.info(f"  - Average Sharpe Ratio: {metrics.get('average_sharpe_ratio', 'N/A'):.4f}" if metrics.get('average_sharpe_ratio') is not None else "  - Average Sharpe Ratio: N/A")

    if 'action_distribution' in metrics and not metrics['action_distribution'].empty:
         logging.info(f"  - Action Distribution:\n{metrics['action_distribution']}")

    return metrics

def generate_visualizations(db_data: dict[str, pd.DataFrame], metrics: dict, plots_dir: str, run_id: str):
    """Generates and saves visualizations using data queried from the database."""
    if not db_data or not metrics:
        logging.warning("DB data or metrics are empty, skipping visualization generation.")
        return

    episodes_df = db_data['episodes']
    steps_df = db_data.get('steps', pd.DataFrame())

    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Saving plots for run '{run_id}' to directory: '{plots_dir}'")
    plot_prefix = f"{run_id}_" # Prefix plots with run_id

    # 1. Plot: Total rewards per episode
    if 'rewards_per_episode' in metrics and not metrics['rewards_per_episode'].empty:
        plt.figure(figsize=(10, 5))
        metrics['rewards_per_episode'].plot(kind='line', marker='o')
        plt.title(f'Total Reward per Episode (Run: {run_id})')
        plt.xlabel('Episode ID') # Use Episode ID from index
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{plot_prefix}rewards_per_episode.png'))
        plt.close()
        logging.info(f"  - Saved '{plot_prefix}rewards_per_episode.png'")
    else:
        logging.warning("Skipping rewards per episode plot (no data).")


    # 2. Plot: Portfolio value over time (for the last episode)
    if 'portfolio_value_data' in metrics and not metrics['portfolio_value_data'].empty:
        last_episode_id = episodes_df['episode_id'].max()
        last_episode_steps = steps_df[steps_df['episode_id'] == last_episode_id]

        if not last_episode_steps.empty:
            plt.figure(figsize=(12, 6))
            # Use timestamp for x-axis if available and correct type
            if 'timestamp' in last_episode_steps.columns and pd.api.types.is_datetime64_any_dtype(last_episode_steps['timestamp']):
                 plt.plot(last_episode_steps['timestamp'], last_episode_steps['portfolio_value'])
                 plt.xlabel('Time')
            else:
                 # Fallback to step index if timestamp is missing/invalid
                 plt.plot(last_episode_steps.index, last_episode_steps['portfolio_value']) # Assuming steps are ordered
                 plt.xlabel('Step Index')

            plt.title(f'Portfolio Value Over Time (Episode ID: {last_episode_id}, Run: {run_id})')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{plot_prefix}portfolio_value_episode_{last_episode_id}.png'))
            plt.close()
            logging.info(f"  - Saved '{plot_prefix}portfolio_value_episode_{last_episode_id}.png'")
        else:
             logging.warning(f"Skipping portfolio value plot (no step data for last episode {last_episode_id}).")
    else:
        logging.warning("Skipping portfolio value plot (no step data).")

    # 3. Plot: Action distribution (Pie Chart)
    if 'action_distribution' in metrics and not metrics['action_distribution'].empty:
        plt.figure(figsize=(7, 7))
        # Ensure labels match the data index
        metrics['action_distribution'].plot(kind='pie', labels=metrics['action_distribution'].index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Action Distribution (Run: {run_id})')
        plt.ylabel('') # Hide the y-label
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{plot_prefix}action_distribution_pie.png'))
        plt.close()
        logging.info(f"  - Saved '{plot_prefix}action_distribution_pie.png'")

        # Optional: Bar Chart for Action Distribution
        plt.figure(figsize=(8, 5))
        metrics['action_distribution'].plot(kind='bar')
        plt.title(f'Action Distribution (Run: {run_id} %)')
        plt.xlabel('Action')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{plot_prefix}action_distribution_bar.png'))
        plt.close()
        logging.info(f"  - Saved '{plot_prefix}action_distribution_bar.png'")
    else:
        logging.warning("Skipping action distribution plots (no data).")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze training results from the database.")
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific Run ID to analyze. If not provided, the latest completed run will be used."
    )
    args = parser.parse_args()

    logging.info("Starting training results analysis from database...")

    run_id_to_analyze = args.run_id
    db_data = {}
    metrics = {}

    try:
        with get_db_session() as db:
            if not run_id_to_analyze:
                logging.info("No Run ID provided, attempting to find the latest completed run...")
                run_id_to_analyze = get_latest_run_id(db)
                if not run_id_to_analyze:
                    logging.error("Could not find any completed runs in the database.")
                    return
                logging.info(f"Analyzing latest completed run: {run_id_to_analyze}")
            else:
                 logging.info(f"Analyzing specified run: {run_id_to_analyze}")

            db_data = load_data_from_db(db, run_id_to_analyze)

    except Exception as e:
        logging.error(f"Failed to connect to or query the database: {e}")
        return # Exit if DB connection fails

    if not db_data:
        logging.error(f"Failed to load data for run {run_id_to_analyze}. Exiting analysis.")
        return

    metrics = calculate_metrics_from_db(db_data)

    if not metrics:
        logging.error("Failed to calculate metrics. Exiting analysis.")
        return

    generate_visualizations(db_data, metrics, PLOTS_DIR, run_id_to_analyze)

    logging.info("Analysis complete.")

if __name__ == "__main__":
    main()
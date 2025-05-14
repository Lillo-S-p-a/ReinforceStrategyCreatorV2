import argparse
import logging
from sqlalchemy.orm import sessionmaker
from reinforcestrategycreator.db_models import Episode  # Assuming Episode model is in db_models
from reinforcestrategycreator.db_utils import get_db_session # For database session
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_episode_details(db_session, run_id_to_check: str):
    """
    Queries episode details for a given run_id and prints them.
    """
    logging.info(f"Querying episode details for run_id: {run_id_to_check}")

    episodes = db_session.query(Episode).filter(Episode.run_id == run_id_to_check).all()

    if not episodes:
        logging.warning(f"No episodes found for run_id: {run_id_to_check}")
        print(f"No episodes found for run_id: {run_id_to_check}")
        return False

    logging.info(f"Found {len(episodes)} episodes for run_id {run_id_to_check}.")
    print(f"\nDetails for episodes with run_id: {run_id_to_check}")
    
    all_filled = True
    key_columns = [
        'rllib_episode_id', 'start_time', 'end_time', 
        'initial_portfolio_value', 'final_portfolio_value', 'status', 'pnl', 
        'sharpe_ratio', 'max_drawdown', 'total_reward', 'total_steps', 'win_rate'
    ]

    for i, episode in enumerate(episodes):
        print(f"\n--- Episode {i+1} (DB ID: {episode.episode_id}) ---")
        episode_all_filled = True
        for col in key_columns:
            value = getattr(episode, col, 'N/A')
            if value is None:
                print(f"  {col}: NULL (MISSING!)")
                all_filled = False
                episode_all_filled = False
            elif isinstance(value, datetime):
                print(f"  {col}: {value.isoformat()}")
            else:
                print(f"  {col}: {value}")
        if episode_all_filled:
            print(f"  Episode {i+1} appears fully populated.")
        else:
            print(f"  Episode {i+1} has NULL values for some key metrics.")
            
    return all_filled

def main():
    parser = argparse.ArgumentParser(description="Check episode details for a specific run_id.")
    parser.add_argument("run_id", type=str, help="The run_id to check episode details for.")
    args = parser.parse_args()

    run_id_to_check = args.run_id

    with get_db_session() as db:
        try:
            overall_status = get_episode_details(db, run_id_to_check)
            if overall_status:
                print(f"\nAll checked episodes for run_id {run_id_to_check} appear to have key metrics populated.")
            else:
                print(f"\nSome episodes for run_id {run_id_to_check} have NULL values for key metrics. Please review the output above.")
        except Exception as e:
            logging.error(f"Error querying database for episode details: {e}", exc_info=True)
            print(f"An error occurred while checking episode details: {e}")

if __name__ == "__main__":
    main()
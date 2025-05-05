# analyze_latest_run.py
import os
import sys
from sqlalchemy import create_engine, func, desc, case
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import NoResultFound
import numpy as np
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
# Assuming the script is run from the project root, otherwise adjust path
sys.path.insert(0, project_root)

try:
    from reinforcestrategycreator.db_models import Base, TrainingRun, Episode, TradingOperation, OperationType
    # Assuming get_db is not needed directly here, we'll create a session locally
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the project root or the path is correctly set.")
    sys.exit(1)

# Get DATABASE_URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set.")
    print("Please ensure a .env file exists with DATABASE_URL=postgresql://user:pass@host:port/db")
    sys.exit(1)

def get_db_session():
    """Creates a new SQLAlchemy session."""
    try:
        # Removed connect_args={"check_same_thread": False} as it's SQLite specific
        engine = create_engine(DATABASE_URL)
        # Base.metadata.create_all(bind=engine) # Usually not needed for analysis script
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print(f"Successfully connected to database: {DATABASE_URL}")
        return SessionLocal()
    except Exception as e:
        print(f"Error connecting to database {DATABASE_URL}: {e}")
        return None

def analyze_latest_run(db: Session):
    """Analyzes the latest completed training run."""
    print("Analyzing latest completed training run...")
    run_id = None
    avg_sharpe_ratio = None
    avg_non_hold_ops = None

    try:
        # 1. Find the latest completed TrainingRun
        latest_completed_run = db.query(TrainingRun)\
            .filter(TrainingRun.status == 'completed')\
            .order_by(desc(TrainingRun.end_time))\
            .first()

        if not latest_completed_run:
            print("No completed training runs found.")
            return None, None, None

        run_id = latest_completed_run.run_id
        print(f"Found latest completed run: {run_id} (Ended: {latest_completed_run.end_time})")

        # 2. Calculate average Sharpe Ratio for this run
        sharpe_ratios_query = db.query(Episode.sharpe_ratio)\
            .filter(Episode.run_id == run_id)\
            .filter(Episode.sharpe_ratio.isnot(None))

        sharpe_ratios = sharpe_ratios_query.all()

        if not sharpe_ratios:
            print(f"No episodes with Sharpe ratios found for run {run_id}.")
            avg_sharpe_ratio = None
        else:
            # Extract values from tuples and handle potential NaNs/Infs if necessary
            valid_sharpe_ratios = [r[0] for r in sharpe_ratios if r[0] is not None and np.isfinite(r[0])]
            if not valid_sharpe_ratios:
                 avg_sharpe_ratio = None
                 print(f"No finite Sharpe ratios found for run {run_id}.")
            else:
                avg_sharpe_ratio = np.mean(valid_sharpe_ratios)
                print(f"Average Sharpe Ratio: {avg_sharpe_ratio:.4f} (from {len(valid_sharpe_ratios)} episodes with valid ratios out of {len(sharpe_ratios)} total)")


        # 3. Calculate average number of non-HOLD operations per episode
        # Get episode IDs for the run
        # Corrected: Use Episode.episode_id instead of Episode.id
        episode_ids_query = db.query(Episode.episode_id).filter(Episode.run_id == run_id)
        episode_ids = [e.episode_id for e in episode_ids_query.all()] # Corrected: Use e.episode_id
        num_episodes_in_run = len(episode_ids)


        if not episode_ids:
            print(f"No episodes found for run {run_id}.")
            avg_non_hold_ops = None
        else:
            print(f"Found {num_episodes_in_run} episodes for run {run_id}.")
            # Count non-HOLD operations per episode
            # Using case to count only non-HOLD operations
            ops_per_episode_query = db.query(
                    TradingOperation.episode_id,
                    # Corrected: Use TradingOperation.operation_id instead of TradingOperation.id
                    func.count(case((TradingOperation.operation_type != OperationType.HOLD, TradingOperation.operation_id))).label('non_hold_count')
                )\
                .filter(TradingOperation.episode_id.in_(episode_ids))\
                .group_by(TradingOperation.episode_id)

            ops_per_episode = ops_per_episode_query.all()

            # Create a dictionary {episode_id: count}
            ops_count_dict = {ep_id: count for ep_id, count in ops_per_episode}

            # Ensure all episodes from the run are considered, even those with 0 non-HOLD ops
            non_hold_counts = []
            for ep_id in episode_ids:
                 count = ops_count_dict.get(ep_id, 0) # Default to 0 if no non-HOLD ops found for an episode
                 non_hold_counts.append(count)

            if not non_hold_counts: # Should not happen if episode_ids is not empty, but safety check
                 avg_non_hold_ops = 0.0
            else:
                 avg_non_hold_ops = np.mean(non_hold_counts)

            print(f"Average Non-HOLD Operations per Episode: {avg_non_hold_ops:.2f} (Total non-HOLD ops: {sum(non_hold_counts)} across {num_episodes_in_run} episodes)")


        return run_id, avg_sharpe_ratio, avg_non_hold_ops

    except NoResultFound:
        print("Error: Could not find required data in the database.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return run_id, avg_sharpe_ratio, avg_non_hold_ops # Return potentially partial results
    finally:
        if db:
            db.close()
            print("Database session closed.")

if __name__ == "__main__":
    print("Starting analysis script...")
    db_session = get_db_session()
    if db_session:
        run_id, avg_sharpe, avg_ops = analyze_latest_run(db_session)
        if run_id is not None: # Check if run_id was found, even if metrics are None
            print("\n--- Analysis Summary ---")
            print(f"Latest Completed Run ID: {run_id}")
            print(f"Average Sharpe Ratio: {f'{avg_sharpe:.4f}' if avg_sharpe is not None else 'N/A'}")
            print(f"Average Non-HOLD Operations per Episode: {f'{avg_ops:.2f}' if avg_ops is not None else 'N/A'}")
            print("------------------------")
        else:
            print("Analysis could not be completed (no completed run found or error occurred).")
    else:
        print("Failed to create database session. Analysis aborted.")
    print("Analysis script finished.")
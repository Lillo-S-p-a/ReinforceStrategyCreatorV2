# get_episode_metrics.py
import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The Run ID from the test run we just executed
RUN_ID = 'RUN-SPY-20250505225840-5ffc28d0' # <-- Updated Run ID

def get_episode_metrics(run_id):
    """Fetches episode metrics for a specific run_id from the episodes table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
        return None

    engine = None
    try:
        # Mask credentials for printing
        db_identifier = db_url.split('@')[1] if '@' in db_url else db_url
        print(f"Connecting to database: {db_identifier}")
        engine = create_engine(db_url)

        with engine.connect() as connection:
            # Select relevant metrics from the episodes table
            query = text("""
                SELECT
                    episode_id,
                    run_id,
                    start_time,
                    end_time,
                    initial_portfolio_value,
                    final_portfolio_value,
                    pnl,
                    sharpe_ratio,
                    max_drawdown,
                    total_reward,
                    total_steps,
                    win_rate
                FROM episodes
                WHERE run_id = :run_id
                ORDER BY episode_id
            """)
            df = pd.read_sql_query(query, connection, params={'run_id': run_id})

            if df.empty:
                print(f"No episode data found for run_id: {run_id}")
                return None

            print(f"\nEpisode Metrics for run_id: {run_id}")
            # Print the DataFrame in a readable format
            # Set display options for better readability
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.colheader_justify', 'center')
            pd.set_option('display.precision', 4)
            print(df.to_string(index=False)) # Use to_string to print the full DataFrame without index

            # Calculate and print average metrics
            print("\nAverage Metrics:")
            # Select only numeric columns suitable for averaging
            numeric_cols = ['pnl', 'sharpe_ratio', 'max_drawdown', 'total_reward', 'win_rate', 'total_steps']
            avg_metrics = df[numeric_cols].mean()
            print(avg_metrics.to_string())

            return df

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if engine:
            engine.dispose() # Close connection pool

if __name__ == "__main__":
    get_episode_metrics(RUN_ID)
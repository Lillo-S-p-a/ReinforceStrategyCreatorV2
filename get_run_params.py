# get_run_params.py
import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# RUN_ID = 'RUN-SPY-20250505215547-118de704' # No longer needed

def get_episode_id_for_run(run_id, episode_index_target):
    """Fetches the database episode_id for a given 1-based episode index within a specific run_id."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
        return None

    engine = None
    try:
        print(f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else db_url}") # Mask credentials
        engine = create_engine(db_url)

        with engine.connect() as connection:
            # Use text() for literal SQL and bind parameters for safety
            # Query to get the episode_id for the target episode index within the run
            # We order by start_time assuming episodes are created sequentially
            # We use OFFSET to get the Nth episode (OFFSET is 0-based, so target-1)
            query = text("""
                SELECT episode_id
                FROM episodes
                WHERE run_id = :run_id
                ORDER BY start_time
                LIMIT 1 OFFSET :offset
            """)
            params = {'run_id': run_id, 'offset': episode_index_target - 1}
            df = pd.read_sql_query(query, connection, params=params)

            if df.empty:
                print(f"No episode found for run_id '{run_id}' at index {episode_index_target}.")
                # Fallback: Get the latest episode ID for this run if the target index doesn't exist
                print("Attempting to fetch the latest episode ID for this run...")
                query_latest = text("""
                    SELECT episode_id
                    FROM episodes
                    WHERE run_id = :run_id
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                df_latest = pd.read_sql_query(query_latest, connection, params={'run_id': run_id})
                if df_latest.empty:
                    print(f"No episodes found at all for run_id: {run_id}")
                    return None
                else:
                    latest_episode_id = df_latest.iloc[0]['episode_id']
                    print(f"Found latest episode ID instead: {latest_episode_id}")
                    return latest_episode_id
            else:
                episode_id = df.iloc[0]['episode_id']
                print(f"\nFound DB Episode ID {episode_id} for run '{run_id}', episode index {episode_index_target}.")
                return episode_id

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if engine:
            engine.dispose() # Close connection pool

# Define get_latest_run_info before it's called
def get_latest_run_info():
    """Fetches the run_id and parameters for the most recent run from the training_runs table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
        return None
    engine = None
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            query = text("SELECT * FROM training_runs ORDER BY start_time DESC LIMIT 1")
            df = pd.read_sql_query(query, connection)
            if df.empty:
                print("No training runs found in the database.")
                return None
            run_data = df.iloc[0].to_dict()
            if 'parameters' in run_data and isinstance(run_data['parameters'], str):
                 try:
                     run_data['parameters'] = json.loads(run_data['parameters'])
                 except json.JSONDecodeError:
                     print("Warning: Could not parse 'parameters' column as JSON.")
            # Also print the info when called directly
            latest_run_id = run_data.get('run_id', 'N/A')
            print(f"\nLatest Run Info (run_id: {latest_run_id}):")
            print(json.dumps(run_data, indent=4, default=str))
            return run_data
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    # Example: Get the DB ID for the 376th episode of the latest run
    # First, get the latest run_id
    latest_run_info = get_latest_run_info() # Now defined above
    if latest_run_info:
        latest_run_id = latest_run_info['run_id']
        target_episode_index = 376 # The episode number we saw completed
        get_episode_id_for_run(latest_run_id, target_episode_index)
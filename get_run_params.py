# get_run_params.py
import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

RUN_ID = 'RUN-SPY-20250505215547-118de704'

def get_run_parameters(run_id):
    """Fetches parameters for a specific run_id from the training_runs table using DATABASE_URL."""
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
            query = text("SELECT * FROM training_runs WHERE run_id = :run_id")
            df = pd.read_sql_query(query, connection, params={'run_id': run_id})

            if df.empty:
                print(f"No data found for run_id: {run_id}")
                return None

            # Convert DataFrame row to a dictionary
            # Assuming only one row matches the unique run_id
            run_data = df.iloc[0].to_dict()

            # Convert parameters from string back to dict if stored as JSON string
            if 'parameters' in run_data and isinstance(run_data['parameters'], str):
                 try:
                     run_data['parameters'] = json.loads(run_data['parameters'])
                 except json.JSONDecodeError:
                     print("Warning: Could not parse 'parameters' column as JSON.")


            print(f"\nParameters for run_id: {run_id}")
            # Pretty print the dictionary, handling potential non-serializable types if any
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
            engine.dispose() # Close connection pool

if __name__ == "__main__":
    get_run_parameters(RUN_ID)
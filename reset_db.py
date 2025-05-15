# reset_db.py
"""
Script to reliably and completely empty all data from the relevant database tables
for the Reinforce Strategy Creator project.

This script connects to the database using the configuration in the .env file
and executes DELETE statements for the tables:
- trading_operations
- trades
- steps
- episodes
- training_runs

It is intended for use in autonomous testing loops to ensure a clean database
state before each test run.
"""
import sys
import os
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Add project root to path to allow imports from reinforcestrategycreator
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Attempting to reset database tables...")

try:
    # Import after setting path
    from reinforcestrategycreator.db_utils import get_db_session, engine
    # Import models to ensure they are loaded and Base.metadata is populated
    from reinforcestrategycreator.db_models import (
        TradingOperation, Trade, Step, Episode, TrainingRun
    )

    if engine is None:
        print("Database engine not initialized. Check .env file and DB connection.")
        sys.exit(1)

    print(f"Using database engine: {engine.url}")

    # Define the order of deletion to respect foreign key constraints
    # Delete from child tables first
    tables_to_clear = [
        TradingOperation.__table__,
        Trade.__table__,
        Step.__table__,
        Episode.__table__,
        TrainingRun.__table__,
    ]

    with get_db_session() as db:
        for table in tables_to_clear:
            print(f"Clearing table: {table.name}...")
            # Use SQLAlchemy's delete construct for clarity and safety
            delete_stmt = table.delete()
            result = db.execute(delete_stmt)
            print(f"Cleared {result.rowcount} rows from {table.name}.")

        db.commit()
        print("Database tables reset successfully.")

except ImportError as e:
    print(f"Import error: {e}. Make sure you are running this script from the project root "
          "or the necessary modules are in the Python path.")
    sys.exit(1)
except OperationalError as e:
    print(f"Database operational error: {e}. Check database connection and permissions.")
    sys.exit(1)
except SQLAlchemyError as e:
    print(f"A SQLAlchemy error occurred: {e}")
    db.rollback() # Rollback changes in case of error
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during database reset: {e}")
    if 'db' in locals() and db.in_transaction():
        db.rollback() # Rollback if a session was created and a transaction is active
    sys.exit(1)
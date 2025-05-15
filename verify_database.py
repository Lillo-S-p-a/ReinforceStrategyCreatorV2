import os
import sys
import yaml
import argparse
from sqlalchemy import text, func
from sqlalchemy.orm import aliased

# Add the parent directory to the sys.path to import reinforcestrategycreator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Base, TrainingRun, Episode, Step, Trade, TradingOperation

# Define critical columns based on TASK-DB-20250515-154500.md
CRITICAL_COLUMNS = {
    "training_runs": [
        {"column": "run_id", "reason": "Primary key, unique identifier for the run.", "validation": "Must be a non-empty string."},
        {"column": "start_time", "reason": "Records when the training run began.", "validation": "Must be a valid timestamp."},
        {"column": "end_time", "reason": "Records when the training run finished. Essential for completed runs.", "validation": "Must be a valid timestamp, not NULL for status 'completed'."},
        {"column": "status", "reason": "Indicates the final state of the training run.", "validation": "Must be a non-empty string, expected to be 'completed' for successful runs."},
    ],
    "episodes": [
        {"column": "episode_id", "reason": "Primary key, unique identifier for the episode.", "validation": "Must be a non-NULL integer."},
        {"column": "run_id", "reason": "Foreign key linking to the parent training run.", "validation": "Must be a non-empty string, referencing a valid run_id."},
        {"column": "rllib_episode_id", "reason": "Unique identifier assigned by RLlib.", "validation": "Must be a non-empty string."},
        {"column": "start_time", "reason": "Records when the episode began.", "validation": "Must be a valid timestamp."},
        {"column": "end_time", "reason": "Records when the episode finished. Essential for completed episodes.", "validation": "Must be a valid timestamp, not NULL for status 'completed'."},
        {"column": "initial_portfolio_value", "reason": "Starting portfolio value for the episode.", "validation": "Must be a non-NULL float."},
        {"column": "final_portfolio_value", "reason": "Ending portfolio value for the episode. Essential for completed episodes.", "validation": "Must be a non-NULL float, not NULL for status 'completed'."},
        {"column": "status", "reason": "Indicates the final state of the episode.", "validation": "Must be a non-empty string, expected to be 'completed' for completed episodes."},
        {"column": "pnl", "reason": "Profit and Loss for the episode. Essential for completed episodes.", "validation": "Must be a non-NULL float, not NULL for status 'completed'."},
        {"column": "total_reward", "reason": "Total reward accumulated during the episode. Essential for completed episodes.", "validation": "Must be a non-NULL float, not NULL for status 'completed'."},
        {"column": "total_steps", "reason": "Total number of steps executed in the episode. Essential for completed episodes.", "validation": "Must be a non-NULL integer, not NULL for status 'completed'."},
    ],
    "steps": [
        {"column": "step_id", "reason": "Primary key, unique identifier for the step.", "validation": "Must be a non-NULL integer."},
        {"column": "episode_id", "reason": "Foreign key linking to the parent episode.", "validation": "Must be a non-NULL integer, referencing a valid episode_id."},
        {"column": "timestamp", "reason": "Records when the step occurred.", "validation": "Must be a valid timestamp."},
        {"column": "portfolio_value", "reason": "Portfolio value at this step.", "validation": "Must be a non-NULL float."},
        {"column": "reward", "reason": "Reward received at this step.", "validation": "Must be a non-NULL float."},
        {"column": "action", "reason": "Action taken by the agent at this step.", "validation": "Must be a non-empty string."},
        {"column": "position", "reason": "Agent's position after the action.", "validation": "Must be a non-empty string."},
    ],
    "trades": [
        {"column": "trade_id", "reason": "Primary key, unique identifier for the trade.", "validation": "Must be a non-NULL integer."},
        {"column": "episode_id", "reason": "Foreign key linking to the parent episode.", "validation": "Must be a non-NULL integer, referencing a valid episode_id."},
        {"column": "entry_time", "reason": "Timestamp when the trade was entered.", "validation": "Must be a valid timestamp."},
        {"column": "exit_time", "reason": "Timestamp when the trade was exited. Essential for closed trades.", "validation": "Must be a valid timestamp, not NULL for closed trades."},
        {"column": "entry_price", "reason": "Price at which the trade was entered.", "validation": "Must be a non-NULL float."},
        {"column": "exit_price", "reason": "Price at which the trade was exited. Essential for closed trades.", "validation": "Must be a non-NULL float, not NULL for closed trades."},
        {"column": "quantity", "reason": "Quantity of the asset traded.", "validation": "Must be a non-NULL float, greater than 0."},
        {"column": "direction", "reason": "Direction of the trade (long/short).", "validation": "Must be a non-empty string ('long' or 'short')."},
        {"column": "pnl", "reason": "Profit or Loss for the trade. Essential for closed trades.", "validation": "Must be a non-NULL float, not NULL for closed trades."},
        {"column": "costs", "reason": "Transaction costs associated with the trade. Essential for closed trades.", "validation": "Must be a non-NULL float, not NULL for closed trades."},
    ],
    "trading_operations": [
        {"column": "operation_id", "reason": "Primary key, unique identifier for the operation.", "validation": "Must be a non-NULL integer."},
        {"column": "step_id", "reason": "Foreign key linking to the parent step.", "validation": "Must be a non-NULL integer, referencing a valid step_id."},
        {"column": "episode_id", "reason": "Foreign key linking to the parent episode.", "validation": "Must be a non-NULL integer, referencing a valid episode_id."},
        {"column": "timestamp", "reason": "Records when the operation occurred.", "validation": "Must be a valid timestamp."},
        {"column": "operation_type", "reason": "Type of trading operation (e.g., ENTRY_LONG, EXIT_SHORT).", "validation": "Must be a valid OperationType enum value."},
        {"column": "size", "reason": "Size of the operation (e.g., quantity of shares).", "validation": "Must be a non-NULL float, greater than or equal to 0."},
        {"column": "price", "reason": "Execution price of the operation.", "validation": "Must be a non-NULL float, greater than or equal to 0."},
    ],
}

TABLE_MODELS = {
    "training_runs": TrainingRun,
    "episodes": Episode,
    "steps": Step,
    "trades": Trade,
    "trading_operations": TradingOperation,
}

def check_table_population(session, run_id):
    """Checks if target tables are populated for a given run_id."""
    print("\n--- Checking Table Population ---")
    all_populated = True
    for table_name, model in TABLE_MODELS.items():
        count = session.query(model).filter(model.run_id == run_id).count() if hasattr(model, 'run_id') else session.query(model).join(Episode).filter(Episode.run_id == run_id).count()
        if count == 0:
            print(f"❌ FAIL: Table '{table_name}' is empty for run_id '{run_id}'.")
            all_populated = False
        else:
            print(f"✅ PASS: Table '{table_name}' is populated ({count} records) for run_id '{run_id}'.")
    return all_populated

def check_null_values_in_critical_columns(session, run_id):
    """Checks for NULL values in critical columns for a given run_id."""
    print("\n--- Checking Critical Columns for NULL Values ---")
    all_non_null = True
    for table_name, columns in CRITICAL_COLUMNS.items():
        model = TABLE_MODELS.get(table_name)
        if not model:
            print(f"⚠️ WARNING: Model not found for table '{table_name}'. Skipping NULL check.")
            continue

        print(f"Checking table '{table_name}'...")
        
        # Filter by run_id if the model has a run_id column, otherwise join with Episode
        if hasattr(model, 'run_id'):
            query = session.query(model).filter(model.run_id == run_id)
        elif table_name in ['episodes', 'steps', 'trades', 'trading_operations']: # These tables are linked to episodes which have run_id
             query = session.query(model).join(Episode).filter(Episode.run_id == run_id)
        else:
             print(f"⚠️ WARNING: Cannot filter table '{table_name}' by run_id. Skipping NULL check.")
             continue

        records = query.all()

        for record in records:
            record_id = getattr(record, f"{table_name[:-1]}_id", "N/A") # Get primary key for logging
            for col_info in columns:
                col_name = col_info["column"]
                # Special handling for columns that are allowed to be NULL under certain conditions (e.g., end_time for running episodes)
                if table_name == 'training_runs' and col_name in ['end_time'] and record.status != 'completed':
                     continue
                if table_name == 'episodes' and col_name in ['end_time', 'final_portfolio_value', 'pnl', 'total_reward', 'total_steps'] and record.status != 'completed':
                     continue
                if table_name == 'trades' and col_name in ['exit_time', 'exit_price', 'pnl', 'costs'] and record.exit_time is None: # Assuming exit_time is NULL for open trades
                     continue

                value = getattr(record, col_name)
                if value is None:
                    print(f"❌ FAIL: NULL value found in table '{table_name}', column '{col_name}', record ID '{record_id}'. Reason: {col_info['reason']}")
                    all_non_null = False
    return all_non_null

def count_episodes(session, run_id):
    """Counts the number of episodes for a given run_id."""
    print("\n--- Counting Episodes ---")
    count = session.query(Episode).filter(Episode.run_id == run_id).count()
    print(f"📊 Episode Count: Found {count} episodes for run_id '{run_id}'.")
    return count # Return count for potential further checks

def check_data_consistency(session, run_id):
    """Performs basic data consistency checks for a given run_id."""
    print("\n--- Checking Data Consistency ---")
    all_consistent = True

    # Check if episode_id in steps, trades, and trading_operations exists in episodes table
    print("Checking foreign key consistency for episode_id...")
    for related_table, related_model in {"steps": Step, "trades": Trade, "trading_operations": TradingOperation}.items():
        # Find records in related table whose episode_id does not exist in the Episode table for the given run
        invalid_records = session.query(related_model).join(Episode).filter(Episode.run_id == run_id).filter(~related_model.episode_id.in_(session.query(Episode.episode_id).filter(Episode.run_id == run_id))).limit(10).all() # Limit to 10 for brevity
        if invalid_records:
            print(f"❌ FAIL: Found records in '{related_table}' with episode_id not present in 'episodes' for run_id '{run_id}'. Examples:")
            for record in invalid_records:
                 record_id = getattr(record, f"{related_table[:-1]}_id", "N/A")
                 print(f"  - Table: '{related_table}', Record ID: '{record_id}', Invalid episode_id: {record.episode_id}")
            all_consistent = False
        else:
            print(f"✅ PASS: Foreign key consistency for episode_id in '{related_table}' is valid for run_id '{run_id}'.")

    # Add more consistency checks here as needed
    # Example: Check if step_id in trading_operations exists in steps table
    print("Checking foreign key consistency for step_id in trading_operations...")
    invalid_operation_steps = session.query(TradingOperation).join(Step).join(Episode).filter(Episode.run_id == run_id).filter(~TradingOperation.step_id.in_(session.query(Step.step_id).join(Episode).filter(Episode.run_id == run_id))).limit(10).all()
    if invalid_operation_steps:
         print(f"❌ FAIL: Found records in 'trading_operations' with step_id not present in 'steps' for run_id '{run_id}'. Examples:")
         for record in invalid_operation_steps:
              print(f"  - Table: 'trading_operations', Record ID: '{record.operation_id}', Invalid step_id: {record.step_id}")
         all_consistent = False
    else:
         print(f"✅ PASS: Foreign key consistency for step_id in 'trading_operations' is valid for run_id '{run_id}'.")


    return all_consistent

def main():
    parser = argparse.ArgumentParser(
        description="""
Perform comprehensive database verification checks for a specific training run.

This script connects to the database specified by the DATABASE_URL environment variable
and performs checks on the 'training_runs', 'episodes', 'steps', 'trades', and
'trading_operations' tables related to the provided run_id.

Checks include:
- Ensuring tables are populated for the given run_id.
- Checking for NULL values in critical columns.
- Verifying foreign key consistency between related tables (episodes, steps, trades, trading_operations).

The script exits with status 0 on success and 1 if any verification check fails or
if the specified run_id is not found.
"""
    )
    parser.add_argument(
        "run_id",
        help="""
The unique ID of the training run to verify.
This ID is typically generated at the start of a training run and can be found
in training logs or by querying the 'training_runs' table in the database.
Example: RUN-SPY-YYYYMMDDHHMMSS-HASH
"""
    )
    args = parser.parse_args()

    run_id = args.run_id
    overall_status = True

    try:
        with get_db_session() as session:
            print(f"Starting database verification for run_id: {run_id}")

            # Check if the run_id exists
            run_exists = session.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run_exists:
                print(f"❌ ERROR: Run ID '{run_id}' not found in the database.")
                sys.exit(1)

            # Perform checks
            overall_status &= check_table_population(session, run_id)
            overall_status &= check_null_values_in_critical_columns(session, run_id)
            episode_count = count_episodes(session, run_id) # Store count if needed later
            overall_status &= check_data_consistency(session, run_id)

            print("\n--- Verification Summary ---")
            if overall_status:
                print(f"✅ Overall Status: PASS for run_id '{run_id}'. Database verification completed successfully.")
            else:
                print(f"❌ Overall Status: FAIL for run_id '{run_id}'. Database verification found issues.")
                sys.exit(1) # Exit with a non-zero code on failure

    except Exception as e:
        print(f"\n❌ An error occurred during verification: {e}")
        sys.exit(1) # Exit with a non-zero code on error

if __name__ == "__main__":
    main()
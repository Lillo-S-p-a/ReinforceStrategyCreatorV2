# verify_operations.py
import sys
from sqlalchemy import exists, select, and_
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import TradingOperation

def check_operations_exist(episode_ids):
    """
    Checks if TradingOperation records exist for the given episode IDs.

    Args:
        episode_ids (list): A list of episode IDs to check.

    Returns:
        dict: A dictionary mapping episode_id to a boolean indicating existence.
    """
    results = {}
    try:
        with get_db_session() as db:
            print("Successfully obtained database session.")
            for ep_id in episode_ids:
                # Construct an exists query for the specific episode_id
                stmt = select(exists().where(TradingOperation.episode_id == ep_id))
                # Execute the query and get the scalar result (True or False)
                operation_exists = db.execute(stmt).scalar()
                results[ep_id] = operation_exists
                print(f"Checked episode_id {ep_id}: Exists = {operation_exists}")
        return results
    except Exception as e:
        print(f"An error occurred during database query: {e}", file=sys.stderr)
        # Ensure partial results are not returned on error, or handle as needed
        return {ep_id: "Error" for ep_id in episode_ids} # Indicate error for all

if __name__ == "__main__":
    target_episode_ids = [45, 46]
    print(f"Checking for TradingOperation data for episode IDs: {target_episode_ids}")
    existence_results = check_operations_exist(target_episode_ids)

    print("\n--- Verification Results ---")
    for ep_id, exists_flag in existence_results.items():
        if exists_flag == "Error":
             print(f"Data exists for {ep_id}: Error during check")
        else:
            print(f"Data exists for {ep_id}: {'Yes' if exists_flag else 'No'}")
    print("--------------------------")

    # Exit with non-zero code if any check resulted in an error
    if any(v == "Error" for v in existence_results.values()):
        sys.exit(1)
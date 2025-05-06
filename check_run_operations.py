import argparse
import logging
from sqlalchemy import create_engine, text # create_engine and text are not used if Option 2 is commented
from sqlalchemy.orm import sessionmaker # sessionmaker is not used if Option 2 is commented
from reinforcestrategycreator.db_models import TradingOperation, OperationType, Episode # Assuming OperationType is an Enum, added Episode
from reinforcestrategycreator.db_utils import get_db_session # For direct engine creation if needed
from collections import Counter

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_operations_summary(db_session, run_id_to_check: str):
    """
    Queries trading operations for a given run_id and returns a summary.
    """
    logging.info(f"Querying trading operations for run_id: {run_id_to_check}")

    # Join TradingOperation with Episode to filter by run_id
    operations = db_session.query(TradingOperation.operation_type)\
        .join(Episode, TradingOperation.episode_id == Episode.episode_id)\
        .filter(Episode.run_id == run_id_to_check)\
        .all()

    if not operations:
        logging.warning(f"No trading operations found for run_id: {run_id_to_check}")
        return None

    # Count occurrences of each operation type
    # Assuming operation_type is stored as an Enum or string that can be directly counted
    operation_counts = Counter([op.operation_type for op in operations])

    logging.info(f"Operations summary for run_id {run_id_to_check}:")
    for op_type, count in operation_counts.items():
        # If OperationType is an Enum, op_type.name might be better for display
        op_type_name = op_type.name if hasattr(op_type, 'name') else str(op_type)
        logging.info(f"  {op_type_name}: {count}")

    return operation_counts

def main():
    parser = argparse.ArgumentParser(description="Check trading operations for a specific run_id.")
    parser.add_argument("run_id", type=str, help="The run_id to check operations for.")
    args = parser.parse_args()

    run_id_to_check = args.run_id

    # Option 1: Use existing get_db_session utility
    with get_db_session() as db:
        try:
            summary = get_operations_summary(db, run_id_to_check)
            if summary is None:
                print(f"No operations found for run_id: {run_id_to_check}")
            else:
                print(f"\nSummary for run_id: {run_id_to_check}")
                for op_type, count in summary.items():
                    op_type_name = op_type.name if hasattr(op_type, 'name') else str(op_type)
                    print(f"  {op_type_name}: {count}")
        except Exception as e:
            logging.error(f"Error querying database: {e}", exc_info=True)
            print(f"An error occurred: {e}")

    # Option 2: Direct engine creation (if get_db_session is problematic or for standalone use)
    # engine = create_engine(DATABASE_URL)
    # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # db = SessionLocal()
    # try:
    #     summary = get_operations_summary(db, run_id_to_check)
    #     # ... print summary ...
    # finally:
    #     db.close()

if __name__ == "__main__":
    main()
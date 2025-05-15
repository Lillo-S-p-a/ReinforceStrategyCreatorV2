import os
import logging
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Assuming db_models.py defines Step, TrainingRun, and Episode models
from reinforcestrategycreator.db_models import Step, TrainingRun, Episode, Base

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chats.db")

if not DATABASE_URL:
    logger.error("DATABASE_URL not set in .env file.")
    exit(1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables are created (idempotent)
Base.metadata.create_all(bind=engine)

def verify_steps_for_run(run_id_to_check: str):
    """
    Queries the steps table for a given run_id and checks for NULL values
    in critical columns.
    """
    db: Session = SessionLocal()
    try:
        logger.info(f"Verifying steps for run_id: {run_id_to_check}")

        # First, check if the run exists
        training_run_obj = db.query(TrainingRun).filter(TrainingRun.run_id == run_id_to_check).first()
        if not training_run_obj:
            logger.warning(f"TrainingRun with ID '{run_id_to_check}' not found in the database.")
            return False, 0, 0 # Found, Non-Null, Null

        stmt = (
            select(
                Step.portfolio_value,
                Step.asset_price,
                Step.action,
                Step.position,
                Step.step_id # For context, using step_id instead of current_step
            )
            .join(Episode, Step.episode_id == Episode.episode_id) # Join Step to Episode
            .join(TrainingRun, Episode.run_id == TrainingRun.run_id) # Join Episode to TrainingRun
            .where(TrainingRun.run_id == run_id_to_check)
            .order_by(Step.step_id) # Order by step_id
        )
        results = db.execute(stmt).fetchall()

        if not results:
            logger.warning(f"No steps found for run_id: {run_id_to_check}")
            return True, 0, 0 # Run found, but no steps

        logger.info(f"Found {len(results)} steps for run_id: {run_id_to_check}")

        null_entries_count = 0
        non_null_entries_count = 0

        for i, row in enumerate(results):
            portfolio_value, asset_price, action, position, step_id_val = row # Renamed current_step to step_id_val
            is_null = False
            if portfolio_value is None:
                logger.error(f"Step ID {step_id_val}: portfolio_value is NULL")
                is_null = True
            if asset_price is None:
                logger.error(f"Step ID {step_id_val}: asset_price is NULL")
                is_null = True
            if action is None: # Action can be 0 (HOLD), so check for None explicitly
                logger.error(f"Step ID {step_id_val}: action is NULL")
                is_null = True
            if position is None: # Position can be 0, so check for None explicitly
                logger.error(f"Step ID {step_id_val}: position is NULL")
                is_null = True
            
            if is_null:
                null_entries_count += 1
            else:
                non_null_entries_count +=1
                if i < 5 or i > len(results) - 6 : # Log first 5 and last 5 non-null entries
                    logger.info(
                        f"Step ID {step_id_val}: PV={portfolio_value}, Price={asset_price}, "
                        f"Action={action}, Pos={position}"
                    )
        
        if null_entries_count > 0:
            logger.error(
                f"Verification FAILED: Found {null_entries_count} steps with NULL "
                f"values out of {len(results)} total steps."
            )
            return True, non_null_entries_count, null_entries_count
        else:
            logger.info(
                "Verification SUCCESS: All queried steps have non-NULL values for "
                "portfolio_value, asset_price, action, and position."
            )
            return True, non_null_entries_count, null_entries_count

    except Exception as e:
        logger.error(f"An error occurred during verification: {e}", exc_info=True)
        return False, 0, 0
    finally:
        db.close()

if __name__ == "__main__":
    target_run_id = "RLlibDBG-SPY-20250515105755-6f797c51"
    run_found, non_null_count, null_count = verify_steps_for_run(target_run_id)

    if run_found:
        if null_count == 0 and non_null_count > 0:
            logger.info(f"Successfully verified {non_null_count} steps. All critical fields are non-NULL.")
        elif non_null_count == 0 and null_count == 0:
             logger.info(f"Run {target_run_id} found, but no steps recorded for it.")
        else:
            logger.error(f"Verification for run {target_run_id} completed with issues: {null_count} NULL steps, {non_null_count} non-NULL steps.")
    else:
        logger.error(f"Could not perform verification as run {target_run_id} was not found or an error occurred.")
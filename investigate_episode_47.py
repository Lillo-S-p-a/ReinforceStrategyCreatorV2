import sys
import os

# Add the project root to the Python path to allow importing reinforcestrategycreator
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from reinforcestrategycreator.db_utils import get_db_session
    from reinforcestrategycreator.db_models import Episode, TrainingRun
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure the script is run from the project root directory or the path is correctly set.")
    sys.exit(1)

EPISODE_ID_TO_CHECK = 47

print(f"--- Investigating Episode ID: {EPISODE_ID_TO_CHECK} ---")

try:
    with get_db_session() as db:
        print(f"1. Checking for Episode with id={EPISODE_ID_TO_CHECK}...")
        episode = db.query(Episode).filter(Episode.episode_id == EPISODE_ID_TO_CHECK).first()

        if episode:
            print(f"   ✅ Found Episode with id={EPISODE_ID_TO_CHECK}.")
            print(f"2. Checking associated training_run_id...")
            run_id = episode.run_id
            if run_id:
                print(f"   ✅ Episode has training_run_id: '{run_id}'.")
                print(f"3. Checking for TrainingRun with run_id='{run_id}'...")
                training_run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
                if training_run:
                    print(f"   ✅ Found corresponding TrainingRun.")
                    print("\n--- Summary ---")
                    print(f"Episode {EPISODE_ID_TO_CHECK} exists and has a valid, linked TrainingRun ('{run_id}').")
                    print("The 404 error is likely NOT due to missing related records in the database.")
                else:
                    print(f"   ❌ ERROR: Corresponding TrainingRun with run_id='{run_id}' NOT FOUND.")
                    print("\n--- Summary ---")
                    print(f"Episode {EPISODE_ID_TO_CHECK} exists but its associated TrainingRun ('{run_id}') is missing.")
                    print("This is the likely cause of the 404 error.")
            else:
                print(f"   ❌ ERROR: Episode exists but has a NULL or empty training_run_id.")
                print("\n--- Summary ---")
                print(f"Episode {EPISODE_ID_TO_CHECK} exists but is missing its training_run_id.")
                print("This is the likely cause of the 404 error.")
        else:
            print(f"   ❌ ERROR: Episode with id={EPISODE_ID_TO_CHECK} NOT FOUND.")
            print("\n--- Summary ---")
            print(f"Episode {EPISODE_ID_TO_CHECK} does not exist in the database.")
            print("This is the likely cause of the 404 error.")

except Exception as e:
    print(f"\n--- An error occurred during database investigation ---")
    print(f"Error: {e}")
    sys.exit(1)

print("\n--- Investigation Complete ---")
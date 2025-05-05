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

EPISODE_ID_TO_CHECK = 47  # Same as in the investigation script

print(f"--- Testing Fixed Relationship for Episode ID: {EPISODE_ID_TO_CHECK} ---")

try:
    with get_db_session() as db:
        print(f"1. Querying Episode with id={EPISODE_ID_TO_CHECK}...")
        episode = db.query(Episode).filter(Episode.episode_id == EPISODE_ID_TO_CHECK).first()

        if episode:
            print(f"   ✅ Found Episode with id={EPISODE_ID_TO_CHECK}.")
            print(f"2. Checking run_id value...")
            run_id = episode.run_id
            print(f"   ✅ Episode has run_id: '{run_id}'.")
            
            print(f"3. Testing the fixed relationship by accessing episode.training_run...")
            training_run = episode.training_run
            
            if training_run:
                print(f"   ✅ SUCCESS: episode.training_run correctly returns the TrainingRun object!")
                print(f"   Training Run ID: {training_run.run_id}")
                print(f"   Training Run Parameters: {training_run.parameters}")
                print("\n--- Summary ---")
                print(f"The relationship fix was successful. The API endpoint should now work correctly.")
            else:
                print(f"   ❌ ERROR: episode.training_run is still None despite the fix.")
                print("\n--- Summary ---")
                print(f"The relationship fix did not resolve the issue. Further investigation needed.")
        else:
            print(f"   ❌ ERROR: Episode with id={EPISODE_ID_TO_CHECK} NOT FOUND.")
            print("\n--- Summary ---")
            print(f"Cannot test the relationship fix because Episode {EPISODE_ID_TO_CHECK} does not exist.")

except Exception as e:
    print(f"\n--- An error occurred during testing ---")
    print(f"Error: {e}")
    sys.exit(1)

print("\n--- Test Complete ---")
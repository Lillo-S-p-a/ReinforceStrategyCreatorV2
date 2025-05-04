# init_db.py
import sys
import os

# Add project root to path to allow imports from reinforcestrategycreator
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Attempting to initialize database schema...")

try:
    # Import after setting path
    from reinforcestrategycreator.db_utils import init_db, engine
    from reinforcestrategycreator.db_models import Base # Ensure models are loaded

    if engine is None:
        print("Database engine not initialized. Check .env file and DB connection.")
        sys.exit(1)

    print(f"Using database engine: {engine.url}")
    # The init_db function uses the global engine by default
    init_db()
    print("Database schema initialization complete.")

except ImportError as e:
    print(f"Import error: {e}. Make sure you are running this script from the project root "
          "or the necessary modules are in the Python path.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during database initialization: {e}")
    sys.exit(1)
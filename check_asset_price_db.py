import sys
from sqlalchemy import desc, func
from reinforcestrategycreator.db_utils import get_db_session, SessionLocal # Import SessionLocal for check
from reinforcestrategycreator.db_models import Episode, Step

def check_latest_episode_asset_prices():
    print("Checking database session factory initialization...")
    if SessionLocal is None:
        print("Database session factory (SessionLocal) is not initialized. "
              "Check DATABASE_URL environment variable and database connectivity.", file=sys.stderr)
        sys.exit(1)
    print("Session factory seems initialized.")

    print("Attempting to connect to database and query...")
    try:
        with get_db_session() as db:
            print("Successfully connected. Finding the latest episode...")
            latest_episode = db.query(Episode).order_by(desc(Episode.episode_id)).first()

            if not latest_episode:
                print("No episodes found in the database.")
                return

            episode_id = latest_episode.episode_id
            print(f"Latest episode ID found: {episode_id}")
            print(f"Querying steps for episode {episode_id} to check asset_price...")

            # Query to count total steps and null/non-null asset_price
            step_counts = db.query(
                func.count(Step.step_id).label("total_steps"),
                func.count(Step.asset_price).label("non_null_asset_price_count") # func.count(column) ignores NULLs
            ).filter(Step.episode_id == episode_id).one()

            total_steps = step_counts.total_steps
            non_null_count = step_counts.non_null_asset_price_count
            null_count = total_steps - non_null_count

            print(f"\n--- Results for Episode {episode_id} ---")
            print(f"Total steps recorded: {total_steps}")
            print(f"Steps with non-NULL asset_price: {non_null_count}")
            print(f"Steps with NULL asset_price: {null_count}")
            print("------------------------------------")


            if total_steps == 0:
                 print("\nNo steps found for this episode.")
            elif non_null_count == 0:
                 print("\nConclusion: All recorded asset_price values are NULL in the database for the latest episode.")
            else:
                print("\nConclusion: Valid (non-NULL) asset_price values exist in the database.")
                print("Fetching first 5 non-NULL asset_price values as examples:")
                non_null_steps = db.query(Step.asset_price)\
                                   .filter(Step.episode_id == episode_id, Step.asset_price.isnot(None))\
                                   .limit(5).all()
                for i, price in enumerate(non_null_steps):
                    print(f" - Example {i+1}: {price[0]}")

    except Exception as e:
        print(f"\nAn error occurred during database query: {e}", file=sys.stderr)
        print("Please check database connection details and table existence.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    check_latest_episode_asset_prices()
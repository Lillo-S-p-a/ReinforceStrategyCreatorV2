import psycopg2
import os

# Database connection parameters
DB_HOST = "localhost"
DB_PORT = "5434"
DB_NAME = "trading_db"
DB_USER = "postgres"
DB_PASSWORD = "mysecretpassword"  # Consider using environment variables for sensitive data

# Run ID to check
RUN_ID_TO_CHECK = "RLlibDBG-SPY-20250515104140-38b8da7e"

def get_episode_counts(run_id):
    """
    Connects to the PostgreSQL database and retrieves episode counts for a given run_id.

    Args:
        run_id (str): The run_id to filter episodes by.

    Returns:
        tuple: (total_episodes, completed_episodes) or (None, None) if an error occurs.
    """
    conn = None
    total_episodes = None
    completed_episodes = None

    try:
        # Establish connection
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()

        # Query for total episodes
        query_total = "SELECT COUNT(*) FROM episodes WHERE run_id = %s;"
        cur.execute(query_total, (run_id,))
        total_episodes_result = cur.fetchone()
        if total_episodes_result:
            total_episodes = total_episodes_result[0]

        # Query for completed episodes
        query_completed = "SELECT COUNT(*) FROM episodes WHERE run_id = %s AND status = %s;"
        cur.execute(query_completed, (run_id, 'completed'))
        completed_episodes_result = cur.fetchone()
        if completed_episodes_result:
            completed_episodes = completed_episodes_result[0]

        cur.close()
    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL or executing query: {error}")
        return None, None
    finally:
        if conn:
            conn.close()
            # print("PostgreSQL connection is closed")
    return total_episodes, completed_episodes

if __name__ == "__main__":
    print(f"Checking episode counts for run_id: {RUN_ID_TO_CHECK}")
    total, completed = get_episode_counts(RUN_ID_TO_CHECK)

    if total is not None:
        print(f"Total episodes for run_id '{RUN_ID_TO_CHECK}': {total}")
    else:
        print(f"Could not retrieve total episode count for run_id '{RUN_ID_TO_CHECK}'.")

    if completed is not None:
        print(f"Completed episodes for run_id '{RUN_ID_TO_CHECK}': {completed}")
    else:
        print(f"Could not retrieve completed episode count for run_id '{RUN_ID_TO_CHECK}'.")
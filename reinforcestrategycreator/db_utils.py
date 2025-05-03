import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv
from .db_models import Base # Import Base from the models file

# Load environment variables from .env file if it exists
load_dotenv()

def get_database_url():
    """
    Retrieves the database connection URL from environment variables.
    Raises an error if the variable is not set.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set. "
                         "Please set it (e.g., postgresql://user:password@host:port/database)")
    return db_url

def get_engine(db_url=None):
    """
    Creates and returns a SQLAlchemy engine.
    Uses the DATABASE_URL environment variable by default.
    """
    if db_url is None:
        db_url = get_database_url()
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        # Test connection
        with engine.connect() as connection:
            pass # Connection successful if no exception
        print("Database connection successful.")
        return engine
    except OperationalError as e:
        print(f"Error connecting to the database: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during engine creation: {e}")
        raise

# Create engine instance (can be imported elsewhere)
# Handle potential errors during initial import/setup
try:
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except (ValueError, OperationalError, Exception) as e:
    print(f"Failed to initialize database engine on module load: {e}")
    engine = None
    SessionLocal = None # Ensure SessionLocal is None if engine fails

def get_db_session():
    """
    Provides a database session.
    Handles session creation and closing.
    Usage:
        with get_db_session() as db:
            # use db session here
            db.query(...)
    """
    if SessionLocal is None:
        raise RuntimeError("Database session factory not initialized. Check DATABASE_URL and connection.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(engine_instance=None):
    """
    Initializes the database by creating tables defined in db_models.py.
    Uses the global engine by default.
    """
    if engine_instance is None:
        engine_instance = engine

    if engine_instance is None:
         raise RuntimeError("Database engine not initialized. Cannot create tables.")

    print("Attempting to create database tables...")
    try:
        Base.metadata.create_all(bind=engine_instance)
        print("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise

# Example usage (e.g., in a main script or setup utility)
# if __name__ == "__main__":
#     print("Running DB Utils setup...")
#     try:
#         # Ensure DATABASE_URL is set in your environment or .env file
#         # Example: export DATABASE_URL="postgresql://user:password@localhost:5432/mydatabase"
#         init_db()
#         print("Database initialization check complete.")
#
#         # Example session usage
#         with get_db_session() as session:
#             print("Successfully obtained database session.")
#             # Perform some basic query or operation
#             # result = session.execute(text("SELECT 1"))
#             # print(f"Test query result: {result.scalar()}")
#     except Exception as e:
#         print(f"An error occurred during setup: {e}")
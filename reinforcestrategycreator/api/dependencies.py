from typing import Generator, Annotated
import os

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from reinforcestrategycreator.db_utils import SessionLocal

# --- Database Dependency ---

def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a SQLAlchemy database session per request.
    Ensures the session is closed afterwards.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DBSession = Annotated[Session, Depends(get_db)]

# --- Security Dependency ---

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY") # Load from environment variable for security

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """
    Dependency that extracts and validates the API key from the header.
    Raises HTTPException 401 if the key is missing or invalid.
    """
    if not API_KEY:
        # This should not happen in production, indicates server misconfiguration
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server.",
        )

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header

APIKey = Annotated[str, Depends(get_api_key)]
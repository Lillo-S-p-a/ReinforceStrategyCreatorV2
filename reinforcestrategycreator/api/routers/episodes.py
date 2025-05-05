from typing import List, Optional, Annotated, Dict, Any # Added Dict, Any
import datetime # Add datetime import
from math import ceil
import logging # Add logging import at the top

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session, class_mapper # Added class_mapper

from reinforcestrategycreator import db_models
from reinforcestrategycreator.db_models import TradingOperation, Episode # Added Episode and TradingOperation
from reinforcestrategycreator.api import schemas
from reinforcestrategycreator.api.schemas import episodes as episode_schemas # Import the new schemas
from reinforcestrategycreator.api.schemas import TradingOperationRead # Added TradingOperationRead
from reinforcestrategycreator.api.dependencies import DBSession, APIKey, get_api_key # Import get_api_key
# Removed import of PaginationParams, will define params directly

# Import constants from runs or define them here if preferred
from .runs import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(
    prefix="/episodes",
    tags=["Episodes"],
    # Removed router-level dependency
)

# Import the new schema
from reinforcestrategycreator.api.schemas.episodes import EpisodeIdList

# Helper function to convert SQLAlchemy model instance to dict
def model_to_dict(obj):
    if obj is None:
        return None
    # Get columns from the mapper
    mapper = class_mapper(obj.__class__)
    columns = [c.key for c in mapper.columns]
    # Create dict using column names and getattr
    # Convert datetime objects to ISO format strings for JSON serialization
    d = {}
    for c in columns:
        val = getattr(obj, c)
        if isinstance(val, datetime.datetime):
             d[c] = val.isoformat()
        else:
             d[c] = val
    return d


@router.get("/ids", response_model=EpisodeIdList)
async def get_all_episode_ids(
    db: DBSession,
    api_key: str = Depends(get_api_key), # Reuse existing dependency for consistency
):
    """
    Retrieve a list of all distinct episode IDs available in the database,
    sorted in descending order (newest first).
    """
    try:
        # Query distinct episode IDs and order them descendingly
        query = (
            select(db_models.Episode.episode_id)
            .distinct()
            .order_by(db_models.Episode.episode_id.desc())
        )
        result = db.execute(query).scalars().all()

        # Return the list within the defined schema
        return EpisodeIdList(episode_ids=result)
    except Exception as e:
        # Basic error handling, can be expanded
        print(f"Error fetching episode IDs: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail="Internal server error fetching episode IDs")
# Define constants for this endpoint's pagination (as per task)
DEFAULT_OPERATIONS_PAGE_SIZE = 100
MAX_OPERATIONS_PAGE_SIZE = 1000
@router.get("/{episode_id}", response_model=schemas.EpisodeDetail)
async def get_episode_details(
    episode_id: Annotated[int, Path(description="The ID of the episode to retrieve")],
    db: DBSession,
    api_key: str = Depends(get_api_key), # Add dependency here
):
    """
    Retrieve details for a specific episode by its ID.
    """
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")
    return episode


# Change response_model to expect dict instead of schemas.Step
@router.get("/{episode_id}/steps/", response_model=schemas.PaginatedResponse[dict])
async def list_episode_steps(
    episode_id: Annotated[int, Path(description="The ID of the episode whose steps to retrieve")],
    db: DBSession,
    api_key: str = Depends(get_api_key), # Add dependency here
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")] = DEFAULT_PAGE_SIZE,
):
    """
    Retrieve a paginated list of steps for a specific episode.
    """
    # Check if episode exists first
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    # Calculate skip and limit directly
    limit = min(page_size, MAX_PAGE_SIZE)
    skip = (page - 1) * limit

    # Select the whole Step model object
    query = select(db_models.Step).where(db_models.Step.episode_id == episode_id)

    count_query = select(func.count()).select_from(db_models.Step).where(db_models.Step.episode_id == episode_id)

    # Get total count
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    current_page = (skip // limit) + 1

    # Get paginated Step model objects
    db_steps = db.execute(
        query.order_by(db_models.Step.timestamp.asc()) # Order steps chronologically
             .offset(skip)
             .limit(limit)
    ).scalars().all() # Use scalars().all() to get model instances

    # Convert SQLAlchemy objects to dictionaries
    response_items_as_dicts = []
    for db_row in db_steps:
        try:
            row_dict = model_to_dict(db_row)
            if row_dict:
                 # Ensure asset_price is float or None before adding to dict list
                 if 'asset_price' in row_dict and row_dict['asset_price'] is not None:
                     try:
                         row_dict['asset_price'] = float(row_dict['asset_price'])
                     except (ValueError, TypeError):
                         row_dict['asset_price'] = None # Set to None if conversion fails
                 response_items_as_dicts.append(row_dict)
            else:
                 logging.warning(f"Could not convert step object to dict: {db_row}")

        except Exception as e:
            logging.error(f"Error processing step {getattr(db_row, 'step_id', 'N/A')}: {e}", exc_info=True)
            continue

    # --- Logging the constructed dictionaries ---
    logging.warning(f"API Endpoint: Preparing to return dict steps (first 5): {response_items_as_dicts[:5]}")
    # --- END LOGGING ---

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        items=response_items_as_dicts # Return the list of dictionaries
    )


@router.get("/{episode_id}/trades/", response_model=schemas.PaginatedResponse[schemas.Trade])
async def list_episode_trades(
    episode_id: Annotated[int, Path(description="The ID of the episode whose trades to retrieve")],
    db: DBSession,
    api_key: str = Depends(get_api_key), # Add dependency here
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")] = DEFAULT_PAGE_SIZE,
):
    """
    Retrieve a paginated list of trades for a specific episode.
    """
    # Check if episode exists first
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    # Calculate skip and limit directly
    limit = min(page_size, MAX_PAGE_SIZE)
    skip = (page - 1) * limit

    query = select(db_models.Trade).where(db_models.Trade.episode_id == episode_id)
    count_query = select(func.count()).select_from(db_models.Trade).where(db_models.Trade.episode_id == episode_id)

    # Get total count
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    current_page = (skip // limit) + 1

    # Get paginated items
    trades = db.execute(
        query.order_by(db_models.Trade.entry_time.asc()) # Order trades chronologically
             .offset(skip)
             .limit(limit)
    ).scalars().all()

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        items=trades
    )
@router.get("/{episode_id}/operations/", response_model=schemas.PaginatedResponse[schemas.TradingOperationRead])
async def read_episode_operations(
    episode_id: Annotated[int, Path(description="The ID of the episode whose operations to retrieve")],
    db: DBSession,
    api_key: str = Depends(get_api_key), # Use str as per existing code
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_OPERATIONS_PAGE_SIZE, description="Items per page")] = DEFAULT_OPERATIONS_PAGE_SIZE,
):
    """
    Retrieve a paginated list of trading operations for a specific episode, ordered by timestamp.
    """
    # Check if episode exists first
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    # Calculate skip and limit
    limit = min(page_size, MAX_OPERATIONS_PAGE_SIZE)
    skip = (page - 1) * limit

    # Base query for items
    query = (
        select(db_models.TradingOperation)
        .where(db_models.TradingOperation.episode_id == episode_id)
    )

    # Count query
    count_query = (
        select(func.count())
        .select_from(db_models.TradingOperation)
        .where(db_models.TradingOperation.episode_id == episode_id)
    )

    # Get total count
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    # Ensure current_page calculation is correct even if page requested is too high
    current_page = page if page <= total_pages else total_pages
    if total_items == 0:
        current_page = 1 # Or page, depending on desired behavior for empty results

    # Get paginated items ordered by timestamp
    operations = db.execute(
        query.order_by(db_models.TradingOperation.timestamp.asc())
             .offset(skip)
             .limit(limit)
    ).scalars().all()

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page, # Use calculated current_page
        page_size=limit,
        items=operations # Ensure this matches the schema name 'TradingOperationRead' implicitly
    )
@router.get("/{episode_id}/model/", response_model=episode_schemas.ModelParameters)
async def get_episode_model_parameters(
    episode_id: Annotated[int, Path(description="The ID of the episode whose model parameters to retrieve")],
    db: DBSession,
    api_key: str = Depends(get_api_key), # Add dependency here
):
    """
    Retrieve the training parameters (model configuration) associated with a specific episode's training run.
    """
    # Fetch the episode with its related training run eagerly or check relationship after fetch
    episode = db.query(db_models.Episode).options(
        # Use joinedload to fetch the related training_run in the same query
        # from sqlalchemy.orm import joinedload
        # joinedload(db_models.Episode.training_run) # Uncomment if needed and import joinedload
    ).get(episode_id)

    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    # Access the related training run
    training_run = episode.training_run
    if training_run is None:
        # This case should ideally not happen if the relationship is set up correctly
        # and data integrity is maintained, but it's good practice to check.
        raise HTTPException(status_code=404, detail=f"Training run associated with episode {episode_id} not found")

    # Return the parameters from the training run
    # The ModelParameters schema expects a dictionary with a 'parameters' key
    return episode_schemas.ModelParameters(parameters=training_run.parameters)
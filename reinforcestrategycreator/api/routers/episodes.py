from typing import List, Optional, Annotated
from math import ceil

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session

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


@router.get("/{episode_id}/steps/", response_model=schemas.PaginatedResponse[schemas.Step])
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

    query = select(db_models.Step).where(db_models.Step.episode_id == episode_id)
    count_query = select(func.count()).select_from(db_models.Step).where(db_models.Step.episode_id == episode_id)

    # Get total count
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    current_page = (skip // limit) + 1

    # Get paginated items
    steps = db.execute(
        query.order_by(db_models.Step.timestamp.asc()) # Order steps chronologically
             .offset(skip)
             .limit(limit)
    ).scalars().all()

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        items=steps
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
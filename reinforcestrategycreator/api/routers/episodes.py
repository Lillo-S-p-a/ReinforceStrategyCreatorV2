from typing import List, Optional, Annotated
from math import ceil

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from reinforcestrategycreator import db_models
from reinforcestrategycreator.api import schemas
from reinforcestrategycreator.api.dependencies import DBSession, APIKey
from .runs import PaginationParams # Reuse pagination helper from runs router

router = APIRouter(
    prefix="/episodes",
    tags=["Episodes"],
    dependencies=[Depends(APIKey)] # Apply API Key security
)

@router.get("/{episode_id}", response_model=schemas.EpisodeDetail)
async def get_episode_details(
    episode_id: Annotated[int, Path(description="The ID of the episode to retrieve")],
    db: DBSession,
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
    pagination: PaginationParams,
):
    """
    Retrieve a paginated list of steps for a specific episode.
    """
    # Check if episode exists first
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    skip, limit = pagination
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
    pagination: PaginationParams,
):
    """
    Retrieve a paginated list of trades for a specific episode.
    """
    # Check if episode exists first
    episode = db.get(db_models.Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode with ID {episode_id} not found")

    skip, limit = pagination
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
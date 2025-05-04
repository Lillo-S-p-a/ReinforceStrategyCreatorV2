import datetime
from typing import List, Optional, Annotated
from math import ceil

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from reinforcestrategycreator import db_models
from reinforcestrategycreator.api import schemas
from reinforcestrategycreator.api.dependencies import DBSession, APIKey

router = APIRouter(
    prefix="/runs",
    tags=["Training Runs"],
    dependencies=[Depends(APIKey)] # Apply API Key security to all routes in this router
)

# --- Helper for Pagination ---
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Removed get_pagination_params dependency function and PaginationParams alias

# --- Endpoints ---

@router.get("/", response_model=schemas.PaginatedResponse[schemas.TrainingRun])
async def list_training_runs(
    db: DBSession,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")] = DEFAULT_PAGE_SIZE,
    start_date: Annotated[Optional[datetime.date], Query(description="Filter runs started on or after this date (YYYY-MM-DD)")] = None,
    end_date: Annotated[Optional[datetime.date], Query(description="Filter runs started on or before this date (YYYY-MM-DD)")] = None,
    status: Annotated[Optional[str], Query(description="Filter runs by status (e.g., 'completed', 'running')")] = None,
):
    """
    Retrieve a paginated list of training runs, optionally filtered by date range and status.
    """
    # Calculate skip and limit directly
    limit = min(page_size, MAX_PAGE_SIZE)
    skip = (page - 1) * limit

    query = select(db_models.TrainingRun)
    count_query = select(func.count()).select_from(db_models.TrainingRun)

    # Apply filters
    if start_date:
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        query = query.where(db_models.TrainingRun.start_time >= start_datetime)
        count_query = count_query.where(db_models.TrainingRun.start_time >= start_datetime)
    if end_date:
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        query = query.where(db_models.TrainingRun.start_time <= end_datetime)
        count_query = count_query.where(db_models.TrainingRun.start_time <= end_datetime)
    if status:
        query = query.where(db_models.TrainingRun.status == status)
        count_query = count_query.where(db_models.TrainingRun.status == status)

    # Get total count for pagination
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    current_page = (skip // limit) + 1

    # Get paginated items
    runs = db.execute(
        query.order_by(db_models.TrainingRun.start_time.desc())
             .offset(skip)
             .limit(limit)
    ).scalars().all()

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        items=runs
    )


@router.get("/{run_id}", response_model=schemas.TrainingRunDetail)
async def get_training_run_details(
    run_id: Annotated[str, Path(description="The ID of the training run to retrieve")],
    db: DBSession,
):
    """
    Retrieve details for a specific training run by its ID.
    """
    run = db.get(db_models.TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run with ID '{run_id}' not found")
    return run


# --- Endpoints for Episodes within a Run ---

@router.get("/{run_id}/episodes/", response_model=schemas.PaginatedResponse[schemas.Episode])
async def list_run_episodes(
    run_id: Annotated[str, Path(description="The ID of the training run whose episodes to retrieve")],
    db: DBSession,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")] = DEFAULT_PAGE_SIZE,
    min_pnl: Annotated[Optional[float], Query(description="Filter episodes with PnL greater than or equal to this value")] = None,
    max_sharpe: Annotated[Optional[float], Query(description="Filter episodes with Sharpe Ratio less than or equal to this value")] = None,
    start_date: Annotated[Optional[datetime.date], Query(description="Filter episodes started on or after this date (YYYY-MM-DD)")] = None,
    end_date: Annotated[Optional[datetime.date], Query(description="Filter episodes started on or before this date (YYYY-MM-DD)")] = None,
):
    """
    Retrieve a paginated list of episodes for a specific training run,
    optionally filtered by performance metrics and date range.
    """
    # First, check if the run exists
    run = db.get(db_models.TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run with ID '{run_id}' not found")

    # Calculate skip and limit directly
    limit = min(page_size, MAX_PAGE_SIZE)
    skip = (page - 1) * limit

    query = select(db_models.Episode).where(db_models.Episode.run_id == run_id)
    count_query = select(func.count()).select_from(db_models.Episode).where(db_models.Episode.run_id == run_id)

    # Apply filters
    if min_pnl is not None:
        query = query.where(db_models.Episode.pnl >= min_pnl)
        count_query = count_query.where(db_models.Episode.pnl >= min_pnl)
    if max_sharpe is not None:
        # Handle potential NULL sharpe ratios if necessary, assuming they don't match
        query = query.where(db_models.Episode.sharpe_ratio <= max_sharpe)
        count_query = count_query.where(db_models.Episode.sharpe_ratio <= max_sharpe)
    if start_date:
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        query = query.where(db_models.Episode.start_time >= start_datetime)
        count_query = count_query.where(db_models.Episode.start_time >= start_datetime)
    if end_date:
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        query = query.where(db_models.Episode.start_time <= end_datetime)
        count_query = count_query.where(db_models.Episode.start_time <= end_datetime)

    # Get total count for pagination
    total_items = db.execute(count_query).scalar_one_or_none() or 0
    total_pages = ceil(total_items / limit) if total_items > 0 else 1
    current_page = (skip // limit) + 1

    # Get paginated items
    episodes = db.execute(
        query.order_by(db_models.Episode.episode_id.asc()) # Or order by start_time?
             .offset(skip)
             .limit(limit)
    ).scalars().all()

    return schemas.PaginatedResponse(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        items=episodes
    )


@router.get("/{run_id}/episodes/summary/", response_model=schemas.EpisodeSummary)
async def get_run_episodes_summary(
    run_id: Annotated[str, Path(description="The ID of the training run whose episodes to summarize")],
    db: DBSession,
):
    """
    Calculate and retrieve aggregated performance metrics for all episodes
    within a specific training run.
    """
    # First, check if the run exists
    run = db.get(db_models.TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run with ID '{run_id}' not found")

    # Query for aggregated metrics
    # Note: func.percentile_cont(0.5).within_group(...) is standard SQL for median
    # Ensure your PostgreSQL version supports this.
    summary_query = select(
        func.count(db_models.Episode.episode_id).label("total_episodes"),
        func.avg(db_models.Episode.pnl).label("average_pnl"),
        func.percentile_cont(0.5).within_group(db_models.Episode.pnl.asc()).label("median_pnl"),
        func.avg(db_models.Episode.sharpe_ratio).label("average_sharpe_ratio"),
        func.percentile_cont(0.5).within_group(db_models.Episode.sharpe_ratio.asc()).label("median_sharpe_ratio"),
        func.avg(db_models.Episode.max_drawdown).label("average_max_drawdown"),
        func.percentile_cont(0.5).within_group(db_models.Episode.max_drawdown.asc()).label("median_max_drawdown"),
        func.avg(db_models.Episode.win_rate).label("average_win_rate"),
        func.percentile_cont(0.5).within_group(db_models.Episode.win_rate.asc()).label("median_win_rate"),
    ).where(db_models.Episode.run_id == run_id)

    summary_result = db.execute(summary_query).first() # Use first() as it returns one row

    if not summary_result or summary_result.total_episodes == 0:
        # Return default values or raise 404 if no episodes found for the run?
        # Returning default seems more informative.
        return schemas.EpisodeSummary(run_id=run_id, total_episodes=0)

    # Map the result row to the Pydantic schema
    return schemas.EpisodeSummary(
        run_id=run_id,
        total_episodes=summary_result.total_episodes,
        average_pnl=summary_result.average_pnl,
        median_pnl=summary_result.median_pnl,
        average_sharpe_ratio=summary_result.average_sharpe_ratio,
        median_sharpe_ratio=summary_result.median_sharpe_ratio,
        average_max_drawdown=summary_result.average_max_drawdown,
        median_max_drawdown=summary_result.median_max_drawdown,
        average_win_rate=summary_result.average_win_rate,
        median_win_rate=summary_result.median_win_rate,
    )
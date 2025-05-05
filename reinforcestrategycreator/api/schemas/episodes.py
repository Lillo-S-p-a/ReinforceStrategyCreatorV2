from typing import Any, Dict
from pydantic import BaseModel
import datetime

# Re-using existing schemas if needed by importing them
from .operations import TradingOperationRead # Example if needed later
from . import PaginatedResponse # Import from the schemas package level

# Schema for basic Episode details (can be expanded)
class EpisodeBase(BaseModel):
    episode_id: int
    run_id: str
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    initial_portfolio_value: float | None = None
    final_portfolio_value: float | None = None
    pnl: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_reward: float | None = None
    total_steps: int | None = None
    win_rate: float | None = None

    class Config:
        from_attributes = True # Use orm_mode for compatibility with SQLAlchemy models
        # Use from_attributes=True for Pydantic v2

class EpisodeDetail(EpisodeBase):
    # Add relationships if needed, e.g., list of steps or trades
    pass

# Schema for the model parameters associated with an episode's training run
class ModelParameters(BaseModel):
    parameters: Dict[str, Any] | None = None

    class Config:
        from_attributes = True
        # Use from_attributes=True for Pydantic v2

# You might also need schemas for Step and Trade if not defined elsewhere
class Step(BaseModel):
    step_id: int
    episode_id: int
    timestamp: datetime.datetime
    portfolio_value: float | None = None
    reward: float | None = None
    action: str | None = None
    position: str | None = None
    asset_price: float | None = None # Added as per TASK-API-20250505-212200

    class Config:
        from_attributes = True

class Trade(BaseModel):
    trade_id: int
    episode_id: int
    entry_time: datetime.datetime
    exit_time: datetime.datetime | None = None
    entry_price: float
    exit_price: float | None = None
    quantity: float
    direction: str
    pnl: float | None = None
    costs: float | None = None

    class Config:
        from_attributes = True

from typing import List # Ensure List is imported if not already

# Schema for the list of episode IDs
class EpisodeIdList(BaseModel):
    episode_ids: List[int]
# Ensure PaginatedResponse is defined, e.g., in schemas/base.py or here
# from typing import Generic, TypeVar, List
# T = TypeVar('T')
# class PaginatedResponse(BaseModel, Generic[T]):
#     total_items: int
#     total_pages: int
#     current_page: int
#     page_size: int
#     items: List[T]
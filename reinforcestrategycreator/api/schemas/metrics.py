import datetime
from typing import List, Optional, TypeVar, Generic, Dict, Any
from pydantic import BaseModel, Field

# --- Base Schemas (mirroring db_models.py fields) ---

class StepBase(BaseModel):
    step_id: int
    episode_id: int
    timestamp: datetime.datetime
    portfolio_value: Optional[float] = None
    reward: Optional[float] = None
    action: Optional[str] = None
    position: Optional[str] = None

class Step(StepBase):
    class Config:
        from_attributes = True # Pydantic V2 equivalent of orm_mode

class TradeBase(BaseModel):
    trade_id: int
    episode_id: int
    entry_time: datetime.datetime
    exit_time: Optional[datetime.datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    direction: str
    pnl: Optional[float] = None
    costs: Optional[float] = None

class Trade(TradeBase):
    class Config:
        from_attributes = True

class EpisodeBase(BaseModel):
    episode_id: int
    run_id: str
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    initial_portfolio_value: Optional[float] = None
    final_portfolio_value: Optional[float] = None
    pnl: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_reward: Optional[float] = None
    total_steps: Optional[int] = None
    win_rate: Optional[float] = None

class Episode(EpisodeBase):
    # Include related data if needed for specific endpoints, but keep base simple
    # steps: List[Step] = []
    # trades: List[Trade] = []
    class Config:
        from_attributes = True

class TrainingRunBase(BaseModel):
    run_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    parameters: Optional[Dict[str, Any]] = None # Use Dict for JSONB
    status: Optional[str] = None
    notes: Optional[str] = None

class TrainingRun(TrainingRunBase):
    # episodes: List[Episode] = [] # Avoid deep nesting by default
    class Config:
        from_attributes = True

# --- Schemas for Specific Endpoint Responses ---

class TrainingRunDetail(TrainingRun):
    # Potentially add aggregated episode info here if needed
    pass

class EpisodeDetail(Episode):
    # Potentially add related steps/trades here if needed for the detail view
    pass

class EpisodeSummary(BaseModel):
    run_id: str
    total_episodes: int
    average_pnl: Optional[float] = None
    median_pnl: Optional[float] = None
    average_sharpe_ratio: Optional[float] = None
    median_sharpe_ratio: Optional[float] = None
    average_max_drawdown: Optional[float] = None
    median_max_drawdown: Optional[float] = None
    average_win_rate: Optional[float] = None
    median_win_rate: Optional[float] = None
    # Add other aggregated metrics as needed

# --- Pagination Schema ---

DataType = TypeVar('DataType')

class PaginatedResponse(BaseModel, Generic[DataType]):
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    items: List[DataType]
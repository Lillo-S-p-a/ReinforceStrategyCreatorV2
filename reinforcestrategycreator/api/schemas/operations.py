from datetime import datetime
from pydantic import BaseModel
from reinforcestrategycreator.db_models import OperationType

class TradingOperationRead(BaseModel):
    """
    Pydantic schema for reading trading operation data.
    """
    operation_id: int
    step_id: int
    timestamp: datetime
    operation_type: OperationType
    size: float
    price: float

    class Config:
        orm_mode = True # Enable ORM mode for compatibility with SQLAlchemy models
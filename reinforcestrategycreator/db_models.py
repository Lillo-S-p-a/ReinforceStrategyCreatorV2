import datetime
import enum  # Add enum import
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Index,
    Enum,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class TrainingRun(Base):
    __tablename__ = "training_runs"

    run_id = Column(String, primary_key=True)  # Consider UUID if generating IDs
    start_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    end_time = Column(DateTime)
    parameters = Column(
        JSONB
    )  # Store training parameters (e.g., learning rate, features used)
    status = Column(String)  # e.g., 'running', 'completed', 'failed'
    notes = Column(String)  # Optional field for user notes

    episodes = relationship("Episode", back_populates="training_run")


class Episode(Base):
    __tablename__ = "episodes"

    episode_id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Simpler auto-incrementing ID
    run_id = Column(
        String, ForeignKey("training_runs.run_id"), nullable=False, index=True
    )
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)
    initial_portfolio_value = Column(Float)
    final_portfolio_value = Column(Float)
    pnl = Column(Float)  # Profit and Loss
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_reward = Column(Float)
    total_steps = Column(Integer)
    win_rate = Column(Float)  # Percentage of winning trades
    # Add other summary metrics as needed from metrics_definitions.md

    training_run = relationship("TrainingRun",
                               foreign_keys=[run_id],
                               primaryjoin="Episode.run_id == TrainingRun.run_id",
                               back_populates="episodes")
    steps = relationship("Step", back_populates="episode", cascade="all, delete-orphan")
    trades = relationship(
        "Trade", back_populates="episode", cascade="all, delete-orphan"
    )
    operations = relationship(
        "TradingOperation", back_populates="episode", cascade="all, delete-orphan"
    )  # Add this line


class Step(Base):
    __tablename__ = "steps"

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey("episodes.episode_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    portfolio_value = Column(Float)
    reward = Column(Float)
    action = Column(String)  # e.g., 'buy', 'sell', 'hold'
    position = Column(String)  # e.g., 'long', 'short', 'flat'
    # Add other step-level details if necessary

    episode = relationship("Episode", back_populates="steps")
    operations = relationship(
        "TradingOperation", back_populates="step", cascade="all, delete-orphan"
    )  # Add this line

    __table_args__ = (
        Index("ix_steps_episode_id_timestamp", "episode_id", "timestamp"),
    )


class Trade(Base):
    __tablename__ = "trades"

    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(
        Integer, ForeignKey("episodes.episode_id"), nullable=False, index=True
    )
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(
        DateTime
    )  # Nullable if trade is still open? Or only log closed trades? Assuming closed.
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)  # Shares, contracts, etc.
    direction = Column(String)  # 'long' or 'short'
    pnl = Column(Float)  # Profit and Loss for this specific trade
    costs = Column(Float)  # Commissions, fees, slippage

    episode = relationship("Episode", back_populates="trades")


# Add OperationType Enum (before TradingOperation class)
class OperationType(enum.Enum):
    ENTRY_LONG = "ENTRY_LONG"
    EXIT_LONG = "EXIT_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"  # Or maybe just don't log holds? Let's include for now.


# Add TradingOperation class (e.g., after Trade class)
class TradingOperation(Base):
    __tablename__ = "trading_operations"

    operation_id = Column(Integer, primary_key=True, autoincrement=True)
    step_id = Column(Integer, ForeignKey("steps.step_id"), nullable=False)
    episode_id = Column(
        Integer, ForeignKey("episodes.episode_id"), nullable=False
    )  # As requested, though step implies episode
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    operation_type = Column(Enum(OperationType), nullable=False)  # Use Enum here
    size = Column(Float, nullable=False)  # e.g., number of shares/contracts
    price = Column(Float, nullable=False)  # Execution price

    step = relationship("Step", back_populates="operations")
    episode = relationship("Episode", back_populates="operations")

    __table_args__ = (
        Index("ix_trading_operations_step_id", "step_id"),
        Index(
            "ix_trading_operations_episode_id", "episode_id"
        ),  # Index for direct episode queries
        Index(
            "ix_trading_operations_timestamp", "timestamp"
        ),  # Index for time-based queries
    )


# Example of how to create the engine (connection details should come from config/env vars)
# DATABASE_URL = "postgresql://user:password@host:port/database"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create tables (can be called from an initialization script)
# def create_db_tables():
#     Base.metadata.create_all(bind=engine)

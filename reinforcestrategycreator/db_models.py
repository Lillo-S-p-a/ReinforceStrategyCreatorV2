import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class TrainingRun(Base):
    __tablename__ = 'training_runs'

    run_id = Column(String, primary_key=True) # Consider UUID if generating IDs
    start_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    end_time = Column(DateTime)
    parameters = Column(JSONB) # Store training parameters (e.g., learning rate, features used)
    status = Column(String) # e.g., 'running', 'completed', 'failed'
    notes = Column(String) # Optional field for user notes

    episodes = relationship("Episode", back_populates="training_run")

class Episode(Base):
    __tablename__ = 'episodes'

    episode_id = Column(Integer, primary_key=True, autoincrement=True) # Simpler auto-incrementing ID
    run_id = Column(String, ForeignKey('training_runs.run_id'), nullable=False, index=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)
    initial_portfolio_value = Column(Float)
    final_portfolio_value = Column(Float)
    pnl = Column(Float) # Profit and Loss
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_reward = Column(Float)
    total_steps = Column(Integer)
    win_rate = Column(Float) # Percentage of winning trades
    # Add other summary metrics as needed from metrics_definitions.md

    training_run = relationship("TrainingRun", back_populates="episodes")
    steps = relationship("Step", back_populates="episode", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="episode", cascade="all, delete-orphan")


class Step(Base):
    __tablename__ = 'steps'

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey('episodes.episode_id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    portfolio_value = Column(Float)
    reward = Column(Float)
    action = Column(String) # e.g., 'buy', 'sell', 'hold'
    position = Column(String) # e.g., 'long', 'short', 'flat'
    # Add other step-level details if necessary

    episode = relationship("Episode", back_populates="steps")

    __table_args__ = (
        Index('ix_steps_episode_id_timestamp', 'episode_id', 'timestamp'),
    )

class Trade(Base):
    __tablename__ = 'trades'

    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey('episodes.episode_id'), nullable=False, index=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime) # Nullable if trade is still open? Or only log closed trades? Assuming closed.
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False) # Shares, contracts, etc.
    direction = Column(String) # 'long' or 'short'
    pnl = Column(Float) # Profit and Loss for this specific trade
    costs = Column(Float) # Commissions, fees, slippage

    episode = relationship("Episode", back_populates="trades")

# Example of how to create the engine (connection details should come from config/env vars)
# DATABASE_URL = "postgresql://user:password@host:port/database"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create tables (can be called from an initialization script)
# def create_db_tables():
#     Base.metadata.create_all(bind=engine)
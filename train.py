# train.py
import numpy as np
import pandas as pd
import logging
import datetime
import uuid
# import csv # Removed for DB logging
from reinforcestrategycreator.data_fetcher import fetch_historical_data # Corrected import
from reinforcestrategycreator.technical_analyzer import calculate_indicators # Corrected import
from reinforcestrategycreator.trading_environment import TradingEnv # Corrected import
from reinforcestrategycreator.rl_agent import StrategyAgent
from reinforcestrategycreator.db_utils import get_db_session, SessionLocal # Import session management
from reinforcestrategycreator.db_models import TrainingRun, Episode, Step, Trade # Import DB models
from reinforcestrategycreator.metrics_calculator import (
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
)

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
TRAINING_EPISODES = 10 # Start with a small number for testing
SHARPE_WINDOW_SIZE = 100 # Example value, adjust if needed based on env implementation

# Agent Hyperparameters (Example values, use defaults or tune later)
STATE_SIZE = None # Will be determined from env
ACTION_SIZE = None # Will be determined from env
AGENT_MEMORY_SIZE = 2000
AGENT_BATCH_SIZE = 32
AGENT_GAMMA = 0.95
AGENT_EPSILON = 1.0
AGENT_EPSILON_DECAY = 0.995
AGENT_EPSILON_MIN = 0.01
AGENT_LEARNING_RATE = 0.001
AGENT_TARGET_UPDATE_FREQ = 5 # Example value

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info(f"Starting training for {TICKER} from {START_DATE} to {END_DATE}")

    # --- 1. Data Pipeline ---
    logging.info("Fetching historical data...")
    try:
        # Call the function directly
        data = fetch_historical_data(TICKER, START_DATE, END_DATE)
        if data.empty:
            logging.error("Failed to fetch data or data is empty.")
            return
        logging.info(f"Data fetched successfully: {data.shape[0]} rows")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return

    # Rename columns to lowercase for consistency (Moved outside except block)
    logging.info(f"Original columns: {data.columns}") # Log original columns
    # Handle MultiIndex by accessing the first element of the tuple
    data.columns = [col[0].lower() for col in data.columns]
    logging.info(f"Renamed columns to lowercase: {list(data.columns)}")

    logging.info("Adding technical indicators...")
    try:
        # Call the function directly and use the correct name
        data_with_indicators = calculate_indicators(data.copy()) # Use copy to avoid modifying original
        if data_with_indicators.empty or data_with_indicators.isnull().values.any():
             logging.warning("Data after adding indicators is empty or contains NaNs. Check indicator calculations and data range.")
             # Decide how to handle NaNs, e.g., dropna or adjust start date
             initial_rows = len(data_with_indicators)
             data_with_indicators.dropna(inplace=True)
             rows_dropped = initial_rows - len(data_with_indicators)
             if rows_dropped > 0:
                 logging.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation.")
             if data_with_indicators.empty:
                 logging.error("Data is empty after dropping NaNs from indicators.")
                 return
        logging.info("Technical indicators added successfully.")
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return

    # --- 2. Environment Setup ---
    logging.info("Initializing trading environment...")
    try:
        env = TradingEnv(data_with_indicators, sharpe_window_size=SHARPE_WINDOW_SIZE, transaction_fee_percent=0.001) # Corrected class name and added fee
        STATE_SIZE = env.observation_space.shape[0]
        ACTION_SIZE = env.action_space.n
        logging.info(f"Environment initialized. State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    except Exception as e:
        logging.error(f"Error initializing environment: {e}")
        return

    # --- 3. Agent Setup ---
    logging.info("Initializing RL agent...")
    try:
        agent = StrategyAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            memory_size=AGENT_MEMORY_SIZE,
            batch_size=AGENT_BATCH_SIZE,
            gamma=AGENT_GAMMA,
            epsilon=AGENT_EPSILON,
            epsilon_decay=AGENT_EPSILON_DECAY,
            epsilon_min=AGENT_EPSILON_MIN,
            learning_rate=AGENT_LEARNING_RATE,
            target_update_freq=AGENT_TARGET_UPDATE_FREQ
        )
        logging.info("Agent initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        return

    # --- 4. Training Loop ---
    logging.info(f"Starting training loop for {TRAINING_EPISODES} episodes...")

    # --- Setup DB Logging ---
    run_id = f"RUN-{TICKER}-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logging.info(f"Generated Run ID: {run_id}")

    # Store parameters for logging
    run_params = {
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "training_episodes": TRAINING_EPISODES,
        "sharpe_window_size": SHARPE_WINDOW_SIZE,
        "agent_memory_size": AGENT_MEMORY_SIZE,
        "agent_batch_size": AGENT_BATCH_SIZE,
        "agent_gamma": AGENT_GAMMA,
        "agent_epsilon_start": AGENT_EPSILON,
        "agent_epsilon_decay": AGENT_EPSILON_DECAY,
        "agent_epsilon_min": AGENT_EPSILON_MIN,
        "agent_learning_rate": AGENT_LEARNING_RATE,
        "agent_target_update_freq": AGENT_TARGET_UPDATE_FREQ,
        "env_transaction_fee": env.transaction_fee_percent,
        "env_window_size": env.window_size
    }

    db = None # Initialize db session variable
    try:
        # Get a single session for the entire run
        db = next(get_db_session()) # Get session from generator

        # Create TrainingRun record
        training_run = TrainingRun(
            run_id=run_id,
            start_time=datetime.datetime.utcnow(),
            parameters=run_params,
            status='running'
        )
        db.add(training_run)
        # Don't commit yet, commit at the end of the run

        # --- Episode Loop ---
        for episode_num in range(TRAINING_EPISODES):
            state, reset_info = env.reset() # Capture reset info if needed
            initial_portfolio_value = reset_info.get('portfolio_value', env.initial_balance)

            # Create Episode record for this episode
            current_episode = Episode(
                run_id=run_id,
                start_time=datetime.datetime.utcnow(),
                initial_portfolio_value=initial_portfolio_value,
                # Other fields will be updated at the end
            )
            db.add(current_episode)
            db.flush() # Flush to get the episode_id assigned
            current_episode_id = current_episode.episode_id
            logging.debug(f"Started Episode {episode_num+1}, DB Episode ID: {current_episode_id}")

            # Reshape state for the agent's Keras model (expects batch dimension)
            state = np.reshape(state, [1, STATE_SIZE])
            total_reward = 0.0
            done = False
            step_count = 0
            episode_portfolio_values = [initial_portfolio_value] # Start with initial value for MDD calc
            episode_step_rewards = [] # Collect step rewards for Sharpe

            # --- Step Loop ---
            while not done:
                # Agent selects action
                action = agent.select_action(state)

                # Environment steps
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Reshape next_state for the agent
                next_state = np.reshape(next_state, [1, STATE_SIZE])

                # Agent remembers the transition
                agent.remember(state, action, reward, next_state, done)

                # Update state
                state = next_state
                total_reward += reward
                episode_step_rewards.append(float(reward)) # Store reward for Sharpe
                step_count += 1
                # Store portfolio value for MDD calculation
                if info.get('portfolio_value') is not None:
                    episode_portfolio_values.append(info['portfolio_value'])

                # Agent learns (if enough memory samples)
                if len(agent.memory) > AGENT_BATCH_SIZE:
                    agent.learn()

                # --- Log Step Data to DB ---
                step_time = env.df.index[info['step']] if isinstance(env.df.index, pd.DatetimeIndex) else datetime.datetime.utcnow()
                # Map action/position ints to strings (adjust if env changes)
                action_map = {0: 'flat', 1: 'long', 2: 'short'}
                position_map = {0: 'flat', 1: 'long', -1: 'short'}
                db_step = Step(
                    episode_id=current_episode_id,
                    timestamp=step_time,
                    portfolio_value=info.get('portfolio_value'),
                    reward=float(reward), # Ensure float
                    action=action_map.get(action, str(action)), # Store string representation
                    position=position_map.get(info.get('current_position'), str(info.get('current_position')))
                )
                db.add(db_step)

                if done:
                    # --- Log Completed Trades for Episode ---
                    completed_trades = env.get_completed_trades()
                    logging.debug(f"Episode {episode_num+1} completed. Trades: {len(completed_trades)}")
                    for trade_data in completed_trades:
                        db_trade = Trade(
                            episode_id=current_episode_id,
                            entry_time=trade_data['entry_time'],
                            exit_time=trade_data['exit_time'],
                            entry_price=trade_data['entry_price'],
                            exit_price=trade_data['exit_price'],
                            quantity=trade_data['quantity'],
                            direction=trade_data['direction'],
                            pnl=trade_data['pnl'],
                            costs=trade_data['costs']
                        )
                        db.add(db_trade)

                    # --- Update Episode Summary Metrics ---
                    # Calculate metrics
                    # Note: Using step rewards for Sharpe as per env reward definition
                    sharpe = calculate_sharpe_ratio(pd.Series(episode_step_rewards))
                    mdd = calculate_max_drawdown(episode_portfolio_values)
                    win_rate = calculate_win_rate(completed_trades)
                    # Trade Frequency (per episode) is simply the number of trades
                    trade_frequency = len(completed_trades)
                    # Success Rate (episode level)
                    episode_pnl = info.get('portfolio_value', initial_portfolio_value) - initial_portfolio_value
                    success_rate = 100.0 if episode_pnl > 0 else 0.0 # 100% if profitable, 0% otherwise

                    current_episode.end_time = datetime.datetime.utcnow()
                    current_episode.final_portfolio_value = info.get('portfolio_value')
                    current_episode.pnl = episode_pnl
                    current_episode.total_reward = total_reward
                    current_episode.total_steps = step_count
                    current_episode.sharpe_ratio = sharpe
                    current_episode.max_drawdown = mdd
                    current_episode.win_rate = win_rate
                    # Add trade_frequency and success_rate if columns exist in Episode model
                    # Assuming they exist based on metrics_definitions.md:
                    # current_episode.trade_frequency = trade_frequency # Uncomment if column exists
                    # current_episode.success_rate = success_rate # Uncomment if column exists

                    logging.info(f"Episode: {episode_num+1}/{TRAINING_EPISODES}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.2f}, Steps: {step_count}, Final Portfolio: {info.get('portfolio_value', 'N/A'):.2f}")

            # Commit changes at the end of each episode
            db.commit()
            logging.debug(f"Committed DB changes for Episode {episode_num+1}")

        # --- Final Run Update ---
        training_run.end_time = datetime.datetime.utcnow()
        training_run.status = 'completed'
        db.commit()
        logging.info(f"Training finished. Run ID: {run_id}. Results logged to database.")

    except Exception as e:
        logging.error(f"An error occurred during training or DB logging: {e}", exc_info=True)
        if db:
            db.rollback() # Rollback any partial changes for the run
            # Optionally update run status to 'failed'
            try:
                # Need to query the run again in case the session was invalidated
                failed_run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
                if failed_run:
                    failed_run.status = 'failed'
                    failed_run.end_time = datetime.datetime.utcnow()
                    db.commit()
            except Exception as e_update:
                 logging.error(f"Failed to update run status to 'failed': {e_update}")
    finally:
        if db:
            db.close() # Ensure session is closed

    # Optional: Save the trained model
    # agent.save_model("spy_rl_model.keras") # Or .weights.h5
    # logging.info("Agent model saved.")

if __name__ == "__main__":
    main()
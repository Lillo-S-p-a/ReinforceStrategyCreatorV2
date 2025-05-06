# train_fix.py
import numpy as np
import pandas as pd
import logging
import datetime
import uuid
# import csv # Removed for DB logging
import os
from reinforcestrategycreator.data_fetcher import fetch_historical_data # Corrected import
from reinforcestrategycreator.technical_analyzer import calculate_indicators # Corrected import
from reinforcestrategycreator.trading_environment import TradingEnv # Corrected import
from reinforcestrategycreator.rl_agent import StrategyAgent
from reinforcestrategycreator.db_utils import get_db_session, SessionLocal # Import session management
from reinforcestrategycreator.db_models import TrainingRun, Episode, Step, Trade, TradingOperation, OperationType # Import DB models
from reinforcestrategycreator.metrics_calculator import (
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
)

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
TRAINING_EPISODES = 1000 # Iteration 2: Increased for better convergence
SHARPE_WINDOW_SIZE = 100 # Example value, adjust if needed based on env implementation

# Environment Tuning Parameters (Phase 1 Debug)
ENV_DRAWDOWN_PENALTY = 0.05 # Iteration 2: Increased penalty for better risk management
ENV_TRADING_PENALTY = 0.005 # Iteration 2: Increased penalty to reduce excessive trading
ENV_RISK_FRACTION = 0.1    # Increased from 0.02, back to original default
ENV_STOP_LOSS_PCT = 5.0     # Enabled, was None (disabled)

# Agent Hyperparameters (Example values, use defaults or tune later)
STATE_SIZE = None # Will be determined from env
ACTION_SIZE = None # Will be determined from env
AGENT_MEMORY_SIZE = 2000
AGENT_BATCH_SIZE = 32
AGENT_GAMMA = 0.99 # Iteration 2: Higher gamma for better long-term reward consideration
AGENT_EPSILON = 1.0
AGENT_EPSILON_DECAY = 0.9999 # Iteration 2: Slower decay for more exploration
AGENT_EPSILON_MIN = 0.01
AGENT_LEARNING_RATE = 0.001 # Iteration 4: Reverted learning rate
AGENT_TARGET_UPDATE_FREQ = 100 # Iteration 2: Increased for more stable target network

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
        env = TradingEnv(
            data_with_indicators,
            sharpe_window_size=SHARPE_WINDOW_SIZE,
            transaction_fee_percent=0.001, # Keep existing fee
            drawdown_penalty=ENV_DRAWDOWN_PENALTY,
            trading_frequency_penalty=ENV_TRADING_PENALTY,
            risk_fraction=ENV_RISK_FRACTION,
            stop_loss_pct=ENV_STOP_LOSS_PCT,
            use_sharpe_ratio=True # Iteration 2: Enable Sharpe ratio for better risk-adjusted rewards
            # Note: take_profit_pct remains None (default) for now
        )
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
        "env_window_size": env.window_size,
        "env_drawdown_penalty": env.drawdown_penalty, # Log tuned value
        "env_trading_penalty": env.trading_frequency_penalty, # Log tuned value
        "env_risk_fraction": env.risk_fraction, # Log tuned value
        "env_stop_loss_pct": env.stop_loss_pct # Log tuned value
    }

    # Use a 'with' statement to manage the DB session lifecycle
    with get_db_session() as db:
        training_run = None # Initialize training_run to handle potential early exceptions
        try:
            # Create TrainingRun record
            training_run = TrainingRun(
                run_id=run_id,
                start_time=datetime.datetime.now(datetime.UTC), # Use timezone-aware UTC now
                parameters=run_params,
                status='running'
            )
            db.add(training_run)
            # Commit the initial run record immediately to ensure it exists even if episodes fail
            db.commit()
            db.refresh(training_run) # Refresh to get the generated ID if needed elsewhere

            # --- Episode Loop ---
            for episode_num in range(TRAINING_EPISODES):
                current_episode = None # Initialize for potential early errors in episode setup
                try:
                    state, reset_info = env.reset() # Capture reset info if needed
                    initial_portfolio_value = reset_info.get('portfolio_value', env.initial_balance)

                    # Create Episode record for this episode
                    current_episode = Episode(
                        run_id=run_id, # Use run_id directly
                        start_time=datetime.datetime.now(datetime.UTC), # Use timezone-aware UTC now
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
                    
                    # Track previous position for detecting changes
                    prev_position = 0  # Start with flat position (0)

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
                        step_time = env.df.index[info['step']] if isinstance(env.df.index, pd.DatetimeIndex) else datetime.datetime.now(datetime.UTC) # Use timezone-aware UTC now
                        # Map action/position ints to strings (adjust if env changes)
                        action_map = {0: 'flat', 1: 'long', 2: 'short'}
                        position_map = {0: 'flat', 1: 'long', -1: 'short'}
                        
                        # Get current position
                        current_position = info.get('current_position', 0)
                        
                        # Create Step record
                        db_step = Step(
                            episode_id=current_episode_id,
                            timestamp=step_time,
                            portfolio_value=info.get('portfolio_value'),
                            reward=float(reward), # Ensure float
                            asset_price=info.get('current_price'), # <-- ADDED THIS LINE
                            action=action_map.get(action, str(action)), # Store string representation
                            position=position_map.get(current_position, str(current_position))
                        )
                        db.add(db_step)
                        db.flush()  # Flush to get the step_id
                        
                        # --- NEW CODE: Log TradingOperation if position changed ---
                        # Check if position changed
                        if current_position != prev_position:
                            # Determine operation type based on position transition
                            operation_type = None
                            
                            if prev_position == 0 and current_position == 1:
                                # Flat to Long
                                operation_type = OperationType.ENTRY_LONG
                            elif prev_position == 0 and current_position == -1:
                                # Flat to Short
                                operation_type = OperationType.ENTRY_SHORT
                            elif prev_position == 1 and current_position == 0:
                                # Long to Flat
                                operation_type = OperationType.EXIT_LONG
                            elif prev_position == -1 and current_position == 0:
                                # Short to Flat
                                operation_type = OperationType.EXIT_SHORT
                            elif prev_position == 1 and current_position == -1:
                                # Long to Short (two operations)
                                # First, exit long
                                exit_long_op = TradingOperation(
                                    step_id=db_step.step_id,
                                    episode_id=current_episode_id,
                                    timestamp=step_time,
                                    operation_type=OperationType.EXIT_LONG,
                                    size=abs(info.get('shares_held', 0)),  # Use actual shares if available
                                    price=info.get('current_price', 0.0)  # Use actual price if available
                                )
                                db.add(exit_long_op)
                                
                                # Then, enter short
                                operation_type = OperationType.ENTRY_SHORT
                            elif prev_position == -1 and current_position == 1:
                                # Short to Long (two operations)
                                # First, exit short
                                exit_short_op = TradingOperation(
                                    step_id=db_step.step_id,
                                    episode_id=current_episode_id,
                                    timestamp=step_time,
                                    operation_type=OperationType.EXIT_SHORT,
                                    size=abs(info.get('shares_held', 0)),  # Use actual shares if available
                                    price=info.get('current_price', 0.0)  # Use actual price if available
                                )
                                db.add(exit_short_op)
                                
                                # Then, enter long
                                operation_type = OperationType.ENTRY_LONG
                            
                            # Create the trading operation if an operation type was determined
                            if operation_type:
                                trading_op = TradingOperation(
                                    step_id=db_step.step_id,
                                    episode_id=current_episode_id,
                                    timestamp=step_time,
                                    operation_type=operation_type,
                                    size=abs(info.get('shares_held', 0)),  # Use actual shares if available
                                    price=info.get('current_price', 0.0)  # Use actual price if available
                                )
                                db.add(trading_op)
                            
                            # Update previous position for next iteration
                            prev_position = current_position

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
                            sharpe = calculate_sharpe_ratio(pd.Series(episode_step_rewards)) if episode_step_rewards else 0.0
                            mdd = calculate_max_drawdown(episode_portfolio_values) if len(episode_portfolio_values) > 1 else 0.0
                            win_rate = calculate_win_rate(completed_trades) if completed_trades else 0.0
                            # Trade Frequency (per episode) is simply the number of trades
                            trade_frequency = len(completed_trades)
                            # Success Rate (episode level)
                            episode_pnl = info.get('portfolio_value', initial_portfolio_value) - initial_portfolio_value
                            success_rate = 100.0 if episode_pnl > 0 else 0.0 # 100% if profitable, 0% otherwise

                            current_episode.end_time = datetime.datetime.now(datetime.UTC) # Use timezone-aware UTC now
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
# --- ADDED: Save Model Weights for this Episode ---
                            try:
                                model_save_path = f"models/episode_{current_episode_id}_model.keras" # Save as Keras native format # Adjust extension if needed (.h5, .keras, etc.)
                                os.makedirs("models", exist_ok=True) # Ensure the 'models' directory exists
                                # !!! IMPORTANT: Replace 'save_weights' with the actual method name of your StrategyAgent class !!!
                                agent.model.save(model_save_path)  # Use save() for SavedModel format 
                                logging.info(f"Saved Keras model for episode {current_episode_id} to {model_save_path}")
                            except AttributeError:
                                logging.error(f"Agent does not have a 'save_weights' method. Please implement or correct the method name.")
                            except Exception as save_err:
                                logging.error(f"Error saving model weights for episode {current_episode_id}: {save_err}")
                            # --- END OF ADDED PART ---

                    # Commit changes at the end of each episode (steps, trades, episode summary)
                    db.commit()
                    logging.debug(f"Committed DB changes for Episode {episode_num+1}")

                except Exception as episode_err:
                    logging.error(f"Error during Episode {episode_num+1}: {episode_err}", exc_info=True)
                    db.rollback() # Rollback changes for the failed episode
                    # Optionally mark the episode as failed if the record exists
                    if current_episode and current_episode.episode_id:
                         try:
                             # Re-fetch in case session state is weird after rollback
                             failed_episode = db.query(Episode).filter(Episode.episode_id == current_episode.episode_id).first()
                             if failed_episode:
                                 failed_episode.end_time = datetime.datetime.now(datetime.UTC)
                                 # Add a status field to Episode model if you want to mark it 'failed'
                                 # failed_episode.status = 'failed'
                                 db.commit()
                         except Exception as ep_update_err:
                             logging.error(f"Failed to mark episode {current_episode_id} after error: {ep_update_err}")
                             db.rollback() # Rollback again if marking failed
                    # Continue to the next episode
                    continue

            # --- Final Run Update (after all episodes attempted) ---
            training_run.end_time = datetime.datetime.now(datetime.UTC) # Use timezone-aware UTC now
            training_run.status = 'completed' # Mark as completed even if some episodes failed
            db.commit()
            logging.info(f"Training finished. Run ID: {run_id}. Results logged to database.")

        except Exception as e:
            logging.error(f"An error occurred during training run setup or DB logging: {e}", exc_info=True)
            # Rollback is handled automatically by the 'with' statement on exception
            # Update status to failed if possible and if the run was initially committed
            if training_run and training_run.run_id: # Check if training_run exists and has an ID
                try:
                    # The session might be in a rolled-back state, but we can try to update
                    # Re-fetch the run to ensure we have the correct object state
                    run_to_update = db.query(TrainingRun).filter(TrainingRun.run_id == training_run.run_id).first()
                    if run_to_update:
                        run_to_update.status = 'failed'
                        run_to_update.end_time = datetime.datetime.now(datetime.UTC) # Use timezone-aware UTC now
                        db.commit() # Try committing the failure status
                    else:
                        logging.warning(f"Could not find run {training_run.run_id} to mark as failed (might not have been committed initially).")
                except Exception as update_err:
                    logging.error(f"Failed to update run status to 'failed' after initial error: {update_err}", exc_info=True)
                    # Rollback might happen again here if commit fails, handled by 'with' exit

        # The 'with' block automatically handles db.close() or rollback on error.
        logging.info("Database session context exited.")

    # Optional: Save the trained model
    # agent.save_model("spy_rl_model.keras") # Or .weights.h5
    # logging.info("Agent model saved.")

if __name__ == "__main__":
    main()
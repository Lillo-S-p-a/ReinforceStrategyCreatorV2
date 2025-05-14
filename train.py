import numpy as np
import pandas as pd
import logging
import datetime
import uuid
import os
import ray
from ray.rllib.algorithms.dqn import DQNConfig # Using DQN as per existing agent
from ray.tune.logger import pretty_print

from reinforcestrategycreator.data_fetcher import fetch_historical_data
from reinforcestrategycreator.technical_analyzer import calculate_indicators
# Import TradingEnv and the registration function
from reinforcestrategycreator.trading_environment import TradingEnv, register_rllib_env
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import TrainingRun, Episode # Step, Trade, TradingOperation, OperationType
from reinforcestrategycreator.callbacks import DatabaseLoggingCallbacks # Import the new callback
# Metrics calculator might be used in callbacks or post-analysis
# from reinforcestrategycreator.metrics_calculator import (
#     calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
# )

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
# RLlib training is often defined by iterations or total timesteps, not just episodes
NUM_TRAINING_ITERATIONS = 10 # Example: 10 training iterations for RLlib
INITIAL_TRAINING_EPISODES_FOR_DATA_EST = 5 # Used for estimating data length if needed

# Environment parameters (to be passed in env_config)
ENV_INITIAL_BALANCE = 10000.0
ENV_TRANSACTION_FEE_PERCENT = 0.001
ENV_WINDOW_SIZE = 5 # Default from original TradingEnv
ENV_SHARPE_WINDOW_SIZE = 100
ENV_DRAWDOWN_PENALTY = 0.005
ENV_TRADING_PENALTY = 0.002
ENV_RISK_FRACTION = 0.1
ENV_STOP_LOSS_PCT = 5.0
ENV_USE_SHARPE_RATIO = False # As per last setting in original train.py
ENV_NORMALIZATION_WINDOW_SIZE = 20 # Default from TradingEnv

# Agent Hyperparameters for RLlib DQN
AGENT_LEARNING_RATE = 0.001
AGENT_GAMMA = 0.90
AGENT_BUFFER_SIZE = 10000 # RLlib default is 50000, original was 2000. Let's use a bit more.
AGENT_TRAIN_BATCH_SIZE = 32 # RLlib default is 32
AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS = 500 # RLlib DQN: target_network_update_freq in timesteps
AGENT_INITIAL_EPSILON = 1.0
AGENT_FINAL_EPSILON = 0.01
# Estimate epsilon timesteps: if avg 250 steps/ep, 5 eps = 1250 steps.
# Let's make it decay over a few iterations. If 1 iteration = 4000 timesteps (RLlib default)
# then 1250 is too short. Let's use a common RLlib default or a fraction of total expected timesteps.
# Total timesteps = NUM_TRAINING_ITERATIONS * timesteps_per_iteration (e.g., 4000) = 40000
AGENT_EPSILON_TIMESTEPS = 10000 # Anneal over 10k timesteps

# Parallelism
NUM_ROLLOUT_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1 # Use N-1 cores or 1 if single core

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

def main():
    logger.info(f"Starting RLlib training for {TICKER} from {START_DATE} to {END_DATE}")
    ray.init(ignore_reinit_error=True, log_to_driver=True) # Log to driver for easier debugging initially
    logger.info(f"Ray initialized. Dashboard URL: {ray.dashboard}")

    # --- 1. Data Pipeline ---
    logger.info("Fetching historical data...")
    try:
        data_df = fetch_historical_data(TICKER, START_DATE, END_DATE)
        if data_df.empty:
            logger.error("Failed to fetch data or data is empty.")
            ray.shutdown()
            return
        logger.info(f"Data fetched successfully: {data_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        ray.shutdown()
        return

    data_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data_df.columns]
    logger.info(f"Renamed columns to lowercase: {list(data_df.columns)}")

    logger.info("Adding technical indicators...")
    try:
        data_with_indicators = calculate_indicators(data_df.copy())
        initial_rows = len(data_with_indicators)
        data_with_indicators.dropna(inplace=True)
        rows_dropped = initial_rows - len(data_with_indicators)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation.")
        if data_with_indicators.empty:
            logger.error("Data is empty after dropping NaNs from indicators.")
            ray.shutdown()
            return
        logger.info("Technical indicators added successfully.")
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}", exc_info=True)
        ray.shutdown()
        return

    # --- Register Custom Environment ---
    register_rllib_env() # Call the registration function

    # --- RLlib Algorithm Configuration ---
    env_config_params = {
        "df": data_with_indicators,
        "initial_balance": ENV_INITIAL_BALANCE,
        "transaction_fee_percent": ENV_TRANSACTION_FEE_PERCENT,
        "window_size": ENV_WINDOW_SIZE,
        "sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "trading_frequency_penalty": ENV_TRADING_PENALTY,
        "drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "risk_fraction": ENV_RISK_FRACTION,
        "stop_loss_pct": ENV_STOP_LOSS_PCT,
        "normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
        # Add any other parameters TradingEnv expects
    }
    
    # --- Database Logging Setup (run_id needs to be defined before config) ---
    run_id = f"RLlibRUN-{TICKER}-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated Run ID: {run_id}")
    
    # Create the callback config using the run_id
    callback_config = {"run_id": run_id}
    logger.info(f"Created callback_config with run_id: {run_id}")

    config = (
        DQNConfig()
        # Explicitly specify we want the new API stack to silence the warning
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment(env="TradingEnv-v0", env_config=env_config_params)
        .framework("torch") # Switch to PyTorch
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            exploration_config={} # Use empty dict for default exploration with new API
        )
        .training(
            lr=AGENT_LEARNING_RATE,
            gamma=AGENT_GAMMA,
            train_batch_size=AGENT_TRAIN_BATCH_SIZE,
            # Using default replay buffer for single-agent DQN by removing explicit config.
            # The 'capacity' will be taken from the default or other general buffer settings if applicable.
            target_network_update_freq=AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS
            # Model config now under .rl_module()
        )
        .rl_module( # Model config under .rl_module() for new API stack
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        # Top-level .exploration() removed
        .resources(
            num_gpus=1 if ray.is_initialized() and ray.cluster_resources().get("GPU", 0) > 0 else 0
        )
        # Initialize DatabaseLoggingCallbacks with a dictionary containing run_id
        .callbacks(lambda: DatabaseLoggingCallbacks(callback_config)) # Use callback_config here
        .reporting(min_time_s_per_iteration=10) # Report metrics at least every 10s
    )
    
    # For debugging the config
    # logger.info(f"RLlib Config: {pretty_print(config.to_dict())}")

    run_params = {
        "ticker": TICKER, "start_date": START_DATE, "end_date": END_DATE,
        "rllib_num_iterations": NUM_TRAINING_ITERATIONS,
        "rllib_num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "agent_learning_rate": AGENT_LEARNING_RATE, "agent_gamma": AGENT_GAMMA,
        "agent_buffer_size": AGENT_BUFFER_SIZE, "agent_train_batch_size": AGENT_TRAIN_BATCH_SIZE,
        "agent_target_network_update_freq": AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS,
        "agent_initial_epsilon": AGENT_INITIAL_EPSILON, "agent_final_epsilon": AGENT_FINAL_EPSILON,
        "agent_epsilon_timesteps": AGENT_EPSILON_TIMESTEPS,
        # Env params
        "env_initial_balance": ENV_INITIAL_BALANCE,
        "env_transaction_fee": ENV_TRANSACTION_FEE_PERCENT,
        "env_window_size": ENV_WINDOW_SIZE,
        "env_sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "env_drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "env_trading_penalty": ENV_TRADING_PENALTY,
        "env_risk_fraction": ENV_RISK_FRACTION,
        "env_stop_loss_pct": ENV_STOP_LOSS_PCT,
        "env_use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "env_normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
    }

    training_run_db_id = None
    with get_db_session() as db:
        try:
            training_run_record = TrainingRun(
                run_id=run_id,
                start_time=datetime.datetime.now(datetime.UTC),
                parameters=run_params,
                status='starting_rllib'
            )
            db.add(training_run_record)
            db.commit()
            db.refresh(training_run_record) # Refresh is good practice, though PK is manually set
            training_run_db_id = training_run_record.run_id # Correct attribute is run_id
            logger.info(f"TrainingRun record created in DB with run_id: {training_run_db_id}")
        except Exception as e:
            logger.error(f"Failed to create TrainingRun DB record: {e}", exc_info=True)
            ray.shutdown()
            return
    
    # Both set callbacks_config (for algorithm config) and directly pass run_id to the callback via lambda
    config.callbacks_config = callback_config
    # Directly set the config value to ensure it's available
    logger.info(f"Set config.callbacks_config to: {config.callbacks_config}")
    
    # Explicitly disable new API stack warnings
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
    
    # Pass environment variables to worker processes to disable warnings
    config.extra_python_environs_for_driver = {
        "RAY_DISABLE_DEPRECATION_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning",
        "RAY_DEDUP_LOGS": "0",
        "RLLIB_DISABLE_API_STACK_WARNING": "1",
        "RAY_LOGGING_LEVEL": "ERROR"
    }
    
    # Also pass to workers
    config.extra_python_environs_for_worker = {
        "RAY_DISABLE_DEPRECATION_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning",
        "RAY_DEDUP_LOGS": "0",
        "RLLIB_DISABLE_API_STACK_WARNING": "1",
        "RAY_LOGGING_LEVEL": "ERROR"
    }

    # --- Build and Train Algorithm ---
    try:
        algo = config.build_algo()
        logger.info("RLlib DQN Algorithm built successfully.")

        # Training loop
        for i in range(NUM_TRAINING_ITERATIONS):
            result = algo.train()
            logger.info(f"Iteration {i+1}/{NUM_TRAINING_ITERATIONS}:")
            logger.info(pretty_print(result))

            # TODO: Implement DB logging for episode summaries using callbacks
            # Example: result['custom_metrics'], result['episode_reward_mean'], etc.
            # This would involve defining a custom callback class and adding it to .callbacks(YourCustomCallbacks)

            if (i + 1) % 5 == 0:  # Save checkpoint every 5 iterations
                checkpoint_dir = algo.save()
                logger.info(f"Checkpoint saved at iteration {i+1} in {checkpoint_dir}")
        
        final_checkpoint_dir = algo.save()
        logger.info(f"Final model checkpoint saved in {final_checkpoint_dir}")

        # Update TrainingRun status to completed
        if training_run_db_id: # training_run_db_id stores the string run_id
            with get_db_session() as db:
                run_to_update = db.query(TrainingRun).filter(TrainingRun.run_id == training_run_db_id).first() # Corrected to use run_id
                if run_to_update:
                    run_to_update.end_time = datetime.datetime.now(datetime.UTC)
                    run_to_update.status = 'completed_rllib'
                    # TODO: Add final metrics to TrainingRun if available from RLlib results
                    # run_to_update.final_metrics = { ... }
                    db.commit()
                    logger.info(f"TrainingRun {run_id} status updated to completed_rllib.")

    except Exception as e:
        logger.error(f"Error during RLlib training: {e}", exc_info=True)
        if training_run_db_id:
            with get_db_session() as db:
                run_to_update = db.query(TrainingRun).filter(TrainingRun.run_id == training_run_db_id).first() # Corrected to use run_id
                if run_to_update:
                    run_to_update.end_time = datetime.datetime.now(datetime.UTC)
                    run_to_update.status = 'failed_rllib'
                    db.commit()
                    logger.info(f"TrainingRun {run_id} status updated to failed_rllib due to error.")
    finally:
        # Ensure algo is defined before calling stop()
        if 'algo' in locals() and algo is not None: # Check if algo was successfully assigned
            algo.stop()
        ray.shutdown()
        logger.info("Ray shut down. Training process finished.")

if __name__ == "__main__":
    main()
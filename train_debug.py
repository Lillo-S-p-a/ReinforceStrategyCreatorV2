import numpy as np
import pandas as pd
import logging
import datetime
import uuid
import os
import ray
from ray.rllib.algorithms.dqn import DQNConfig # Using DQN as per existing agent
from ray.tune.logger import pretty_print

# Import our debug wrapper
from debug_replay_buffer import debug_sampling_process

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
NUM_TRAINING_ITERATIONS = 1  # REDUCED TO 1 FOR DEBUGGING
INITIAL_TRAINING_EPISODES_FOR_DATA_EST = 5 # Used for estimating data length if needed
GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS = 1 # Number of extra train() calls for draining
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 30 # Timeout for the draining phase
 
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

# Parallelism - Reduce for debugging
NUM_ROLLOUT_WORKERS = 0  # Use 0 for debugging to avoid multi-process complexity

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

def main():
    # Clear any lingering system-wide shutdown flag from previous runs
    TradingEnv.clear_system_wide_graceful_shutdown()
    logger.info(f"Starting RLlib debug training run for {TICKER} from {START_DATE} to {END_DATE}")
    
    # Monkey patch removed as direct modification to RLlib source is used.
    logger.info("Direct modification to RLlib source is used for debugging EpisodeReplayBuffer.")
    
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
    run_id = f"RLlibDBG-{TICKER}-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated Debug Run ID: {run_id}")
    
    # Create the callback config using the run_id
    callback_config = {"run_id": run_id}
    logger.info(f"Created callback_config with run_id: {run_id}")

    config = (
        DQNConfig()
        # New API stack is enabled by default, so removing the explicit disabling.
        .environment(env="TradingEnv-v0", env_config=env_config_params)
        .framework("torch") # Switch to PyTorch
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length=50, # Explicitly set rollout_fragment_length
            exploration_config={} # Use empty dict for default exploration with new API
        )
        .training(
            lr=AGENT_LEARNING_RATE,
            gamma=AGENT_GAMMA,
            train_batch_size=AGENT_TRAIN_BATCH_SIZE,
            n_step=1,  # IMPORTANT: Set to 1 for debugging, as this is what causes the issue
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",  # We use the monkey-patched version
                "capacity": 50000,
                "worker_side_prioritization": False,
            },
            target_network_update_freq=AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        .resources(
            num_gpus=0  # Don't use GPU for debugging
        )
        .callbacks(lambda: DatabaseLoggingCallbacks(callback_config))
        .reporting(min_time_s_per_iteration=10)
    )
    config.log_level = "ERROR"
    
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

    # --- Create TrainingRun DB Record ---
    # This must happen BEFORE algo.build_algo() or algo.train()
    # so that callbacks can find the parent run_id.
    run_params = {
        "ticker": TICKER, "start_date": START_DATE, "end_date": END_DATE,
        "rllib_num_iterations": NUM_TRAINING_ITERATIONS,
        "rllib_num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "agent_learning_rate": AGENT_LEARNING_RATE, "agent_gamma": AGENT_GAMMA,
        "agent_buffer_size": AGENT_BUFFER_SIZE, "agent_train_batch_size": AGENT_TRAIN_BATCH_SIZE,
        "agent_target_network_update_freq": AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS,
        "agent_initial_epsilon": AGENT_INITIAL_EPSILON, "agent_final_epsilon": AGENT_FINAL_EPSILON,
        "agent_epsilon_timesteps": AGENT_EPSILON_TIMESTEPS,
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
                run_id=run_id, # This is the run_id generated earlier
                start_time=datetime.datetime.now(datetime.UTC),
                parameters=run_params,
                status='starting_rllib_debug' # Indicate it's a debug run
            )
            db.add(training_run_record)
            db.commit() # Commit immediately to make it available for foreign key constraints
            db.refresh(training_run_record)
            training_run_db_id = training_run_record.run_id
            logger.info(f"TrainingRun record created in DB with run_id: {training_run_db_id}")
        except Exception as e:
            logger.error(f"Failed to create TrainingRun DB record: {e}", exc_info=True)
            ray.shutdown()
            return

    # --- Build and Train Algorithm ---
    try:
        # Ensure callback_config is part of the main config if not already set by .callbacks()
        # This might be redundant if .callbacks(lambda: DatabaseLoggingCallbacks(callback_config)) works as expected
        # but let's ensure it's explicitly there.
        if not hasattr(config, "callbacks_config") or not config.callbacks_config:
             config.callbacks(lambda: DatabaseLoggingCallbacks(callback_config))
        
        algo = config.build_algo()
        logger.info("RLlib DQN Algorithm built successfully.")

        # Training loop - stop after 10 episodes
        episodes_completed_count = 0
        max_episodes_to_run = 10
        max_iterations = 100 # Safety break for the loop
        logger.info(f"Starting training loop, will stop after {max_episodes_to_run} episodes or {max_iterations} iterations.")

        for i in range(max_iterations):
            result = algo.train()
            logger.info(f"Training iteration {i+1} complete.")
            # logger.info(pretty_print(result)) # Can be verbose

            episodes_this_iter = result.get("episodes_this_iteration", 0)
            episodes_completed_count = result.get("episodes_total", 0)
            
            logger.info(f"Episodes this iteration: {episodes_this_iter}, Total episodes completed: {episodes_completed_count}")

            if episodes_completed_count >= max_episodes_to_run:
                logger.info(f"Reached {episodes_completed_count} episodes (target {max_episodes_to_run}). Stopping training loop.")
                break
            if i == max_iterations - 1:
                logger.warning(f"Reached max_iterations ({max_iterations}) before completing {max_episodes_to_run} episodes. Stopping.")

        # --- Graceful Shutdown Logic ---
        logger.info("Initiating graceful shutdown...")
        # Access workers property to get the EnvRunnerGroup (or equivalent)
        # This object structure can be tricky with num_workers=0 vs >0
        
        if NUM_ROLLOUT_WORKERS == 0:
            logger.info("Signaling driver's environment to initiate graceful shutdown (NUM_ROLLOUT_WORKERS == 0).")
            try:
                driver_env_runner = algo.env_runner # This is a SingleAgentEnvRunner
                actual_env = None
                
                if hasattr(driver_env_runner, 'env'):
                    wrapped_env = driver_env_runner.env
                    logger.info(f"Retrieved wrapped_env from driver_env_runner.env, type: {type(wrapped_env)}")
                    
                    potential_vector_env = wrapped_env
                    if hasattr(wrapped_env, 'unwrapped'):
                        potential_vector_env = wrapped_env.unwrapped
                        logger.info(f"Unwrapped to potential_vector_env, type: {type(potential_vector_env)}")
                    
                    # Check if it's a vectorized environment that supports 'call'
                    if hasattr(potential_vector_env, 'call') and callable(potential_vector_env.call):
                        logger.info(f"Attempting to call 'signal_graceful_shutdown' on sub-environments of vector_env (type: {type(potential_vector_env)}).")
                        try:
                            # The 'call' method executes a named method on each sub-environment.
                            # We assume 'signal_graceful_shutdown' exists on the sub-environments (our TradingEnv).
                            potential_vector_env.call('signal_graceful_shutdown')
                            logger.info("Signal 'signal_graceful_shutdown' called on sub-environments via vector_env.call().")
                        except Exception as e_vec_call:
                            logger.error(f"Error calling 'signal_graceful_shutdown' via vector_env.call(): {e_vec_call}", exc_info=True)
                    elif hasattr(potential_vector_env, 'signal_graceful_shutdown'): # Fallback for non-vectorized but unwrapped
                         logger.info(f"Potential vector env (type: {type(potential_vector_env)}) is not a vector env with 'call', but trying direct call.")
                         try:
                            potential_vector_env.signal_graceful_shutdown()
                            logger.info(f"Signal sent directly to environment (type: {type(potential_vector_env)}).")
                         except AttributeError:
                            logger.error(f"Environment (type: {type(potential_vector_env)}) does NOT have 'signal_graceful_shutdown' method.", exc_info=True)
                         except Exception as e_direct_call:
                            logger.error(f"Error calling signal_graceful_shutdown directly on env (type: {type(potential_vector_env)}): {e_direct_call}", exc_info=True)
                    else:
                        logger.warning(f"Environment (type: {type(potential_vector_env)}) does not support 'call' and does not have 'signal_graceful_shutdown' directly.")
                else:
                    logger.warning("driver_env_runner does not have an 'env' attribute.")
            except Exception as e_local_signal:
                logger.error(f"Error during local environment signaling setup: {e_local_signal}", exc_info=True)
        else: # NUM_ROLLOUT_WORKERS > 0
            worker_manager = algo.workers # This should be the EnvRunnerGroup
            logger.info(f"Signaling {worker_manager.num_remote_workers()} remote workers to initiate graceful shutdown in their environments.")
            try:
                # Use a lambda that checks if 'signal_graceful_shutdown' exists
                def safe_signal(env):
                    if hasattr(env, 'signal_graceful_shutdown'):
                        env.signal_graceful_shutdown()
                    else:
                        # This log will appear in worker logs, not driver, if it happens.
                        # Consider a way to aggregate such warnings if needed.
                        print(f"Warning: Env {type(env)} in remote worker does not have signal_graceful_shutdown.")

                worker_manager.foreach_env(safe_signal)
                logger.info("Signal attempt sent to all remote environments via foreach_env.")
            except Exception as e_remote_signal:
                logger.error(f"Error signaling remote workers via foreach_env: {e_remote_signal}", exc_info=True)

        logger.info(f"Performing {GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS} drain iterations...")
        drain_start_time = datetime.datetime.now()
        for i in range(GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS): # Should be 1 iteration
            logger.info(f"Drain iteration {i+1}/{GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS}...")
            # Perform a training step to allow episodes to complete and be logged
            algo.train()
            if (datetime.datetime.now() - drain_start_time).total_seconds() > GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                logger.warning(f"Graceful shutdown drain timeout ({GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS}s) reached during drain train() call.")
                break
        logger.info("Graceful shutdown drain phase complete.")
  
    except Exception as e:
        logger.error(f"Error during RLlib training or graceful shutdown: {e}", exc_info=True)
    finally:
        # Clear the system-wide flag again to ensure clean state for any subsequent processes
        TradingEnv.clear_system_wide_graceful_shutdown()

        # Ensure algo is defined before calling stop()
        if 'algo' in locals() and algo is not None:
            logger.info("Stopping RLlib algorithm...")
            algo.stop()
            logger.info("RLlib algorithm stopped.")
        else:
            logger.warning("'algo' object not found or is None, skipping algo.stop().")

        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("Ray shut down. Debug training process finished.")
        logger.info("Check replay_buffer_debug.log for detailed debug information about the replay buffer.")

if __name__ == "__main__":
    main()
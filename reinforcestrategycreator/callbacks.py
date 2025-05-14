import datetime
import inspect # Keep for potential future use when inspecting episode object
import os # Add this import
import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
# from ray.rllib.env.episode import Episode # Reverted this import
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.evaluation.rollout_worker import RolloutWorker # Corrected import path
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch # Import SampleBatch
from ray.rllib.env.single_agent_episode import SingleAgentEpisode # Import SingleAgentEpisode
from typing import Dict, Optional, Any # Keep Any for now, or use Union[Episode, SingleAgentEpisode] if needed

# Forward-declare RLModule and EnvRunner if full import causes issues or for cleaner typing
# These are typically available in newer RLlib versions.
# from ray.rllib.core.rl_module.rl_module import RLModule
# from ray.rllib.env.env_runner import EnvRunner

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode as DbEpisode, Trade as DbTrade, TrainingRun

# Set up a specific logger for this module
logger = logging.getLogger('callbacks') # Use a specific name
logger.setLevel(logging.INFO) # Set default level for this logger

# Create a file handler for this logger
log_file_path = 'callbacks_debug.log'
try:
    # Remove old log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
except OSError as e:
    print(f"Error removing old log file {log_file_path}: {e}")

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers: # Avoid adding multiple handlers if script is reloaded
    logger.addHandler(file_handler)

logger.info("Callbacks logger initialized and configured to write to callbacks_debug.log")

class DatabaseLoggingCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict = None): # Re-added legacy_callbacks_dict
        super().__init__()
        self.run_id = None # Initialize self.run_id
        
        # Debug info about what we're receiving
        logger.info(f"DatabaseLoggingCallbacks init with: {legacy_callbacks_dict}")
        
        # Try multiple ways to access the run_id from different locations
        if legacy_callbacks_dict:
            if "run_id" in legacy_callbacks_dict:
                self.run_id = legacy_callbacks_dict["run_id"]
                logger.info(f"Found run_id in legacy_callbacks_dict: {self.run_id}")
            elif "callbacks_config" in legacy_callbacks_dict and legacy_callbacks_dict["callbacks_config"]:
                if "run_id" in legacy_callbacks_dict["callbacks_config"]:
                    self.run_id = legacy_callbacks_dict["callbacks_config"]["run_id"]
                    logger.info(f"Found run_id in callbacks_config: {self.run_id}")
        
        # Note: Even if run_id is not found during initialization, it may be set later
        # via the set_run_id method or retrieved from algorithm config
        if not self.run_id:
            logger.warning("WARNING: run_id not found in callback initialization! Will attempt to get it during execution.")
            
        logger.info(f"DatabaseLoggingCallbacks initialized with run_id: {self.run_id}")
        
    def set_run_id(self, run_id: str) -> None:
        """Allow setting the run_id after initialization"""
        self.run_id = run_id
        logger.info(f"Run ID set to: {self.run_id}")

    def get_run_id_from_algorithm(self, worker) -> str:
        """Try to get run_id from algorithm's config if available"""
        # This method is reinstated as 'worker' will be available with the old signature.
        try:
            # Try getting from callbacks_config on the worker's algorithm_config
            if hasattr(worker, "config") and worker.config and "callbacks_config" in worker.config:
                 if worker.config["callbacks_config"] and "run_id" in worker.config["callbacks_config"]:
                    logger.info(f"Found run_id in worker.config['callbacks_config']")
                    return worker.config["callbacks_config"]["run_id"]

            # Fallback: Try getting from worker.env.callbacks_config if worker is an EnvRunner and env has it
            # This is more speculative.
            if hasattr(worker, "env") and hasattr(worker.env, "config") and worker.env.config:
                if "callbacks_config" in worker.env.config and worker.env.config["callbacks_config"]:
                    if "run_id" in worker.env.config["callbacks_config"]:
                        logger.info(f"Found run_id in worker.env.config['callbacks_config']")
                        return worker.env.config["callbacks_config"]["run_id"]
            
            # Try algorithm instance directly if worker is an Algorithm object (less likely for episode callbacks)
            if hasattr(worker, "get_policy") and hasattr(worker.get_policy(), "config"):
                config = worker.get_policy().config
                if config and "callbacks_config" in config and config["callbacks_config"]:
                    if "run_id" in config["callbacks_config"]:
                        logger.info(f"Found run_id in worker.get_policy().config['callbacks_config']")
                        return config["callbacks_config"]["run_id"]
            
            logger.warning("Could not find run_id through worker object examination.")
            return None
        except Exception as e:
            logger.warning(f"Error retrieving run_id from algorithm via worker: {e}")
            return None

    def on_episode_start(
        self,
        *,
        base_env: Optional[BaseEnv] = None, # Make it optional to bypass TypeError
        episode: Any,  # Reverted to Any
        env_index: Optional[int] = None,
        env_runner: Optional["EnvRunner"] = None, # New API stack specific
        rl_module: Optional["RLModule"] = None,   # New API stack specific
        # For compatibility / older RLlib versions that might still pass these:
        worker: Optional["RolloutWorker"] = None,
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs,
    ) -> None:
        """Called at the beginning of each episode.
        Handles both new (EnvRunner) and old (RolloutWorker) API stacks.
        """
        logger.info(f"on_episode_start: received base_env parameter. Type: {type(base_env)}, Is None: {base_env is None}")
        if "base_env" in kwargs: # Check if RLlib somehow passed it via **kwargs despite named param
             logger.warning("on_episode_start: 'base_env' key was also found directly in **kwargs. This is unexpected.")
             # If base_env param is None but it's in kwargs, we might want to use kwargs['base_env']
             if base_env is None and kwargs.get("base_env") is not None: # Use .get for safety
                 logger.info("on_episode_start: Using 'base_env' from **kwargs as parameter was None.")
                 base_env = kwargs.pop("base_env") # Use it and remove from kwargs to avoid issues

        try:
            # Use getattr for episode.id_ for robustness as episode_id is preferred
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.info(f"--- on_episode_start CALLED for RLlib episode_id: {rllib_episode_id_str} ---")
            
            # Log details about received parameters (base_env type logged above)
            logger.info(f"on_episode_start details: env_runner type: {type(env_runner)}, rl_module type: {type(rl_module)}, worker type: {type(worker)}")
            # Note: kwargs might have had 'base_env' popped if the above condition was met
            kwarg_keys = list(kwargs.keys())
            logger.info(f"on_episode_start additional kwargs received (after potential pop): {kwarg_keys}")

            # Determine the correct runner object (EnvRunner for new, RolloutWorker for old)
            # The `env_runner` argument is preferred for the new stack.
            # `worker` is the argument for the old stack.
            current_runner_obj = env_runner if env_runner is not None else worker

            if current_runner_obj:
                logger.info(f"Using runner object of type: {type(current_runner_obj)} for run_id retrieval.")
            else:
                logger.warning("No env_runner or worker object explicitly passed to on_episode_start.")

            # Ensure self.run_id is set.
            if not self.run_id:
                logger.warning("self.run_id not set during __init__. Attempting to retrieve from runner object or legacy dict...")
                if current_runner_obj: # Try getting from the resolved runner object
                    # Check if callbacks_config is directly on the runner_obj (e.g., if it's an Algorithm instance)
                    if hasattr(current_runner_obj, "callbacks_config") and current_runner_obj.callbacks_config and "run_id" in current_runner_obj.callbacks_config:
                        self.run_id = current_runner_obj.callbacks_config["run_id"]
                        logger.info(f"Found run_id in current_runner_obj.callbacks_config: {self.run_id}")
                    # Check if it's on runner_obj.config.callbacks_config (e.g., EnvRunner.config or RolloutWorker.config)
                    elif hasattr(current_runner_obj, "config") and hasattr(current_runner_obj.config, "callbacks_config"):
                        cb_cfg = current_runner_obj.config.callbacks_config
                        if cb_cfg and "run_id" in cb_cfg:
                            self.run_id = cb_cfg["run_id"]
                            logger.info(f"Found run_id in current_runner_obj.config.callbacks_config: {self.run_id}")
                
                # Fallback to legacy_callbacks_dict if still no run_id
                if not self.run_id and hasattr(self, '_legacy_callbacks_dict') and self._legacy_callbacks_dict:
                    if "run_id" in self._legacy_callbacks_dict:
                        self.run_id = self._legacy_callbacks_dict["run_id"]
                        logger.info(f"Found run_id in _legacy_callbacks_dict (fallback in on_episode_start): {self.run_id}")
                    elif "callbacks_config" in self._legacy_callbacks_dict and \
                         isinstance(self._legacy_callbacks_dict["callbacks_config"], dict) and \
                         "run_id" in self._legacy_callbacks_dict["callbacks_config"]:
                        self.run_id = self._legacy_callbacks_dict["callbacks_config"]["run_id"]
                        logger.info(f"Found run_id in _legacy_callbacks_dict['callbacks_config'] (fallback in on_episode_start): {self.run_id}")

            if not self.run_id:
                logger.error("CRITICAL: self.run_id could not be determined in on_episode_start. Skipping DB logging.")
                return
            
            # Prepare kwargs for _log_episode_start_data, removing explicitly passed args
            call_kwargs = {k: v for k, v in kwargs.items() if k not in ['worker', 'base_env', 'policies', 'episode', 'env_index', 'env_runner', 'rl_module']}
            
            # The `policies` argument is not available in the new stack's on_episode_start.
            # It's also not used by _log_episode_start_data's new signature.
            # The `worker` argument to _log_episode_start_data will now be `env_runner` from this scope.
            self._log_episode_start_data(
                episode=episode,
                base_env=base_env,
                env_runner=current_runner_obj, # Pass the resolved runner object
                env_index=env_index,
                rl_module=rl_module, # Pass rl_module if available
                **call_kwargs
            )
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_episode_start: {e_outer}", exc_info=True)

    def _log_episode_start_data(
            self,
            *,
            episode: Any, # Reverted to Any
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env_index: Optional[int] = None,
            rl_module: Optional["RLModule"] = None,
            **kwargs,
        ) -> None:
            """Helper method to log episode start data."""
            try:
                rllib_episode_id = getattr(episode, 'id_', None)
                if not rllib_episode_id:
                    # Fallback for older RLlib versions or different episode objects
                    rllib_episode_id = getattr(episode, 'episode_id', f"unknown_rllib_ep_id_{datetime.datetime.now().timestamp()}")
                    logger.warning(f"Using fallback rllib_episode_id: {rllib_episode_id}")

                logger.info(f"--- _log_episode_start_data ENTERED for RLlib episode_id: {rllib_episode_id} ---")
                # Log the type of env_runner received for debugging
                logger.info(f"_log_episode_start_data received env_runner of type: {type(env_runner)}")
                logger.info(f"Run ID for this episode: {self.run_id}")


                if not self.run_id: # Attempt to get run_id from env_runner if not already set on self
                    if env_runner and hasattr(env_runner, "config") and hasattr(env_runner.config, "callbacks_config"):
                        cb_cfg = env_runner.config.callbacks_config
                        if cb_cfg and "run_id" in cb_cfg:
                            self.run_id = cb_cfg["run_id"]
                            logger.info(f"Retrieved run_id '{self.run_id}' from env_runner.config.callbacks_config in _log_episode_start_data")
                    
                    if not self.run_id and hasattr(self, '_legacy_callbacks_dict') and self._legacy_callbacks_dict: # Final fallback
                        if "run_id" in self._legacy_callbacks_dict:
                            self.run_id = self._legacy_callbacks_dict["run_id"]
                            logger.info(f"Retrieved run_id '{self.run_id}' from _legacy_callbacks_dict in _log_episode_start_data")
                        elif "callbacks_config" in self._legacy_callbacks_dict and \
                            isinstance(self._legacy_callbacks_dict["callbacks_config"], dict) and \
                            "run_id" in self._legacy_callbacks_dict["callbacks_config"]:
                            self.run_id = self._legacy_callbacks_dict["callbacks_config"]["run_id"]
                            logger.info(f"Retrieved run_id '{self.run_id}' from _legacy_callbacks_dict['callbacks_config'] in _log_episode_start_data")

                if not self.run_id:
                    logger.error("CRITICAL: self.run_id not set in _log_episode_start_data. Cannot log episode start.")
                    return

                initial_balance = None
                env_config_for_balance = None

                # Attempt 1: From the episode object itself (if populated by RLlib)
                if hasattr(episode, 'user_data') and 'env_config' in episode.user_data:
                    env_config_for_balance = episode.user_data['env_config']
                    if env_config_for_balance and 'initial_balance' in env_config_for_balance:
                        initial_balance = env_config_for_balance['initial_balance']
                        logger.info(f"Extracted initial_balance ({initial_balance}) from episode.user_data['env_config'].")

                # Attempt 2: From actual_env via base_env (primary method for direct env access)
                if initial_balance is None and base_env and hasattr(base_env, "get_sub_environments") and callable(base_env.get_sub_environments):
                    sub_envs = base_env.get_sub_environments()
                    if sub_envs:
                        current_actual_env = None
                        if env_index is not None and env_index < len(sub_envs):
                            current_actual_env = sub_envs[env_index]
                            logger.info(f"Accessing sub_env at index {env_index}, type: {type(current_actual_env)}")
                        elif sub_envs:
                            current_actual_env = sub_envs[0]
                            logger.info(f"Accessing first sub_env (index 0) as env_index was {env_index}, type: {type(current_actual_env)}")
                        
                        if current_actual_env:
                            if hasattr(current_actual_env, 'initial_balance'):
                                initial_balance = current_actual_env.initial_balance
                                logger.info(f"Extracted initial_balance ({initial_balance}) directly from actual_env.initial_balance.")
                            elif hasattr(current_actual_env, 'env_config') and isinstance(current_actual_env.env_config, dict) and 'initial_balance' in current_actual_env.env_config:
                                initial_balance = current_actual_env.env_config['initial_balance']
                                logger.info(f"Extracted initial_balance ({initial_balance}) from actual_env.env_config.")
                    else:
                        logger.warning("base_env.get_sub_environments() returned empty or None.")
                elif initial_balance is None:
                    logger.warning("base_env is None or does not support get_sub_environments for initial_balance retrieval.")

                # Attempt 3: Fallback to env_runner's AlgorithmConfig
                if initial_balance is None:
                    logger.info("Initial balance still None, trying env_runner's AlgorithmConfig.")
                    if not env_runner:
                        logger.warning("EnvRunner object (env_runner) is None in _log_episode_start_data.")
                    elif not hasattr(env_runner, 'config'):
                        logger.warning(f"EnvRunner object (type: {type(env_runner)}) does not have 'config' attribute.")
                    else:
                        runner_config = env_runner.config # This should be the AlgorithmConfig
                        logger.info(f"EnvRunner.config type: {type(runner_config)}")
                        if hasattr(runner_config, 'env_config'):
                            algo_env_config = runner_config.env_config
                            logger.info(f"runner_config.env_config type: {type(algo_env_config)}")
                            if algo_env_config and isinstance(algo_env_config, dict) and 'initial_balance' in algo_env_config:
                                initial_balance = algo_env_config['initial_balance']
                                logger.info(f"Extracted initial_balance ({initial_balance}) from env_runner.config.env_config.")
                            elif algo_env_config:
                                logger.warning(f"env_runner.config.env_config found (type {type(algo_env_config)}), but 'initial_balance' missing or not a dict. Keys: {list(algo_env_config.keys()) if isinstance(algo_env_config, dict) else 'N/A'}")
                            else:
                                logger.warning("env_runner.config.env_config is None or empty.")
                        elif isinstance(runner_config, dict) and 'env_config' in runner_config:
                            algo_env_config = runner_config['env_config']
                            logger.info(f"env_runner.config is a dict, accessed env_runner.config['env_config'], type: {type(algo_env_config)}")
                            if algo_env_config and isinstance(algo_env_config, dict) and 'initial_balance' in algo_env_config:
                                initial_balance = algo_env_config['initial_balance']
                                logger.info(f"Extracted initial_balance ({initial_balance}) from env_runner.config['env_config'].")
                            elif algo_env_config:
                                logger.warning(f"env_runner.config['env_config'] found (type {type(algo_env_config)}), but 'initial_balance' missing or not a dict. Keys: {list(algo_env_config.keys()) if isinstance(algo_env_config, dict) else 'N/A'}")
                            else:
                                logger.warning("env_runner.config['env_config'] is None or empty.")
                        else:
                            logger.warning(f"env_runner.config (type: {type(runner_config)}) does not have 'env_config' attribute and is not a dict with 'env_config' key. env_runner.config keys (if dict): {list(runner_config.keys()) if isinstance(runner_config, dict) else 'N/A'}")
                
                if initial_balance is None:
                    logger.error("CRITICAL: Could not determine initial_balance for episode start after all attempts. Will be Null in DB.")
                else:
                    logger.info(f"Final initial_balance to be logged: {initial_balance}")


                with get_db_session() as db:
                    # Create a new episode record
                    db_episode = DbEpisode(
                        rllib_episode_id=str(rllib_episode_id), # Ensure it's a string
                        run_id=self.run_id,
                        start_time=datetime.datetime.now(datetime.timezone.utc),
                        initial_portfolio_value=initial_balance, # Corrected field name
                        status="started"
                    )
                    db.add(db_episode)
                    db.commit()
                    db.refresh(db_episode)
                    # Store the database episode ID in the RLlib episode's custom_data
                    # This is crucial for linking data in on_episode_end
                    episode.custom_data["db_episode_id"] = db_episode.episode_id # Corrected attribute
                    episode.custom_data["db_run_id"] = self.run_id # Also store run_id for safety
                    logger.info(f"Episode {db_episode.episode_id} (RLlib ID: {rllib_episode_id}) started and logged to DB. Initial balance: {initial_balance}. DB ID stored in custom_data.")

            except Exception as e:
                logger.error(f"Error in _log_episode_start_data for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e}", exc_info=True)
                # Do not re-raise, allow the main callback to continue if possible
    def _log_episode_end_data(
        self,
        *,
        episode: Any,
        worker: Optional[EnvRunnerGroup] = None, # Keep for signature consistency if called from on_episode_end
        base_env: Optional[BaseEnv] = None,    # Keep for signature consistency
        policies: Optional[Dict[str, Policy]] = None, # Keep for signature consistency
        env_index: Optional[int] = None,       # Keep for signature consistency
        **kwargs,
    ) -> None:
        """Helper method to log episode end data. Can be called from on_episode_end or on_sample_end."""
        try:
            logger.info(f"--- _log_episode_end_data ENTERED for RLlib episode_id: {getattr(episode, 'id_', 'UNKNOWN_RLIB_EPISODE_ID')} ---")
            # Log available arguments for debugging, especially if called from different contexts
            logger.info(f"Args for _log_episode_end_data: Worker: {worker}, BaseEnv: {base_env}, EnvIndex: {env_index}, Policies: {policies is not None}, Kwargs: {kwargs.keys()}")
            logger.info(f"Episode object type: {type(episode)}")
            # logger.info(f"Episode available attributes: {dir(episode)}") # Can be very verbose

            # Ensure run_id is set (it should be from __init__ or on_episode_start)
            if not self.run_id:
                # Attempt to get run_id if it's somehow not set (should be rare here)
                if worker:
                    self.run_id = self.get_run_id_from_algorithm(worker)
                if not self.run_id and "callbacks_config" in kwargs:
                     callbacks_cfg = kwargs["callbacks_config"]
                     if callbacks_cfg and "run_id" in callbacks_cfg:
                         self.run_id = callbacks_cfg["run_id"]
                
                if not self.run_id:
                    logger.error(f"CRITICAL: self.run_id is NOT SET in _log_episode_end_data for RLlib episode_id {getattr(episode, 'id_', 'unknown_rllib_id')}. Aborting DB log.")
                    return

            db_episode_id = episode.custom_data.get("db_episode_id")
            
            # Check if this episode's end has already been logged by this callback instance
            if episode.custom_data.get("_db_logged_end", False):
                logger.info(f"Episode {getattr(episode, 'id_', 'unknown_rllib_id')} (DB ID: {db_episode_id}) end already processed by this callback. Skipping _log_episode_end_data.")
                return

            if not db_episode_id:
                logger.warning(f"db_episode_id not found in episode.custom_data for RLlib episode_id {getattr(episode, 'id_', 'unknown_rllib_id')} in _log_episode_end_data. This episode may have already been processed or started incorrectly.")
                # Attempt to find an open episode as a fallback, though this is less reliable if called from on_sample_end for multiple episodes
                try:
                    with get_db_session() as db:
                        open_episode_q = db.query(DbEpisode).filter(
                            DbEpisode.run_id == self.run_id,
                            DbEpisode.end_time.is_(None),
                            # Heuristic: try to match based on some episode property if available, e.g. start time if stored in custom_data
                            # For now, just take the latest open one, but this is risky.
                        ).order_by(DbEpisode.start_time.desc()).first()
                        
                        if open_episode_q:
                            db_episode_id = open_episode_q.episode_id
                            logger.info(f"Fallback: Found open DB episode with ID {db_episode_id}")
                        else:
                            logger.error(f"Fallback: Could not find an open DB episode for run_id {self.run_id}. Cannot log episode end for RLlib ID {getattr(episode, 'id_', 'unknown_rllib_id')}.")
                            return
                except Exception as e_find_ep:
                    logger.error(f"Fallback: Error finding open DB episode: {e_find_ep}", exc_info=True)
                    return

            # Check if this db_episode_id has already been processed (i.e., has an end_time in DB)
            # This check is still useful as a safeguard against race conditions or multiple callback instances
            try:
                with get_db_session() as db_check:
                    existing_db_ep = db_check.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                    if existing_db_ep and existing_db_ep.end_time is not None:
                        logger.warning(f"DB Episode {db_episode_id} (RLlib ID: {getattr(episode, 'id_', 'unknown_rllib_id')}) already has an end_time in DB. Skipping update from _log_episode_end_data.")
                        episode.custom_data["_db_logged_end"] = True # Mark as processed by callback even if DB said so
                        return
            except Exception as e_check:
                logger.error(f"Error checking existing DB episode {db_episode_id}: {e_check}", exc_info=True)
                # Proceed with caution or return, depending on desired robustness

            final_portfolio_value = None
            initial_portfolio_value = None
            actual_env_for_value = None
            total_reward_val = None
            episode_length_val = None
            custom_metrics_val = {}
            sharpe_ratio_val = None
            max_drawdown_val = None
            win_rate_val = None
            completed_trades = []

            # Determine the actual environment instance for metrics
            if hasattr(episode, 'env') and episode.env is not None:
                 actual_env_for_value = episode.env
                 logger.info("Using episode.env as actual_env_for_value for metrics in _log_episode_end_data.")
            elif 'env' in kwargs and kwargs['env'] is not None:
                actual_env_for_value = kwargs['env']
                logger.info("Using env from kwargs as actual_env_for_value for metrics in _log_episode_end_data.")
            elif 'env_runner' in kwargs and hasattr(kwargs['env_runner'], 'env') and kwargs['env_runner'].env is not None:
                actual_env_for_value = kwargs['env_runner'].env
                logger.info("Using env_runner.env from kwargs as actual_env_for_value for metrics in _log_episode_end_data.")
            elif base_env and env_index is not None:
                try:
                    actual_env_for_value = base_env.get_sub_environments()[env_index]
                    logger.info(f"Using sub_environment at index {env_index} from base_env for metrics in _log_episode_end_data.")
                except Exception as e:
                    logger.warning(f"Error getting sub_environment from base_env at index {env_index} for metrics in _log_episode_end_data: {e}")
            else:
                logger.warning("Could not determine specific environment instance for metrics in _log_episode_end_data.")

            if isinstance(episode, SingleAgentEpisode):
                logger.info("Processing metrics for SingleAgentEpisode.")
                if callable(getattr(episode, 'get_infos', None)) and episode.get_infos():
                    last_info = episode.get_infos()[-1] # Get the info dict from the last step
                    if isinstance(last_info, dict):
                        final_portfolio_value = last_info.get('portfolio_value')
                        initial_portfolio_value = last_info.get('initial_balance') # Assuming env puts it in last info
                        sharpe_ratio_val = last_info.get("sharpe_ratio")
                        max_drawdown_val = last_info.get("max_drawdown")
                        win_rate_val = last_info.get("win_rate")
                        completed_trades = last_info.get("completed_trades", [])
                        logger.info(f"From last_info: pf={final_portfolio_value}, ib={initial_portfolio_value}, sr={sharpe_ratio_val}, mdd={max_drawdown_val}, wr={win_rate_val}, trades={len(completed_trades)}")
                    else:
                        logger.warning(f"Last info from episode.get_infos() is not a dict: {last_info}")
                else:
                    logger.warning("SingleAgentEpisode does not have get_infos() or it's empty.")

                if callable(getattr(episode, 'get_rewards', None)):
                    total_reward_val = sum(episode.get_rewards())
                    logger.info(f"Calculated total_reward_val: {total_reward_val} from episode.get_rewards()")
                else:
                    logger.warning("SingleAgentEpisode does not have get_rewards().")
                
                episode_length_val = len(episode) # Use len(episode) for SingleAgentEpisode
                logger.info(f"Episode length: {episode_length_val}")

            else: # Fallback for older Episode types or other structures
                logger.info("Processing metrics for non-SingleAgentEpisode (old API or custom).")
                # 1. Try episode.last_info_for() - Old API
                try:
                    if hasattr(episode, 'last_info_for') and callable(getattr(episode, 'last_info_for')):
                        agent_ids_to_try = [None, Policy.DEFAULT_POLICY_ID]
                        if hasattr(episode, 'agent_ids') and episode.agent_ids:
                            agent_ids_to_try.extend(list(episode.agent_ids))
                        
                        for agent_id in agent_ids_to_try:
                            try:
                                last_info = episode.last_info_for(agent_id=agent_id) if agent_id is not None else episode.last_info_for()
                                if last_info and isinstance(last_info, dict):
                                    if 'portfolio_value' in last_info: final_portfolio_value = last_info['portfolio_value']
                                    if initial_portfolio_value is None and 'initial_balance' in last_info: initial_portfolio_value = last_info['initial_balance']
                                    if final_portfolio_value is not None: break
                            except Exception: pass # Ignore if agent_id fails
                        if final_portfolio_value is None: logger.warning("portfolio_value not in last_info_for().")
                    else:
                        logger.warning("episode does not have last_info_for attribute.")
                except Exception as e:
                    logger.error(f"Error with last_info_for: {e}")

                total_reward_val = getattr(episode, 'total_reward', None) # Old API direct attribute
                episode_length_val = getattr(episode, 'length', None) # Old API direct attribute
                custom_metrics_val = getattr(episode, 'custom_metrics', {}) # Old API direct attribute
                sharpe_ratio_val = custom_metrics_val.get("sharpe_ratio")
                max_drawdown_val = custom_metrics_val.get("max_drawdown")
                win_rate_val = custom_metrics_val.get("win_rate")

            # Fallback for initial_portfolio_value if not found in info from SingleAgentEpisode.get_infos()[-1]
            if initial_portfolio_value is None:
                logger.info(f"initial_portfolio_value is None after checking last_info. Attempting DB fallback for DB episode {db_episode_id}.")
                try:
                    with get_db_session() as db:
                        db_ep_for_initial = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                        if db_ep_for_initial and db_ep_for_initial.initial_portfolio_value is not None:
                            initial_portfolio_value = db_ep_for_initial.initial_portfolio_value
                            logger.info(f"Got initial_balance {initial_portfolio_value} from DB for episode {db_episode_id}.")
                        else:
                            logger.warning(f"initial_portfolio_value not found or is None in DB for episode {db_episode_id}. Will try actual_env_for_value.")
                except Exception as e_db_initial:
                     logger.warning(f"Error fetching initial_portfolio_value from DB for episode {db_episode_id}: {e_db_initial}")

            if initial_portfolio_value is None: # Try actual_env_for_value as a further fallback
                if actual_env_for_value:
                    initial_portfolio_value = getattr(actual_env_for_value, 'initial_balance', None)
                    if initial_portfolio_value is not None:
                        logger.info(f"Got initial_balance {initial_portfolio_value} from actual_env_for_value (e.g., env instance passed in kwargs).")
                    else:
                        logger.warning(f"initial_balance not found in actual_env_for_value for episode {db_episode_id}.")
                else:
                    logger.warning(f"actual_env_for_value is None, cannot attempt to get initial_balance from it for episode {db_episode_id}.")

            if initial_portfolio_value is None: # Final default if all other sources fail
                initial_portfolio_value = 10000.0
                logger.warning(f"Using default initial_balance {initial_portfolio_value} for episode {db_episode_id} after all fallbacks failed.")
            
            # Fallback for final_portfolio_value if still None
            if final_portfolio_value is None and actual_env_for_value:
                final_portfolio_value = getattr(actual_env_for_value, 'portfolio_value', None)
                if final_portfolio_value is not None: logger.info(f"Got final_portfolio_value {final_portfolio_value} from actual_env_for_value.")
                else: logger.warning("portfolio_value not found in actual_env_for_value.")

            if final_portfolio_value is None and total_reward_val is not None and initial_portfolio_value is not None:
                # If total_reward is absolute (like final balance), use it. If it's relative PnL, add to initial.
                # Assuming total_reward_val might be PnL if portfolio_value is missing.
                # This part is heuristic and depends on env's reward structure.
                # For now, if final_pf is None but total_reward_val is available, we might assume total_reward_val is the PnL.
                # Or, if the environment's reward is the change in portfolio value, then final_pf = initial_pf + total_reward
                # Let's assume for now if final_pf is None, we can't reliably set it from total_reward without more info.
                logger.warning(f"final_portfolio_value is None. total_reward_val is {total_reward_val}. Cannot reliably set final_pf from total_reward without knowing reward structure.")


            pnl = None
            if final_portfolio_value is not None and initial_portfolio_value is not None:
                pnl = final_portfolio_value - initial_portfolio_value
            else:
                logger.warning(f"Cannot calculate PnL. final_pf: {final_portfolio_value}, initial_pf: {initial_portfolio_value}")

            # Fallback for custom metrics if not in last_info (for SingleAgentEpisode) or episode.custom_metrics (old)
            if actual_env_for_value:
                if sharpe_ratio_val is None and hasattr(actual_env_for_value, 'sharpe_ratio'): sharpe_ratio_val = getattr(actual_env_for_value, 'sharpe_ratio', None)
                if max_drawdown_val is None and hasattr(actual_env_for_value, 'max_drawdown'): max_drawdown_val = getattr(actual_env_for_value, 'max_drawdown', None)
                if win_rate_val is None and hasattr(actual_env_for_value, 'win_rate'): win_rate_val = getattr(actual_env_for_value, 'win_rate', None)

            # Fallback for completed_trades if not in last_info (for SingleAgentEpisode)
            if not completed_trades: # if it's still an empty list from episode.get_infos()
                # Try getting from the environment instance directly using its public method
                if actual_env_for_value and callable(getattr(actual_env_for_value, 'get_completed_trades', None)):
                    try:
                        completed_trades = actual_env_for_value.get_completed_trades()
                        logger.info(f"Found {len(completed_trades)} trades via actual_env_for_value.get_completed_trades().")
                    except Exception as e_get_trades:
                        logger.warning(f"Error calling actual_env_for_value.get_completed_trades(): {e_get_trades}")
                elif actual_env_for_value and hasattr(actual_env_for_value, '_completed_trades'): # Less ideal direct access
                    completed_trades = actual_env_for_value._completed_trades
                    logger.info(f"Found {len(completed_trades)} trades in actual_env_for_value._completed_trades (direct access fallback).")
                elif hasattr(episode, 'user_data') and "completed_trades" in episode.user_data: # Old API style
                    completed_trades = episode.user_data.get("completed_trades", [])
                    logger.info(f"Found {len(completed_trades)} trades in episode.user_data (old API fallback).")
            
            if not completed_trades:
                 logger.info("No completed_trades found via any method. This may be normal if no trades occurred in the episode.")
            else:
                 logger.info(f"Found {len(completed_trades)} completed_trades.")

            logger.info(f"Metrics to log for DB episode {db_episode_id} in _log_episode_end_data: final_pf={final_portfolio_value}, pnl={pnl}, total_reward={total_reward_val}, length={episode_length_val}, sharpe={sharpe_ratio_val}, drawdown={max_drawdown_val}, win_rate={win_rate_val}, trades_count={len(completed_trades)}")

            with get_db_session() as db:
                db_ep_to_update = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                if db_ep_to_update:
                    if db_ep_to_update.end_time is not None:
                        logger.warning(f"DB Episode {db_episode_id} (RLlib ID: {getattr(episode, 'id_', 'unknown_rllib_id')}) end_time already set. Re-logging attempt from _log_episode_end_data. This might indicate multiple calls.")
                    
                    db_ep_to_update.end_time = datetime.datetime.now(datetime.timezone.utc)
                    db_ep_to_update.final_portfolio_value = final_portfolio_value
                    db_ep_to_update.pnl = pnl
                    db_ep_to_update.total_reward = total_reward_val
                    db_ep_to_update.total_steps = episode_length_val
                    db_ep_to_update.sharpe_ratio = sharpe_ratio_val
                    db_ep_to_update.max_drawdown = max_drawdown_val
                    db_ep_to_update.win_rate = win_rate_val

                    logger.info(f"Attempting to update DB episode {db_episode_id} with collected metrics in _log_episode_end_data.")
                                        
                    for trade_info in completed_trades: # Iterate over the already determined completed_trades list
                        if not isinstance(trade_info, dict):
                            logger.warning(f"Skipping non-dict trade_info: {trade_info} in _log_episode_end_data")
                            continue
                        db_trade = DbTrade(
                            episode_id=db_episode_id,
                            entry_time=trade_info.get("entry_time"),
                            exit_time=trade_info.get("exit_time"),
                            entry_price=trade_info.get("entry_price"),
                            exit_price=trade_info.get("exit_price"),
                            quantity=trade_info.get("quantity"),
                            direction=trade_info.get("direction"),
                            pnl=trade_info.get("pnl"),
                            costs=trade_info.get("costs", 0.0)
                        )
                        db.add(db_trade)
                    
                    if actual_env_for_value and hasattr(actual_env_for_value, '_clear_episode_trades'):
                        try:
                            actual_env_for_value._clear_episode_trades()
                            logger.info("Cleared trades from actual_env_for_value in _log_episode_end_data.")
                        except Exception as e_clear:
                            logger.warning(f"Error clearing trades from actual_env_for_value in _log_episode_end_data: {e_clear}")
                    
                    db.commit()
                    episode.custom_data["_db_logged_end"] = True # Mark as successfully processed
                    logger.info(f"Successfully updated DB episode {db_episode_id} for RLlib episode_id {getattr(episode, 'id_', 'unknown_rllib_id')} with {len(completed_trades)} trades in _log_episode_end_data.")
                else:
                    logger.warning(f"Could not find DB episode with ID {db_episode_id} to update in _log_episode_end_data.")
        
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in _log_episode_end_data: {e_outer}", exc_info=True)

    def on_episode_end(
        self,
        *,
        episode: Any, # Make episode the primary required kwarg
        **kwargs,    # Capture all other arguments
    ) -> None:
        """Called at the end of each episode.
        Simplified signature to rely on kwargs for other parameters.
        """
        try: # Outer try-except for the entire method
            logger.info(f"--- on_episode_end CALLED for RLlib episode_id: {getattr(episode, 'id_', 'UNKNOWN_RLIB_EPISODE_ID')} ---")
            logger.info(f"on_episode_end kwargs received: {list(kwargs.keys())}")

            # Extract known optional params from kwargs if they exist, otherwise pass None
            worker = kwargs.get("worker")
            base_env = kwargs.get("base_env")
            policies = kwargs.get("policies")
            env_index = kwargs.get("env_index")

            self._log_episode_end_data(
                episode=episode,
                worker=worker, # Pass extracted or None
                base_env=base_env, # Pass extracted or None
                policies=policies, # Pass extracted or None
                env_index=env_index, # Pass extracted or None
                **kwargs # Pass along all original kwargs for flexibility in _log_episode_end_data
            )
        except Exception as e_outer: # Catch-all for the entire method
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_episode_end: {e_outer}", exc_info=True)

    def on_sample_end(
        self,
        *,
        worker: Optional[RolloutWorker] = None, # Changed to keyword-only with default, RolloutWorker type
        samples: Optional["SampleBatch"] = None, # Changed to keyword-only with default
        **kwargs
    ) -> None:
        """Processes completed episodes from a sample batch."""
        try:
            logger.info(f"on_sample_end called. kwargs received: {list(kwargs.keys())}")
            
            worker_to_use = kwargs.get("env_runner") # Prioritize env_runner from kwargs for new API stack
            if worker_to_use is not None:
                logger.info(f"Using 'env_runner' from kwargs (type: {type(worker_to_use)}) as worker_to_use.")
            elif worker is not None: # Fallback to direct worker argument if env_runner not in kwargs
                worker_to_use = worker
                logger.info(f"Using 'worker' argument (type: {type(worker_to_use)}) as worker_to_use.")
            else: # Only log warning if neither is available
                logger.error("on_sample_end: 'env_runner' not in kwargs and 'worker' argument is None. Cannot process samples.")
                return
            
            if samples is None:
                logger.error("on_sample_end called but 'samples' is None. Cannot process.")
                return

            logger.info(f"on_sample_end using worker_to_use type: {type(worker_to_use)}, samples type: {type(samples)}")

            logger.info("on_sample_end: Episode processing logic is currently commented out for testing on_episode_end reliability.")
            # episodes_to_process = []
            # if isinstance(samples, SampleBatch):
            #     logger.info(f"Samples is a SampleBatch. Processing {len(samples.get_terminated_episodes())} terminated episodes.")
            #     episodes_to_process.extend(samples.get_terminated_episodes())
            # elif isinstance(samples, list):
            #     logger.info(f"Samples is a list of length {len(samples)}. Iterating through items.")
            #     for i, item in enumerate(samples):
            #         # Assuming items in the list are individual episode objects if not SampleBatch
            #         if hasattr(item, 'episode_id') or hasattr(item, 'id_'): # Heuristic for episode-like objects
            #             logger.info(f"Item {i} (type: {type(item)}) appears to be an episode. Adding to process list.")
            #             episodes_to_process.append(item)
            #         elif isinstance(item, SampleBatch): # Should not happen if outer samples is a list of episodes, but good check
            #             num_terminated = len(item.get_terminated_episodes())
            #             logger.info(f"Item {i} is a SampleBatch, found {num_terminated} terminated episodes.")
            #             episodes_to_process.extend(item.get_terminated_episodes())
            #         else:
            #             logger.warning(f"Item {i} in samples list is neither an identifiable episode nor a SampleBatch (type: {type(item)}). Skipping.")
            # else:
            #     logger.error(f"Samples object is of unexpected type: {type(samples)}. Cannot extract episodes.")
            #     return

            # if not episodes_to_process:
            #     logger.info("No episodes found to process in on_sample_end.")
            #     return

            # for episode_obj in episodes_to_process: # Renamed to avoid conflict with outer scope 'episode' if any
            #     episode_id_str = getattr(episode_obj, 'id_', getattr(episode_obj, 'episode_id', 'unknown_rllib_id'))
            #     if getattr(episode_obj, 'is_done', False):
            #         logger.info(f"Processing episode {episode_id_str} (marked as is_done=True) from on_sample_end.")
            #         # actual_env determination is handled within _log_episode_end_data from kwargs
            #         self._log_episode_end_data(
            #             episode=episode_obj,
            #             worker=worker_to_use, # Pass the worker/env_runner context
            #             base_env=None,        # Let _log_episode_end_data determine from kwargs
            #             policies=None,
            #             env_index=None,       # Let _log_episode_end_data determine from kwargs
            #             **kwargs
            #         )
            #     else:
            #         logger.info(f"Skipping episode {episode_id_str} from on_sample_end as it is not marked as is_done=True (is_done: {getattr(episode_obj, 'is_done', 'N/A')}).")
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_sample_end: {e_outer}", exc_info=True)


    # on_episode_step could be used for very granular logging if needed,
    # but often on_episode_end is sufficient for aggregated metrics and trades.
    # def on_episode_step(self, *, worker: "WorkerSet", base_env: BaseEnv, episode: Episode, env_index: Optional[int] = None, **kwargs) -> None:
    #     # Example: log step-specific data if necessary
    #     # db_episode_id = episode.user_data.get("db_episode_id")
    #     # if db_episode_id:
    #     #     # Log step data
    #     pass
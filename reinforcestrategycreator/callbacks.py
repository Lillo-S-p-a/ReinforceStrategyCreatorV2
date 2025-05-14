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
from reinforcestrategycreator.db_models import (
    Episode as DbEpisode,
    Trade as DbTrade,
    TrainingRun,
    Step as DbStep, # Added
    TradingOperation as DbTradingOperation, # Added
    OperationType) # Added

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
        worker: Optional[EnvRunnerGroup] = None, 
        base_env: Optional[BaseEnv] = None,    
        policies: Optional[Dict[str, Policy]] = None, 
        env_index: Optional[int] = None,       
        **kwargs,
    ) -> None:
        """Helper method to log episode end data. Can be called from on_episode_end or on_sample_end."""
        try:
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.info(f"--- _log_episode_end_data ENTERED for RLlib episode_id: {rllib_episode_id_str} ---")
            logger.info(f"Args for _log_episode_end_data: Worker type: {type(worker)}, BaseEnv type: {type(base_env)}, EnvIndex: {env_index}, Policies: {policies is not None}, Kwarg keys: {list(kwargs.keys())}")
            logger.info(f"Episode object type: {type(episode)}")

            if not self.run_id:
                if worker: self.run_id = self.get_run_id_from_algorithm(worker)
                if not self.run_id and "callbacks_config" in kwargs and kwargs["callbacks_config"] and "run_id" in kwargs["callbacks_config"]:
                    self.run_id = kwargs["callbacks_config"]["run_id"]
                if not self.run_id:
                    logger.error(f"CRITICAL: self.run_id is NOT SET in _log_episode_end_data for RLlib episode_id {rllib_episode_id_str}. Aborting DB log.")
                    return

            db_episode_id = episode.custom_data.get("db_episode_id")
            if episode.custom_data.get("_db_logged_end", False):
                logger.info(f"Episode {rllib_episode_id_str} (DB ID: {db_episode_id}) end already processed by this callback. Skipping.")
                return

            if not db_episode_id:
                logger.warning(f"db_episode_id not found in episode.custom_data for RLlib episode_id {rllib_episode_id_str}. Attempting fallback.")
                try:
                    with get_db_session() as db_fallback:
                        open_episode_q = db_fallback.query(DbEpisode).filter(
                            DbEpisode.run_id == self.run_id,
                            DbEpisode.rllib_episode_id == rllib_episode_id_str,
                            DbEpisode.end_time.is_(None)
                        ).order_by(DbEpisode.start_time.desc()).first()
                        if open_episode_q:
                            db_episode_id = open_episode_q.episode_id
                            logger.info(f"Fallback: Found open DB episode with ID {db_episode_id} by matching rllib_episode_id.")
                            episode.custom_data["db_episode_id"] = db_episode_id
                        else:
                            logger.error(f"Fallback: Could not find an open DB episode for run_id {self.run_id} and rllib_episode_id {rllib_episode_id_str}. Cannot log episode end.")
                            return
                except Exception as e_find_ep:
                    logger.error(f"Fallback: Error finding open DB episode: {e_find_ep}", exc_info=True)
                    return
            
            try:
                with get_db_session() as db_check:
                    existing_db_ep = db_check.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                    if existing_db_ep and existing_db_ep.end_time is not None and existing_db_ep.status == "completed":
                        logger.warning(f"DB Episode {db_episode_id} (RLlib ID: {rllib_episode_id_str}) already has an end_time and status 'completed' in DB. Skipping update.")
                        episode.custom_data["_db_logged_end"] = True 
                        return
            except Exception as e_check:
                logger.error(f"Error checking existing DB episode {db_episode_id}: {e_check}", exc_info=True)

            initial_portfolio_value = None
            final_portfolio_value = None
            pnl_val = None
            sharpe_ratio_val = None
            max_drawdown_val = None
            total_reward_val = None
            total_steps_val = None
            win_rate_val = None
            completed_trades = []
            last_info_dict = {}
            
            actual_env_for_value = None
            if hasattr(episode, 'env') and episode.env is not None: actual_env_for_value = episode.env
            elif 'env' in kwargs and kwargs['env'] is not None: actual_env_for_value = kwargs['env']
            elif base_env and env_index is not None and hasattr(base_env, "get_sub_environments") and callable(base_env.get_sub_environments):
                try:
                    sub_envs = base_env.get_sub_environments()
                    if sub_envs and env_index < len(sub_envs): actual_env_for_value = sub_envs[env_index]
                except Exception as e: logger.warning(f"Error getting sub_environment: {e}")
            if actual_env_for_value: logger.info(f"Resolved actual_env_for_value: {type(actual_env_for_value)}")
            else: logger.warning("Could not resolve actual_env_for_value for direct metric access.")

            if isinstance(episode, SingleAgentEpisode):
                if callable(getattr(episode, 'get_infos', None)) and episode.get_infos():
                    temp_info = episode.get_infos()[-1]
                    if isinstance(temp_info, dict): last_info_dict = temp_info
                    else: logger.warning(f"SingleAgentEpisode.get_infos()[-1] not a dict: {temp_info}")
                else: logger.warning("SingleAgentEpisode: get_infos() not callable or empty.")
            elif hasattr(episode, 'last_info_for') and callable(getattr(episode, 'last_info_for')):
                agent_ids_to_try = [None, Policy.DEFAULT_POLICY_ID]
                if hasattr(episode, 'agent_ids') and episode.agent_ids: agent_ids_to_try.extend(list(episode.agent_ids))
                for agent_id in agent_ids_to_try:
                    try:
                        temp_info = episode.last_info_for(agent_id=agent_id) if agent_id is not None else episode.last_info_for()
                        if temp_info and isinstance(temp_info, dict):
                            last_info_dict = temp_info
                            logger.info(f"Retrieved last_info_dict from episode.last_info_for(agent_id={agent_id})")
                            break
                    except Exception: pass
                if not last_info_dict: logger.warning("Could not retrieve dict from episode.last_info_for().")
            else:
                logger.warning("Episode has neither get_infos() nor last_info_for() to get info dict.")

            if last_info_dict:
                logger.info(f"Processing last_info_dict with keys: {list(last_info_dict.keys())}")
                initial_portfolio_value = last_info_dict.get('initial_portfolio_value', last_info_dict.get('initial_balance'))
                final_portfolio_value = last_info_dict.get('final_portfolio_value', last_info_dict.get('portfolio_value'))
                pnl_val = last_info_dict.get('pnl')
                sharpe_ratio_val = last_info_dict.get('sharpe_ratio')
                max_drawdown_val = last_info_dict.get('max_drawdown')
                win_rate_val = last_info_dict.get('win_rate')
                if 'total_reward' in last_info_dict: total_reward_val = last_info_dict.get('total_reward')
                if 'total_steps' in last_info_dict: total_steps_val = last_info_dict.get('total_steps')
                elif 'episode_length' in last_info_dict: total_steps_val = last_info_dict.get('episode_length')
                
                temp_trades = last_info_dict.get("completed_trades", [])
                if isinstance(temp_trades, list): completed_trades = temp_trades
                else: logger.warning(f"completed_trades in last_info_dict is not a list: {temp_trades}")
            else:
                logger.warning("last_info_dict is empty. Metrics will rely on other sources.")

            if total_reward_val is None:
                if isinstance(episode, SingleAgentEpisode) and callable(getattr(episode, 'get_rewards', None)):
                    total_reward_val = sum(episode.get_rewards())
                    logger.info(f"Retrieved total_reward: {total_reward_val} from episode.get_rewards()")
                elif hasattr(episode, 'total_reward'):
                    total_reward_val = episode.total_reward
                    logger.info(f"Retrieved total_reward: {total_reward_val} from episode.total_reward (attribute)")
                else: logger.warning("total_reward_val remains None after checking episode attributes/methods.")

            if total_steps_val is None:
                if isinstance(episode, SingleAgentEpisode):
                    total_steps_val = len(episode)
                    logger.info(f"Retrieved total_steps: {total_steps_val} from len(episode)")
                elif hasattr(episode, 'length'):
                    total_steps_val = episode.length
                    logger.info(f"Retrieved total_steps: {total_steps_val} from episode.length (attribute)")
                else: logger.warning("total_steps_val remains None after checking episode attributes/methods.")

            if initial_portfolio_value is None:
                logger.info(f"initial_portfolio_value is None. Attempting DB/env fallback for DB episode {db_episode_id}.")
                try:
                    with get_db_session() as db_session_initial:
                        db_ep_for_initial = db_session_initial.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                        if db_ep_for_initial and db_ep_for_initial.initial_portfolio_value is not None:
                            initial_portfolio_value = db_ep_for_initial.initial_portfolio_value
                            logger.info(f"Retrieved initial_portfolio_value: {initial_portfolio_value} from DB.")
                        elif actual_env_for_value and hasattr(actual_env_for_value, 'initial_balance'):
                            initial_portfolio_value = getattr(actual_env_for_value, 'initial_balance')
                            logger.info(f"Retrieved initial_portfolio_value: {initial_portfolio_value} from actual_env_for_value.initial_balance.")
                        else: logger.error(f"CRITICAL: initial_portfolio_value not found for episode {db_episode_id}. It will be NULL.")
                except Exception as e_db_initial: logger.warning(f"Error fetching initial_portfolio_value from DB/env: {e_db_initial}")
            
            if pnl_val is None and final_portfolio_value is not None and initial_portfolio_value is not None:
                pnl_val = final_portfolio_value - initial_portfolio_value
                logger.info(f"Calculated PnL: {pnl_val}")
            elif pnl_val is None: logger.warning(f"PnL could not be determined/calculated. final_pf: {final_portfolio_value}, initial_pf: {initial_portfolio_value}.")

            custom_metrics_ep = getattr(episode, 'custom_metrics', {})
            if custom_metrics_ep: logger.info(f"Found episode.custom_metrics: {custom_metrics_ep}")
            if sharpe_ratio_val is None: sharpe_ratio_val = custom_metrics_ep.get("sharpe_ratio")
            if max_drawdown_val is None: max_drawdown_val = custom_metrics_ep.get("max_drawdown")
            if win_rate_val is None: win_rate_val = custom_metrics_ep.get("win_rate")

            if actual_env_for_value:
                if sharpe_ratio_val is None and hasattr(actual_env_for_value, 'sharpe_ratio'): sharpe_ratio_val = getattr(actual_env_for_value, 'sharpe_ratio', None)
                if max_drawdown_val is None and hasattr(actual_env_for_value, 'max_drawdown'): max_drawdown_val = getattr(actual_env_for_value, 'max_drawdown', None)
                if win_rate_val is None and hasattr(actual_env_for_value, 'win_rate'): win_rate_val = getattr(actual_env_for_value, 'win_rate', None)
                if not completed_trades and callable(getattr(actual_env_for_value, 'get_completed_trades', None)):
                    try:
                        temp_trades_env = actual_env_for_value.get_completed_trades()
                        if isinstance(temp_trades_env, list): completed_trades = temp_trades_env
                        logger.info(f"Retrieved {len(completed_trades)} trades via actual_env_for_value.get_completed_trades().")
                    except Exception as e_get_trades: logger.warning(f"Error calling actual_env_for_value.get_completed_trades(): {e_get_trades}")
            
            if not completed_trades and hasattr(episode, 'user_data') and "completed_trades" in episode.user_data:
                temp_trades_ud = episode.user_data.get("completed_trades", [])
                if isinstance(temp_trades_ud, list): completed_trades = temp_trades_ud
                logger.info(f"Retrieved {len(completed_trades)} trades from episode.user_data (final fallback).")

            current_end_time = datetime.datetime.now(datetime.timezone.utc)
            
            logger.info(f"FINAL METRICS for DB episode {db_episode_id} (RLlib ID: {rllib_episode_id_str}): "
                        f"run_id='{self.run_id}', end_time='{current_end_time}', "
                        f"initial_portfolio_value={initial_portfolio_value}, final_portfolio_value={final_portfolio_value}, "
                        f"pnl={pnl_val}, sharpe_ratio={sharpe_ratio_val}, max_drawdown={max_drawdown_val}, "
                        f"total_reward={total_reward_val}, total_steps={total_steps_val}, win_rate={win_rate_val}, "
                        f"status='completed', trades_count={len(completed_trades)}")

            with get_db_session() as db:
                db_ep_to_update = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                if db_ep_to_update:
                    logger.info(f"DB Episode {db_episode_id} BEFORE update: status='{db_ep_to_update.status}', end_time='{db_ep_to_update.end_time}', final_value='{db_ep_to_update.final_portfolio_value}', pnl='{db_ep_to_update.pnl}', total_reward='{db_ep_to_update.total_reward}'")
                    
                    # Convert numpy types to standard Python types before assignment
                    db_ep_to_update.end_time = current_end_time
                    if initial_portfolio_value is not None:
                        db_ep_to_update.initial_portfolio_value = float(initial_portfolio_value)
                    if final_portfolio_value is not None:
                        db_ep_to_update.final_portfolio_value = float(final_portfolio_value)
                    if pnl_val is not None:
                        db_ep_to_update.pnl = float(pnl_val)
                    if total_reward_val is not None:
                        db_ep_to_update.total_reward = float(total_reward_val)
                    if total_steps_val is not None:
                        db_ep_to_update.total_steps = int(total_steps_val) # total_steps should be an integer
                    if sharpe_ratio_val is not None:
                        db_ep_to_update.sharpe_ratio = float(sharpe_ratio_val)
                    if max_drawdown_val is not None:
                        db_ep_to_update.max_drawdown = float(max_drawdown_val)
                    if win_rate_val is not None:
                        db_ep_to_update.win_rate = float(win_rate_val)
                    
                    db_ep_to_update.status = "completed"
 
                    logger.info(f"DB Episode {db_episode_id} AFTER assignments (before commit): status='{db_ep_to_update.status}', end_time='{db_ep_to_update.end_time}', final_value='{db_ep_to_update.final_portfolio_value}', pnl='{db_ep_to_update.pnl}', total_reward='{db_ep_to_update.total_reward}', total_steps='{db_ep_to_update.total_steps}', sharpe='{db_ep_to_update.sharpe_ratio}', max_drawdown='{db_ep_to_update.max_drawdown}', win_rate='{db_ep_to_update.win_rate}'")
                                        
                    if completed_trades:
                        logger.info(f"Processing {len(completed_trades)} trades for episode {db_episode_id}.")
                        for trade_info in completed_trades: 
                            if not isinstance(trade_info, dict):
                                logger.warning(f"Skipping non-dict trade_info: {trade_info}")
                                continue
                            required_trade_fields = ["entry_time", "exit_time", "entry_price", "exit_price", "quantity", "direction"]
                            if any(trade_info.get(field) is None for field in required_trade_fields):
                                logger.warning(f"Skipping trade due to missing required fields: {trade_info}")
                                continue
                            db_trade = DbTrade(
                                episode_id=db_episode_id,
                                entry_time=trade_info.get("entry_time"), exit_time=trade_info.get("exit_time"),
                                entry_price=trade_info.get("entry_price"), exit_price=trade_info.get("exit_price"),
                                quantity=trade_info.get("quantity"), direction=trade_info.get("direction"),
                                pnl=trade_info.get("pnl"), costs=trade_info.get("costs", 0.0)
                            )
                            db.add(db_trade)
                    else:
                        logger.info(f"No trades to log for episode {db_episode_id}.")
                    
                    if actual_env_for_value and callable(getattr(actual_env_for_value, '_clear_episode_trades', None)):
                        try:
                            actual_env_for_value._clear_episode_trades()
                            logger.info("Called _clear_episode_trades() on actual_env_for_value.")
                        except Exception as e_clear: logger.warning(f"Error calling _clear_episode_trades(): {e_clear}")
                    
                    logger.info(f"Attempting db.commit() for DB Episode {db_episode_id} (RLlib ID: {rllib_episode_id_str})")
                    db.commit()
                    episode.custom_data["_db_logged_end"] = True 
                    logger.info(f"Successfully committed updates for DB episode {db_episode_id} (RLlib ID: {rllib_episode_id_str}).")
                else:
                    logger.error(f"CRITICAL: Could not find DB episode with ID {db_episode_id} to update. Metrics for RLlib episode {rllib_episode_id_str} will be lost.")
        
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in _log_episode_end_data for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e_outer}", exc_info=True)

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
            # These will be passed explicitly.
            extracted_worker = kwargs.get("worker")
            extracted_base_env = kwargs.get("base_env")
            extracted_policies = kwargs.get("policies")
            extracted_env_index = kwargs.get("env_index")

            # Prepare a new kwargs dict for spreading, excluding the ones we pass explicitly.
            remaining_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ['worker', 'base_env', 'policies', 'env_index', 'episode'] # 'episode' is also explicit
            }
            logger.info(f"Explicitly passing to _log_episode_end_data: worker={type(extracted_worker)}, base_env={type(extracted_base_env)}, policies={'set' if extracted_policies else 'None'}, env_index={extracted_env_index}")
            logger.info(f"Spreading remaining_kwargs to _log_episode_end_data: {list(remaining_kwargs.keys())}")

            self._log_episode_end_data(
                episode=episode,
                worker=extracted_worker,
                base_env=extracted_base_env,
                policies=extracted_policies,
                env_index=extracted_env_index,
                **remaining_kwargs # Pass only the remaining, unhandled kwargs
            )
        except Exception as e_outer: # Catch-all for the entire method
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_episode_end: {e_outer}", exc_info=True)

    def on_sample_end(
        self,
        *,
        worker: Optional[RolloutWorker] = None, # Changed to keyword-only with default, RolloutWorker type
        samples: Optional["SampleBatch"] = None, # Changed to keyword-only with default
        **kwargs,
    ) -> None:
        """Processes completed episodes from a sample batch."""
        try:
            logger.info(f"--- on_sample_end CALLED --- (Worker type: {type(worker)}, Samples type: {type(samples)})")

            # Enhanced logging for samples object
            logger.info(f"on_sample_end: Received samples of type: {type(samples)}")
            if isinstance(samples, list):
                logger.info(f"on_sample_end: samples is a list with length {len(samples)}")
            if hasattr(samples, "count"): # SampleBatch has a 'count' attribute
                logger.info(f"on_sample_end: samples has a count attribute: {samples.count}")

            if not samples:
                logger.warning("on_sample_end called with no samples. Nothing to process.")
                return

            # Ensure run_id is available
            if not self.run_id and worker:
                self.run_id = self.get_run_id_from_algorithm(worker)
            if not self.run_id and "callbacks_config" in kwargs:
                 callbacks_cfg = kwargs["callbacks_config"]
                 if callbacks_cfg and "run_id" in callbacks_cfg:
                     self.run_id = callbacks_cfg["run_id"]
            
            if not self.run_id:
                logger.error("CRITICAL: self.run_id could not be determined in on_sample_end. Cannot process episodes from batch.")
                return

            # Iterate over episodes in the batch
            # The structure of `samples` can vary. For `SampleBatch.TYPE_EPISODES`, it's a list of Episode objects.
            # For `SampleBatch.TYPE_TRAJECTORIES`, we need to reconstruct or identify episodes.
            # RLlib's default behavior is to provide `SingleAgentEpisode` objects when `batch_mode="complete_episodes"`.
            
            # Check if samples is a SampleBatch and contains episode objects directly
            # This is common with `batch_mode="complete_episodes"`
            if isinstance(samples, SampleBatch) and samples.is_single_trajectory(): # A single trajectory might be a full episode
                # This case is less common for 'complete_episodes' but good to check
                # We need to check if this trajectory represents a completed episode
                if samples.terminateds[-1] or samples.truncateds[-1]:
                    # This is tricky because SampleBatch itself isn't an 'Episode' object in the same way.
                    # We might need to rely on episode IDs if they are present in the batch.
                    # For now, this path is less robust for direct episode metric logging.
                    # The `split_by_episode` method is more reliable.
                    logger.info("on_sample_end: Received a single trajectory SampleBatch. Attempting to process if it's a completed episode.")
                    # This requires more complex logic to map SampleBatch to an Episode-like structure for _log_episode_end_data
                    # For now, we'll rely on split_by_episode for more robust handling.
                    pass # Placeholder for potential future handling if needed

            # More robust: Use split_by_episode if available on the samples object
            # This method is designed to yield individual Episode objects from a batch.
            if hasattr(samples, "split_by_episode") and callable(samples.split_by_episode):
                num_episodes_in_batch = 0
                processed_episode_ids_in_batch = set()

                for episode_batch in samples.split_by_episode(batch_size=None): # Process one episode at a time
                    num_episodes_in_batch += 1
                    # `episode_batch` here should be an Episode-like object (e.g., SingleAgentEpisode)
                    # or a SampleBatch representing a single episode.
                    # We need to ensure we pass an Episode-like object to _log_episode_end_data.

                    # If episode_batch is a SampleBatch representing one episode, we might need to adapt it
                    # or rely on the fact that on_episode_end would have been called for it.
                    # However, if on_episode_end is not reliably called, this is a fallback.

                    # Let's assume episode_batch is an Episode-like object here.
                    # We need to get its unique ID.
                    current_rllib_episode_id = getattr(episode_batch, 'id_', getattr(episode_batch, 'episode_id', None))
                    if not current_rllib_episode_id:
                        logger.warning("on_sample_end: Could not get RLlib episode ID from episode_batch. Skipping.")
                        continue
                    
                    if current_rllib_episode_id in processed_episode_ids_in_batch:
                        logger.info(f"on_sample_end: RLlib episode {current_rllib_episode_id} already processed in this batch. Skipping.")
                        continue

                    # Check if the episode is actually terminated or truncated
                    # For SingleAgentEpisode, check `is_terminated` and `is_truncated`
                    is_done = False
                    if isinstance(episode_batch, SingleAgentEpisode):
                        is_done = episode_batch.is_terminated or episode_batch.is_truncated
                        logger.info(f"on_sample_end: Processing SingleAgentEpisode {current_rllib_episode_id}. is_terminated={episode_batch.is_terminated}, is_truncated={episode_batch.is_truncated}")
                    elif isinstance(episode_batch, SampleBatch): # If it's a SampleBatch for an episode
                        if episode_batch.count > 0: # Ensure not empty
                           is_done = episode_batch.terminateds[-1] or episode_batch.truncateds[-1]
                           logger.info(f"on_sample_end: Processing SampleBatch for episode {current_rllib_episode_id}. terminateds[-1]={episode_batch.terminateds[-1]}, truncateds[-1]={episode_batch.truncateds[-1]}")
                        else:
                            logger.warning(f"on_sample_end: SampleBatch for episode {current_rllib_episode_id} is empty. Skipping.")
                            continue
                    else:
                        logger.warning(f"on_sample_end: episode_batch is of unexpected type {type(episode_batch)}. Cannot determine if done. Skipping.")
                        continue # Cannot determine if done

                    if is_done:
                        logger.info(f"on_sample_end: Episode {current_rllib_episode_id} is marked as done. Calling _log_episode_end_data.")
                        # Pass relevant parts of the worker or kwargs if needed by _log_episode_end_data
                        # The `worker` here is the RolloutWorker. `base_env` might be on worker.env.
                        # `policies` might be on worker.policy_map.
                        
                        # Construct a minimal set of arguments for _log_episode_end_data
                        # It primarily needs the 'episode' object.
                        # Other arguments like base_env, policies, env_index might be harder to get reliably here
                        # if they are not part of the episode_batch object itself.
                        # _log_episode_end_data is designed to be robust to missing optional args.
                        
                        # Try to get base_env and policies from the worker if available
                        current_base_env = getattr(worker, 'env', None) if worker else None
                        current_policies = getattr(worker, 'policy_map', None) if worker else None
                        # env_index is tricky here, might not be directly available per episode from a batch.
                        # _log_episode_end_data should handle env_index=None.

                        self._log_episode_end_data(
                            episode=episode_batch, # This is the crucial part
                            worker=worker, # Pass the RolloutWorker
                            base_env=current_base_env,
                            policies=current_policies,
                            env_index=None, # env_index is harder to determine here
                            **kwargs # Pass original kwargs
                        )
                        processed_episode_ids_in_batch.add(current_rllib_episode_id)
                    else:
                        logger.info(f"on_sample_end: Episode {current_rllib_episode_id} from batch is not done. Not logging end.")
                
                if num_episodes_in_batch == 0:
                    logger.info("on_sample_end: samples.split_by_episode() yielded no episodes.")
                else:
                    logger.info(f"on_sample_end: Processed {len(processed_episode_ids_in_batch)} completed episodes from a batch of {num_episodes_in_batch} episode parts.")

            else: # Fallback if split_by_episode is not available (older RLlib or different SampleBatch type)
                logger.warning("on_sample_end: samples object does not have split_by_episode. Trying to iterate if it's a list of episodes (less common).")
                logger.info(f"on_sample_end: samples type: {type(samples)}")
                if isinstance(samples, list): # e.g. if samples was already a list of Episode objects
                    if len(samples) > 0:
                        logger.info(f"on_sample_end: samples[0] type: {type(samples[0])}")
                        if isinstance(samples[0], dict):
                            logger.info(f"on_sample_end: samples[0] keys: {list(samples[0].keys())}")
                    processed_episode_ids_in_batch = set()
                    for i, episode_obj in enumerate(samples):
                        if not hasattr(episode_obj, 'is_terminated') or not hasattr(episode_obj, 'is_truncated'): # Basic check
                            logger.warning(f"Item {i} in samples list is not an episode-like object with is_terminated/is_truncated. Type: {type(episode_obj)}. Skipping.")
                            continue
                        
                        current_rllib_episode_id = getattr(episode_obj, 'id_', getattr(episode_obj, 'episode_id', None))
                        if not current_rllib_episode_id:
                             logger.warning(f"on_sample_end (list iteration): Could not get RLlib episode ID from episode object at index {i}. Skipping.")
                             continue
                        
                        if current_rllib_episode_id in processed_episode_ids_in_batch:
                            logger.info(f"on_sample_end (list iteration): RLlib episode {current_rllib_episode_id} already processed in this list. Skipping.")
                            continue

                        is_done = episode_obj.is_terminated or episode_obj.is_truncated
                        if is_done:
                            logger.info(f"on_sample_end (list iteration): Episode {current_rllib_episode_id} is done. Calling _log_episode_end_data.")
                            current_base_env = getattr(worker, 'env', None) if worker else None
                            current_policies = getattr(worker, 'policy_map', None) if worker else None
                            self._log_episode_end_data(
                                episode=episode_obj,
                                worker=worker,
                                base_env=current_base_env,
                                policies=current_policies,
                                env_index=None, # env_index is hard to determine here
                                **kwargs
                            )
                            processed_episode_ids_in_batch.add(current_rllib_episode_id)
                        else:
                             logger.info(f"on_sample_end (list iteration): Episode {current_rllib_episode_id} from list is not done. Not logging end.")
                    logger.info(f"on_sample_end (list iteration): Processed {len(processed_episode_ids_in_batch)} completed episodes from list.")
                else:
                    logger.warning("on_sample_end: samples object is not a list and does not have split_by_episode. Cannot process episodes from this SampleBatch for _log_episode_end_data.")

        except Exception as e:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_sample_end: {e}", exc_info=True)


    def on_episode_step(
        self,
        *,
        episode: Any, # Make episode the primary required kwarg
        **kwargs,    # Capture all other arguments
    ) -> None:
        """Called after each step within an episode."""
        try:
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.debug(f"--- on_episode_step CALLED for RLlib episode_id: {rllib_episode_id_str}, Length: {len(episode) if hasattr(episode, '__len__') else 'N/A'} ---")

            if not self.run_id:
                worker = kwargs.get("worker") # Try to get worker from kwargs
                if worker: self.run_id = self.get_run_id_from_algorithm(worker)
                if not self.run_id and "callbacks_config" in kwargs and kwargs["callbacks_config"] and "run_id" in kwargs["callbacks_config"]:
                     self.run_id = kwargs["callbacks_config"]["run_id"]

                if not self.run_id:
                    logger.error(f"CRITICAL: self.run_id is NOT SET in on_episode_step for RLlib episode {rllib_episode_id_str}. Cannot log step.")
                    return

            db_episode_id = episode.custom_data.get("db_episode_id")
            if not db_episode_id:
                # Try to find it if not set (e.g., if on_episode_start failed or was missed)
                # This is a fallback and might be slow if called every step.
                logger.warning(f"db_episode_id not in custom_data for RLlib episode {rllib_episode_id_str} during on_episode_step. Attempting DB lookup.")
                try:
                    with get_db_session() as db_find_ep:
                        db_ep_obj = db_find_ep.query(DbEpisode.episode_id).filter(
                            DbEpisode.run_id == self.run_id,
                            DbEpisode.rllib_episode_id == str(rllib_episode_id_str)
                        ).first()
                        if db_ep_obj:
                            db_episode_id = db_ep_obj.episode_id
                            episode.custom_data["db_episode_id"] = db_episode_id # Cache it
                            logger.info(f"Found and cached db_episode_id {db_episode_id} for RLlib_ep {rllib_episode_id_str} in on_episode_step.")
                        else:
                            logger.error(f"Could not find matching DB episode for RLlib_ep {rllib_episode_id_str} and run {self.run_id} in on_episode_step. Cannot log step.")
                            return
                except Exception as e_find:
                    logger.error(f"Error looking up db_episode_id in on_episode_step: {e_find}", exc_info=True)
                    return
            
            # Get the last observation, action, reward, and info
            # For SingleAgentEpisode, these are typically accessed via methods
            last_observation = None
            last_action = None
            last_reward = None
            last_info_for_step = {} # Store info dict for this step

            if isinstance(episode, SingleAgentEpisode):
                if episode.get_observations(): last_observation = episode.get_observations()[-1]
                if episode.get_actions(): last_action = episode.get_actions()[-1]
                if episode.get_rewards(): last_reward = episode.get_rewards()[-1]
                if episode.get_infos(): 
                    temp_info = episode.get_infos()[-1]
                    if isinstance(temp_info, dict): last_info_for_step = temp_info
            else: # Older Episode API might have these as direct attributes or via agent_id methods
                # This part is more complex for older multi-agent; focusing on single agent for now
                # last_observation = episode.last_observation_for() # Needs agent_id for multi-agent
                # last_action = episode.last_action_for()
                # last_reward = episode.last_reward_for()
                # last_info_for_step = episode.last_info_for()
                logger.debug("Using direct attribute access for obs/act/reward/info for non-SingleAgentEpisode (may be limited).")
                if hasattr(episode, '_agent_to_last_obs'): last_observation = episode._agent_to_last_obs.get(Policy.DEFAULT_POLICY_ID) # Example
                if hasattr(episode, '_agent_to_last_action'): last_action = episode._agent_to_last_action.get(Policy.DEFAULT_POLICY_ID)
                if hasattr(episode, '_agent_to_last_reward'): last_reward = episode._agent_to_last_reward.get(Policy.DEFAULT_POLICY_ID)
                if hasattr(episode, '_agent_to_last_info'): 
                    temp_info = episode._agent_to_last_info.get(Policy.DEFAULT_POLICY_ID, {})
                    if isinstance(temp_info, dict): last_info_for_step = temp_info


            current_step_number = len(episode) # RLlib episode length is 1-based for current step
            
            # Extract trading operation details if present in info
            operation_type_str = last_info_for_step.get("operation_type")
            operation_price = last_info_for_step.get("operation_price")
            operation_quantity = last_info_for_step.get("operation_quantity")
            operation_cost = last_info_for_step.get("operation_cost", 0.0) # Default to 0 if not present
            current_balance_after_op = last_info_for_step.get("balance_after_operation")
            current_position_after_op = last_info_for_step.get("position_after_operation")
            
            # Log step data
            with get_db_session() as db:
                db_step = DbStep(
                    episode_id=db_episode_id,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    reward=float(last_reward) if last_reward is not None else None,
                    # obs, action can be large; consider how/if to store them (e.g., hash, summary, or omit)
                    # observation_data=str(last_observation)[:255] if last_observation is not None else None, # Example: truncate
                    # action_data=str(last_action)[:255] if last_action is not None else None # Example: truncate
                )
                db.add(db_step)
                
                # Log trading operation if details are present
                if operation_type_str and operation_price is not None and operation_quantity is not None:
                    try:
                        op_type_enum = OperationType[operation_type_str.upper()] # Convert string to Enum
                        db_trading_op = DbTradingOperation(
                            episode_id=db_episode_id,
                            step_number=current_step_number,
                            timestamp=datetime.datetime.now(datetime.timezone.utc), # Could also use env's current time if available
                            operation_type=op_type_enum,
                            price=operation_price,
                            quantity=operation_quantity,
                            cost=operation_cost,
                            balance_after_operation=current_balance_after_op,
                            position_after_operation=current_position_after_op
                        )
                        db.add(db_trading_op)
                        logger.debug(f"Logged trading operation: {op_type_enum} at step {current_step_number}")
                    except KeyError:
                        logger.warning(f"Invalid operation_type string '{operation_type_str}' at step {current_step_number}. Cannot log operation.")
                    except Exception as e_op_log:
                        logger.error(f"Error logging trading operation at step {current_step_number}: {e_op_log}", exc_info=True)
                
                db.commit() # Commit step and any operation
                logger.debug(f"Step {current_step_number} for episode {db_episode_id} logged. Reward: {last_reward}")

        except Exception as e:
            logger.error(f"Error in on_episode_step for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e}", exc_info=True)
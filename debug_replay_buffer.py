"""
Debug wrapper for EpisodeReplayBuffer to diagnose reward issues
"""
import logging
import numpy as np
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
debug_logger = logging.getLogger("debug-replay-buffer")
debug_logger.setLevel(logging.DEBUG)

# Add a file handler to save logs
file_handler = logging.FileHandler("replay_buffer_debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(file_handler)


class DebugEpisodeReplayBuffer(EpisodeReplayBuffer):
    """
    Debug version of EpisodeReplayBuffer that adds logging to diagnose issues
    with empty rewards in n-step sampling.
    """

    def add(self, episodes):
        """Add logging to the add method"""
        import copy
        from ray.rllib.utils import force_list
        
        episodes_list = force_list(episodes)
        
        # Log episodes being added
        debug_logger.info(f"Adding {len(episodes_list)} episode(s) to buffer")
        
        for eps in episodes_list:
            # Make a copy as per original method
            eps = copy.deepcopy(eps)
            
            # Log episode details
            try:
                debug_logger.info(f"ADD: Episode ID: {eps.id_}, Length: {len(eps)}, Is Done: {eps.is_done}")
                
                # Try to access rewards in different ways
                rewards_list = None
                rewards_array = None
                
                # Try to get rewards as list
                if hasattr(eps, "_rewards_list"):
                    rewards_list = eps._rewards_list
                    debug_logger.info(f"  _rewards_list: {rewards_list}")
                
                # Try to get rewards as array
                if hasattr(eps, "_rewards"):
                    rewards_array = eps._rewards
                    debug_logger.info(f"  _rewards array: {rewards_array}")
                
                # Try the get_rewards method
                try:
                    method_rewards = eps.get_rewards()
                    debug_logger.info(f"  get_rewards() result: {method_rewards}")
                except Exception as e:
                    debug_logger.error(f"  Error getting rewards via get_rewards(): {e}")
                    
            except Exception as e:
                debug_logger.error(f"Error logging episode details: {e}")
        
        # Call the parent method to actually add the episodes
        return super().add(episodes)

    def _sample_episodes(self, num_items=None, **kwargs):
        """Add logging to the _sample_episodes method"""
        debug_logger.info(f"SAMPLE: _sample_episodes called with num_items={num_items}, kwargs={kwargs}")
        debug_logger.info(f"SAMPLE: Number of episodes in buffer: {len(self.episodes)}")
        
        # Get n_step from kwargs
        n_step = kwargs.get("n_step", 1)
        debug_logger.info(f"SAMPLE: n_step parameter: {n_step}")
        
        # Log episode details in buffer (first few only to avoid overwhelming logs)
        max_log = min(3, len(self.episodes))
        for i in range(max_log):
            ep = self.episodes[i]
            try:
                debug_logger.info(f"SAMPLE: Buffer Episode {i} - ID: {ep.id_}, Length: {len(ep)}")
                rewards = ep.get_rewards()
                debug_logger.info(f"SAMPLE: Buffer Episode {i} - Rewards: {rewards}")
            except Exception as e:
                debug_logger.error(f"Error logging episode {i} details: {e}")
        
        # Call the original method with a try-except to catch errors
        try:
            results = super()._sample_episodes(num_items=num_items, **kwargs)
            return results
        except Exception as e:
            debug_logger.error(f"Exception in _sample_episodes: {e}", exc_info=True)
            # Re-raise to maintain original behavior
            raise
            

# Helper method to monkey-patch the sampling function for extended debugging
def debug_sampling_process():
    """
    More invasive debugging that monkey-patches the EpisodeReplayBuffer._sample_episodes method
    to add detailed logging just before the scipy.signal.lfilter call.
    
    Call this function before training starts.
    """
    original_sample_episodes = EpisodeReplayBuffer._sample_episodes
    
    def patched_sample_episodes(self, num_items=None, **kwargs):
        # Log standard info
        debug_logger.info(f"PATCH: _sample_episodes called with num_items={num_items}, kwargs={kwargs}")
        debug_logger.info(f"PATCH: Number of episodes in buffer: {len(self.episodes)}")
        
        # Monkey-patch the SingleAgentEpisode.get_rewards method temporarily
        original_get_rewards = SingleAgentEpisode.get_rewards
        
        def instrumented_get_rewards(self):
            try:
                rewards = original_get_rewards(self)
                debug_logger.info(f"REWARDS CALL: Episode ID: {self.id_}, get_rewards() returned: {rewards}")
                return rewards
            except Exception as e:
                debug_logger.error(f"REWARDS ERROR: Episode ID: {self.id_}, Error in get_rewards(): {e}")
                raise
        
        # Apply the monkey patch
        SingleAgentEpisode.get_rewards = instrumented_get_rewards
        
        try:
            # Call the original method
            return original_sample_episodes(self, num_items=num_items, **kwargs)
        except Exception as e:
            debug_logger.error(f"PATCH: Exception in _sample_episodes: {e}", exc_info=True)
            raise
        finally:
            # Restore the original method
            SingleAgentEpisode.get_rewards = original_get_rewards
    
    # Apply the monkey patch
    EpisodeReplayBuffer._sample_episodes = patched_sample_episodes
    debug_logger.info("Applied monkey patch to EpisodeReplayBuffer._sample_episodes")
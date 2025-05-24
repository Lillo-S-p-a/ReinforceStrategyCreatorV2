"""
Reinforcement Learning Agent Module (PyTorch Version)

This module provides a reinforcement learning agent for strategy creation.
:ComponentRole RLAgent
:Context RL Core (Req 3.1)
"""

import logging
import numpy as np
import random
from typing import List, Tuple, Any, Union, Dict
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # For loss function if needed, or use nn.MSELoss

# Configure logger
logger = logging.getLogger(__name__)

class StrategyAgent:
    """
    Reinforcement Learning agent for trading strategy creation using DQN (PyTorch).
    :Algorithm DQN

    This class implements a Deep Q-Network (DQN) agent that learns to create
    trading strategies based on market data and technical indicators.

    Attributes:
        state_size (int): Dimension of the state space.
        action_size (int): Dimension of the action space.
        device (torch.device): Device to run the model on (e.g., 'cuda', 'mps', 'cpu').
        model (nn.Module): The Q-network model.
        target_model (nn.Module): The target Q-network model for stable learning.
        optimizer (optim.Optimizer): Optimizer for training the model.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Current exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Factor to decrease epsilon after each step.
        memory (deque): Replay buffer for storing experiences. :Algorithm ExperienceReplay
        memory_size (int): Maximum size of the replay buffer.
        batch_size (int): Size of the mini-batch sampled from memory for training.
        gamma (float): Discount factor for future rewards.
        target_update_freq (int): Frequency (in learning steps) to update the target network.
        update_counter (int): Counter for tracking steps until the next target network update.
    """

    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 2000, batch_size: int = 32,
                 gamma: float = 0.95,
                 target_update_freq: int = 100,
                 # Enhanced DQN features
                 use_dueling: bool = False,
                 use_double_q: bool = False,
                 use_prioritized_replay: bool = False,
                 prioritized_replay_alpha: float = 0.6,
                 prioritized_replay_beta: float = 0.4,
                 prioritized_replay_beta_annealing: bool = False,
                 prioritized_replay_beta_annealing_steps: int = 10000):
        """
        Initialize the DQN agent with Experience Replay and Target Network (PyTorch).

        Builds the main Q-network and the target Q-network using PyTorch,
        sets up epsilon-greedy parameters, initializes the experience replay memory,
        and configures the target network update frequency.
        :Algorithm EpsilonGreedyExploration
        :Algorithm ExperienceReplay
        :Algorithm TargetNetwork
        :Algorithm DuelingDQN (when use_dueling=True)
        :Algorithm DoubleDQN (when use_double_q=True)
        :Algorithm PrioritizedExperienceReplay (when use_prioritized_replay=True)

        Args:
            state_size (int): Dimension of the state space (input features).
            action_size (int): Dimension of the action space (possible actions).
            learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.001.
            epsilon (float): Initial exploration rate. Defaults to 1.0.
            epsilon_min (float): Minimum exploration rate. Defaults to 0.01.
            epsilon_decay (float): Decay rate for exploration probability. Defaults to 0.995.
            memory_size (int): Maximum size of the experience replay buffer. Defaults to 2000.
            batch_size (int): Size of the mini-batch sampled for learning. Defaults to 32.
            gamma (float): Discount factor for future rewards. Defaults to 0.95.
            target_update_freq (int): How often (in learning steps) to update the target network. Defaults to 100.
            use_dueling (bool): Whether to use Dueling DQN architecture. Defaults to False.
            use_double_q (bool): Whether to use Double Q-learning. Defaults to False.
            use_prioritized_replay (bool): Whether to use Prioritized Experience Replay. Defaults to False.
            prioritized_replay_alpha (float): Alpha parameter for PER, controls how much prioritization is used. Defaults to 0.6.
            prioritized_replay_beta (float): Beta parameter for PER, controls importance sampling. Defaults to 0.4.
            prioritized_replay_beta_annealing (bool): Whether to anneal beta to 1.0 over time. Defaults to False.
            prioritized_replay_beta_annealing_steps (int): Number of steps over which to anneal beta. Defaults to 10000.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Enhanced DQN features
        self.use_dueling = use_dueling
        self.use_double_q = use_double_q
        self.use_prioritized_replay = use_prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.initial_prioritized_replay_beta = prioritized_replay_beta  # Store initial beta for annealing
        self.prioritized_replay_beta_annealing = prioritized_replay_beta_annealing
        self.prioritized_replay_beta_annealing_steps = prioritized_replay_beta_annealing_steps
        self.prioritized_replay_epsilon = 1e-6  # Small epsilon to add to priorities
        self.beta_step = 0  # Counter for beta annealing
        
        # PER metrics tracking
        self.per_metrics = {'td_error': 0.0, 'mean_priority': 0.0}

        # --- Device Configuration ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("PyTorch CUDA device found. Using GPU.")
        elif torch.backends.mps.is_available(): # For Apple Silicon
            self.device = torch.device("mps")
            logger.info("PyTorch MPS device found. Using Apple Silicon GPU.")
        else:
            self.device = torch.device("cpu")
            logger.info("No GPU or MPS found. PyTorch will use CPU.")
        # --- End Device Configuration ---

        # Initialize replay buffer based on whether PER is used
        if self.use_prioritized_replay:
            # For PER, we store (experience, priority)
            self.memory = []  # List for more flexibility with priorities
            self.priorities = np.zeros(self.memory_size)  # Store priorities separately
            self.memory_indices = []  # Track indices for update
            logger.info(f"Using Prioritized Experience Replay with alpha={self.prioritized_replay_alpha}, beta={self.prioritized_replay_beta}")
        else:
            # Standard uniform replay buffer
            self.memory = deque(maxlen=self.memory_size)

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize target model weights by copying from main model
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Target model weights initialized from main model.")

        logger.info(f"StrategyAgent (PyTorch DQN with Target Network) initialized: state_size={state_size}, "
                    f"action_size={action_size}, learning_rate={learning_rate}, device={self.device}. "
                    "Models built. Target model initialized. Replay memory initialized.")

    def _build_model(self) -> nn.Module:
        """
        Builds the PyTorch model for the Q-network.
        
        If use_dueling is True, uses a Dueling network architecture.
        Otherwise, uses a standard DQN architecture.
        """
        if self.use_dueling:
            # Dueling network architecture
            class DuelingDQN(nn.Module):
                def __init__(self, state_size, action_size):
                    super(DuelingDQN, self).__init__()
                    self.feature_layer = nn.Sequential(
                        nn.Linear(state_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU()
                    )
                    
                    # State value stream
                    self.value_stream = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)  # Single state value
                    )
                    
                    # Action advantage stream
                    self.advantage_stream = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, action_size)  # Advantage for each action
                    )
                
                def forward(self, state):
                    features = self.feature_layer(state)
                    value = self.value_stream(features)
                    advantage = self.advantage_stream(features)
                    
                    # Combine value and advantage to get Q-values
                    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
                    q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                    return q_values
            
            model = DuelingDQN(self.state_size, self.action_size)
            logger.debug("PyTorch Dueling DQN model built.")
        else:
            # Standard DQN architecture
            model = nn.Sequential(
                nn.Linear(self.state_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)  # Output layer, linear activation for Q-values
            )
            logger.debug("PyTorch Standard DQN model built.")
            
        return model
    def remember(self, state: Union[List[float], np.ndarray, tuple],
                 action: int,
                 reward: float,
                 next_state: Union[List[float], np.ndarray, tuple],
                 done: bool) -> None:
        """
        Store an experience tuple in the replay memory.
        
        If using Prioritized Experience Replay, adds with max priority.
        Ensures states have consistent shapes before storing.
        Handles various input types including tuples from Gymnasium API.
        """
        # Handle tuple case (from Gymnasium API)
        if isinstance(state, tuple):
            # Assume first element is the observation
            state = state[0]
        if isinstance(next_state, tuple):
            # Assume first element is the observation
            next_state = next_state[0]
            
        # Ensure states are numpy arrays for consistency before potential tensor conversion
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        if isinstance(next_state, list):
            next_state = np.array(next_state, dtype=np.float32)
            
        # Ensure states have the expected shape (state_size,)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        if state.shape != (self.state_size,):
            logger.warning(f"State shape mismatch: expected ({self.state_size},), got {state.shape}. Reshaping...")
            # Try to reshape if total size matches
            if state.size == self.state_size:
                state = state.reshape(self.state_size)
            else:
                # If sizes don't match, pad or truncate
                new_state = np.zeros(self.state_size, dtype=np.float32)
                copy_size = min(state.size, self.state_size)
                new_state[:copy_size] = state.flatten()[:copy_size]
                state = new_state
                
        if next_state.shape != (self.state_size,):
            # Apply the same reshaping to next_state
            if next_state.size == self.state_size:
                next_state = next_state.reshape(self.state_size)
            else:
                new_next_state = np.zeros(self.state_size, dtype=np.float32)
                copy_size = min(next_state.size, self.state_size)
                new_next_state[:copy_size] = next_state.flatten()[:copy_size]
                next_state = new_next_state
        
        experience = (state, action, reward, next_state, done)
        
        if self.use_prioritized_replay:
            if len(self.memory) < self.memory_size:
                # Add to memory with max priority
                self.memory.append(experience)
                max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0
                self.priorities = np.append(self.priorities, max_priority)
            else:
                # Replace oldest experience
                idx = len(self.memory) % self.memory_size
                self.memory[idx] = experience
                max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0
                self.priorities[idx] = max_priority
        else:
            # Standard uniform replay buffer
            self.memory.append(experience)

    def select_action(self, state: Union[List[float], np.ndarray], return_confidence: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Select an action based on the current state using an epsilon-greedy policy.
        
        Args:
            state: The current state observation
            return_confidence: If True, returns a tuple of (action, confidence)
                               where confidence is a value between 0-1
        
        Returns:
            Either the selected action (int) or a tuple of (action, confidence)
        """
        action = None
        confidence = 0.0
        q_values_np = None
        
        if np.random.rand() <= self.epsilon:
            # Exploration - random action with low confidence
            action = np.random.randint(self.action_size)
            confidence = 0.1  # Low confidence for random actions
            # logger.debug(f"Exploration: Selected random action {action}")
        else:
            # Exploitation - use model
            if isinstance(state, list) or isinstance(state, np.ndarray):
                # Ensure state is a flat numpy array first if it's a list
                if isinstance(state, list):
                    state = np.array(state, dtype=np.float32)
                # Convert to PyTorch tensor, add batch dimension, and send to device
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else: # Assuming state is already a tensor (e.g. from RLlib)
                state_tensor = state.float().unsqueeze(0).to(self.device)

            self.model.eval() # Set model to evaluation mode for inference
            with torch.no_grad(): # Disable gradient calculations for inference
                q_values = self.model(state_tensor)
            self.model.train() # Set model back to training mode

            # Convert to numpy for easier processing
            q_values_np = q_values[0].cpu().numpy()
            
            # Get action with max Q-value
            action = int(np.argmax(q_values_np))
            
            # Calculate confidence based on Q-values
            # Softmax normalization for Q-values to get probabilities
            exp_q_values = np.exp(q_values_np - np.max(q_values_np))  # Subtract max for numerical stability
            softmax_probs = exp_q_values / np.sum(exp_q_values)
            
            # Confidence is the softmax probability of the selected action
            confidence = float(softmax_probs[action])
            
            # Additional scaling - avoid overconfidence by keeping max confidence to 0.9
            confidence = min(0.9, confidence)
            
            # logger.debug(f"Exploitation: Q-values={q_values_np}, Selected action {action}, Confidence={confidence:.4f}")
        
        # Epsilon decay is typically handled by RLlib's exploration config when integrated
        # If running standalone, uncomment:
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     logger.debug(f"Epsilon decayed to {self.epsilon}")
            
        if return_confidence:
            return int(action), confidence
        else:
            return int(action)

    def learn(self, return_stats: bool = False) -> Union[None, dict]:
        """
        Samples a batch from memory, calculates target Q-values, and trains the Q-network.
        
        If using Double DQN:
          Uses main network to select actions and target network to evaluate them.
          
        If using PER:
          Samples based on priorities and uses importance sampling weights.
          Updates priorities based on TD errors.
          
        Args:
            return_stats: If True, returns a dictionary of training statistics
            
        Returns:
            Dictionary of training statistics if return_stats=True, otherwise None
        """
        # Check if we have enough experiences
        if self.use_prioritized_replay:
            memory_size = len(self.memory)
            if memory_size < self.batch_size:
                if return_stats:
                    return {'td_error': 0.0, 'mean_priority': 0.0}
                return
        else:
            if len(self.memory) < self.batch_size:
                if return_stats:
                    return {'td_error': 0.0, 'mean_priority': 0.0}
                return

        # Beta annealing for Prioritized Experience Replay
        if self.use_prioritized_replay and self.prioritized_replay_beta_annealing:
            self.beta_step += 1
            progress = min(1.0, self.beta_step / self.prioritized_replay_beta_annealing_steps)
            self.prioritized_replay_beta = self.initial_prioritized_replay_beta + progress * (1.0 - self.initial_prioritized_replay_beta)

        # Sample batch based on whether we're using PER
        if self.use_prioritized_replay:
            # Convert priorities to probabilities via softmax
            probs = self.priorities[:len(self.memory)] ** self.prioritized_replay_alpha
            
            # Add safety check to prevent division by zero/NaN
            sum_probs = np.sum(probs)
            if sum_probs <= 0 or np.isnan(sum_probs):
                # If sum is zero or NaN, fall back to uniform distribution
                probs = np.ones(len(self.memory)) / len(self.memory)
            else:
                probs = probs / sum_probs
            
            # Sample based on priorities
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs, replace=False)
            self.memory_indices = indices  # Store indices for priority update
            
            # Calculate importance sampling weights
            weights = (len(self.memory) * probs[indices]) ** (-self.prioritized_replay_beta)
            weights = weights / np.max(weights)  # Normalize
            weights_tensor = torch.from_numpy(weights).float().to(self.device)
            
            # Get experiences from sampled indices
            minibatch = [self.memory[idx] for idx in indices]
        else:
            # Standard uniform sampling
            minibatch = random.sample(self.memory, self.batch_size)
            weights_tensor = torch.ones(self.batch_size).to(self.device)  # No weighting
        
        # Safely create arrays from experiences, ensuring consistent shapes
        try:
            states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
            next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
            dones = np.array([experience[4] for experience in minibatch], dtype=np.uint8)
        except ValueError as e:
            # If we still encounter shape issues, log and handle them
            logger.error(f"Shape error in minibatch: {e}")
            
            # Verify and fix shapes if needed
            fixed_states = []
            fixed_next_states = []
            
            for exp in minibatch:
                state = exp[0]
                next_state = exp[3]
                
                # Ensure state has correct shape
                if state.shape != (self.state_size,):
                    new_state = np.zeros(self.state_size, dtype=np.float32)
                    copy_size = min(state.size, self.state_size)
                    new_state[:copy_size] = state.flatten()[:copy_size]
                    fixed_states.append(new_state)
                else:
                    fixed_states.append(state)
                
                # Ensure next_state has correct shape
                if next_state.shape != (self.state_size,):
                    new_next_state = np.zeros(self.state_size, dtype=np.float32)
                    copy_size = min(next_state.size, self.state_size)
                    new_next_state[:copy_size] = next_state.flatten()[:copy_size]
                    fixed_next_states.append(new_next_state)
                else:
                    fixed_next_states.append(next_state)
            
            # Create arrays with fixed shapes
            states = np.array(fixed_states, dtype=np.float32)
            next_states = np.array(fixed_next_states, dtype=np.float32)

        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device) # Actions are indices
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device) # Float for (1 - dones)

        # Implement Double DQN if enabled
        if self.use_double_q:
            # Use main network to SELECT action
            next_q_values_main = self.model(next_states_tensor)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            # Use target network to EVALUATE action
            next_q_values_target = self.target_model(next_states_tensor).detach()
            max_next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            next_q_values_target = self.target_model(next_states_tensor).detach()
            max_next_q_values = torch.max(next_q_values_target, dim=1)[0]
        
        # Compute target Q-values: R + gamma * max_a' Q_target(s', a')
        target_q_val = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Get current Q-values from main model
        current_q_values = self.model(states_tensor)
        # Gather Q-values for the actions taken
        action_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute TD errors for PER
        td_errors = (target_q_val - action_q_values).detach().cpu().numpy()
        
        # Apply weights if using PER
        if self.use_prioritized_replay:
            # Weighted MSE loss
            losses = weights_tensor * (target_q_val - action_q_values) ** 2
            loss = losses.mean()
            
            # Update priorities with new TD errors
            new_priorities = np.abs(td_errors) + self.prioritized_replay_epsilon
            for idx, priority in zip(self.memory_indices, new_priorities):
                self.priorities[idx] = priority
        else:
            # Standard MSE loss
            loss = F.mse_loss(action_q_values, target_q_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._update_target_if_needed()
        
        # Update PER metrics
        mean_td_error = np.mean(np.abs(td_errors))
        mean_priority = np.mean(self.priorities[:len(self.memory)]) if self.use_prioritized_replay else 0.0
        self.per_metrics = {
            'td_error': mean_td_error,
            'mean_priority': mean_priority
        }
        
        if return_stats:
            return self.per_metrics

    def _update_target_if_needed(self) -> None:
        """Checks the counter and updates the target network weights if the frequency is met."""
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
            # logger.info(f"Target network updated at step {self.update_counter}")

    def update_target_model(self) -> None:
        """Copies weights from the main model to the target model."""
        # logger.debug("Updating target model weights from main model.")
        self.target_model.load_state_dict(self.model.state_dict())

    # Placeholder for saving/loading, RLlib will handle this when integrated
    def save_model(self, path: str):
        """Saves the model state_dict."""
        # torch.save(self.model.state_dict(), path)
        # logger.info(f"PyTorch model saved to {path}")
        pass # RLlib handles saving

    def load_model(self, path: str):
        """Loads the model state_dict."""
        # self.model.load_state_dict(torch.load(path, map_location=self.device))
        # self.update_target_model() # Ensure target model is also updated
        # logger.info(f"PyTorch model loaded from {path}")
        pass # RLlib handles loading
        
    def get_per_metrics(self) -> Dict[str, float]:
        """
        Returns metrics related to Prioritized Experience Replay.
        
        Returns:
            Dict containing PER metrics like TD error and mean priority
        """
        return self.per_metrics
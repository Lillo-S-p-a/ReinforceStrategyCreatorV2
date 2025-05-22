"""
Reinforcement Learning Agent Module (PyTorch Version)

This module provides a reinforcement learning agent for strategy creation.
:ComponentRole RLAgent
:Context RL Core (Req 3.1)
"""

import logging
import numpy as np
import random
from typing import List, Tuple, Any, Union
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
                 target_update_freq: int = 100):
        """
        Initialize the DQN agent with Experience Replay and Target Network (PyTorch).

        Builds the main Q-network and the target Q-network using PyTorch,
        sets up epsilon-greedy parameters, initializes the experience replay memory,
        and configures the target network update frequency.
        :Algorithm EpsilonGreedyExploration
        :Algorithm ExperienceReplay
        :Algorithm TargetNetwork

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

        self.memory = deque(maxlen=self.memory_size)

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model() # Initialize target model weights

        logger.info(f"StrategyAgent (PyTorch DQN with Target Network) initialized: state_size={state_size}, "
                    f"action_size={action_size}, learning_rate={learning_rate}, device={self.device}. "
                    "Models built. Target model initialized. Replay memory initialized.")

    def _build_model(self) -> nn.Module:
        """Builds the PyTorch model for the Q-network."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)  # Output layer, linear activation for Q-values
        )
        logger.debug("PyTorch Q-Network model built.")
        return model

    def remember(self, state: Union[List[float], np.ndarray],
                 action: int,
                 reward: float,
                 next_state: Union[List[float], np.ndarray],
                 done: bool) -> None:
        """Store an experience tuple in the replay memory."""
        # Ensure states are numpy arrays for consistency before potential tensor conversion
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        if isinstance(next_state, list):
            next_state = np.array(next_state, dtype=np.float32)
        
        # No need to reshape here if RLlib handles batching correctly later
        self.memory.append((state, action, reward, next_state, done))
        # logger.debug(f"Remembered experience. Memory size: {len(self.memory)}")


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

    def learn(self) -> None:
        """Samples a batch from memory, calculates target Q-values, and trains the Q-network."""
        if len(self.memory) < self.batch_size:
            # logger.debug(f"Learn called but not enough samples in memory ({len(self.memory)}/{self.batch_size}).")
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
        dones = np.array([experience[4] for experience in minibatch], dtype=np.uint8)

        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device) # Actions are indices
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device) # Float for (1 - dones)

        # Get Q-values for next states from target model
        next_q_values_target = self.target_model(next_states_tensor).detach()
        # Select max Q-value for next states (Double DQN would use main model here for action selection)
        max_next_q_values = torch.max(next_q_values_target, dim=1)[0]
        
        # Compute target Q-values: R + gamma * max_a' Q_target(s', a')
        target_q_val = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Get current Q-values from main model
        current_q_values = self.model(states_tensor)
        # Gather Q-values for the actions taken
        action_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute loss (e.g., MSELoss or SmoothL1Loss)
        loss = F.mse_loss(action_q_values, target_q_val)
        # loss = nn.MSELoss()(action_q_values, target_q_val)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # logger.debug(f"Training batch complete. Loss: {loss.item()}")

        self._update_target_if_needed()

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
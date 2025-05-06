"""
Reinforcement Learning Agent Module

This module provides a reinforcement learning agent for strategy creation.
:ComponentRole RLAgent
:Context RL Core (Req 3.1)
"""

import logging
import numpy as np
import random
from typing import List, Tuple, Any, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input # Added Input
from tensorflow.keras.optimizers import Adam

from collections import deque # Added import

# Configure logger
logger = logging.getLogger(__name__)

class StrategyAgent:
    """
    Reinforcement Learning agent for trading strategy creation using DQN.
    :Algorithm DQN

    This class implements a Deep Q-Network (DQN) agent that learns to create
    trading strategies based on market data and technical indicators.

    Attributes:
        state_size (int): Dimension of the state space.
        action_size (int): Dimension of the action space.
        model (keras.Model): The Q-network model.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Current exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Factor to decrease epsilon after each step.
        memory (deque): Replay buffer for storing experiences. :Algorithm ExperienceReplay
        memory_size (int): Maximum size of the replay buffer.
        batch_size (int): Size of the mini-batch sampled from memory for training.
        gamma (float): Discount factor for future rewards.
        target_model (keras.Model): The target Q-network model for stable learning.
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
                 target_update_freq: int = 100): # <-- Add target_update_freq
        """
        Initialize the DQN agent with Experience Replay and Target Network.

        Builds and compiles the main Q-network and the target Q-network,
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
        self.update_counter = 0 # Counter for target network updates

        # --- Device Check & Configuration (MPS for Apple Silicon) ---
        try:
            physical_gpus = tf.config.list_physical_devices('GPU')
            if physical_gpus:
                # Standard GPU (CUDA) found - less likely on Mac but check first
                logger.info(f"TensorFlow GPU device(s) found: {physical_gpus}")
                # Configuration for standard GPUs usually happens automatically or via CUDA env vars
            elif tf.config.backends.mps.is_available():
                logger.info("TensorFlow MPS (Metal Performance Shaders) device found.")
                # Explicitly enable memory growth for MPS if needed, although often default
                # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('MPS')[0], True) # MPS doesn't have explicit memory growth setting like CUDA
                logger.info("TensorFlow should automatically utilize MPS device on compatible versions.")
                # We don't typically need tf.device('/MPS:0') context manager for the whole class,
                # Keras/TF should handle placement if MPS is available and tensorflow-metal is installed.
                # Logging confirms awareness. Performance will be the real test.
            else:
                logger.warning("No GPU or MPS device found. TensorFlow will use CPU.")
        except Exception as e:
            logger.error(f"Error during device check/configuration: {e}")
        # --- End Device Check ---

        # Initialize replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Build the main Q-Network model
        self.model = self._build_model()

        # Build the target Q-Network model
        self.target_model = self._build_model()
        # Initialize target model weights to match the main model
        self.update_target_model()

        logger.info(f"StrategyAgent (DQN with Target Network) initialized: state_size={state_size}, "
                    f"action_size={action_size}, learning_rate={learning_rate}, "
                    f"epsilon={epsilon}, epsilon_min={epsilon_min}, epsilon_decay={epsilon_decay}, "
                    f"memory_size={memory_size}, batch_size={batch_size}, gamma={gamma}, "
                    f"target_update_freq={target_update_freq}. "
                    "Models built and compiled. Target model initialized. Replay memory initialized.")

    def _build_model(self) -> keras.Model:
        """Builds the Keras model for the Q-network."""
        model = Sequential([
            Input(shape=(self.state_size,)), # Explicit Input layer
            Dense(64, activation='relu'),     # No input_dim needed here
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        logger.debug("Q-Network model built and compiled.")
        return model

    def remember(self, state: Union[List[float], np.ndarray],
                 action: int,
                 reward: float,
                 next_state: Union[List[float], np.ndarray],
                 done: bool) -> None:
        """
        Store an experience tuple in the replay memory.
        :Algorithm ExperienceReplay

        Args:
            state (Union[List[float], np.ndarray]): The starting state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Union[List[float], np.ndarray]): The next state observed.
            done (bool): Whether the episode has ended.
        """
        # Ensure states are numpy arrays for consistency in memory
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(next_state, list):
            next_state = np.array(next_state)

        # Reshape states to ensure they have the expected shape (1, state_size)
        # This helps maintain consistency when sampling batches later
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        self.memory.append((state, action, reward, next_state, done))
        logger.debug(f"Remembered experience: state shape={state.shape}, action={action}, reward={reward}, "
                     f"next_state shape={next_state.shape}, done={done}. Memory size: {len(self.memory)}")


    def select_action(self, state: Union[List[float], np.ndarray]) -> int:
        """
        Select an action based on the current state using an epsilon-greedy policy.
        
        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest predicted Q-value is chosen (exploitation).
        
        Args:
            state (Union[List[float], np.ndarray]): The current state observation.
            
        Returns:
            int: The selected action index.
        """
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
            logger.debug(f"Exploration: Selected random action {action}")
        else:
            # Ensure state is a numpy array and reshape for the model
            if isinstance(state, list):
                state = np.array(state)
            if state.ndim == 1: # Reshape if it's a flat array
                 state = np.reshape(state, [1, self.state_size])
            elif state.shape[0] != 1: # Ensure batch dimension is 1 if already 2D+
                 # This case might indicate an issue elsewhere, but we handle reshaping defensively
                 logger.warning(f"Unexpected state shape {state.shape} received in select_action. Reshaping first element.")
                 state = np.reshape(state[0], [1, self.state_size]) # Attempt to use the first element

            # Predict Q-values for the state
            q_values = self.model.predict(state, verbose=0) # verbose=0 suppresses Keras prediction logs
            action = np.argmax(q_values[0])
            logger.debug(f"Exploitation: Q-values={q_values[0]}, Selected action {action}")

        # Decay epsilon (optional placement, could be in learn method)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logger.debug(f"Epsilon decayed to {self.epsilon}")
            
        return int(action) # Ensure standard Python int is returned

    def learn(self) -> None:
        """
        Samples a batch from memory, calculates target Q-values, and trains the Q-network.

        This method implements the core DQN learning step:
        1. Samples a minibatch of experiences from the replay memory.
        2. Predicts Q-values for the next states using the main Q-network.
        3. Calculates the target Q-values using the Bellman equation:
           target = reward + gamma * max(Q(next_state, a')) * (1 - done)
        4. Predicts the current Q-values for the sampled states.
        5. Creates a target Q-value array where only the Q-value for the action
           actually taken is updated with the calculated Bellman target.
        6. Trains the main Q-network model using the states and the constructed
           target Q-value array.

        :Algorithm DQN
        :Algorithm ExperienceReplay
        :Context ModelTraining
        :Context BatchProcessing
        """
        # Check if enough samples are available in memory to form a batch
        if len(self.memory) < self.batch_size:
            logger.debug(f"Learn called but not enough samples in memory ({len(self.memory)}/{self.batch_size}). Skipping learning step.")
            return # Not enough samples to train

        # Sample a random minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        logger.debug(f"Sampled minibatch of size {self.batch_size}")

        # Extract components from the minibatch
        # States and next_states are already shaped (1, state_size) in memory
        # We stack them vertically to get (batch_size, state_size)
        states = np.vstack([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.vstack([experience[3] for experience in minibatch])
        # Convert dones to uint8 for (1 - dones) calculation
        dones = np.array([experience[4] for experience in minibatch]).astype(np.uint8)

        logger.debug(f"Prepared batch data: states shape {states.shape}, actions shape {actions.shape}, "
                     f"rewards shape {rewards.shape}, next_states shape {next_states.shape}, dones shape {dones.shape}")

        # --- Q-Value Target Calculation and Training ---
        # 1. Predict Q-values for the next states using the *target* model for stability
        next_q_values = self.target_model.predict(next_states, verbose=0) # Use target_model here
        logger.debug(f"Predicted next_q_values (using target model) shape: {next_q_values.shape}")

        # 2. Calculate the target Q-value using the Bellman equation
        # Target is reward if done, otherwise reward + discounted max future Q
        target = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones)
        logger.debug(f"Calculated target shape: {target.shape}")

        # 3. Predict current Q-values for the states in the batch
        current_q_values = self.model.predict(states, verbose=0)
        logger.debug(f"Predicted current_q_values shape: {current_q_values.shape}")

        # 4. Create the target Q-value array for training
        # Start with current Q-values, then update only the Q-value for the action taken
        target_q_values = current_q_values.copy()
        batch_indices = np.arange(self.batch_size)
        target_q_values[batch_indices, actions] = target
        logger.debug(f"Constructed target_q_values shape: {target_q_values.shape}")

        # 5. Train the main model using the states and the calculated target Q-values
        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        logger.debug(f"Training complete for one batch. Loss: {loss}")

        # Increment counter and update target network if needed
        self._update_target_if_needed()

        # Optional: Update epsilon decay here instead of in select_action if preferred
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def _update_target_if_needed(self) -> None:
        """Checks the counter and updates the target network weights if the frequency is met."""
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
            logger.info(f"Target network updated at step {self.update_counter}")
            # Optional: Reset counter if you prefer counting from 0 each time
            # self.update_counter = 0

    def update_target_model(self) -> None:
        """Copies weights from the main model to the target model."""
        logger.debug("Updating target model weights from main model.")
        self.target_model.set_weights(self.model.get_weights())
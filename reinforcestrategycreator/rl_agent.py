"""
Reinforcement Learning Agent Module

This module provides a reinforcement learning agent for strategy creation.
:ComponentRole RLAgent
:Context RL Core (Req 3.1)
"""

import logging
import numpy as np
from typing import List, Tuple, Any, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
    """

    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize the DQN agent.

        Builds and compiles the Q-network model and sets up epsilon-greedy parameters.
        :Algorithm EpsilonGreedyExploration

        Args:
            state_size (int): Dimension of the state space (input features).
            action_size (int): Dimension of the action space (possible actions).
            learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.001.
            epsilon (float): Initial exploration rate. Defaults to 1.0.
            epsilon_min (float): Minimum exploration rate. Defaults to 0.01.
            epsilon_decay (float): Decay rate for exploration probability. Defaults to 0.995.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Build the Q-Network model
        self.model = self._build_model()

        logger.info(f"StrategyAgent (DQN) initialized: state_size={state_size}, "
                    f"action_size={action_size}, learning_rate={learning_rate}, "
                    f"epsilon={epsilon}, epsilon_min={epsilon_min}, epsilon_decay={epsilon_decay}. "
                    "Model built and compiled.")
    
    def _build_model(self) -> keras.Model:
        """Builds the Keras model for the Q-network."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        logger.debug("Q-Network model built and compiled.")
        return model

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
    
    def learn(self, state: Union[List[float], np.ndarray], 
              action: int, 
              reward: float, 
              next_state: Union[List[float], np.ndarray], 
              done: bool) -> None:
        """
        Update the agent's knowledge based on the observed transition.
        
        Args:
            state (Union[List[float], np.ndarray]): The starting state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Union[List[float], np.ndarray]): The next state observed.
            done (bool): Whether the episode has ended.
        """
        # Placeholder for future implementation
        # Will implement actual learning algorithm in future tasks
        logger.debug(f"Learning from transition: state shape={np.shape(state)}, "
                    f"action={action}, reward={reward}, "
                    f"next_state shape={np.shape(next_state)}, done={done}")
        pass
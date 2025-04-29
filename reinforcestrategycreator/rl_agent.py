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
    """

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the DQN agent.

        Builds and compiles the Q-network model.

        Args:
            state_size (int): Dimension of the state space (input features).
            action_size (int): Dimension of the action space (possible actions).
            learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.001.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Build the Q-Network model
        self.model = self._build_model()

        logger.info(f"StrategyAgent (DQN) initialized: state_size={state_size}, "
                    f"action_size={action_size}, learning_rate={learning_rate}. Model built and compiled.")
    
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
        Select an action based on the current state.
        
        Args:
            state (Union[List[float], np.ndarray]): The current state observation.
            
        Returns:
            int: The selected action index.
        """
        # Placeholder implementation - random action selection
        # Will be replaced with actual policy in future implementation
        action = np.random.randint(0, self.action_size)
        
        logger.debug(f"Selected action: {action}")
        return action
    
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
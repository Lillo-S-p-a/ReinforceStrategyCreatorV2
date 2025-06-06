"""Deep Q-Network (DQN) implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import json
from collections import deque
import random

from ..base import ModelBase


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched experiences
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)


class DQN(ModelBase):
    """Deep Q-Network implementation.
    
    This is a simplified DQN implementation for demonstration purposes.
    In a real implementation, you would use a deep learning framework
    like PyTorch or TensorFlow.
    """
    
    model_type = "DQN"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DQN model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.hidden_layers = self.hyperparameters.get("hidden_layers", [256, 128, 64])
        self.activation = self.hyperparameters.get("activation", "relu")
        self.dropout_rate = self.hyperparameters.get("dropout_rate", 0.2)
        self.double_dqn = self.hyperparameters.get("double_dqn", True)
        self.dueling_dqn = self.hyperparameters.get("dueling_dqn", False)
        self.prioritized_replay = self.hyperparameters.get("prioritized_replay", True)
        
        # Memory settings
        self.memory_size = self.hyperparameters.get("memory_size", 10000)
        self.update_frequency = self.hyperparameters.get("update_frequency", 4)
        self.target_update_frequency = self.hyperparameters.get("target_update_frequency", 100)
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.steps = 0
        self.episodes = 0
        
        # Training history
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilon_values": []
        }
    
    def build(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """Build the Q-network architecture.
        
        Args:
            input_shape: Shape of state input
            output_shape: Shape of action output (number of actions)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_actions = output_shape[0] if len(output_shape) > 0 else output_shape
        
        # Initialize networks with proper structure
        self._initialize_networks(input_shape, output_shape)
        
        # Copy weights to target network
        self._update_target_network()
    
    def _initialize_networks(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """Initialize Q-network and target network structures.
        
        Args:
            input_shape: Shape of state input
            output_shape: Shape of action output
        """
        # Initialize Q-network
        self.q_network = {
            "weights": self._initialize_weights(input_shape, output_shape),
            "input_shape": input_shape,
            "output_shape": output_shape
        }
        
        # Initialize target network
        self.target_network = {
            "weights": self._initialize_weights(input_shape, output_shape),
            "input_shape": input_shape,
            "output_shape": output_shape
        }
    
    def _initialize_weights(self, input_shape: Tuple[int, ...], 
                          output_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Initialize network weights.
        
        Args:
            input_shape: Input shape
            output_shape: Output shape
            
        Returns:
            Dictionary of weight matrices
        """
        # Simplified weight initialization
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        
        weights = {}
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_layers):
            weights[f"W{i}"] = np.random.randn(prev_size, hidden_size) * 0.01
            weights[f"b{i}"] = np.zeros(hidden_size)
            prev_size = hidden_size
        
        # Output layer
        weights["W_out"] = np.random.randn(prev_size, output_size) * 0.01
        weights["b_out"] = np.zeros(output_size)
        
        return weights
    
    def _update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        if self.q_network and self.target_network:
            self.target_network["weights"] = {
                k: v.copy() for k, v in self.q_network["weights"].items()
            }
    
    def _forward(self, state: np.ndarray, network: Dict[str, Any]) -> np.ndarray:
        """Forward pass through network.
        
        Args:
            state: Input state
            network: Network to use (q_network or target_network)
            
        Returns:
            Q-values for all actions
        """
        # Ensure network has proper structure
        if not network or "weights" not in network:
            raise ValueError("Network structure is invalid or not initialized")
        
        weights = network["weights"]
        
        # Validate weights structure
        if not isinstance(weights, dict):
            raise ValueError("Network weights must be a dictionary")
        
        # Check if required weight keys exist
        for i in range(len(self.hidden_layers)):
            if f"W{i}" not in weights or f"b{i}" not in weights:
                raise KeyError(f"Missing weight key W{i} or b{i} in network weights. Available keys: {list(weights.keys())}")
        
        if "W_out" not in weights or "b_out" not in weights:
            raise KeyError(f"Missing output layer weights. Available keys: {list(weights.keys())}")
        
        # Simplified forward pass
        x = state.flatten()
        self.logger.debug(f"_forward: input state flattened sample: {x[:5]}")
        if np.isnan(x).any(): self.logger.warning(f"_forward: NaN in input state x: {x}")

        # Hidden layers
        for i in range(len(self.hidden_layers)):
            x_prev = x
            W = weights[f"W{i}"]
            b = weights[f"b{i}"]
            if np.isnan(W).any(): self.logger.warning(f"_forward: NaN in W{i}")
            if np.isnan(b).any(): self.logger.warning(f"_forward: NaN in b{i}")
            
            x = np.dot(x_prev, W) + b
            self.logger.debug(f"_forward: after layer {i} (pre-activation) x sample: {x[:5]}")
            if np.isnan(x).any():
                self.logger.warning(f"_forward: NaN in x after layer {i} (pre-activation). x_prev: {x_prev[:5]}, W: {W[:2,:2]}, b: {b[:5]}") # Log parts of W and b
                return x # Return early if NaN detected

            # ReLU activation
            if self.activation == "relu":
                x = np.maximum(0, x)
            elif self.activation == "tanh":
                x = np.tanh(x)
            self.logger.debug(f"_forward: after layer {i} (post-activation) x sample: {x[:5]}")
            if np.isnan(x).any():
                self.logger.warning(f"_forward: NaN in x after layer {i} (post-activation)")
                return x # Return early
        
        # Output layer
        W_out = weights["W_out"]
        b_out = weights["b_out"]
        if np.isnan(W_out).any(): self.logger.warning(f"_forward: NaN in W_out")
        if np.isnan(b_out).any(): self.logger.warning(f"_forward: NaN in b_out")

        q_values = np.dot(x, W_out) + b_out
        self.logger.debug(f"_forward: final q_values sample: {q_values[:5]}")
        if np.isnan(q_values).any():
            self.logger.error(f"_forward: NaN in final q_values. x: {x[:5]}, W_out: {W_out[:2,:2]}, b_out: {b_out[:5]}")
        
        return q_values
    
    def predict(self, data: Any, **kwargs) -> Any:
        """Predict Q-values for given states.
        
        Args:
            data: State or batch of states
            **kwargs: Additional arguments (e.g., use_target_network)
            
        Returns:
            Q-values or selected actions
        """
        if not self.q_network:
            raise ValueError("Model must be built before prediction")
        
        use_target = kwargs.get("use_target_network", False)
        network = self.target_network if use_target else self.q_network
        
        # Handle single state or batch
        if isinstance(data, np.ndarray):
            if len(data.shape) == len(self.input_shape):
                # Single state
                return self._forward(data, network)
            else:
                # Batch of states
                return np.array([self._forward(s, network) for s in data])
        else:
            raise ValueError("Data must be numpy array")
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.predict(state)
            return np.argmax(q_values)
    
    def train(self, train_data: Any, validation_data: Optional[Any] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the DQN model.
        
        Args:
            train_data: Training environment or data
            validation_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history and metrics
        """
        # Ensure model is built
        if self.q_network is None or self.target_network is None:
            raise ValueError("Model must be built before training. Call build() first.")
        
        # Check and reinitialize if weights are missing
        if "weights" not in self.q_network or not self.q_network["weights"]:
            print("WARNING: Q-network weights were not properly initialized. Reinitializing...")
            if hasattr(self, 'input_shape') and hasattr(self, 'output_shape'):
                self._initialize_networks(self.input_shape, self.output_shape)
            else:
                raise ValueError("Cannot reinitialize networks: input_shape and output_shape not set")
        
        # Extract training parameters
        episodes = kwargs.get("episodes", 100)
        batch_size = kwargs.get("batch_size", 32)
        gamma = kwargs.get("gamma", 0.99)
        epsilon_start = kwargs.get("epsilon_start", 1.0)
        epsilon_end = kwargs.get("epsilon_end", 0.01)
        epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        learning_rate = kwargs.get("learning_rate", 0.001)
        
        epsilon = epsilon_start

        # Reward and portfolio parameters
        initial_episode_cash = 100000.0
        transaction_cost_rate = 0.001
        invalid_action_penalty = -1.0
        hold_cash_reward = -0.005 # Small penalty for holding cash
        unrealized_pnl_reward_scaling_factor = 0.1
        close_price_index = 3 # Assuming OHLCV, Close is at index 3

        # Enhanced logging for train_data
        self.logger.info(f"DQN.train() received train_data of type: {type(train_data)}")
        if hasattr(train_data, 'shape'):
            self.logger.info(f"DQN.train() received train_data with shape: {train_data.shape}")
        else:
            self.logger.info(f"DQN.train() received train_data that has no shape attribute.")
        if hasattr(train_data, 'ndim'):
            self.logger.info(f"DQN.train() received train_data with ndim: {train_data.ndim}")
        else:
            self.logger.info(f"DQN.train() received train_data that has no ndim attribute.")

        if not isinstance(train_data, np.ndarray) or train_data.ndim != 2:
            self.logger.error(f"Validation failed for train_data: type is {type(train_data)}, ndim is {getattr(train_data, 'ndim', 'N/A')}, is_ndarray: {isinstance(train_data, np.ndarray)}")
            raise ValueError("train_data must be a 2D numpy array of features.")
        
        self.logger.info(f"DQN.train() train_data shape[1]: {train_data.shape[1]}, close_price_index: {close_price_index}")
        if train_data.shape[1] <= close_price_index:
            raise ValueError(f"close_price_index {close_price_index} is out of bounds for train_data with shape {train_data.shape}")

        # Training loop
        for episode in range(episodes):
            cash = initial_episode_cash
            current_position_units = 0.0
            entry_price = 0.0
            previous_unrealized_pnl = 0.0
            
            episode_reward = 0
            episode_length = 0
            losses = []
            
            # Iterate through the training data for the episode
            # Ensure we have enough data for a next_state
            for step_idx in range(len(train_data) - 1):
                state = train_data[step_idx]
                current_price = state[close_price_index]
                next_state = train_data[step_idx + 1]
                next_price = next_state[close_price_index] # Used for liquidation if done

                action = self.select_action(state, epsilon)
                reward = 0.0

                # Trading Logic (0: Hold, 1: Buy, 2: Sell)
                if action == 0: # Hold
                    if current_position_units > 0:
                        unrealized_pnl_at_step = (current_price - entry_price) * current_position_units
                        change_in_unrealized_pnl = unrealized_pnl_at_step - previous_unrealized_pnl
                        reward = change_in_unrealized_pnl * unrealized_pnl_reward_scaling_factor
                        previous_unrealized_pnl = unrealized_pnl_at_step
                    else: # Holding cash
                        reward = hold_cash_reward
                        previous_unrealized_pnl = 0.0
                elif action == 1: # Buy
                    # Check if affordable and not already in position
                    if cash >= current_price * (1 + transaction_cost_rate) and current_position_units == 0 and current_price > 0:
                        units_to_buy = cash / (current_price * (1 + transaction_cost_rate))
                        cost = units_to_buy * current_price * transaction_cost_rate
                        
                        cash -= (units_to_buy * current_price) + cost
                        current_position_units = units_to_buy
                        entry_price = current_price
                        reward = -cost # Negative reward for transaction cost
                        previous_unrealized_pnl = 0.0
                    else:
                        reward = invalid_action_penalty
                elif action == 2: # Sell
                    if current_position_units > 0:
                        realized_pnl = (current_price - entry_price) * current_position_units
                        transaction_value = current_position_units * current_price
                        cost = transaction_value * transaction_cost_rate
                        
                        reward = realized_pnl - cost
                        cash += transaction_value - cost
                        current_position_units = 0.0
                        entry_price = 0.0
                        previous_unrealized_pnl = 0.0
                    else:
                        reward = invalid_action_penalty
                
                done = (step_idx == len(train_data) - 2) # Episode ends at the second to last step

                # If done and holding a position, liquidate
                if done and current_position_units > 0:
                    # PnL from liquidation at next_price (which is the last available price)
                    liquidation_pnl = (next_price - entry_price) * current_position_units
                    liquidation_cost = (current_position_units * next_price) * transaction_cost_rate
                    # Add this to the reward of the last action
                    reward += liquidation_pnl - liquidation_cost
                    cash += (current_position_units * next_price) - liquidation_cost
                    current_position_units = 0 # Position closed
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_length += 1
                
                if len(self.replay_buffer) >= batch_size and self.steps % self.update_frequency == 0:
                    loss = self._train_step(batch_size, gamma, learning_rate)
                    losses.append(loss)
                
                if self.steps % self.target_update_frequency == 0:
                    self._update_target_network()
                
                self.steps += 1
                
                if done:
                    break # End of episode
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            self.episodes += 1
            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_lengths"].append(episode_length)
            if losses: # only extend if there are losses to add
                self.training_history["losses"].extend(losses)
            self.training_history["epsilon_values"].append(epsilon)
            
            if episode % 10 == 0:
                # Ensure there are rewards to average
                avg_reward = np.mean(self.training_history["episode_rewards"][-10:]) if self.training_history["episode_rewards"] else 0.0
                print(f"Episode {self.episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {self.steps}, Buffer: {len(self.replay_buffer)}")

        self.is_trained = True
        self.update_metadata({
            "training_episodes": self.episodes, # Use self.episodes
            "final_epsilon": epsilon,
            "total_steps": self.steps
        })
        
        return self.training_history

    def _train_step(self, batch_size: int, gamma: float,
                    learning_rate: float) -> float:
        """Perform one training step.
        
        Args:
            batch_size: Batch size for training
            gamma: Discount factor
            learning_rate: Learning rate
            
        Returns:
            Training loss
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        self.logger.debug(f"_train_step: Sampled states shape: {states.shape}")
        self.logger.debug(f"_train_step: Sampled actions: {actions[:5]}") # Log first 5
        self.logger.debug(f"_train_step: Sampled rewards: {rewards[:5]}")
        self.logger.debug(f"_train_step: Sampled next_states shape: {next_states.shape}")
        self.logger.debug(f"_train_step: Sampled dones: {dones[:5]}")

        # Compute target Q-values
        if self.double_dqn:
            # Double DQN: use online network to select actions, target network for values
            next_q_values = self.predict(next_states)
            self.logger.debug(f"_train_step (double_dqn): next_q_values (online net for action selection) sample: {next_q_values[0] if next_q_values.size > 0 else 'empty'}")
            if np.isnan(next_q_values).any(): self.logger.warning("_train_step (double_dqn): NaN in next_q_values (online)")

            next_actions = np.argmax(next_q_values, axis=1)
            self.logger.debug(f"_train_step (double_dqn): next_actions sample: {next_actions[:5]}")

            next_q_values_target = self.predict(next_states, use_target_network=True)
            self.logger.debug(f"_train_step (double_dqn): next_q_values_target (target net for value) sample: {next_q_values_target[0] if next_q_values_target.size > 0 else 'empty'}")
            if np.isnan(next_q_values_target).any(): self.logger.warning("_train_step (double_dqn): NaN in next_q_values_target (target)")
            
            next_q_selected = next_q_values_target[np.arange(batch_size), next_actions]
        else:
            # Standard DQN
            next_q_values = self.predict(next_states, use_target_network=True)
            self.logger.debug(f"_train_step (std_dqn): next_q_values (target net) sample: {next_q_values[0] if next_q_values.size > 0 else 'empty'}")
            if np.isnan(next_q_values).any(): self.logger.warning("_train_step (std_dqn): NaN in next_q_values (target)")
            next_q_selected = np.max(next_q_values, axis=1)
        
        self.logger.debug(f"_train_step: next_q_selected sample: {next_q_selected[:5]}")
        if np.isnan(next_q_selected).any(): self.logger.warning("_train_step: NaN in next_q_selected")

        targets = rewards + gamma * next_q_selected * (1 - dones)
        self.logger.debug(f"_train_step: targets sample: {targets[:5]}")
        if np.isnan(targets).any(): self.logger.warning("_train_step: NaN in targets")
        
        # Compute current Q-values
        current_q_values = self.predict(states)
        self.logger.debug(f"_train_step: current_q_values sample: {current_q_values[0] if current_q_values.size > 0 else 'empty'}")
        if np.isnan(current_q_values).any(): self.logger.warning("_train_step: NaN in current_q_values")

        current_q_selected = current_q_values[np.arange(batch_size), actions]
        self.logger.debug(f"_train_step: current_q_selected sample: {current_q_selected[:5]}")
        if np.isnan(current_q_selected).any(): self.logger.warning("_train_step: NaN in current_q_selected")
        
        # Compute loss (simplified - in reality you'd use backpropagation)
        loss = np.mean((targets - current_q_selected) ** 2)
        self.logger.debug(f"_train_step: Calculated loss: {loss}")
        if np.isnan(loss): self.logger.error("_train_step: LOSS IS NAN!")
        
        # Simulate weight update (in reality, you'd use gradient descent)
        # This is a very simplified simulation
        error = targets - current_q_selected
        self.logger.debug(f"_train_step: error sample: {error[:5]}")
        if np.isnan(error).any(): self.logger.warning("_train_step: NaN in error term")
        
        # Ensure q_network and weights exist before updating
        if self.q_network is None or "weights" not in self.q_network:
            raise ValueError("Q-network is not properly initialized")
        
        # Create a copy of keys to avoid dictionary modification during iteration
        weight_keys = list(self.q_network["weights"].keys())
        
        for i in range(batch_size):
            # Simulate gradient update effect
            adjustment = learning_rate * error[i] * 0.01
            for key in weight_keys:
                if key in self.q_network["weights"]:
                    # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
                    if isinstance(self.q_network["weights"][key], list):
                        self.q_network["weights"][key] = np.array(self.q_network["weights"][key])
                    
                    # Apply random gradient update
                    self.q_network["weights"][key] += np.random.randn(*self.q_network["weights"][key].shape) * adjustment
        
        return float(loss)
    
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_data: Test environment or data
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Simulate evaluation
        n_episodes = kwargs.get("n_episodes", 10)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Simulate test episode
            for step in range(100):
                state = np.random.randn(*self.input_shape)
                action = self.select_action(state, epsilon=0.0)  # No exploration
                reward = np.random.randn()
                done = step == 99 or np.random.random() < 0.01
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "min_episode_reward": float(np.min(episode_rewards)),
            "max_episode_reward": float(np.max(episode_rewards))
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current state of the model for serialization.
        
        Returns:
            Dictionary containing model state
        """
        # Create deep copies of networks to avoid modifying the original
        import copy
        
        state = {
            "q_network": copy.deepcopy(self.q_network) if self.q_network else None,
            "target_network": copy.deepcopy(self.target_network) if self.target_network else None,
            "steps": self.steps,
            "episodes": self.episodes,
            "training_history": self.training_history,
            "input_shape": self.input_shape if hasattr(self, "input_shape") else None,
            "output_shape": self.output_shape if hasattr(self, "output_shape") else None,
            "n_actions": self.n_actions if hasattr(self, "n_actions") else None
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if state["q_network"] and "weights" in state["q_network"]:
            weights_dict = {}
            for k, v in state["q_network"]["weights"].items():
                # Handle both numpy arrays and lists
                if isinstance(v, np.ndarray):
                    weights_dict[k] = v.tolist()
                elif isinstance(v, list):
                    weights_dict[k] = v
                else:
                    # For any other type, try to convert to list
                    weights_dict[k] = list(v)
            state["q_network"]["weights"] = weights_dict
        
        if state["target_network"] and "weights" in state["target_network"]:
            weights_dict = {}
            for k, v in state["target_network"]["weights"].items():
                # Handle both numpy arrays and lists
                if isinstance(v, np.ndarray):
                    weights_dict[k] = v.tolist()
                elif isinstance(v, list):
                    weights_dict[k] = v
                else:
                    # For any other type, try to convert to list
                    weights_dict[k] = list(v)
            state["target_network"]["weights"] = weights_dict
        
        return state
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        """Set the model state from a dictionary.
        
        Args:
            state: Dictionary containing model state
        """
        # Restore shapes first
        if state.get("input_shape"):
            self.input_shape = tuple(state["input_shape"])
        if state.get("output_shape"):
            self.output_shape = tuple(state["output_shape"])
        if state.get("n_actions") is not None:
            self.n_actions = state["n_actions"]
        
        # Restore networks
        self.q_network = state.get("q_network")
        self.target_network = state.get("target_network")
        
        # Ensure networks are properly initialized if they're missing structure
        if self.q_network is None or self.target_network is None:
            if hasattr(self, 'input_shape') and hasattr(self, 'output_shape'):
                self._initialize_networks(self.input_shape, self.output_shape)
        
        # Convert lists back to numpy arrays and ensure proper structure
        if self.q_network and "weights" in self.q_network:
            # Ensure weights is a dictionary
            if isinstance(self.q_network["weights"], dict):
                self.q_network["weights"] = {
                    k: np.array(v) if not isinstance(v, np.ndarray) else v
                    for k, v in self.q_network["weights"].items()
                }
            else:
                # If weights is not a dict, reinitialize
                if hasattr(self, 'input_shape') and hasattr(self, 'output_shape'):
                    self.q_network["weights"] = self._initialize_weights(self.input_shape, self.output_shape)
        
        if self.target_network and "weights" in self.target_network:
            # Ensure weights is a dictionary
            if isinstance(self.target_network["weights"], dict):
                self.target_network["weights"] = {
                    k: np.array(v) if not isinstance(v, np.ndarray) else v
                    for k, v in self.target_network["weights"].items()
                }
            else:
                # If weights is not a dict, reinitialize
                if hasattr(self, 'input_shape') and hasattr(self, 'output_shape'):
                    self.target_network["weights"] = self._initialize_weights(self.input_shape, self.output_shape)
        
        # Restore other attributes
        self.steps = state.get("steps", 0)
        self.episodes = state.get("episodes", 0)
        self.training_history = state.get("training_history", {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilon_values": []
        })
        if state.get("output_shape"):
            self.output_shape = tuple(state["output_shape"])
        if state.get("n_actions") is not None:
            self.n_actions = state["n_actions"]
        
        # Reinitialize replay buffer
        self.replay_buffer = ReplayBuffer(self.memory_size)
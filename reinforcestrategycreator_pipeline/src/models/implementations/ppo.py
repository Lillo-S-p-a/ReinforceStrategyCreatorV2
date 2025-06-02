"""Proximal Policy Optimization (PPO) implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from ..base import ModelBase


class PPO(ModelBase):
    """Proximal Policy Optimization implementation.
    
    This is a simplified PPO implementation for demonstration purposes.
    In a real implementation, you would use a deep learning framework
    like PyTorch or TensorFlow.
    """
    
    model_type = "PPO"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PPO model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.policy_layers = self.hyperparameters.get("policy_layers", [256, 128])
        self.value_layers = self.hyperparameters.get("value_layers", [256, 128])
        self.activation = self.hyperparameters.get("activation", "tanh")
        
        # PPO specific parameters
        self.clip_range = self.hyperparameters.get("clip_range", 0.2)
        self.value_coefficient = self.hyperparameters.get("value_coefficient", 0.5)
        self.entropy_coefficient = self.hyperparameters.get("entropy_coefficient", 0.01)
        self.max_grad_norm = self.hyperparameters.get("max_grad_norm", 0.5)
        
        # Training parameters
        self.n_steps = self.hyperparameters.get("n_steps", 2048)
        self.n_epochs = self.hyperparameters.get("n_epochs", 10)
        self.gae_lambda = self.hyperparameters.get("gae_lambda", 0.95)
        
        # Initialize components
        self.policy_network = None
        self.value_network = None
        self.optimizer = None
        self.steps = 0
        self.episodes = 0
        
        # Experience buffer
        self.experience_buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": []
        }
        
        # Training history
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "explained_variance": []
        }
    
    def build(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """Build the policy and value networks.
        
        Args:
            input_shape: Shape of state input
            output_shape: Shape of action output
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_actions = output_shape[0] if len(output_shape) > 0 else output_shape
        
        # Build policy network
        self.policy_network = {
            "weights": self._initialize_network(input_shape, output_shape, self.policy_layers),
            "input_shape": input_shape,
            "output_shape": output_shape
        }
        
        # Build value network
        value_output_shape = (1,)  # Single value output
        self.value_network = {
            "weights": self._initialize_network(input_shape, value_output_shape, self.value_layers),
            "input_shape": input_shape,
            "output_shape": value_output_shape
        }
    
    def _initialize_network(self, input_shape: Tuple[int, ...], 
                          output_shape: Tuple[int, ...],
                          hidden_layers: List[int]) -> Dict[str, np.ndarray]:
        """Initialize network weights.
        
        Args:
            input_shape: Input shape
            output_shape: Output shape
            hidden_layers: List of hidden layer sizes
            
        Returns:
            Dictionary of weight matrices
        """
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        
        weights = {}
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            weights[f"W{i}"] = np.random.randn(prev_size, hidden_size) * 0.01
            weights[f"b{i}"] = np.zeros(hidden_size)
            prev_size = hidden_size
        
        # Output layer
        weights["W_out"] = np.random.randn(prev_size, output_size) * 0.01
        weights["b_out"] = np.zeros(output_size)
        
        return weights
    
    def _forward(self, state: np.ndarray, network: Dict[str, Any]) -> np.ndarray:
        """Forward pass through network.
        
        Args:
            state: Input state
            network: Network to use
            
        Returns:
            Network output
        """
        weights = network["weights"]
        x = state.flatten()
        
        # Get number of hidden layers
        n_hidden = len([k for k in weights.keys() if k.startswith("W") and not k.endswith("_out")])
        
        # Hidden layers
        for i in range(n_hidden):
            x = np.dot(x, weights[f"W{i}"]) + weights[f"b{i}"]
            # Activation
            if self.activation == "tanh":
                x = np.tanh(x)
            elif self.activation == "relu":
                x = np.maximum(0, x)
        
        # Output layer
        output = np.dot(x, weights["W_out"]) + weights["b_out"]
        
        return output
    
    def predict(self, data: Any, **kwargs) -> Any:
        """Predict action probabilities and values.
        
        Args:
            data: State or batch of states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with 'actions', 'values', 'log_probs'
        """
        if not self.policy_network or not self.value_network:
            raise ValueError("Model must be built before prediction")
        
        # Handle single state or batch
        if isinstance(data, np.ndarray):
            single_state = len(data.shape) == len(self.input_shape)
            states = data[np.newaxis, ...] if single_state else data
            
            results = {
                "actions": [],
                "values": [],
                "log_probs": [],
                "action_probs": []
            }
            
            for state in states:
                # Get action logits from policy network
                logits = self._forward(state, self.policy_network)
                
                # Convert to probabilities (softmax)
                exp_logits = np.exp(logits - np.max(logits))
                action_probs = exp_logits / np.sum(exp_logits)
                
                # Sample action
                action = np.random.choice(self.n_actions, p=action_probs)
                log_prob = np.log(action_probs[action] + 1e-8)
                
                # Get value from value network
                value = self._forward(state, self.value_network)[0]
                
                results["actions"].append(action)
                results["values"].append(value)
                results["log_probs"].append(log_prob)
                results["action_probs"].append(action_probs)
            
            # Return single values if single state
            if single_state:
                return {k: v[0] for k, v in results.items()}
            else:
                return {k: np.array(v) for k, v in results.items()}
        else:
            raise ValueError("Data must be numpy array")
    
    def collect_experience(self, env: Any, n_steps: int) -> Dict[str, np.ndarray]:
        """Collect experience by interacting with environment.
        
        Args:
            env: Environment to interact with (simulated here)
            n_steps: Number of steps to collect
            
        Returns:
            Dictionary of collected experience
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Simulate environment interaction
        for step in range(n_steps):
            # Get current state (simulated)
            state = np.random.randn(*self.input_shape)
            
            # Get action from policy
            prediction = self.predict(state)
            action = prediction["actions"]
            value = prediction["values"]
            log_prob = prediction["log_probs"]
            
            # Simulate environment step
            reward = np.random.randn() * 0.1
            done = np.random.random() < 0.01  # 1% chance of episode ending
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            self.steps += 1
            
            if done:
                self.episodes += 1
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones)
        }
    
    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                          dones: np.ndarray, gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards from experience
            values: Value estimates
            dones: Episode termination flags
            gamma: Discount factor
            
        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        
        # Compute advantages backwards
        last_advantage = 0
        last_value = values[-1]
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def train(self, train_data: Any, validation_data: Optional[Any] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the PPO model.
        
        Args:
            train_data: Training environment or data
            validation_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history and metrics
        """
        # Extract training parameters
        total_timesteps = kwargs.get("total_timesteps", 10000)
        batch_size = kwargs.get("batch_size", 64)
        gamma = kwargs.get("gamma", 0.99)
        learning_rate = kwargs.get("learning_rate", 0.0003)
        
        n_updates = total_timesteps // self.n_steps
        
        # Training loop
        for update in range(n_updates):
            # Collect experience
            experience = self.collect_experience(train_data, self.n_steps)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(
                experience["rewards"],
                experience["values"],
                experience["dones"],
                gamma
            )
            
            # Store experience for training
            self.experience_buffer = {
                "states": experience["states"],
                "actions": experience["actions"],
                "old_log_probs": experience["log_probs"],
                "advantages": advantages,
                "returns": returns
            }
            
            # Train for multiple epochs
            policy_losses = []
            value_losses = []
            entropy_losses = []
            
            for epoch in range(self.n_epochs):
                # Create mini-batches
                indices = np.random.permutation(self.n_steps)
                
                for start_idx in range(0, self.n_steps, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    
                    # Get batch data
                    batch_states = self.experience_buffer["states"][batch_indices]
                    batch_actions = self.experience_buffer["actions"][batch_indices]
                    batch_old_log_probs = self.experience_buffer["old_log_probs"][batch_indices]
                    batch_advantages = self.experience_buffer["advantages"][batch_indices]
                    batch_returns = self.experience_buffer["returns"][batch_indices]
                    
                    # Compute losses
                    losses = self._compute_losses(
                        batch_states, batch_actions, batch_old_log_probs,
                        batch_advantages, batch_returns
                    )
                    
                    policy_losses.append(losses["policy_loss"])
                    value_losses.append(losses["value_loss"])
                    entropy_losses.append(losses["entropy_loss"])
                    
                    # Simulate gradient update
                    self._update_networks(losses, learning_rate)
            
            # Record training metrics
            episode_rewards = []
            current_reward = 0
            for i, (reward, done) in enumerate(zip(experience["rewards"], experience["dones"])):
                current_reward += reward
                if done:
                    episode_rewards.append(current_reward)
                    current_reward = 0
            
            if episode_rewards:
                self.training_history["episode_rewards"].extend(episode_rewards)
            
            self.training_history["policy_losses"].extend(policy_losses)
            self.training_history["value_losses"].extend(value_losses)
            self.training_history["entropy_losses"].extend(entropy_losses)
            
            # Compute explained variance
            explained_var = self._explained_variance(experience["values"], returns)
            self.training_history["explained_variance"].append(explained_var)
            
            # Log progress
            if update % 10 == 0:
                avg_reward = np.mean(self.training_history["episode_rewards"][-10:]) if self.training_history["episode_rewards"] else 0
                print(f"Update {update}/{n_updates}, Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {np.mean(policy_losses):.4f}, "
                      f"Value Loss: {np.mean(value_losses):.4f}")
        
        self.is_trained = True
        self.update_metadata({
            "total_timesteps": total_timesteps,
            "total_episodes": self.episodes,
            "total_updates": n_updates
        })
        
        return self.training_history
    
    def _compute_losses(self, states: np.ndarray, actions: np.ndarray,
                       old_log_probs: np.ndarray, advantages: np.ndarray,
                       returns: np.ndarray) -> Dict[str, float]:
        """Compute PPO losses.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probabilities from old policy
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Dictionary of losses
        """
        # Get current policy predictions
        predictions = self.predict(states)
        
        # Compute policy loss
        new_log_probs = []
        entropy_values = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            pred = self.predict(state)
            action_probs = pred["action_probs"]
            
            # Log probability of taken action
            new_log_prob = np.log(action_probs[action] + 1e-8)
            new_log_probs.append(new_log_prob)
            
            # Entropy for exploration
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            entropy_values.append(entropy)
        
        new_log_probs = np.array(new_log_probs)
        entropy_values = np.array(entropy_values)
        
        # Compute ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Value loss
        values = np.array([self.predict(state)["values"] for state in states])
        value_loss = np.mean((returns - values) ** 2)
        
        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -np.mean(entropy_values)
        
        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss) * self.value_coefficient,
            "entropy_loss": float(entropy_loss) * self.entropy_coefficient,
            "total_loss": float(policy_loss + 
                               value_loss * self.value_coefficient + 
                               entropy_loss * self.entropy_coefficient)
        }
    
    def _update_networks(self, losses: Dict[str, float], learning_rate: float) -> None:
        """Simulate network weight updates.
        
        Args:
            losses: Dictionary of losses
            learning_rate: Learning rate
        """
        # Simulate gradient updates (simplified)
        total_loss = losses["total_loss"]
        
        # Update policy network
        for key in self.policy_network["weights"]:
            # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
            if isinstance(self.policy_network["weights"][key], list):
                self.policy_network["weights"][key] = np.array(self.policy_network["weights"][key])
            gradient_sim = np.random.randn(*self.policy_network["weights"][key].shape)
            self.policy_network["weights"][key] -= learning_rate * gradient_sim * total_loss * 0.01
        
        # Update value network
        for key in self.value_network["weights"]:
            # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
            if isinstance(self.value_network["weights"][key], list):
                self.value_network["weights"][key] = np.array(self.value_network["weights"][key])
            gradient_sim = np.random.randn(*self.value_network["weights"][key].shape)
            self.value_network["weights"][key] -= learning_rate * gradient_sim * losses["value_loss"] * 0.01
    
    def _explained_variance(self, values: np.ndarray, returns: np.ndarray) -> float:
        """Compute explained variance.
        
        Args:
            values: Predicted values
            returns: Actual returns
            
        Returns:
            Explained variance ratio
        """
        var_returns = np.var(returns)
        if var_returns == 0:
            return 0
        return 1 - np.var(returns - values) / var_returns
    
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
            for step in range(1000):  # Max steps
                state = np.random.randn(*self.input_shape)
                prediction = self.predict(state)
                action = prediction["actions"]
                
                # Simulate reward
                reward = np.random.randn() * 0.1
                done = np.random.random() < 0.01
                
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
        state = {
            "policy_network": self.policy_network,
            "value_network": self.value_network,
            "steps": self.steps,
            "episodes": self.episodes,
            "training_history": self.training_history,
            "input_shape": self.input_shape if hasattr(self, "input_shape") else None,
            "output_shape": self.output_shape if hasattr(self, "output_shape") else None,
            "n_actions": self.n_actions if hasattr(self, "n_actions") else None
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if state["policy_network"]:
            state["policy_network"]["weights"] = {
                k: v.tolist() for k, v in state["policy_network"]["weights"].items()
            }
        if state["value_network"]:
            state["value_network"]["weights"] = {
                k: v.tolist() for k, v in state["value_network"]["weights"].items()
            }
        
        return state
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        """Set the model state from a dictionary.
        
        Args:
            state: Dictionary containing model state
        """
        # Restore networks
        self.policy_network = state.get("policy_network")
        self.value_network = state.get("value_network")
        
        # Convert lists back to numpy arrays
        if self.policy_network and "weights" in self.policy_network:
            self.policy_network["weights"] = {
                k: np.array(v) for k, v in self.policy_network["weights"].items()
            }
        if self.value_network and "weights" in self.value_network:
            self.value_network["weights"] = {
                k: np.array(v) for k, v in self.value_network["weights"].items()
            }
        
        # Restore other attributes
        self.steps = state.get("steps", 0)
        self.episodes = state.get("episodes", 0)
        self.training_history = state.get("training_history", {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "explained_variance": []
        })
        
        if state.get("input_shape"):
            self.input_shape = tuple(state["input_shape"])
        if state.get("output_shape"):
            self.output_shape = tuple(state["output_shape"])
        if state.get("n_actions") is not None:
            self.n_actions = state["n_actions"]
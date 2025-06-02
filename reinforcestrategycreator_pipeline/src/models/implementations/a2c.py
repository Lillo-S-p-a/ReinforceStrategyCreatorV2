"""Advantage Actor-Critic (A2C) implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..base import ModelBase


class A2C(ModelBase):
    """Advantage Actor-Critic implementation.
    
    This is a simplified A2C implementation for demonstration purposes.
    In a real implementation, you would use a deep learning framework
    like PyTorch or TensorFlow.
    """
    
    model_type = "A2C"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize A2C model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.shared_layers = self.hyperparameters.get("shared_layers", [256, 128])
        self.policy_head_layers = self.hyperparameters.get("policy_head_layers", [64])
        self.value_head_layers = self.hyperparameters.get("value_head_layers", [64])
        self.activation = self.hyperparameters.get("activation", "relu")
        
        # A2C specific parameters
        self.value_coefficient = self.hyperparameters.get("value_coefficient", 0.5)
        self.entropy_coefficient = self.hyperparameters.get("entropy_coefficient", 0.01)
        self.max_grad_norm = self.hyperparameters.get("max_grad_norm", 0.5)
        
        # Training parameters
        self.n_steps = self.hyperparameters.get("n_steps", 5)
        self.use_rms_prop = self.hyperparameters.get("use_rms_prop", True)
        self.rms_prop_eps = self.hyperparameters.get("rms_prop_eps", 1e-5)
        
        # Initialize components
        self.network = None
        self.optimizer_state = None
        self.steps = 0
        self.episodes = 0
        
        # Training history
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropy_losses": [],
            "advantages": []
        }
    
    def build(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """Build the actor-critic network.
        
        Args:
            input_shape: Shape of state input
            output_shape: Shape of action output
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_actions = output_shape[0] if len(output_shape) > 0 else output_shape
        
        # Build shared network with separate heads
        self.network = {
            "shared_weights": self._initialize_layers(input_shape, self.shared_layers),
            "policy_head_weights": self._initialize_head(
                self.shared_layers[-1] if self.shared_layers else np.prod(input_shape),
                self.n_actions,
                self.policy_head_layers
            ),
            "value_head_weights": self._initialize_head(
                self.shared_layers[-1] if self.shared_layers else np.prod(input_shape),
                1,
                self.value_head_layers
            ),
            "input_shape": input_shape,
            "output_shape": output_shape
        }
        
        # Initialize optimizer state if using RMSprop
        if self.use_rms_prop:
            self.optimizer_state = self._initialize_optimizer_state()
    
    def _initialize_layers(self, input_shape: Tuple[int, ...], 
                          layer_sizes: List[int]) -> Dict[str, np.ndarray]:
        """Initialize shared layers.
        
        Args:
            input_shape: Input shape
            layer_sizes: List of layer sizes
            
        Returns:
            Dictionary of weight matrices
        """
        weights = {}
        prev_size = np.prod(input_shape)
        
        for i, size in enumerate(layer_sizes):
            weights[f"W{i}"] = np.random.randn(prev_size, size) * np.sqrt(2.0 / prev_size)
            weights[f"b{i}"] = np.zeros(size)
            prev_size = size
        
        return weights
    
    def _initialize_head(self, input_size: int, output_size: int, 
                        hidden_layers: List[int]) -> Dict[str, np.ndarray]:
        """Initialize a network head (policy or value).
        
        Args:
            input_size: Size of input from shared layers
            output_size: Size of output
            hidden_layers: List of hidden layer sizes for the head
            
        Returns:
            Dictionary of weight matrices
        """
        weights = {}
        prev_size = input_size
        
        # Hidden layers in head
        for i, size in enumerate(hidden_layers):
            weights[f"W{i}"] = np.random.randn(prev_size, size) * np.sqrt(2.0 / prev_size)
            weights[f"b{i}"] = np.zeros(size)
            prev_size = size
        
        # Output layer
        weights["W_out"] = np.random.randn(prev_size, output_size) * 0.01
        weights["b_out"] = np.zeros(output_size)
        
        return weights
    
    def _initialize_optimizer_state(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Initialize RMSprop optimizer state.
        
        Returns:
            Dictionary of optimizer states for each parameter
        """
        optimizer_state = {}
        
        # For shared weights
        for key in self.network["shared_weights"]:
            optimizer_state[f"shared_{key}"] = {
                "square_avg": np.zeros_like(self.network["shared_weights"][key])
            }
        
        # For policy head
        for key in self.network["policy_head_weights"]:
            optimizer_state[f"policy_{key}"] = {
                "square_avg": np.zeros_like(self.network["policy_head_weights"][key])
            }
        
        # For value head
        for key in self.network["value_head_weights"]:
            optimizer_state[f"value_{key}"] = {
                "square_avg": np.zeros_like(self.network["value_head_weights"][key])
            }
        
        return optimizer_state
    
    def _forward_shared(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through shared layers.
        
        Args:
            state: Input state
            
        Returns:
            Shared representation
        """
        weights = self.network["shared_weights"]
        x = state.flatten()
        
        # Pass through shared layers
        n_layers = len([k for k in weights.keys() if k.startswith("W")])
        for i in range(n_layers):
            x = np.dot(x, weights[f"W{i}"]) + weights[f"b{i}"]
            # Activation
            if self.activation == "relu":
                x = np.maximum(0, x)
            elif self.activation == "tanh":
                x = np.tanh(x)
        
        return x
    
    def _forward_head(self, shared_features: np.ndarray, 
                     head_weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass through a head network.
        
        Args:
            shared_features: Features from shared layers
            head_weights: Weights for the head
            
        Returns:
            Head output
        """
        x = shared_features
        
        # Hidden layers in head
        n_hidden = len([k for k in head_weights.keys() if k.startswith("W") and not k.endswith("_out")])
        for i in range(n_hidden):
            x = np.dot(x, head_weights[f"W{i}"]) + head_weights[f"b{i}"]
            if self.activation == "relu":
                x = np.maximum(0, x)
            elif self.activation == "tanh":
                x = np.tanh(x)
        
        # Output layer
        output = np.dot(x, head_weights["W_out"]) + head_weights["b_out"]
        
        return output
    
    def predict(self, data: Any, **kwargs) -> Any:
        """Predict action probabilities and values.
        
        Args:
            data: State or batch of states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with 'actions', 'values', 'action_probs'
        """
        if not self.network:
            raise ValueError("Model must be built before prediction")
        
        # Handle single state or batch
        if isinstance(data, np.ndarray):
            single_state = len(data.shape) == len(self.input_shape)
            states = data[np.newaxis, ...] if single_state else data
            
            results = {
                "actions": [],
                "values": [],
                "action_probs": [],
                "log_probs": []
            }
            
            for state in states:
                # Get shared features
                shared_features = self._forward_shared(state)
                
                # Get action logits from policy head
                logits = self._forward_head(shared_features, self.network["policy_head_weights"])
                
                # Convert to probabilities (softmax)
                exp_logits = np.exp(logits - np.max(logits))
                action_probs = exp_logits / np.sum(exp_logits)
                
                # Sample action
                action = np.random.choice(self.n_actions, p=action_probs)
                log_prob = np.log(action_probs[action] + 1e-8)
                
                # Get value from value head
                value = self._forward_head(shared_features, self.network["value_head_weights"])[0]
                
                results["actions"].append(action)
                results["values"].append(value)
                results["action_probs"].append(action_probs)
                results["log_probs"].append(log_prob)
            
            # Return single values if single state
            if single_state:
                return {k: v[0] for k, v in results.items()}
            else:
                return {k: np.array(v) for k, v in results.items()}
        else:
            raise ValueError("Data must be numpy array")
    
    def train(self, train_data: Any, validation_data: Optional[Any] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the A2C model.
        
        Args:
            train_data: Training environment or data
            validation_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history and metrics
        """
        # Extract training parameters
        total_timesteps = kwargs.get("total_timesteps", 10000)
        gamma = kwargs.get("gamma", 0.99)
        learning_rate = kwargs.get("learning_rate", 0.0007)
        
        n_updates = total_timesteps // self.n_steps
        
        # Training loop
        for update in range(n_updates):
            # Collect experience for n_steps
            states = []
            actions = []
            rewards = []
            values = []
            dones = []
            log_probs = []
            
            for step in range(self.n_steps):
                # Simulate environment interaction
                state = np.random.randn(*self.input_shape)
                
                # Get action and value
                prediction = self.predict(state)
                action = prediction["actions"]
                value = prediction["values"]
                log_prob = prediction["log_probs"]
                
                # Simulate environment step
                reward = np.random.randn() * 0.1
                done = np.random.random() < 0.01
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                log_probs.append(log_prob)
                
                self.steps += 1
                if done:
                    self.episodes += 1
            
            # Convert to arrays
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            values = np.array(values)
            dones = np.array(dones)
            log_probs = np.array(log_probs)
            
            # Compute returns and advantages
            returns = self._compute_returns(rewards, values, dones, gamma)
            advantages = returns - values
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Compute losses
            actor_loss, critic_loss, entropy_loss = self._compute_losses(
                states, actions, log_probs, advantages, returns
            )
            
            # Update networks
            self._update_networks(
                states, actions, advantages, returns,
                learning_rate, actor_loss, critic_loss, entropy_loss
            )
            
            # Record metrics
            self.training_history["actor_losses"].append(actor_loss)
            self.training_history["critic_losses"].append(critic_loss)
            self.training_history["entropy_losses"].append(entropy_loss)
            self.training_history["advantages"].extend(advantages.tolist())
            
            # Track episode rewards
            episode_reward = 0
            for reward, done in zip(rewards, dones):
                episode_reward += reward
                if done:
                    self.training_history["episode_rewards"].append(episode_reward)
                    episode_reward = 0
            
            # Log progress
            if update % 100 == 0:
                avg_reward = np.mean(self.training_history["episode_rewards"][-10:]) if self.training_history["episode_rewards"] else 0
                print(f"Update {update}/{n_updates}, Avg Reward: {avg_reward:.2f}, "
                      f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        
        self.is_trained = True
        self.update_metadata({
            "total_timesteps": total_timesteps,
            "total_episodes": self.episodes,
            "total_updates": n_updates
        })
        
        return self.training_history
    
    def _compute_returns(self, rewards: np.ndarray, values: np.ndarray,
                        dones: np.ndarray, gamma: float) -> np.ndarray:
        """Compute discounted returns.
        
        Args:
            rewards: Rewards for each step
            values: Value estimates
            dones: Episode termination flags
            gamma: Discount factor
            
        Returns:
            Array of returns
        """
        returns = np.zeros_like(rewards)
        running_return = 0
        
        # Compute returns backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _compute_losses(self, states: np.ndarray, actions: np.ndarray,
                       log_probs: np.ndarray, advantages: np.ndarray,
                       returns: np.ndarray) -> Tuple[float, float, float]:
        """Compute actor and critic losses.
        
        Args:
            states: States
            actions: Actions taken
            log_probs: Log probabilities of actions
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Tuple of (actor_loss, critic_loss, entropy_loss)
        """
        # Recompute predictions for current policy
        predictions = self.predict(states)
        current_values = predictions["values"]
        action_probs = predictions["action_probs"]
        
        # Actor loss (negative because we want to maximize)
        actor_loss = -np.mean(log_probs * advantages)
        
        # Critic loss (MSE)
        critic_loss = np.mean((returns - current_values) ** 2)
        
        # Entropy loss for exploration
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
        entropy_loss = -np.mean(entropy)  # Negative because we want to maximize
        
        return float(actor_loss), float(critic_loss), float(entropy_loss)
    
    def _update_networks(self, states: np.ndarray, actions: np.ndarray,
                        advantages: np.ndarray, returns: np.ndarray,
                        learning_rate: float, actor_loss: float,
                        critic_loss: float, entropy_loss: float) -> None:
        """Update network weights.
        
        Args:
            states: Batch of states
            actions: Actions taken
            advantages: Advantages
            returns: Returns
            learning_rate: Learning rate
            actor_loss: Actor loss value
            critic_loss: Critic loss value
            entropy_loss: Entropy loss value
        """
        # Compute total loss
        total_loss = (actor_loss + 
                     self.value_coefficient * critic_loss + 
                     self.entropy_coefficient * entropy_loss)
        
        # Simulate gradient computation and update
        # In reality, you would use automatic differentiation
        
        # Update shared weights
        for key in self.network["shared_weights"]:
            # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
            if isinstance(self.network["shared_weights"][key], list):
                self.network["shared_weights"][key] = np.array(self.network["shared_weights"][key])
            gradient = np.random.randn(*self.network["shared_weights"][key].shape) * total_loss * 0.01
            
            if self.use_rms_prop:
                # RMSprop update
                opt_key = f"shared_{key}"
                # Ensure square_avg is a numpy array
                if isinstance(self.optimizer_state[opt_key]["square_avg"], list):
                    self.optimizer_state[opt_key]["square_avg"] = np.array(self.optimizer_state[opt_key]["square_avg"])
                self.optimizer_state[opt_key]["square_avg"] = (
                    0.99 * self.optimizer_state[opt_key]["square_avg"] +
                    0.01 * gradient ** 2
                )
                self.network["shared_weights"][key] -= (
                    learning_rate * gradient / 
                    (np.sqrt(self.optimizer_state[opt_key]["square_avg"]) + self.rms_prop_eps)
                )
            else:
                # Simple gradient descent
                self.network["shared_weights"][key] -= learning_rate * gradient
        
        # Update policy head
        for key in self.network["policy_head_weights"]:
            # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
            if isinstance(self.network["policy_head_weights"][key], list):
                self.network["policy_head_weights"][key] = np.array(self.network["policy_head_weights"][key])
            gradient = np.random.randn(*self.network["policy_head_weights"][key].shape) * actor_loss * 0.01
            
            if self.use_rms_prop:
                opt_key = f"policy_{key}"
                # Ensure square_avg is a numpy array
                if isinstance(self.optimizer_state[opt_key]["square_avg"], list):
                    self.optimizer_state[opt_key]["square_avg"] = np.array(self.optimizer_state[opt_key]["square_avg"])
                self.optimizer_state[opt_key]["square_avg"] = (
                    0.99 * self.optimizer_state[opt_key]["square_avg"] +
                    0.01 * gradient ** 2
                )
                self.network["policy_head_weights"][key] -= (
                    learning_rate * gradient / 
                    (np.sqrt(self.optimizer_state[opt_key]["square_avg"]) + self.rms_prop_eps)
                )
            else:
                self.network["policy_head_weights"][key] -= learning_rate * gradient
        
        # Update value head
        for key in self.network["value_head_weights"]:
            # Ensure weights are numpy arrays (they might be lists after loading from checkpoint)
            if isinstance(self.network["value_head_weights"][key], list):
                self.network["value_head_weights"][key] = np.array(self.network["value_head_weights"][key])
            gradient = np.random.randn(*self.network["value_head_weights"][key].shape) * critic_loss * 0.01
            
            if self.use_rms_prop:
                opt_key = f"value_{key}"
                # Ensure square_avg is a numpy array
                if isinstance(self.optimizer_state[opt_key]["square_avg"], list):
                    self.optimizer_state[opt_key]["square_avg"] = np.array(self.optimizer_state[opt_key]["square_avg"])
                self.optimizer_state[opt_key]["square_avg"] = (
                    0.99 * self.optimizer_state[opt_key]["square_avg"] +
                    0.01 * gradient ** 2
                )
                self.network["value_head_weights"][key] -= (
                    learning_rate * gradient / 
                    (np.sqrt(self.optimizer_state[opt_key]["square_avg"]) + self.rms_prop_eps)
                )
            else:
                self.network["value_head_weights"][key] -= learning_rate * gradient
    
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
        value_estimates = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            episode_values = []
            
            # Simulate test episode
            for step in range(200):  # Max steps
                state = np.random.randn(*self.input_shape)
                prediction = self.predict(state)
                action = prediction["actions"]
                value = prediction["values"]
                
                # Simulate reward
                reward = np.random.randn() * 0.1
                done = np.random.random() < 0.01
                
                episode_reward += reward
                episode_length += 1
                episode_values.append(value)
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            value_estimates.extend(episode_values)
        
        return {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "min_episode_reward": float(np.min(episode_rewards)),
            "max_episode_reward": float(np.max(episode_rewards)),
            "mean_value_estimate": float(np.mean(value_estimates))
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current state of the model for serialization.
        
        Returns:
            Dictionary containing model state
        """
        state = {
            "network": self.network,
            "optimizer_state": self.optimizer_state,
            "steps": self.steps,
            "episodes": self.episodes,
            "training_history": self.training_history,
            "input_shape": self.input_shape if hasattr(self, "input_shape") else None,
            "output_shape": self.output_shape if hasattr(self, "output_shape") else None,
            "n_actions": self.n_actions if hasattr(self, "n_actions") else None
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if state["network"]:
            for component in ["shared_weights", "policy_head_weights", "value_head_weights"]:
                if component in state["network"]:
                    state["network"][component] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in state["network"][component].items()
                    }
        
        if state["optimizer_state"]:
            for key in state["optimizer_state"]:
                for sub_key in state["optimizer_state"][key]:
                    if isinstance(state["optimizer_state"][key][sub_key], np.ndarray):
                        state["optimizer_state"][key][sub_key] = state["optimizer_state"][key][sub_key].tolist()
        
        return state
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        """Set the model state from a dictionary.
        
        Args:
            state: Dictionary containing model state
        """
        # Restore network
        self.network = state.get("network")
        
        # Convert lists back to numpy arrays
        if self.network:
            for component in ["shared_weights", "policy_head_weights", "value_head_weights"]:
                if component in self.network:
                    self.network[component] = {
                        k: np.array(v) for k, v in self.network[component].items()
                    }
        
        # Restore optimizer state
        self.optimizer_state = state.get("optimizer_state")
        if self.optimizer_state:
            for key in self.optimizer_state:
                for sub_key in self.optimizer_state[key]:
                    if isinstance(self.optimizer_state[key][sub_key], list):
                        self.optimizer_state[key][sub_key] = np.array(self.optimizer_state[key][sub_key])
        
        # Restore other attributes
        self.steps = state.get("steps", 0)
        self.episodes = state.get("episodes", 0)
        self.training_history = state.get("training_history", {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropy_losses": [],
            "advantages": []
        })
        
        if state.get("input_shape"):
            self.input_shape = tuple(state["input_shape"])
        if state.get("output_shape"):
            self.output_shape = tuple(state["output_shape"])
        if state.get("n_actions") is not None:
            self.n_actions = state["n_actions"]
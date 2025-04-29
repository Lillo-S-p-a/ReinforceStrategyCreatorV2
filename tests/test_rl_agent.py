"""
Tests for the rl_agent module using pytest.
"""

import pytest
import numpy as np
import tensorflow as tf
import unittest.mock # Add this import
from tensorflow import keras

from reinforcestrategycreator.rl_agent import StrategyAgent

# --- Fixtures ---

@pytest.fixture(scope="module")
def state_size():
    """Provides a sample state size."""
    return 10  # Example: 10 features in the state

@pytest.fixture(scope="module")
def action_size():
    """Provides a sample action size."""
    return 3  # Example: 3 possible actions (buy, sell, hold)

@pytest.fixture(scope="module")
def learning_rate():
    """Provides a sample learning rate."""
    return 0.001

@pytest.fixture(scope="module")
def agent(state_size, action_size, learning_rate):
    """Provides a StrategyAgent instance."""
    return StrategyAgent(state_size, action_size, learning_rate)

@pytest.fixture
def sample_state(state_size):
    """Provides sample state data."""
    return np.random.random(state_size)

@pytest.fixture
def sample_next_state(state_size):
    """Provides sample next state data."""
    return np.random.random(state_size)

# --- Test Functions ---

def test_agent_model_initialization_and_structure(agent, action_size, learning_rate):
    """CORE LOGIC TEST: Verify model initialization and structure."""
    # Context: Test the Keras model defined in StrategyAgent.__init__
    # Potential SAPPO :Problems: :ConfigurationIssue, :LogicError

    # 1. Verify model existence and type
    assert agent.model is not None, "Model should be initialized."
    assert isinstance(agent.model, keras.models.Sequential), "Model should be a Keras Sequential model."

    # 2. Verify layer count
    assert len(agent.model.layers) == 3, "Model should have 3 Dense layers."

    # 3. Verify layer types and activation functions
    assert isinstance(agent.model.layers[0], keras.layers.Dense), "Layer 0 should be Dense."
    assert agent.model.layers[0].activation.__name__ == 'relu', "Layer 0 activation should be relu."

    assert isinstance(agent.model.layers[1], keras.layers.Dense), "Layer 1 should be Dense."
    assert agent.model.layers[1].activation.__name__ == 'relu', "Layer 1 activation should be relu."

    assert isinstance(agent.model.layers[2], keras.layers.Dense), "Layer 2 should be Dense."
    assert agent.model.layers[2].activation.__name__ == 'linear', "Layer 2 activation should be linear."

    # 4. Verify layer output shapes (Input shape check is implicit in layer 0 build)
    # Note: Accessing shape via layer.output.shape after model build
    # Output shapes are (None, units), where None is the batch dimension.
    # Need to use .output.shape instead of .output_shape
    # The shape tuple is (batch_size, units), where batch_size is None initially
    assert agent.model.layers[0].output.shape == (None, 64), f"Layer 0 output shape mismatch: Expected {(None, 64)}, Got {agent.model.layers[0].output.shape}"
    assert agent.model.layers[1].output.shape == (None, 64), f"Layer 1 output shape mismatch: Expected {(None, 64)}, Got {agent.model.layers[1].output.shape}"
    assert agent.model.layers[2].output.shape == (None, action_size), f"Layer 2 output shape mismatch: Expected {(None, action_size)}, Got {agent.model.layers[2].output.shape}"


    # 5. Verify compilation parameters (Loss and Optimizer)
    assert agent.model.loss in ['mse', 'mean_squared_error'], "Loss function should be MSE."

    assert isinstance(agent.model.optimizer, keras.optimizers.Adam), "Optimizer should be Adam."
    # Check learning rate
    assert np.isclose(agent.model.optimizer.learning_rate.numpy(), learning_rate, atol=1e-6), "Optimizer learning rate mismatch."


def test_select_action(agent, sample_state, action_size):
    """Test that select_action returns a valid action (placeholder check)."""
    # Get an action from the agent
    action = agent.select_action(sample_state)

    # Assert that the action is within the valid range
    assert isinstance(action, int)
    assert 0 <= action < action_size


def test_learn_method_signature(agent, sample_state, sample_next_state):
    """Test that the learn method accepts the correct parameters."""
    # Call the learn method with valid parameters
    try:
        agent.learn(
            state=sample_state,
            action=0,
            reward=1.0,
            next_state=sample_next_state,
            done=False
        )
        # If we get here, the method signature is correct
        test_passed = True
    except (TypeError, ValueError) as e:
        test_passed = False
        pytest.fail(f"learn method signature incorrect: {e}")

    assert test_passed, "learn method should accept state, action, reward, next_state, and done parameters"


@pytest.mark.parametrize("shape_tuple", [(True,), (False,)]) # True for (state_size,), False for (1, state_size)
def test_integration_with_numpy_arrays(agent, state_size, action_size, shape_tuple):
    """CONTEXTUAL INTEGRATION TEST: Verify compatibility with numpy arrays."""
    # Context: Ensures agent methods handle standard numpy array inputs.
    use_1d_shape = shape_tuple[0]
    shape = (state_size,) if use_1d_shape else (1, state_size)

    state = np.random.random(shape)
    next_state = np.random.random(shape)

    try:
        # Test select_action
        action = agent.select_action(state)
        assert isinstance(action, int)

        # Test learn
        agent.learn(state, action, 1.0, next_state, False)
        test_passed = True
    except Exception as e:
        test_passed = False
        pytest.fail(f"Failed with shape {shape}: {e}")

    assert test_passed, f"Methods should handle numpy arrays of shape {shape}"


# --- Tests for select_action and epsilon handling ---

class TestStrategyAgentSelectAction:
    """Tests focusing on the select_action method and epsilon parameters."""

    @pytest.fixture
    def agent_params(self, state_size, action_size, learning_rate):
        """Provides default parameters for agent creation."""
        return {
            "state_size": state_size,
            "action_size": action_size,
            "learning_rate": learning_rate,
            "epsilon": 0.9, # Non-default start epsilon
            "epsilon_min": 0.05, # Non-default min
            "epsilon_decay": 0.99 # Non-default decay
        }

    @pytest.fixture
    def specific_agent(self, agent_params):
        """Provides an agent instance with specific non-default epsilon params."""
        return StrategyAgent(**agent_params)

    def test_init_epsilon_params(self, specific_agent, agent_params):
        """CORE LOGIC TEST: Verify __init__ sets epsilon parameters correctly."""
        # Context: Check if custom epsilon hyperparameters are stored.
        # Potential SAPPO :Problems: :ConfigurationIssue
        assert specific_agent.epsilon == agent_params["epsilon"]
        assert specific_agent.epsilon_min == agent_params["epsilon_min"]
        assert specific_agent.epsilon_decay == agent_params["epsilon_decay"]

    def test_select_action_random_exploration(self, agent, sample_state, action_size):
        """CORE LOGIC TEST: Verify random action selection when epsilon is 1.0."""
        # Context: Test epsilon-greedy exploration path (:Algorithm EpsilonGreedyExploration).
        # Potential SAPPO :Problems: :LogicError
        agent.epsilon = 1.0 # Force exploration
        num_calls = 100
        actions_taken = set()

        for _ in range(num_calls):
            action = agent.select_action(sample_state)
            assert isinstance(action, int), "Action should be an integer."
            assert 0 <= action < action_size, f"Action {action} out of bounds [0, {action_size})."
            actions_taken.add(action)

        # Check if multiple different actions were chosen (probabilistic check for randomness)
        # With 3 actions and 100 calls, it's highly likely we see more than one action.
        assert len(actions_taken) > 1, "Expected multiple different actions during exploration."
        # Epsilon should decay after calls
        assert agent.epsilon < 1.0, "Epsilon should decay after calls."


    @unittest.mock.patch.object(tf.keras.Model, 'predict')
    def test_select_action_greedy_exploitation(self, mock_predict, agent, sample_state, action_size):
        """CORE LOGIC TEST: Verify greedy action selection when epsilon is 0.0."""
        # Context: Test epsilon-greedy exploitation path (:Algorithm EpsilonGreedyExploration).
        # Potential SAPPO :Problems: :LogicError, :CompatibilityIssue (model output)
        agent.epsilon = 0.0 # Force exploitation
        mock_q_values = np.array([[0.1, 0.8, 0.3]]) # Max Q-value at index 1
        expected_action = 1
        mock_predict.return_value = mock_q_values

        action = agent.select_action(sample_state)

        assert action == expected_action, f"Expected action {expected_action}, got {action}."
        assert isinstance(action, int), "Action should be an integer."
        assert 0 <= action < action_size, f"Action {action} out of bounds [0, {action_size})."
        mock_predict.assert_called_once() # Verify model.predict was called
        # Epsilon should not decay below min (assuming default min is > 0)
        assert agent.epsilon == 0.0, "Epsilon should remain 0.0 when forced."


    @unittest.mock.patch.object(tf.keras.Model, 'predict')
    @pytest.mark.parametrize("state_input", [
        list(np.random.rand(10)), # Python list
        np.random.rand(10),       # 1D NumPy array
        np.random.rand(1, 10)     # 2D NumPy array (correct shape)
    ])
    def test_select_action_state_reshaping(self, mock_predict, agent, state_input, state_size):
        """CORE LOGIC TEST: Verify state is correctly reshaped for model.predict."""
        # Context: Ensure input state format compatibility (:CompatibilityIssue).
        # Potential SAPPO :Problems: :TypeError, :ValueError (shape mismatch)
        agent.epsilon = 0.0 # Force exploitation to ensure predict is called
        # Use agent's action_size for the dummy return value shape
        mock_predict.return_value = np.array([[0.1] * agent.action_size])

        # Use the correct state_size fixture value for list generation
        state_input_processed = state_input
        if isinstance(state_input, list):
             state_input_processed = list(np.random.rand(state_size))
        elif isinstance(state_input, np.ndarray) and state_input.ndim == 1:
             state_input_processed = np.random.rand(state_size)
        elif isinstance(state_input, np.ndarray) and state_input.ndim == 2:
             state_input_processed = np.random.rand(1, state_size)


        agent.select_action(state_input_processed)

        mock_predict.assert_called_once()
        call_args, _ = mock_predict.call_args
        passed_state = call_args[0]
        assert isinstance(passed_state, np.ndarray), "State passed to predict should be ndarray."
        assert passed_state.shape == (1, state_size), f"Expected shape {(1, state_size)}, got {passed_state.shape}."


    def test_select_action_epsilon_decay(self, agent, sample_state):
        """CORE LOGIC TEST: Verify epsilon decays correctly after select_action."""
        # Context: Test epsilon decay logic (:Algorithm EpsilonGreedyExploration).
        # Potential SAPPO :Problems: :LogicError
        initial_epsilon = 0.5
        # Use agent's actual configured decay and min values
        decay = agent.epsilon_decay
        min_epsilon = agent.epsilon_min

        agent.epsilon = initial_epsilon
        agent.select_action(sample_state) # Call action selection
        expected_epsilon = initial_epsilon * decay
        assert np.isclose(agent.epsilon, expected_epsilon), f"Epsilon should decay: expected {expected_epsilon}, got {agent.epsilon}"

        # Test decay stops at min_epsilon
        epsilon_near_min = min_epsilon + 0.001 # Slightly above min
        agent.epsilon = epsilon_near_min
        agent.select_action(sample_state)
        # Calculate expected value correctly considering the minimum
        expected_epsilon_at_min = max(min_epsilon, epsilon_near_min * decay)
        assert np.isclose(agent.epsilon, expected_epsilon_at_min), f"Epsilon decay near min failed: expected {expected_epsilon_at_min}, got {agent.epsilon}"
        assert agent.epsilon >= min_epsilon, "Epsilon should not go below min_epsilon."

        # Test epsilon does not decay if already at min_epsilon
        agent.epsilon = min_epsilon
        agent.select_action(sample_state)
        assert np.isclose(agent.epsilon, min_epsilon), "Epsilon should not decay if already at min_epsilon."


    def test_select_action_integration_numpy(self, agent, state_size):
        """CONTEXTUAL INTEGRATION TEST: Verify select_action runs with numpy state."""
        # Context: Basic check for runtime errors with standard input type.
        # Potential SAPPO :Problems: :TypeError, :ValueError
        state = np.random.rand(state_size) # Standard 1D numpy array
        try:
            action = agent.select_action(state)
            assert isinstance(action, int)
            assert 0 <= action < agent.action_size
        except Exception as e:
            pytest.fail(f"select_action failed with numpy input: {e}")
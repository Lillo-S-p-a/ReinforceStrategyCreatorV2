"""
Tests for the rl_agent module using pytest.
"""

import pytest
import numpy as np
import tensorflow as tf
import unittest.mock as mock # Use alias for clarity
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
def agent(state_size, action_size, learning_rate): # Default agent uses default gamma
    """Provides a StrategyAgent instance."""
    # Default gamma is 0.95 in the implementation
    return StrategyAgent(state_size=state_size, action_size=action_size, learning_rate=learning_rate)

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


def test_learn_method_signature(agent, state_size, action_size):
    """Test that the learn method runs without errors when memory is sufficient."""
    # Context: Verify learn() can be called without arguments after refactoring.
    # Potential SAPPO :Problems: :TypeError (if signature changed incorrectly)
    # Populate memory sufficiently
    batch_size = agent.batch_size # Get batch size from agent
    for i in range(batch_size):
         state = np.random.rand(state_size)
         action = np.random.randint(action_size)
         reward = np.random.rand()
         next_state = np.random.rand(state_size)
         done = (i % 5 == 0) # Some True, some False
         agent.remember(state, action, reward, next_state, done)

    try:
        agent.learn() # Call learn without arguments
        test_passed = True
    except Exception as e: # Catch any exception
        test_passed = False
        pytest.fail(f"learn method raised an unexpected exception: {e}")

    assert test_passed, "learn method should run without arguments when memory is sufficient."


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

        # Populate memory sufficiently to test learn
        batch_size = agent.batch_size
        # Ensure state and next_state are in the format remember expects (1D or list)
        state_for_memory = state[0] if state.ndim == 2 else state
        next_state_for_memory = next_state[0] if next_state.ndim == 2 else next_state
        for _ in range(batch_size):
            agent.remember(state_for_memory, action, 1.0, next_state_for_memory, False)

        # Test learn call
        agent.learn() # Call learn without arguments
        test_passed = True
    except Exception as e:
        test_passed = False
        pytest.fail(f"Failed with shape {shape}: {e}")

    assert test_passed, f"Methods should handle numpy arrays of shape {shape} and learn should run"


from collections import deque # Ensure deque is imported

# --- Tests for Experience Replay Memory ---

class TestStrategyAgentMemory:
    """Tests focusing on the experience replay memory (deque)."""

    @pytest.fixture
    def memory_params(self, state_size, action_size, learning_rate):
        """Provides parameters for agent creation with specific memory size."""
        return {
            "state_size": state_size,
            "action_size": action_size,
            "learning_rate": learning_rate,
            "memory_size": 5, # Use a small size for testing limits
            "batch_size": 2,
"gamma": 0.99 # Specific gamma for testing init
        }

    @pytest.fixture
    def memory_agent(self, memory_params):
        """Provides an agent instance with specific memory parameters."""
        return StrategyAgent(**memory_params)

    def test_init_memory(self, memory_agent, memory_params):
        """CORE LOGIC TEST: Verify __init__ initializes memory correctly."""
        # Context: Check deque initialization and maxlen setting.
        # Potential SAPPO :Problems: :ConfigurationIssue, :TypeError
        assert isinstance(memory_agent.memory, deque), "Memory should be a deque."
        assert memory_agent.memory.maxlen == memory_params["memory_size"], \
            f"Memory maxlen should be {memory_params['memory_size']}."
        assert len(memory_agent.memory) == 0, "Memory should be initialized empty."
    def test_init_gamma(self, memory_agent, memory_params):
        """CORE LOGIC TEST: Verify __init__ stores gamma correctly."""
        # Context: Check gamma hyperparameter storage.
        # Potential SAPPO :Problems: :ConfigurationIssue
        assert hasattr(memory_agent, 'gamma'), "Agent should have a 'gamma' attribute."
        assert memory_agent.gamma == memory_params["gamma"], f"Gamma should be {memory_params['gamma']}."

    def test_remember_single_experience(self, memory_agent, state_size):
        """CORE LOGIC TEST: Verify remember stores a single experience correctly."""
        # Context: Test basic storage and data structure.
        # Potential SAPPO :Problems: :LogicError, :TypeError, :ValueError
        state = np.random.rand(state_size)
        action = 1
        reward = 0.5
        next_state = np.random.rand(state_size)
        done = False

        memory_agent.remember(state, action, reward, next_state, done)

        assert len(memory_agent.memory) == 1, "Memory size should be 1 after one remember call."

        stored_state, stored_action, stored_reward, stored_next_state, stored_done = memory_agent.memory[0]

        # Check reshaping and values
        expected_state_shape = (1, state_size)
        assert isinstance(stored_state, np.ndarray), "Stored state should be ndarray."
        assert stored_state.shape == expected_state_shape, f"Stored state shape mismatch: Expected {expected_state_shape}, Got {stored_state.shape}"
        np.testing.assert_array_equal(stored_state[0], state), "Stored state data mismatch." # Compare the inner array

        assert stored_action == action, "Stored action mismatch."
        assert stored_reward == reward, "Stored reward mismatch."

        assert isinstance(stored_next_state, np.ndarray), "Stored next_state should be ndarray."
        assert stored_next_state.shape == expected_state_shape, f"Stored next_state shape mismatch: Expected {expected_state_shape}, Got {stored_next_state.shape}"
        np.testing.assert_array_equal(stored_next_state[0], next_state), "Stored next_state data mismatch." # Compare the inner array

        assert stored_done == done, "Stored done flag mismatch."

    def test_remember_multiple_experiences_order(self, memory_agent, state_size):
        """CORE LOGIC TEST: Verify multiple experiences are stored in correct FIFO order."""
        # Context: Test storage order.
        # Potential SAPPO :Problems: :LogicError
        experiences = []
        for i in range(3):
            state = np.random.rand(state_size) * (i + 1) # Make states distinct
            action = i
            reward = float(i)
            next_state = np.random.rand(state_size) * (i + 1 + 0.5)
            done = (i % 2 == 0)
            experiences.append((state, action, reward, next_state, done))
            memory_agent.remember(state, action, reward, next_state, done)

        assert len(memory_agent.memory) == 3, "Memory size should be 3."

        # Verify order and content
        for i in range(3):
            original_state, original_action, original_reward, original_next_state, original_done = experiences[i]
            stored_state, stored_action, stored_reward, stored_next_state, stored_done = memory_agent.memory[i]

            np.testing.assert_array_equal(stored_state[0], original_state), f"State mismatch at index {i}."
            assert stored_action == original_action, f"Action mismatch at index {i}."
            assert stored_reward == original_reward, f"Reward mismatch at index {i}."
            np.testing.assert_array_equal(stored_next_state[0], original_next_state), f"Next state mismatch at index {i}."
            assert stored_done == original_done, f"Done flag mismatch at index {i}."

    def test_remember_exceeds_maxlen(self, memory_agent, state_size, memory_params):
        """CORE LOGIC TEST: Verify memory discards oldest experience when full."""
        # Context: Test deque maxlen behavior.
        # Potential SAPPO :Problems: :LogicError, :ConfigurationIssue
        memory_limit = memory_params["memory_size"]
        experiences = []

        # Fill the memory + 1
        for i in range(memory_limit + 1):
            state = np.random.rand(state_size) * (i + 1)
            action = i
            reward = float(i)
            next_state = np.random.rand(state_size) * (i + 1 + 0.5)
            done = (i % 2 == 0)
            experiences.append((state, action, reward, next_state, done))
            memory_agent.remember(state, action, reward, next_state, done)

        assert len(memory_agent.memory) == memory_limit, f"Memory size should be capped at {memory_limit}."

        # Verify the first experience is gone
        first_exp_state = experiences[0][0]
        present = any(np.array_equal(stored_exp[0][0], first_exp_state) for stored_exp in memory_agent.memory)
        assert not present, "The oldest experience (index 0) should have been discarded."

        # Verify the last experience is present
        last_exp_state = experiences[-1][0]
        stored_last_state = memory_agent.memory[-1][0][0] # Get the state from the last tuple in deque
        np.testing.assert_array_equal(stored_last_state, last_exp_state), "The newest experience should be present at the end."

    def test_remember_data_types(self, memory_agent, state_size):
        """CORE LOGIC TEST: Verify data types within the stored experience tuple."""
        # Context: Check type consistency after storage.
        # Potential SAPPO :Problems: :TypeError
        state = np.random.rand(state_size).astype(np.float32) # Use specific float type
        action = 2
        reward = 10.0 # Float
        next_state = np.random.rand(state_size).astype(np.float32)
        done = True # Boolean

        memory_agent.remember(state, action, reward, next_state, done)

        assert len(memory_agent.memory) == 1
        stored_state, stored_action, stored_reward, stored_next_state, stored_done = memory_agent.memory[0]

        assert isinstance(stored_state, np.ndarray), "Stored state type should be ndarray."
        assert stored_state.dtype == np.float32, f"Stored state dtype mismatch: Expected float32, Got {stored_state.dtype}"
        assert isinstance(stored_action, int), f"Stored action type mismatch: Expected int, Got {type(stored_action)}"
        assert isinstance(stored_reward, float), f"Stored reward type mismatch: Expected float, Got {type(stored_reward)}"
        assert isinstance(stored_next_state, np.ndarray), "Stored next_state type should be ndarray."
        assert stored_next_state.dtype == np.float32, f"Stored next_state dtype mismatch: Expected float32, Got {stored_next_state.dtype}"
        assert isinstance(stored_done, bool), f"Stored done type mismatch: Expected bool, Got {type(stored_done)}"


    def test_integration_remember_env_like_data(self, memory_agent, state_size):
        """CONTEXTUAL INTEGRATION TEST: Verify remember handles 1D numpy arrays (like Env output)."""
        # Context: Ensure compatibility with typical environment state outputs before reshaping.
        # Potential SAPPO :Problems: :CompatibilityIssue, :ValueError
        # Mimic TradingEnv output (1D array)
        state_1d = np.random.rand(state_size)
        next_state_1d = np.random.rand(state_size)
        action = 0
        reward = -1.2
        done = False

        try:
            memory_agent.remember(state_1d, action, reward, next_state_1d, done)
        except Exception as e:
            pytest.fail(f"remember failed with 1D numpy array input: {e}")

        assert len(memory_agent.memory) == 1, "Memory should contain the experience."
        stored_state, _, _, stored_next_state, _ = memory_agent.memory[0]

        # Verify that the states were correctly reshaped inside 'remember'
        expected_shape = (1, state_size)
        assert stored_state.shape == expected_shape, f"Stored state should be reshaped to {expected_shape}, got {stored_state.shape}"
        assert stored_next_state.shape == expected_shape, f"Stored next_state should be reshaped to {expected_shape}, got {stored_next_state.shape}"
        np.testing.assert_array_equal(stored_state[0], state_1d), "Stored state data mismatch after reshaping."
        np.testing.assert_array_equal(stored_next_state[0], next_state_1d), "Stored next_state data mismatch after reshaping."
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


# --- Tests for learn method batch sampling and preparation ---

class TestStrategyAgentLearnBatching:
    """Tests focusing on the learn method's batch sampling and preparation logic."""

    @pytest.fixture
    def learn_params(self, state_size, action_size, learning_rate):
        """Provides parameters for agent creation with specific batch/memory size."""
        return {
            "state_size": state_size,
            "action_size": action_size,
            "learning_rate": learning_rate,
            "memory_size": 100, # Sufficient memory capacity
            "batch_size": 16,    # Specific batch size for testing
            "gamma": 0.95        # Default gamma for these tests
        }

    @pytest.fixture
    def learn_agent(self, learn_params):
        """Provides an agent instance with specific parameters for learn tests."""
        return StrategyAgent(**learn_params)

    def _populate_memory(self, agent, num_experiences, state_size):
        """Helper function to populate agent memory."""
        for i in range(num_experiences):
            state = np.random.rand(state_size).astype(np.float32)
            action = np.random.randint(agent.action_size)
            reward = np.random.rand() * 10 - 5 # Random reward +/-
            next_state = np.random.rand(state_size).astype(np.float32)
            done = (np.random.rand() > 0.8) # Randomly True/False
            agent.remember(state, action, reward, next_state, done)

    def test_learn_insufficient_memory(self, learn_agent, learn_params, state_size):
        """CORE LOGIC TEST: Verify learn returns None when memory < batch_size."""
        # Context: Test the initial size check in learn().
        # Potential SAPPO :Problems: :LogicError
        batch_size = learn_params["batch_size"]
        num_samples = batch_size - 1
        assert num_samples > 0, "Batch size must be > 1 for this test"

        self._populate_memory(learn_agent, num_samples, state_size)
        assert len(learn_agent.memory) == num_samples

        # Assuming learn is refactored to return None when insufficient memory
        result = learn_agent.learn()

        assert result is None, f"learn() should return None when memory ({len(learn_agent.memory)}) < batch_size ({batch_size})."

    def test_learn_sufficient_memory_batch_preparation(self, learn_agent, learn_params, state_size):
        """CORE LOGIC TEST: Verify learn samples and prepares batch correctly."""
    def test_learn_integration_no_runtime_error(self, learn_agent, learn_params, state_size):
        """CONTEXTUAL INTEGRATION TEST: Verify learn runs without error after memory population."""
        # Context: Ensure learn() executes sampling/preparation without crashing when memory is full.
        # Potential SAPPO :Problems: Runtime errors (:IndexError, :TypeError, etc.) during batch processing.
        batch_size = learn_params["batch_size"]
        num_samples = batch_size + 5

        self._populate_memory(learn_agent, num_samples, state_size)

        try:
            # Mock fit to avoid actual training, just check if learn runs
            with mock.patch.object(learn_agent.model, 'fit') as mock_fit:
                learn_agent.learn()
            # If memory > batch_size, fit should have been called
            if len(learn_agent.memory) >= learn_params["batch_size"]:
                 mock_fit.assert_called_once()
            else:
                 mock_fit.assert_not_called() # Should not be called if memory too small
        except Exception as e:
            pytest.fail(f"learn() raised an unexpected exception during batch preparation: {e}")
# --- Tests for learn method Q-value calculation and model fitting ---

class TestStrategyAgentLearnLogic:
    """Tests focusing on the learn method's core Q-value calculation and model fitting logic."""

    @pytest.fixture
    def learn_logic_params(self, state_size, action_size, learning_rate):
        """Provides parameters for agent creation with specific batch/memory size."""
        return {
            "state_size": state_size,
            "action_size": action_size,
            "learning_rate": learning_rate,
            "memory_size": 50, # Sufficient memory
            "batch_size": 4,   # Small batch size for manual calculation
            "gamma": 0.9       # Specific gamma for testing calculation
        }

    @pytest.fixture
    def learn_logic_agent(self, learn_logic_params):
        """Provides an agent instance with specific parameters for learn logic tests."""
        return StrategyAgent(**learn_logic_params)

    @mock.patch.object(StrategyAgent, '_build_model') # Mock model building
    def test_learn_q_value_target_calculation_and_fit(self, mock_build, learn_logic_agent, learn_logic_params, state_size, action_size):
        """CORE LOGIC TEST: Verify Q-target calculation and model.fit call."""
        # Context: Test Bellman update and training call (:Algorithm DQN).
        # Potential SAPPO :Problems: :LogicError (calculation), :CompatibilityIssue (shapes/types), :NumericalInstability

        # --- Setup ---
        batch_size = learn_logic_params["batch_size"]
        gamma = learn_logic_params["gamma"]

        # Create a mock model with mock predict and fit methods
        mock_model = mock.MagicMock(spec=tf.keras.Model)
        mock_predict = mock.MagicMock(spec=tf.keras.Model.predict)
        mock_fit = mock.MagicMock(spec=tf.keras.Model.fit)
        mock_model.predict = mock_predict
        mock_model.fit = mock_fit
        learn_logic_agent.model = mock_model # Assign the mock model to the agent

        # Prepare known batch data
        states = np.random.rand(batch_size, state_size).astype(np.float32)
        actions = np.array([0, 1, 2, 0]) # Example actions for batch_size=4
        rewards = np.array([1.0, -0.5, 0.0, 2.0]).astype(np.float32)
        next_states = np.random.rand(batch_size, state_size).astype(np.float32)
        # Make some 'done' True to test the (1 - dones) part
        dones = np.array([False, False, True, False]).astype(bool)

        # Populate memory directly (avoiding remember's reshaping for direct control)
        for i in range(batch_size):
            # Store with the shape expected by vstack in learn()
            learn_logic_agent.memory.append((
                states[i:i+1, :], actions[i], rewards[i], next_states[i:i+1, :], dones[i]
            ))
        assert len(learn_logic_agent.memory) == batch_size

        # Define the *original* Q-values the mock should use
        original_mock_next_q = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float32)
        original_mock_current_q = np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9], [2.0, 2.1, 2.2]], dtype=np.float32)

        # Define a side effect function for the mock predict to return Q-values in the shuffled order
        predict_call_count = 0
        def mock_predict_side_effect(input_batch, **kwargs):
            nonlocal predict_call_count
            predict_call_count += 1
            # Determine which original data and Q-values to use based on the call count
            source_array = next_states if predict_call_count == 1 else states
            source_q_values = original_mock_next_q if predict_call_count == 1 else original_mock_current_q

            # Find original indices corresponding to the input_batch rows
            batch_indices = []
            for row in input_batch:
                found = False
                for i, source_row in enumerate(source_array):
                    if np.allclose(row, source_row, rtol=1e-5, atol=1e-6):
                        batch_indices.append(i)
                        found = True
                        break
                if not found:
                     pytest.fail(f"Mock predict side effect could not map input row: {row}")
            assert len(batch_indices) == input_batch.shape[0], "Mock predict side effect mapping failed"

            # Return the corresponding Q-value rows in the same shuffled order as the input_batch
            return source_q_values[batch_indices]

        mock_predict.side_effect = mock_predict_side_effect

        # --- Execute ---
        learn_logic_agent.learn()

        # --- Assert ---
        # 1. Assert predict calls
        assert mock_predict.call_count == 2
        # Helper function to convert numpy array rows to a set of tuples for comparison
        def array_rows_to_set(arr):
            # Convert float rows to tuples of rounded values to handle potential precision issues in comparison
            # Adjust precision as needed
            return set(tuple(np.round(row, decimals=6)) for row in arr)

        # Call 1: next_states
        actual_next_states = mock_predict.call_args_list[0][0][0]
        assert actual_next_states.shape == next_states.shape, f"Shape mismatch for next_states: Expected {next_states.shape}, Got {actual_next_states.shape}"
        assert actual_next_states.dtype == next_states.dtype, f"Dtype mismatch for next_states: Expected {next_states.dtype}, Got {actual_next_states.dtype}"
        # Compare content ignoring order
        expected_next_states_set = array_rows_to_set(next_states)
        actual_next_states_set = array_rows_to_set(actual_next_states)
        assert actual_next_states_set == expected_next_states_set, "Content mismatch for next_states passed to predict"


        # Call 2: states
        actual_states = mock_predict.call_args_list[1][0][0]
        assert actual_states.shape == states.shape, f"Shape mismatch for states: Expected {states.shape}, Got {actual_states.shape}"
        assert actual_states.dtype == states.dtype, f"Dtype mismatch for states: Expected {states.dtype}, Got {actual_states.dtype}"
        # Compare content ignoring order
        expected_states_set = array_rows_to_set(states)
        actual_states_set = array_rows_to_set(actual_states)
        assert actual_states_set == expected_states_set, "Content mismatch for states passed to predict"

        # 2. Assert fit call
        mock_fit.assert_called_once()
        call_args, call_kwargs = mock_fit.call_args
        fit_states, fit_target_q = call_args # These are the actual, potentially shuffled arrays

        # 3. Verify fit_states (order-independent)
        assert fit_states.shape == states.shape, f"Shape mismatch for fit_states: Expected {states.shape}, Got {fit_states.shape}"
        assert fit_states.dtype == np.float32, f"Dtype mismatch for fit_states: Expected {np.float32}, Got {fit_states.dtype}"
        actual_fit_states_set = array_rows_to_set(fit_states)
        expected_states_set = array_rows_to_set(states)
        assert actual_fit_states_set == expected_states_set, "Content mismatch for states passed to fit"

        # 4. Reconstruct expected target_q based on the actual order of states passed to fit
        # Find the permutation indices that map original states to fit_states
        # This is complex if states are not unique; assume unique for this test or use a more robust mapping
        # For simplicity, let's find the index of each row in fit_states within the original states
        # Note: This assumes states rows are unique, which is likely with random floats
        original_indices = []
        for fit_row in fit_states:
            found = False
            for i, original_row in enumerate(states):
                 # Use np.allclose for float comparison
                 if np.allclose(fit_row, original_row, rtol=1e-5, atol=1e-6):
                     original_indices.append(i)
                     found = True
                     break
            if not found:
                 pytest.fail(f"Could not find matching row in original states for fit_state row: {fit_row}")
        assert len(original_indices) == batch_size, "Could not map all fit_states rows back to original states"
        permutation_indices = np.array(original_indices)

        # Reorder the original actions, rewards, dones according to the actual permutation used in learn()
        shuffled_actions = actions[permutation_indices]
        shuffled_rewards = rewards[permutation_indices]
        shuffled_dones = dones[permutation_indices]

        # Calculate the expected target values using the shuffled rewards/dones
        # and the *original* mock_next_q (since that's what predict returned).
        # We need to select the max Q-value corresponding to the shuffled experience.
        max_next_q = np.amax(original_mock_next_q, axis=1) # Max Q for each original next_state
        expected_target_values_shuffled = shuffled_rewards + gamma * max_next_q[permutation_indices] * (1 - shuffled_dones.astype(np.uint8))

        # Construct the expected target_q array that *should* have been passed to fit.
        # Start with the *original* mock_current_q (since that's what predict returned),
        # but reorder the rows according to the permutation.
        expected_fit_target_q = original_mock_current_q[permutation_indices].copy()
        # Update the values for the actions taken in the shuffled batch
        expected_fit_target_q[np.arange(batch_size), shuffled_actions] = expected_target_values_shuffled

        # 5. Compare the actual fit_target_q with the reconstructed expected_fit_target_q
        assert fit_target_q.shape == expected_fit_target_q.shape, f"Shape mismatch for fit_target_q: Expected {expected_fit_target_q.shape}, Got {fit_target_q.shape}"
        assert fit_target_q.dtype == np.float32, f"Dtype mismatch for fit_target_q: Expected {np.float32}, Got {fit_target_q.dtype}"
        np.testing.assert_allclose(fit_target_q, expected_fit_target_q, rtol=1e-6, err_msg="Content mismatch for target_q_values passed to fit (reconstructed)")

        # 6. Verify other fit call parameters
        assert call_kwargs.get('epochs') == 1
        assert call_kwargs.get('verbose') == 0
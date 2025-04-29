"""
Tests for the rl_agent module.
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from reinforcestrategycreator.rl_agent import StrategyAgent


class TestRLAgent(unittest.TestCase):
    """Test cases for the rl_agent module."""

    def setUp(self):
        """Set up test data."""
        # Define sample dimensions for testing
        self.state_size = 10  # Example: 10 features in the state
        self.action_size = 3  # Example: 3 possible actions (buy, sell, hold)
        
        # Create a sample agent
        self.agent = StrategyAgent(self.state_size, self.action_size)
        
        # Create sample state data
        self.sample_state = np.random.random(self.state_size)
        self.sample_next_state = np.random.random(self.state_size)

    # The setUp method now implicitly tests basic initialization success
    # by creating the agent. The old test_agent_initialization is removed.

    def test_agent_model_initialization_and_structure(self):
        """CORE LOGIC TEST: Verify model initialization and structure."""
        # Context: Test the Keras model defined in StrategyAgent.__init__
        # Potential SAPPO :Problems: :ConfigurationIssue, :LogicError

        # 1. Verify model existence and type
        self.assertIsNotNone(self.agent.model, "Model should be initialized.")
        self.assertIsInstance(self.agent.model, keras.models.Sequential, "Model should be a Keras Sequential model.")

        # 2. Verify layer count
        self.assertEqual(len(self.agent.model.layers), 3, "Model should have 3 Dense layers.")

        # 3. Verify layer types and activation functions
        self.assertIsInstance(self.agent.model.layers[0], keras.layers.Dense, "Layer 0 should be Dense.")
        self.assertEqual(self.agent.model.layers[0].activation.__name__, 'relu', "Layer 0 activation should be relu.")

        self.assertIsInstance(self.agent.model.layers[1], keras.layers.Dense, "Layer 1 should be Dense.")
        self.assertEqual(self.agent.model.layers[1].activation.__name__, 'relu', "Layer 1 activation should be relu.")

        self.assertIsInstance(self.agent.model.layers[2], keras.layers.Dense, "Layer 2 should be Dense.")
        self.assertEqual(self.agent.model.layers[2].activation.__name__, 'linear', "Layer 2 activation should be linear.")

        # 4. Verify layer output shapes (Input shape check is implicit in layer 0 build)
        # Note: Keras builds layers lazily or requires an input shape definition.
        # The input_dim in the first Dense layer defines the input shape.
        # Output shapes are (None, units), where None is the batch dimension.
        self.assertEqual(self.agent.model.layers[0].output_shape, (None, 64), "Layer 0 output shape mismatch.")
        self.assertEqual(self.agent.model.layers[1].output_shape, (None, 64), "Layer 1 output shape mismatch.")
        self.assertEqual(self.agent.model.layers[2].output_shape, (None, self.action_size), "Layer 2 output shape mismatch.")

        # 5. Verify compilation parameters (Loss and Optimizer)
        # Accessing the compiled loss name might vary slightly depending on TF/Keras version
        self.assertIn(self.agent.model.loss, ['mse', 'mean_squared_error'], "Loss function should be MSE.")

        self.assertIsInstance(self.agent.model.optimizer, keras.optimizers.Adam, "Optimizer should be Adam.")
        # Check learning rate - access might differ slightly across TF versions
        # Using .learning_rate.numpy() is common for TF 2.x Eager execution
        np.testing.assert_almost_equal(self.agent.model.optimizer.learning_rate.numpy(), self.agent.learning_rate, decimal=6,
                                        err_msg="Optimizer learning rate mismatch.")

    def test_select_action(self):
        """Test that select_action returns a valid action (placeholder check)."""
        # Get an action from the agent
        action = self.agent.select_action(self.sample_state)
        
        # Assert that the action is within the valid range
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)

    def test_learn_method_signature(self):
        """Test that the learn method accepts the correct parameters."""
        # Call the learn method with valid parameters
        try:
            self.agent.learn(
                state=self.sample_state,
                action=0,
                reward=1.0,
                next_state=self.sample_next_state,
                done=False
            )
            # If we get here, the method signature is correct
            test_passed = True
        except (TypeError, ValueError):
            test_passed = False
        
        self.assertTrue(test_passed, "learn method should accept state, action, reward, next_state, and done parameters")

    def test_integration_with_numpy_arrays(self):
        """CONTEXTUAL INTEGRATION TEST: Verify compatibility with numpy arrays."""
        # Context: Ensures agent methods handle standard numpy array inputs.
        # This test remains relevant. The successful initialization in setUp
        # and the core model tests cover the basic integration aspect of __init__.
        # Test with numpy arrays of different shapes
        for shape in [(self.state_size,), (1, self.state_size)]:
            state = np.random.random(shape)
            next_state = np.random.random(shape)
            
            try:
                # Test select_action
                action = self.agent.select_action(state)
                self.assertIsInstance(action, int)
                
                # Test learn
                self.agent.learn(state, action, 1.0, next_state, False)
                test_passed = True
            except Exception as e:
                test_passed = False
                self.fail(f"Failed with shape {shape}: {e}")
            
            self.assertTrue(test_passed, f"Methods should handle numpy arrays of shape {shape}")


if __name__ == '__main__':
    unittest.main()
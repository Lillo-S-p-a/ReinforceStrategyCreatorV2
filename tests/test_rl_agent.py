"""
Tests for the rl_agent module.
"""

import unittest
import numpy as np

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

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        # Assert that the agent has the correct attributes
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsNone(self.agent.model)  # Model should be None in placeholder implementation

    def test_select_action(self):
        """Test that select_action returns a valid action."""
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
        """Contextual Integration Test: Verify compatibility with numpy arrays."""
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
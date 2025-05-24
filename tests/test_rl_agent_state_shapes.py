#!/usr/bin/env python3
"""
Unit test for RL agent state shape handling.

This test verifies that the RL agent correctly handles states with inconsistent shapes,
preventing the ValueError that was occurring during HPO.
"""

import unittest
import numpy as np
from reinforcestrategycreator.rl_agent import StrategyAgent

class TestRLAgentStateShapes(unittest.TestCase):
    """Test case for RL agent state shape handling."""
    
    def setUp(self):
        """Set up the test case."""
        self.state_size = 20
        self.action_size = 3
        self.agent = StrategyAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            batch_size=4  # Small batch size for testing
        )
    
    def test_remember_with_consistent_shapes(self):
        """Test remember method with states of consistent shapes."""
        # Create states with the correct shape
        state = np.ones(self.state_size, dtype=np.float32)
        next_state = np.ones(self.state_size, dtype=np.float32)
        
        # Remember the experience
        self.agent.remember(state, 1, 1.0, next_state, False)
        
        # Verify the experience was stored correctly
        self.assertEqual(len(self.agent.memory), 1)
        experience = self.agent.memory[0]
        self.assertEqual(experience[0].shape, (self.state_size,))
        self.assertEqual(experience[3].shape, (self.state_size,))
    
    def test_remember_with_inconsistent_shapes(self):
        """Test remember method with states of inconsistent shapes."""
        # Create states with incorrect shapes
        state1 = np.ones(self.state_size, dtype=np.float32)  # Correct shape
        next_state1 = np.ones(self.state_size, dtype=np.float32)  # Correct shape
        
        state2 = np.ones(self.state_size + 5, dtype=np.float32)  # Too large
        next_state2 = np.ones(self.state_size + 5, dtype=np.float32)  # Too large
        
        state3 = np.ones(self.state_size - 5, dtype=np.float32)  # Too small
        next_state3 = np.ones(self.state_size - 5, dtype=np.float32)  # Too small
        
        state4 = np.ones((5, 4), dtype=np.float32)  # 2D array
        next_state4 = np.ones((5, 4), dtype=np.float32)  # 2D array
        
        # Remember the experiences
        self.agent.remember(state1, 1, 1.0, next_state1, False)
        self.agent.remember(state2, 0, 0.5, next_state2, False)
        self.agent.remember(state3, 2, -1.0, next_state3, False)
        self.agent.remember(state4, 1, 0.0, next_state4, False)
        
        # Verify all experiences have consistent shapes
        self.assertEqual(len(self.agent.memory), 4)
        for experience in self.agent.memory:
            self.assertEqual(experience[0].shape, (self.state_size,))
            self.assertEqual(experience[3].shape, (self.state_size,))
    
    def test_learn_with_inconsistent_shapes(self):
        """Test learn method with states of inconsistent shapes."""
        # Create states with inconsistent shapes and remember them
        for i in range(10):
            # Alternate between different shapes
            if i % 3 == 0:
                state = np.ones(self.state_size, dtype=np.float32)
                next_state = np.ones(self.state_size, dtype=np.float32)
            elif i % 3 == 1:
                state = np.ones(self.state_size + 3, dtype=np.float32)
                next_state = np.ones(self.state_size - 2, dtype=np.float32)
            else:
                state = np.ones((4, 5), dtype=np.float32)  # 2D array
                next_state = np.ones((5, 4), dtype=np.float32)  # 2D array with different shape
            
            self.agent.remember(state, i % self.action_size, float(i), next_state, i % 2 == 0)
        
        # Try to learn from these experiences
        # This should not raise ValueError due to our fix
        try:
            self.agent.learn()
            success = True
        except ValueError:
            success = False
        
        self.assertTrue(success, "learn() raised ValueError despite the fix")

if __name__ == "__main__":
    unittest.main()
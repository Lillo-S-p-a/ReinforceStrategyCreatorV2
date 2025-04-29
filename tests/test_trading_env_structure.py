"""
Tests for the TradingEnv class structure.

This module specifically tests the structural aspects of the TradingEnv class
to ensure it meets the requirements of the gymnasium.Env interface.
"""

import inspect
import pytest
import gymnasium as gym
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from reinforcestrategycreator.trading_environment import TradingEnv


def test_trading_env_inherits_from_gym_env():
    """Test that TradingEnv inherits from gymnasium.Env."""
    assert issubclass(TradingEnv, gym.Env), "TradingEnv should inherit from gymnasium.Env"


def test_required_methods_exist():
    """Test that all required methods of gymnasium.Env are implemented."""
    # Create a sample environment with indicators
    df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
                125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
               105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
               115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                  3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700],
        # Add mock indicator columns
        'RSI_14': [50.0] * 28,
        'MACD_12_26_9': [0.5] * 28,
        'MACD_Signal_12_26_9': [0.3] * 28,
        'MACD_Hist_12_26_9': [0.2] * 28,
        'BBL_20_2.0': [95.0] * 28,
        'BBM_20_2.0': [100.0] * 28,
        'BBU_20_2.0': [105.0] * 28
    })
    env = TradingEnv(df)
    
    # Check that required methods exist
    assert hasattr(env, '__init__'), "TradingEnv should have __init__ method"
    assert hasattr(env, 'reset'), "TradingEnv should have reset method"
    assert hasattr(env, 'step'), "TradingEnv should have step method"
    assert hasattr(env, 'render'), "TradingEnv should have render method"
    assert hasattr(env, 'close'), "TradingEnv should have close method"


def test_method_signatures():
    """Test that methods have the correct signatures."""
    # Get method signatures
    reset_sig = inspect.signature(TradingEnv.reset)
    step_sig = inspect.signature(TradingEnv.step)
    render_sig = inspect.signature(TradingEnv.render)
    close_sig = inspect.signature(TradingEnv.close)
    
    # Check reset signature
    assert 'seed' in reset_sig.parameters, "reset method should have 'seed' parameter"
    assert 'options' in reset_sig.parameters, "reset method should have 'options' parameter"
    
    # Check step signature
    assert 'action' in step_sig.parameters, "step method should have 'action' parameter"
    
    # Check render signature
    assert 'mode' in render_sig.parameters, "render method should have 'mode' parameter"
    
    # Check return type annotations
    reset_return = reset_sig.return_annotation
    step_return = step_sig.return_annotation
    
    # Check that reset returns a tuple
    assert reset_return == Tuple[np.ndarray, Dict[str, Any]], "reset should return Tuple[np.ndarray, Dict[str, Any]]"
    
    # Check that step returns a tuple with 5 elements
    assert step_return == Tuple[np.ndarray, float, bool, bool, Dict[str, Any]], \
        "step should return Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]"


def test_spaces_defined():
    """Test that action_space and observation_space are properly defined."""
    # Create a sample environment with indicators
    df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
                125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
               105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
               115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                  3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700],
        # Add mock indicator columns
        'RSI_14': [50.0] * 28,
        'MACD_12_26_9': [0.5] * 28,
        'MACD_Signal_12_26_9': [0.3] * 28,
        'MACD_Hist_12_26_9': [0.2] * 28,
        'BBL_20_2.0': [95.0] * 28,
        'BBM_20_2.0': [100.0] * 28,
        'BBU_20_2.0': [105.0] * 28
    })
    env = TradingEnv(df)
    
    # Check that spaces are defined
    assert hasattr(env, 'action_space'), "TradingEnv should have action_space attribute"
    assert hasattr(env, 'observation_space'), "TradingEnv should have observation_space attribute"
    
    # Check that spaces are instances of gym.spaces.Space
    assert isinstance(env.action_space, gym.spaces.Space), "action_space should be an instance of gym.spaces.Space"
    assert isinstance(env.observation_space, gym.spaces.Space), "observation_space should be an instance of gym.spaces.Space"
    
    # Check specific space types
    assert isinstance(env.action_space, gym.spaces.Discrete), "action_space should be a Discrete space"
    assert isinstance(env.observation_space, gym.spaces.Box), "observation_space should be a Box space"


# No need for this block when using pytest
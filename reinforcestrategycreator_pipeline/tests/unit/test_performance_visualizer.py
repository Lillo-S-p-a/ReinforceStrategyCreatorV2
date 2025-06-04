"""Unit tests for the PerformanceVisualizer class."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from reinforcestrategycreator_pipeline.src.visualization.performance_visualizer import PerformanceVisualizer


class TestPerformanceVisualizer(unittest.TestCase):
    """Test cases for PerformanceVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = PerformanceVisualizer()
        
        # Sample data
        self.portfolio_values = [10000, 10200, 10100, 10500, 10300, 10800]
        self.dates = pd.date_range('2024-01-01', periods=6, freq='D')
        
        # Sample metrics
        self.metrics_dict = {
            'Strategy1': {
                'pnl_percentage': 8.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'win_rate': 0.55
            },
            'Strategy2': {
                'pnl_percentage': 5.2,
                'sharpe_ratio': 0.9,
                'max_drawdown': 0.20,
                'win_rate': 0.48
            }
        }
        
        # Sample training history
        self.training_history = {
            'episode_reward': [100, 120, 110, 130, 125, 140, 135, 150],
            'loss': [0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2]
        }
    
    def test_init(self):
        """Test visualizer initialization."""
        config = {
            'default_figsize': (10, 8),
            'style': 'seaborn-v0_8-darkgrid',
            'save_dpi': 150
        }
        visualizer = PerformanceVisualizer(config)
        
        self.assertEqual(visualizer.default_figsize, (10, 8))
        self.assertEqual(visualizer.save_dpi, 150)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_cumulative_returns(self, mock_savefig, mock_show):
        """Test cumulative returns plotting."""
        # Test without benchmark
        fig = self.visualizer.plot_cumulative_returns(
            self.portfolio_values,
            title="Test Returns",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test with benchmark
        benchmark_values = [10000, 10100, 10050, 10200, 10150, 10300]
        fig = self.visualizer.plot_cumulative_returns(
            self.portfolio_values,
            benchmark_values=benchmark_values,
            dates=self.dates,
            title="Test Returns with Benchmark",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        # Check that both lines are plotted
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)  # Strategy, Benchmark, and zero line
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_drawdown(self, mock_show):
        """Test drawdown plotting."""
        fig = self.visualizer.plot_drawdown(
            self.portfolio_values,
            dates=self.dates,
            title="Test Drawdown",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Check that drawdown is calculated correctly
        ax = fig.axes[0]
        # Should have fill_between and line plot
        self.assertTrue(len(ax.collections) > 0)  # fill_between creates a collection
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_metrics_comparison(self, mock_show):
        """Test metrics comparison plotting."""
        fig = self.visualizer.plot_metrics_comparison(
            self.metrics_dict,
            metrics_to_plot=['sharpe_ratio', 'max_drawdown'],
            title="Test Metrics Comparison",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        # Should have 2 subplots for 2 metrics
        self.assertEqual(len(fig.axes), 2)
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_learning_curves(self, mock_show):
        """Test learning curves plotting."""
        fig = self.visualizer.plot_learning_curves(
            self.training_history,
            title="Test Learning Curves",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        # Should have 2 subplots for 2 metrics
        self.assertEqual(len(fig.axes), 2)
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_risk_return_scatter(self, mock_show):
        """Test risk-return scatter plot."""
        results_list = [
            {
                'model': {'id': 'model1'},
                'metrics': {
                    'volatility': 0.15,
                    'pnl_percentage': 8.5,
                    'sharpe_ratio': 1.2
                }
            },
            {
                'model': {'id': 'model2'},
                'metrics': {
                    'volatility': 0.20,
                    'pnl_percentage': 5.2,
                    'sharpe_ratio': 0.9
                }
            }
        ]
        
        fig = self.visualizer.plot_risk_return_scatter(
            results_list,
            title="Test Risk-Return",
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_create_performance_dashboard(self, mock_show):
        """Test performance dashboard creation."""
        evaluation_results = {
            'evaluation_id': 'test_eval_001',
            'evaluation_name': 'Test Evaluation',
            'timestamp': '2024-01-01T12:00:00',
            'model': {
                'id': 'test_model',
                'version': '1.0',
                'type': 'PPO'
            },
            'metrics': {
                'pnl_percentage': 8.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'win_rate': 0.55,
                'profit_factor': 1.8,
                'trades_count': 150
            },
            'benchmarks': self.metrics_dict
        }
        
        fig = self.visualizer.create_performance_dashboard(
            evaluation_results,
            show=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        # Dashboard should have multiple subplots
        self.assertTrue(len(fig.axes) > 1)
        
        plt.close('all')
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_functionality(self, mock_savefig):
        """Test that plots can be saved."""
        save_path = Path("/tmp/test_plot.png")
        
        fig = self.visualizer.plot_cumulative_returns(
            self.portfolio_values,
            save_path=save_path,
            show=False
        )
        
        # Check that savefig was called with correct parameters
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        self.assertEqual(call_args[0][0], str(save_path))
        self.assertEqual(call_args[1]['dpi'], self.visualizer.save_dpi)
        
        plt.close('all')
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Empty metrics dict
        fig = self.visualizer.plot_metrics_comparison(
            {},
            show=False
        )
        self.assertIsNone(fig)
        
        # Empty training history
        fig = self.visualizer.plot_learning_curves(
            {},
            show=False
        )
        self.assertIsNone(fig)
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
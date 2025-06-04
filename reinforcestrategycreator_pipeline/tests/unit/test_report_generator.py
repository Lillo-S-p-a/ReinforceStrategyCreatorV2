"""Unit tests for the ReportGenerator class."""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
from pathlib import Path
from datetime import datetime

from reinforcestrategycreator_pipeline.src.visualization.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.report_generator = ReportGenerator()
        
        # Sample evaluation results
        self.evaluation_results = {
            'evaluation_id': 'eval_test_20240101_120000',
            'evaluation_name': 'Test Model Evaluation',
            'timestamp': '2024-01-01T12:00:00',
            'model': {
                'id': 'test_model',
                'version': '1.0.0',
                'type': 'PPO',
                'hyperparameters': {
                    'learning_rate': 0.0003,
                    'batch_size': 64
                }
            },
            'data': {
                'source_id': 'test_data',
                'version': '2024.1',
                'shape': [1000, 10],
                'columns': ['feature1', 'feature2', 'feature3']
            },
            'metrics': {
                'pnl': 850.50,
                'pnl_percentage': 8.5,
                'total_return': 0.085,
                'sharpe_ratio': 1.2345,
                'max_drawdown': 0.1523,
                'win_rate': 0.55,
                'trades_count': 150
            },
            'benchmarks': {
                'buy_and_hold': {
                    'pnl': 500.00,
                    'pnl_percentage': 5.0,
                    'sharpe_ratio': 0.8,
                    'max_drawdown': 0.25,
                    'win_rate': 0.0,
                    'trades_count': 1
                }
            },
            'relative_performance': {
                'buy_and_hold': {
                    'absolute_difference': 350.50,
                    'percentage_difference': 70.1,
                    'sharpe_ratio_difference': 0.4345
                }
            }
        }
        
        # Sample visualization paths
        self.visualization_paths = {
            'cumulative_returns': '/tmp/eval_test/viz/cumulative_returns.png',
            'drawdown': '/tmp/eval_test/viz/drawdown.png',
            'metrics_comparison': '/tmp/eval_test/viz/metrics_comparison.png'
        }
    
    def test_init(self):
        """Test report generator initialization."""
        config = {
            'template_dir': '/path/to/templates',
            'pdf_options': {
                'page-size': 'Letter',
                'margin-top': '1in'
            }
        }
        
        with patch('jinja2.FileSystemLoader'):
            with patch('jinja2.Environment'):
                generator = ReportGenerator(config)
                self.assertEqual(generator.config, config)
                self.assertEqual(generator.pdf_options['page-size'], 'Letter')
    
    def test_generate_markdown_report(self):
        """Test Markdown report generation."""
        report = self.report_generator.generate_report(
            self.evaluation_results,
            format_type='markdown',
            include_visualizations=True,
            visualization_paths=self.visualization_paths
        )
        
        # Check that report contains expected content
        self.assertIn('# Test Model Evaluation', report)
        self.assertIn('eval_test_20240101_120000', report)
        self.assertIn('test_model', report)
        self.assertIn('8.5%', report)  # PnL percentage
        self.assertIn('1.2345', report)  # Sharpe ratio
        self.assertIn('Benchmark Comparison', report)
        self.assertIn('buy_and_hold', report)
        
        # Check visualizations are included
        self.assertIn('cumulative_returns.png', report)
        self.assertIn('drawdown.png', report)
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        report = self.report_generator.generate_report(
            self.evaluation_results,
            format_type='html',
            include_visualizations=False
        )
        
        # Check HTML structure
        self.assertIn('<!DOCTYPE html>', report)
        self.assertIn('<html', report)
        self.assertIn('</html>', report)
        self.assertIn('<title>Test Model Evaluation</title>', report)
        
        # Check content
        self.assertIn('test_model', report)
        self.assertIn('8.5%', report)
        self.assertIn('<table', report)
        
        # Check no visualizations when not requested
        self.assertNotIn('cumulative_returns.png', report)
    
    @patch('pdfkit.from_string')
    def test_generate_pdf_report(self, mock_pdfkit):
        """Test PDF report generation."""
        output_path = '/tmp/test_report.pdf'
        
        report = self.report_generator.generate_report(
            self.evaluation_results,
            format_type='pdf',
            output_path=output_path
        )
        
        # Check that pdfkit was called
        mock_pdfkit.assert_called_once()
        call_args = mock_pdfkit.call_args
        
        # First argument should be HTML content
        html_content = call_args[0][0]
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('test_model', html_content)
        
        # Second argument should be output path
        self.assertEqual(call_args[0][1], output_path)
    
    def test_format_metrics_for_display(self):
        """Test metrics formatting."""
        metrics = {
            'win_rate': 0.55,
            'pnl_percentage': 8.505,
            'total_return': 0.08505,
            'sharpe_ratio': 1.2345,
            'max_drawdown': 0.1523,
            'trades_count': 150
        }
        
        formatted = self.report_generator._format_metrics_for_display(metrics)
        
        self.assertEqual(formatted['win_rate'], '55.0%')
        self.assertEqual(formatted['pnl_percentage'], '8.51%')
        self.assertEqual(formatted['sharpe_ratio'], '1.2345')
        self.assertEqual(formatted['max_drawdown'], '0.1523')
        self.assertEqual(formatted['trades_count'], '150')
    
    def test_prepare_template_context(self):
        """Test template context preparation."""
        context = self.report_generator._prepare_template_context(
            self.evaluation_results,
            include_visualizations=True,
            visualization_paths=self.visualization_paths
        )
        
        # Check all expected keys are present
        expected_keys = [
            'evaluation', 'model', 'data', 'metrics', 'benchmarks',
            'relative_performance', 'timestamp', 'evaluation_id',
            'evaluation_name', 'include_visualizations', 'visualization_paths',
            'formatted_metrics', 'formatted_benchmarks'
        ]
        
        for key in expected_keys:
            self.assertIn(key, context)
        
        # Check formatted metrics
        self.assertIn('win_rate', context['formatted_metrics'])
        self.assertEqual(context['formatted_metrics']['win_rate'], '55.0%')
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_report_to_file(self, mock_file):
        """Test saving report to file."""
        output_path = Path('/tmp/test_report.md')
        
        with patch.object(Path, 'mkdir'):
            report = self.report_generator.generate_report(
                self.evaluation_results,
                format_type='markdown',
                output_path=output_path
            )
            
            # Check file was opened for writing
            mock_file.assert_called_with(output_path, 'w', encoding='utf-8')
            
            # Check content was written
            handle = mock_file()
            handle.write.assert_called()
    
    def test_create_summary_report(self):
        """Test summary report creation for multiple evaluations."""
        evaluation_list = [
            self.evaluation_results,
            {
                **self.evaluation_results,
                'model': {'id': 'model2', 'version': '2.0'},
                'metrics': {
                    'pnl_percentage': 5.2,
                    'sharpe_ratio': 0.9,
                    'max_drawdown': 0.20,
                    'win_rate': 0.48
                }
            }
        ]
        
        summary = self.report_generator.create_summary_report(
            evaluation_list,
            format_type='markdown'
        )
        
        # Check summary contains both models
        self.assertIn('test_model', summary)
        self.assertIn('model2', summary)
        self.assertIn('Model Evaluation Summary Report', summary)
        self.assertIn('Total Models Evaluated: 2', summary)
        
        # Check performance comparison table
        self.assertIn('| Model | Version | PnL % | Sharpe | Max DD | Win Rate |', summary)
    
    def test_invalid_format_type(self):
        """Test handling of invalid format type."""
        with self.assertRaises(ValueError) as context:
            self.report_generator.generate_report(
                self.evaluation_results,
                format_type='invalid_format'
            )
        
        self.assertIn('Unsupported report format', str(context.exception))
    
    @patch('pdfkit.from_string', side_effect=Exception('wkhtmltopdf not found'))
    def test_pdf_generation_error(self, mock_pdfkit):
        """Test error handling in PDF generation."""
        with self.assertRaises(Exception) as context:
            self.report_generator.generate_report(
                self.evaluation_results,
                format_type='pdf'
            )
        
        self.assertIn('wkhtmltopdf not found', str(context.exception))
    
    def test_custom_template_usage(self):
        """Test using custom templates."""
        # Create a mock template
        mock_template = MagicMock()
        mock_template.render.return_value = "Custom template output"
        
        # Mock the jinja environment
        mock_env = MagicMock()
        mock_env.get_template.return_value = mock_template
        
        generator = ReportGenerator()
        generator.jinja_env = mock_env
        
        report = generator.generate_report(
            self.evaluation_results,
            format_type='markdown',
            template_name='custom_template.md'
        )
        
        # Check that custom template was used
        mock_env.get_template.assert_called_with('custom_template.md')
        self.assertEqual(report, "Custom template output")
    
    def test_html_summary_report(self):
        """Test HTML summary report generation."""
        evaluation_list = [self.evaluation_results]
        
        summary = self.report_generator.create_summary_report(
            evaluation_list,
            format_type='html'
        )
        
        # Check HTML structure
        self.assertIn('<!DOCTYPE html>', summary)
        self.assertIn('<table>', summary)
        self.assertIn('test_model', summary)
        self.assertIn('Model Evaluation Summary Report', summary)


if __name__ == '__main__':
    unittest.main()
"""
Reporting module for backtesting.

This module provides functionality for generating comprehensive reports
of backtesting results in various formats (HTML, Markdown, PDF).
"""

import os
import logging
import datetime
import jinja2
import markdown
import shutil
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive backtesting reports.
    
    This class provides methods for creating reports in various formats,
    including HTML, Markdown, and PDF.
    """
    
    def __init__(self, reports_dir: str = "reports") -> None:
        """
        Initialize the report generator.
        
        Args:
            reports_dir: Directory to save reports
        """
        self.reports_dir = reports_dir
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_report(self, 
                        data: Dict[str, Any], 
                        format: str = "html") -> str:
        """
        Create a comprehensive backtesting report.
        
        Args:
            data: Dictionary containing report data
            format: Report format ('html', 'markdown', or 'pdf')
            
        Returns:
            str: Path to the generated report file
        """
        logger.info(f"Generating {format} backtesting report")
        
        try:
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate report based on format
            if format.lower() == "html":
                report_path = self._generate_html_report(data, timestamp)
            elif format.lower() == "markdown":
                report_path = self._generate_markdown_report(data, timestamp)
            elif format.lower() == "pdf":
                report_path = self._generate_pdf_report(data, timestamp)
            else:
                raise ValueError(f"Unsupported report format: {format}")
            
            logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            raise
    
    def _generate_html_report(self, data: Dict[str, Any], timestamp: str) -> str:
        """
        Generate HTML report using Jinja2 template.
        
        Args:
            data: Dictionary containing report data
            timestamp: Timestamp string for the report
            
        Returns:
            str: Path to the generated HTML report
        """
        # Define HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Report - {{ data.asset }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .metrics-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .metrics-table th { background-color: #f2f2f2; }
                .plot-container { margin: 20px 0; }
                .plot-container img { max-width: 100%; border: 1px solid #ddd; }
                .benchmark-comparison { display: flex; flex-wrap: wrap; }
                .benchmark-item { margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <h1>Backtesting Report - {{ data.asset }}</h1>
            <p><strong>Period:</strong> {{ data.period }}</p>
            <p><strong>Generated:</strong> {{ data.timestamp }}</p>
            
            <h2>Performance Summary</h2>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>PnL</td>
                    <td>${{ "%.2f"|format(data.test_metrics.pnl) }}</td>
                </tr>
                <tr>
                    <td>PnL (%)</td>
                    <td>{{ "%.2f"|format(data.test_metrics.pnl_percentage) }}%</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{{ "%.2f"|format(data.test_metrics.sharpe_ratio) }}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{{ "%.2f"|format(data.test_metrics.max_drawdown * 100) }}%</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{{ "%.2f"|format(data.test_metrics.win_rate * 100) }}%</td>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{{ data.test_metrics.trades }}</td>
                </tr>
            </table>
            
            <h2>Benchmark Comparison</h2>
            <div class="benchmark-comparison">
                {% for name, metrics in data.benchmark_metrics.items() %}
                <div class="benchmark-item">
                    <h3>{{ name|title }}</h3>
                    <p><strong>PnL:</strong> ${{ "%.2f"|format(metrics.pnl) }}</p>
                    <p><strong>Sharpe Ratio:</strong> {{ "%.2f"|format(metrics.sharpe_ratio) }}</p>
                    <p><strong>Difference:</strong> 
                        <span class="{{ 'positive' if data.test_metrics.pnl > metrics.pnl else 'negative' }}">
                            ${{ "%.2f"|format(data.test_metrics.pnl - metrics.pnl) }}
                            ({{ "%.2f"|format((data.test_metrics.pnl / metrics.pnl - 1) * 100 if metrics.pnl != 0 else 0) }}%)
                        </span>
                    </p>
                </div>
                {% endfor %}
            </div>
            
            <h2>Visualizations</h2>
            
            <div class="plot-container">
                <h3>Benchmark Comparison</h3>
                <img src="{{ data.plots.benchmark_comparison }}" alt="Benchmark Comparison">
            </div>
            
            <div class="plot-container">
                <h3>Key Metrics</h3>
                <img src="{{ data.plots.metrics_summary }}" alt="Metrics Summary">
            </div>
            
            {% if data.plots.cv_comparison %}
            <div class="plot-container">
                <h3>Cross-Validation Results</h3>
                <img src="{{ data.plots.cv_comparison }}" alt="CV Comparison">
            </div>
            {% endif %}
            
            <h2>Model Configuration</h2>
            <table class="metrics-table">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {% for key, value in data.best_params.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Cross-Validation Summary</h2>
            <table class="metrics-table">
                <tr>
                    <th>Fold</th>
                    <th>PnL</th>
                    <th>Sharpe Ratio</th>
                    <th>Win Rate</th>
                </tr>
                {% for result in data.cv_results %}
                {% if 'error' not in result %}
                <tr>
                    <td>{{ result.fold + 1 }}</td>
                    <td>${{ "%.2f"|format(result.val_metrics.pnl) }}</td>
                    <td>{{ "%.2f"|format(result.val_metrics.sharpe_ratio) }}</td>
                    <td>{{ "%.2f"|format(result.val_metrics.win_rate * 100) }}%</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        # Create Jinja2 environment and template
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        # Render template with data
        html_content = template.render(data=data)
        
        # Save to file
        report_path = os.path.join(self.reports_dir, f"backtest_report_{timestamp}.html")
        with open(report_path, "w") as f:
            f.write(html_content)
            
        return report_path
    
    def _generate_markdown_report(self, data: Dict[str, Any], timestamp: str) -> str:
        """
        Generate Markdown report.
        
        Args:
            data: Dictionary containing report data
            timestamp: Timestamp string for the report
            
        Returns:
            str: Path to the generated Markdown report
        """
        # Create markdown content
        md_content = f"""
        # Backtesting Report - {data['asset']}
        
        **Period:** {data['period']}  
        **Generated:** {data['timestamp']}
        
        ## Performance Summary
        
        | Metric | Value |
        |--------|-------|
        | PnL | ${data['test_metrics']['pnl']:.2f} |
        | PnL (%) | {data['test_metrics']['pnl_percentage']:.2f}% |
        | Sharpe Ratio | {data['test_metrics']['sharpe_ratio']:.2f} |
        | Max Drawdown | {data['test_metrics']['max_drawdown'] * 100:.2f}% |
        | Win Rate | {data['test_metrics']['win_rate'] * 100:.2f}% |
        | Total Trades | {data['test_metrics']['trades']} |
        
        ## Benchmark Comparison
        
        """
        
        # Add benchmark comparison
        for name, metrics in data['benchmark_metrics'].items():
            diff = data['test_metrics']['pnl'] - metrics['pnl']
            pct_diff = (data['test_metrics']['pnl'] / metrics['pnl'] - 1) * 100 if metrics['pnl'] != 0 else 0
            md_content += f"""
        ### {name.title()}
        
        - PnL: ${metrics['pnl']:.2f}
        - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        - Difference: ${diff:.2f} ({pct_diff:.2f}%)
        
        """
        
        # Add visualizations
        md_content += """
        ## Visualizations
        
        ### Benchmark Comparison
        
        ![Benchmark Comparison](../plots/benchmark_comparison.png)
        
        ### Key Metrics
        
        ![Metrics Summary](../plots/metrics_summary.png)
        
        """
        
        if data['plots']['cv_comparison']:
            md_content += """
        ### Cross-Validation Results
        
        ![CV Comparison](../plots/cv_comparison.png)
        
        """
        
        # Add model configuration
        md_content += """
        ## Model Configuration
        
        | Parameter | Value |
        |-----------|-------|
        """
        
        for key, value in data['best_params'].items():
            md_content += f"| {key} | {value} |\n"
        
        # Add cross-validation summary
        md_content += """
        ## Cross-Validation Summary
        
        | Fold | PnL | Sharpe Ratio | Win Rate |
        |------|-----|--------------|----------|
        """
        
        for result in data['cv_results']:
            if 'error' not in result:
                md_content += f"| {result['fold'] + 1} | ${result['val_metrics']['pnl']:.2f} | {result['val_metrics']['sharpe_ratio']:.2f} | {result['val_metrics']['win_rate'] * 100:.2f}% |\n"
        
        # Save to file
        report_path = os.path.join(self.reports_dir, f"backtest_report_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(md_content)
            
        return report_path
    
    def _generate_pdf_report(self, data: Dict[str, Any], timestamp: str) -> str:
        """
        Generate PDF report (via HTML).
        
        Args:
            data: Dictionary containing report data
            timestamp: Timestamp string for the report
            
        Returns:
            str: Path to the generated PDF report
        """
        # First generate HTML report
        html_path = self._generate_html_report(data, timestamp)
        
        # For a real implementation, convert HTML to PDF using a library like weasyprint
        # For now, we'll just copy the HTML file and rename it
        pdf_path = os.path.join(self.reports_dir, f"backtest_report_{timestamp}.pdf")
        shutil.copy(html_path, pdf_path)
        
        logger.info("Note: PDF generation is a placeholder. In a real implementation, use a HTML-to-PDF converter.")
        
        return pdf_path
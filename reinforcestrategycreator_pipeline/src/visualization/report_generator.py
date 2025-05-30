"""Report generation module for model evaluation results.

This module provides the ReportGenerator class for creating comprehensive
evaluation reports in various formats (Markdown, HTML, PDF).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import markdown
import pdfkit  # Optional dependency for PDF generation

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generator for comprehensive evaluation reports.
    
    Provides methods to create evaluation reports in various formats:
    - Markdown reports
    - HTML reports with styling
    - PDF reports (requires wkhtmltopdf)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the report generator.
        
        Args:
            config: Configuration dictionary with report parameters
        """
        self.config = config or {}
        
        # Template directory
        self.template_dir = self.config.get("template_dir", None)
        if self.template_dir:
            self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        else:
            self.jinja_env = None
        
        # PDF generation options
        self.pdf_options = self.config.get("pdf_options", {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        })
        
        # Default templates (if no external templates provided)
        self.default_templates = {
            'markdown': self._get_default_markdown_template(),
            'html': self._get_default_html_template()
        }
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        format_type: str = "markdown",
        template_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        include_visualizations: bool = True,
        visualization_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate evaluation report in specified format.
        
        Args:
            evaluation_results: Complete evaluation results dictionary
            format_type: Report format ('markdown', 'html', 'pdf')
            template_name: Name of custom template to use
            output_path: Path to save the report
            include_visualizations: Whether to include visualization references
            visualization_paths: Dictionary mapping visualization names to file paths
            
        Returns:
            Report content as string
        """
        logger.info(f"Generating {format_type} report")
        
        # Prepare context for template
        context = self._prepare_template_context(
            evaluation_results, 
            include_visualizations, 
            visualization_paths
        )
        
        # Generate report based on format
        if format_type == "markdown":
            report_content = self._generate_markdown_report(context, template_name)
        elif format_type == "html":
            report_content = self._generate_html_report(context, template_name)
        elif format_type == "pdf":
            # Generate HTML first, then convert to PDF
            html_content = self._generate_html_report(context, template_name)
            report_content = self._generate_pdf_from_html(html_content, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        # Save report if output path provided
        if output_path and format_type != "pdf":  # PDF is saved during generation
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Saved {format_type} report to {output_path}")
        
        return report_content
    
    def _prepare_template_context(
        self,
        evaluation_results: Dict[str, Any],
        include_visualizations: bool,
        visualization_paths: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Prepare context dictionary for template rendering."""
        context = {
            'evaluation': evaluation_results,
            'model': evaluation_results.get('model', {}),
            'data': evaluation_results.get('data', {}),
            'metrics': evaluation_results.get('metrics', {}),
            'benchmarks': evaluation_results.get('benchmarks', {}),
            'relative_performance': evaluation_results.get('relative_performance', {}),
            'timestamp': evaluation_results.get('timestamp', datetime.now().isoformat()),
            'evaluation_id': evaluation_results.get('evaluation_id', 'N/A'),
            'evaluation_name': evaluation_results.get('evaluation_name', 'Model Evaluation'),
            'include_visualizations': include_visualizations,
            'visualization_paths': visualization_paths or {},
            
            # Formatted values for display
            'formatted_metrics': self._format_metrics_for_display(
                evaluation_results.get('metrics', {})
            ),
            'formatted_benchmarks': self._format_benchmarks_for_display(
                evaluation_results.get('benchmarks', {})
            )
        }
        
        return context
    
    def _format_metrics_for_display(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Format metrics for better display in reports."""
        formatted = {}
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if metric in ['win_rate']:
                    formatted[metric] = f"{value:.1%}"
                elif metric in ['pnl_percentage', 'total_return']:
                    formatted[metric] = f"{value:.2f}%"
                elif metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                    formatted[metric] = f"{value:.4f}"
                elif metric in ['max_drawdown', 'volatility']:
                    formatted[metric] = f"{value:.4f}"
                else:
                    formatted[metric] = f"{value:.2f}"
            else:
                formatted[metric] = str(value)
        
        return formatted
    
    def _format_benchmarks_for_display(
        self, 
        benchmarks: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, str]]:
        """Format benchmark metrics for display."""
        formatted = {}
        
        for strategy, metrics in benchmarks.items():
            formatted[strategy] = self._format_metrics_for_display(metrics)
        
        return formatted
    
    def _generate_markdown_report(
        self,
        context: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> str:
        """Generate Markdown format report."""
        # Use custom template if available
        if template_name and self.jinja_env:
            try:
                template = self.jinja_env.get_template(template_name)
                return template.render(**context)
            except Exception as e:
                logger.warning(f"Failed to load custom template: {e}. Using default.")
        
        # Use default template
        template = Template(self.default_templates['markdown'])
        return template.render(**context)
    
    def _generate_html_report(
        self,
        context: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> str:
        """Generate HTML format report."""
        # Use custom template if available
        if template_name and self.jinja_env:
            try:
                template = self.jinja_env.get_template(template_name)
                return template.render(**context)
            except Exception as e:
                logger.warning(f"Failed to load custom template: {e}. Using default.")
        
        # Use default template
        template = Template(self.default_templates['html'])
        return template.render(**context)
    
    def _generate_pdf_from_html(
        self,
        html_content: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Generate PDF from HTML content using wkhtmltopdf."""
        try:
            if not output_path:
                output_path = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate PDF
            pdfkit.from_string(html_content, str(output_path), options=self.pdf_options)
            
            logger.info(f"Generated PDF report: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            logger.info("Make sure wkhtmltopdf is installed: https://wkhtmltopdf.org/")
            raise
    
    def _get_default_markdown_template(self) -> str:
        """Get default Markdown report template."""
        return '''# {{ evaluation_name }}

**Evaluation ID:** {{ evaluation_id }}  
**Generated:** {{ timestamp }}

## Executive Summary

This report presents the evaluation results for model **{{ model.id }}** (version {{ model.version }}).

### Key Performance Metrics

| Metric | Value |
|--------|-------|
{% for metric, value in formatted_metrics.items() %}
| {{ metric.replace('_', ' ').title() }} | {{ value }} |
{% endfor %}

## Model Information

- **Model ID:** {{ model.id }}
- **Version:** {{ model.version }}
- **Type:** {{ model.type }}
- **Hyperparameters:**
{% for param, value in model.hyperparameters.items() %}
  - {{ param }}: {{ value }}
{% endfor %}

## Data Information

- **Source:** {{ data.source_id }}
- **Version:** {{ data.version|default('latest') }}
- **Shape:** {{ data.shape }}
- **Columns:** {{ data.columns|length if data.columns else 'N/A' }}

## Performance Analysis

### Detailed Metrics

The model achieved the following performance metrics:

{% for metric, value in formatted_metrics.items() %}
- **{{ metric.replace('_', ' ').title() }}:** {{ value }}
{% endfor %}

{% if benchmarks %}
## Benchmark Comparison

### Benchmark Performance

| Strategy | PnL % | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|----------|-------|--------------|--------------|----------|--------|
{% for strategy, metrics in formatted_benchmarks.items() %}
| {{ strategy.replace('_', ' ').title() }} | {{ metrics.pnl_percentage|default('N/A') }} | {{ metrics.sharpe_ratio|default('N/A') }} | {{ metrics.max_drawdown|default('N/A') }} | {{ metrics.win_rate|default('N/A') }} | {{ metrics.trades_count|default('N/A') }} |
{% endfor %}

### Relative Performance

{% for strategy, perf in relative_performance.items() %}
**vs {{ strategy.replace('_', ' ').title() }}:**
- PnL Difference: {{ perf.absolute_difference|round(2) }} ({{ perf.percentage_difference|round(2) }}%)
- Sharpe Ratio Difference: {{ perf.sharpe_ratio_difference|round(4) }}

{% endfor %}
{% endif %}

{% if include_visualizations and visualization_paths %}
## Visualizations

{% for viz_name, viz_path in visualization_paths.items() %}
### {{ viz_name.replace('_', ' ').title() }}
![{{ viz_name }}]({{ viz_path }})

{% endfor %}
{% endif %}

## Conclusions

Based on the evaluation results:

1. The model shows {{ 'positive' if metrics.pnl_percentage > 0 else 'negative' }} returns with a PnL of {{ formatted_metrics.pnl_percentage }}.
2. Risk-adjusted performance (Sharpe ratio) is {{ formatted_metrics.sharpe_ratio }}.
3. Maximum drawdown observed was {{ formatted_metrics.max_drawdown }}.
{% if benchmarks %}
4. Compared to benchmarks, the model {{ 'outperforms' if relative_performance and (relative_performance.values()|list)[0]['percentage_difference'] > 0 else 'underperforms' }} in terms of returns.
{% endif %}

---
*Report generated automatically by the Model Evaluation Pipeline*
'''
    
    def _get_default_html_template(self) -> str:
        """Get default HTML report template."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ evaluation_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        .section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .info-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        
        .info-card h3 {
            margin-top: 0;
            color: #3498db;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .positive {
            color: #27ae60;
        }
        
        .negative {
            color: #e74c3c;
        }
        
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        @media print {
            body {
                background-color: white;
            }
            .section {
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
    </style>
</head>
<body>
    <h1>{{ evaluation_name }}</h1>
    
    <div class="section">
        <p><strong>Evaluation ID:</strong> {{ evaluation_id }}</p>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents the evaluation results for model <strong>{{ model.id }}</strong> (version {{ model.version }}).</p>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>Returns</h3>
                <p class="metric-value {{ 'positive' if metrics.pnl_percentage > 0 else 'negative' }}">
                    {{ formatted_metrics.pnl_percentage }}
                </p>
            </div>
            <div class="info-card">
                <h3>Sharpe Ratio</h3>
                <p class="metric-value">{{ formatted_metrics.sharpe_ratio }}</p>
            </div>
            <div class="info-card">
                <h3>Max Drawdown</h3>
                <p class="metric-value negative">{{ formatted_metrics.max_drawdown }}</p>
            </div>
            <div class="info-card">
                <h3>Win Rate</h3>
                <p class="metric-value">{{ formatted_metrics.win_rate|default('N/A') }}</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Model Information</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Model ID</td>
                <td>{{ model.id }}</td>
            </tr>
            <tr>
                <td>Version</td>
                <td>{{ model.version }}</td>
            </tr>
            <tr>
                <td>Type</td>
                <td>{{ model.type }}</td>
            </tr>
            {% for param, value in model.hyperparameters.items() %}
            <tr>
                <td>{{ param }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {% for metric, value in formatted_metrics.items() %}
            <tr>
                <td>{{ metric.replace('_', ' ').title() }}</td>
                <td class="metric-value">{{ value }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    {% if benchmarks %}
    <div class="section">
        <h2>Benchmark Comparison</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>PnL %</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Trades</th>
            </tr>
            {% for strategy, metrics in formatted_benchmarks.items() %}
            <tr>
                <td>{{ strategy.replace('_', ' ').title() }}</td>
                <td>{{ metrics.pnl_percentage|default('N/A') }}</td>
                <td>{{ metrics.sharpe_ratio|default('N/A') }}</td>
                <td>{{ metrics.max_drawdown|default('N/A') }}</td>
                <td>{{ metrics.win_rate|default('N/A') }}</td>
                <td>{{ metrics.trades_count|default('N/A') }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if include_visualizations and visualization_paths %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for viz_name, viz_path in visualization_paths.items() %}
        <div class="visualization">
            <h3>{{ viz_name.replace('_', ' ').title() }}</h3>
            <img src="{{ viz_path }}" alt="{{ viz_name }}">
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Report generated automatically by the Model Evaluation Pipeline</p>
        <p>{{ timestamp }}</p>
    </div>
</body>
</html>
'''
    
    def create_summary_report(
        self,
        evaluation_results_list: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        format_type: str = "markdown"
    ) -> str:
        """Create a summary report comparing multiple evaluation results.
        
        Args:
            evaluation_results_list: List of evaluation results to compare
            output_path: Path to save the report
            format_type: Report format ('markdown' or 'html')
            
        Returns:
            Summary report content
        """
        # Prepare comparison data
        comparison_data = []
        
        for result in evaluation_results_list:
            model_info = result.get('model', {})
            metrics = result.get('metrics', {})
            
            comparison_data.append({
                'model_id': model_info.get('id', 'Unknown'),
                'version': model_info.get('version', 'N/A'),
                'timestamp': result.get('timestamp', 'N/A'),
                'metrics': metrics,
                'formatted_metrics': self._format_metrics_for_display(metrics)
            })
        
        # Sort by performance (e.g., Sharpe ratio)
        comparison_data.sort(
            key=lambda x: x['metrics'].get('sharpe_ratio', 0), 
            reverse=True
        )
        
        # Generate summary report
        if format_type == "markdown":
            report_content = self._generate_markdown_summary(comparison_data)
        elif format_type == "html":
            report_content = self._generate_html_summary(comparison_data)
        else:
            raise ValueError(f"Unsupported format for summary report: {format_type}")
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Saved summary report to {output_path}")
        
        return report_content
    
    def _generate_markdown_summary(self, comparison_data: List[Dict]) -> str:
        """Generate Markdown summary report."""
        lines = ["# Model Evaluation Summary Report"]
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Total Models Evaluated:** {len(comparison_data)}")
        lines.append("\n## Performance Comparison")
        
        # Create comparison table
        lines.append("\n| Model | Version | PnL % | Sharpe | Max DD | Win Rate |")
        lines.append("|-------|---------|-------|--------|--------|----------|")
        
        for data in comparison_data:
            metrics = data['formatted_metrics']
            lines.append(
                f"| {data['model_id']} | {data['version']} | "
                f"{metrics.get('pnl_percentage', 'N/A')} | "
                f"{metrics.get('sharpe_ratio', 'N/A')} | "
                f"{metrics.get('max_drawdown', 'N/A')} | "
                f"{metrics.get('win_rate', 'N/A')} |"
            )
        
        # Best performers
        lines.append("\n## Best Performers")
        
        if comparison_data:
            best_returns = max(comparison_data, 
                             key=lambda x: x['metrics'].get('pnl_percentage', float('-inf')))
            best_sharpe = max(comparison_data, 
                            key=lambda x: x['metrics'].get('sharpe_ratio', float('-inf')))
            
            lines.append(f"\n- **Highest Returns:** {best_returns['model_id']} "
                        f"({best_returns['formatted_metrics'].get('pnl_percentage', 'N/A')})")
            lines.append(f"- **Best Risk-Adjusted:** {best_sharpe['model_id']} "
                        f"(Sharpe: {best_sharpe['formatted_metrics'].get('sharpe_ratio', 'N/A')})")
        
        return "\n".join(lines)
    
    def _generate_html_summary(self, comparison_data: List[Dict]) -> str:
        """Generate HTML summary report."""
        # For brevity, using a simple HTML structure
        # In production, this would use a proper template
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Summary Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Models Evaluated:</strong> {len(comparison_data)}</p>
            
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Version</th>
                    <th>PnL %</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                </tr>
        """
        
        for data in comparison_data:
            metrics = data['formatted_metrics']
            html += f"""
                <tr>
                    <td>{data['model_id']}</td>
                    <td>{data['version']}</td>
                    <td>{metrics.get('pnl_percentage', 'N/A')}</td>
                    <td>{metrics.get('sharpe_ratio', 'N/A')}</td>
                    <td>{metrics.get('max_drawdown', 'N/A')}</td>
                    <td>{metrics.get('win_rate', 'N/A')}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
# Visualization and Reporting Module: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Visualization and Reporting Module of the Trading Model Optimization Pipeline. This component provides comprehensive visualization tools, report generation capabilities, and interactive dashboards to analyze model performance, trading strategy results, and system metrics.

## 2. Component Responsibilities

The Visualization and Reporting Module is responsible for:

- Generating visualizations of model performance metrics
- Creating trading strategy performance reports
- Producing comparative visualizations across multiple models and strategies
- Providing interactive dashboards for real-time monitoring
- Supporting automated report generation for scheduled evaluation
- Exporting visualizations and reports in various formats
- Capturing training and optimization progress
- Visualizing feature importance and model interpretability metrics

## 3. Architecture

### 3.1 Overall Architecture

The Visualization and Reporting Module follows a layered architecture with separation between data access, visualization generation, report formatting, and delivery mechanisms:

```
┌─────────────────────────────────────────┐
│              Report Manager             │ High-level interface for reports
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────┐   ┌───────────────┐  │
│  │ Visualization │   │  Interactive  │  │ Core components for generating
│  │   Generator   │   │   Dashboard   │  │ visualizations and dashboards
│  └───────────────┘   └───────────────┘  │
│                                         │
│  ┌───────────────┐   ┌───────────────┐  │
│  │Report Template│   │  Exporters    │  │ Report templates and export formats
│  │   Engine      │   │               │  │
│  └───────────────┘   └───────────────┘  │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────┐   ┌───────────────┐  │
│  │  Data Access  │   │   Metrics     │  │ Access to databases and metrics
│  │     Layer     │   │  Calculators  │  │
│  └───────────────┘   └───────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── visualization/
    ├── __init__.py
    ├── manager.py                  # High-level report manager
    ├── visualizers/
    │   ├── __init__.py
    │   ├── base.py                 # Base visualizer
    │   ├── model_performance.py    # Model performance visualizers
    │   ├── training_progress.py    # Training progress visualizers
    │   ├── strategy_performance.py # Trading strategy visualizers  
    │   ├── hyperparameter.py       # Hyperparameter tuning visualizers
    │   ├── feature_importance.py   # Feature importance visualizers
    │   ├── correlation.py          # Correlation visualizers
    │   ├── distributions.py        # Distribution visualizers
    │   └── comparison.py           # Comparative visualizers
    ├── dashboards/
    │   ├── __init__.py
    │   ├── base.py                 # Base dashboard
    │   ├── model_dashboard.py      # Model performance dashboard
    │   ├── strategy_dashboard.py   # Trading strategy dashboard
    │   ├── backtest_dashboard.py   # Backtest results dashboard
    │   ├── optimization_dashboard.py # Optimization progress dashboard
    │   └── components/             # Reusable dashboard components
    │       ├── __init__.py
    │       ├── metrics_card.py     # Key metrics display card
    │       ├── performance_chart.py # Performance chart component
    │       ├── confusion_matrix.py # Confusion matrix component
    │       └── data_table.py       # Data table component
    ├── reports/
    │   ├── __init__.py
    │   ├── base.py                 # Base report
    │   ├── model_report.py         # Model evaluation report
    │   ├── strategy_report.py      # Strategy performance report
    │   ├── optimization_report.py  # Optimization results report
    │   ├── backtest_report.py      # Backtest analysis report
    │   └── comparison_report.py    # Model/strategy comparison report
    ├── templates/
    │   ├── __init__.py
    │   ├── html/                   # HTML report templates
    │   │   ├── base_template.html  # Base HTML template
    │   │   ├── model_report.html   # Model report template
    │   │   ├── strategy_report.html # Strategy report template
    │   │   └── comparison_report.html # Comparison report template
    │   ├── pdf/                    # PDF report templates
    │   │   ├── base_template.py    # Base PDF template
    │   │   ├── model_report.py     # Model report template
    │   │   └── strategy_report.py  # Strategy report template
    │   └── notebook/               # Jupyter notebook templates
    │       ├── model_analysis.ipynb # Model analysis notebook
    │       └── strategy_analysis.ipynb # Strategy analysis notebook
    ├── exporters/
    │   ├── __init__.py
    │   ├── base.py                 # Base exporter
    │   ├── html_exporter.py        # HTML report exporter
    │   ├── pdf_exporter.py         # PDF report exporter
    │   ├── image_exporter.py       # Image exporter (PNG, JPEG)
    │   ├── svg_exporter.py         # SVG exporter
    │   ├── excel_exporter.py       # Excel exporter
    │   └── notebook_exporter.py    # Jupyter notebook exporter
    ├── data/
    │   ├── __init__.py
    │   ├── data_adapter.py         # Abstract data adapter
    │   ├── results_db_adapter.py   # Results database adapter
    │   ├── model_data_loader.py    # Model metrics data loader
    │   └── strategy_data_loader.py # Strategy results data loader
    ├── metrics/
    │   ├── __init__.py
    │   ├── calculator.py           # Metrics calculator
    │   ├── model_metrics.py        # Model-specific metrics
    │   └── strategy_metrics.py     # Strategy-specific metrics
    └── utils/
        ├── __init__.py
        ├── color_schemes.py        # Color schemes for visualizations
        ├── plotting_utils.py       # Common plotting utilities
        ├── formatting.py           # Number/text formatting utilities
        └── templates.py            # Template utilities
```

## 4. Core Components Design

### 4.1 Report Manager

High-level interface for generating reports and visualizations:

```python
# manager.py
from typing import Dict, List, Any, Optional, Union
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from trading_optimization.visualization.visualizers.model_performance import ModelPerformanceVisualizer
from trading_optimization.visualization.visualizers.strategy_performance import StrategyPerformanceVisualizer
from trading_optimization.visualization.visualizers.training_progress import TrainingProgressVisualizer
from trading_optimization.visualization.visualizers.hyperparameter import HyperparameterVisualizer
from trading_optimization.visualization.visualizers.feature_importance import FeatureImportanceVisualizer
from trading_optimization.visualization.visualizers.comparison import ComparisonVisualizer

from trading_optimization.visualization.dashboards.model_dashboard import ModelDashboard
from trading_optimization.visualization.dashboards.strategy_dashboard import StrategyDashboard
from trading_optimization.visualization.dashboards.backtest_dashboard import BacktestDashboard
from trading_optimization.visualization.dashboards.optimization_dashboard import OptimizationDashboard

from trading_optimization.visualization.reports.model_report import ModelReport
from trading_optimization.visualization.reports.strategy_report import StrategyReport
from trading_optimization.visualization.reports.optimization_report import OptimizationReport
from trading_optimization.visualization.reports.backtest_report import BacktestReport
from trading_optimization.visualization.reports.comparison_report import ComparisonReport

from trading_optimization.visualization.data.results_db_adapter import ResultsDBAdapter
from trading_optimization.visualization.data.model_data_loader import ModelDataLoader
from trading_optimization.visualization.data.strategy_data_loader import StrategyDataLoader
from trading_optimization.config import ConfigManager


class VisualizationManager:
    """
    High-level manager for generating visualizations and reports.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualization manager.
        
        Args:
            config: Visualization configuration dictionary
        """
        self.config = config
        
        # Directories
        self.output_dir = config.get('output_dir', './reports')
        self.template_dir = config.get('template_dir', './templates')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create data adapters
        results_db_config = config.get('results_db', {})
        self.results_db_adapter = ResultsDBAdapter(results_db_config)
        
        # Create data loaders
        self.model_data_loader = ModelDataLoader(self.results_db_adapter)
        self.strategy_data_loader = StrategyDataLoader(self.results_db_adapter)
        
        # Create visualizers
        self.model_visualizer = ModelPerformanceVisualizer(config.get('model_viz', {}))
        self.strategy_visualizer = StrategyPerformanceVisualizer(config.get('strategy_viz', {}))
        self.training_visualizer = TrainingProgressVisualizer(config.get('training_viz', {}))
        self.hyperparameter_visualizer = HyperparameterVisualizer(config.get('hyperparameter_viz', {}))
        self.feature_visualizer = FeatureImportanceVisualizer(config.get('feature_viz', {}))
        self.comparison_visualizer = ComparisonVisualizer(config.get('comparison_viz', {}))
        
        # Default report formats
        self.default_formats = config.get('default_formats', ['html', 'pdf'])
        
        # Styling/theme
        self.theme = config.get('theme', 'default')
        
    def generate_model_report(
        self,
        model_id: str,
        formats: Optional[List[str]] = None,
        include_plots: bool = True,
        output_path: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate a comprehensive model evaluation report.
        
        Args:
            model_id: ID of the model to report on
            formats: List of output formats ('html', 'pdf', 'notebook')
            include_plots: Whether to include visualizations
            output_path: Custom output path (default: output_dir/model_reports/[model_id])
            additional_data: Additional data to include
            
        Returns:
            Dictionary mapping formats to file paths
        """
        # Set default formats if not provided
        if formats is None:
            formats = self.default_formats
            
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = os.path.join(self.output_dir, 'model_reports', f"{model_id}-{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load model data
        model_data = self.model_data_loader.load_model_evaluation_results(model_id)
        
        if model_data is None:
            raise ValueError(f"No data found for model ID: {model_id}")
        
        # Create report object
        report = ModelReport(
            model_id=model_id,
            model_data=model_data,
            config=self.config,
            template_dir=self.template_dir
        )
        
        # Generate visualizations if requested
        if include_plots:
            # Performance metrics visualizations
            performance_plots = self.model_visualizer.create_performance_plots(model_data)
            
            # Add feature importance plots if available
            if 'feature_importance' in model_data:
                feature_plots = self.feature_visualizer.create_feature_importance_plot(
                    model_data['feature_importance']
                )
                performance_plots.update(feature_plots)
            
            # Add training progress plots if available
            if 'training_history' in model_data:
                training_plots = self.training_visualizer.create_training_plots(
                    model_data['training_history']
                )
                performance_plots.update(training_plots)
                
            # Add plots to report
            report.add_visualizations(performance_plots)
        
        # Add additional data if provided
        if additional_data:
            report.add_additional_data(additional_data)
        
        # Generate reports in requested formats
        output_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                output_file = os.path.join(output_path, f"{model_id}_report.html")
                report.export_html(output_file)
                output_files['html'] = output_file
                
            elif fmt == 'pdf':
                output_file = os.path.join(output_path, f"{model_id}_report.pdf")
                report.export_pdf(output_file)
                output_files['pdf'] = output_file
                
            elif fmt == 'notebook':
                output_file = os.path.join(output_path, f"{model_id}_analysis.ipynb")
                report.export_notebook(output_file)
                output_files['notebook'] = output_file
        
        return output_files
    
    def generate_strategy_report(
        self,
        strategy_id: str,
        formats: Optional[List[str]] = None,
        include_plots: bool = True,
        output_path: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate a comprehensive trading strategy performance report.
        
        Args:
            strategy_id: ID of the strategy to report on
            formats: List of output formats ('html', 'pdf', 'notebook')
            include_plots: Whether to include visualizations
            output_path: Custom output path (default: output_dir/strategy_reports/[strategy_id])
            additional_data: Additional data to include
            
        Returns:
            Dictionary mapping formats to file paths
        """
        # Set default formats if not provided
        if formats is None:
            formats = self.default_formats
            
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = os.path.join(self.output_dir, 'strategy_reports', f"{strategy_id}-{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load strategy data
        strategy_data = self.strategy_data_loader.load_strategy_results(strategy_id)
        
        if strategy_data is None:
            raise ValueError(f"No data found for strategy ID: {strategy_id}")
        
        # Create report object
        report = StrategyReport(
            strategy_id=strategy_id,
            strategy_data=strategy_data,
            config=self.config,
            template_dir=self.template_dir
        )
        
        # Generate visualizations if requested
        if include_plots:
            # Performance metrics visualizations
            performance_plots = self.strategy_visualizer.create_performance_plots(strategy_data)
            
            # Add plots to report
            report.add_visualizations(performance_plots)
        
        # Add additional data if provided
        if additional_data:
            report.add_additional_data(additional_data)
        
        # Generate reports in requested formats
        output_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                output_file = os.path.join(output_path, f"{strategy_id}_report.html")
                report.export_html(output_file)
                output_files['html'] = output_file
                
            elif fmt == 'pdf':
                output_file = os.path.join(output_path, f"{strategy_id}_report.pdf")
                report.export_pdf(output_file)
                output_files['pdf'] = output_file
                
            elif fmt == 'notebook':
                output_file = os.path.join(output_path, f"{strategy_id}_analysis.ipynb")
                report.export_notebook(output_file)
                output_files['notebook'] = output_file
        
        return output_files
    
    def generate_backtest_report(
        self,
        backtest_id: str,
        formats: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a backtest results report.
        
        Args:
            backtest_id: ID of the backtest to report on
            formats: List of output formats ('html', 'pdf', 'notebook')
            output_path: Custom output path
            
        Returns:
            Dictionary mapping formats to file paths
        """
        # Set default formats if not provided
        if formats is None:
            formats = self.default_formats
            
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = os.path.join(self.output_dir, 'backtest_reports', f"{backtest_id}-{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load backtest data
        backtest_data = self.strategy_data_loader.load_backtest_results(backtest_id)
        
        if backtest_data is None:
            raise ValueError(f"No data found for backtest ID: {backtest_id}")
        
        # Create report object
        report = BacktestReport(
            backtest_id=backtest_id,
            backtest_data=backtest_data,
            config=self.config,
            template_dir=self.template_dir
        )
        
        # Generate visualizations
        performance_plots = self.strategy_visualizer.create_backtest_plots(backtest_data)
        
        # Add plots to report
        report.add_visualizations(performance_plots)
        
        # Generate reports in requested formats
        output_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                output_file = os.path.join(output_path, f"{backtest_id}_report.html")
                report.export_html(output_file)
                output_files['html'] = output_file
                
            elif fmt == 'pdf':
                output_file = os.path.join(output_path, f"{backtest_id}_report.pdf")
                report.export_pdf(output_file)
                output_files['pdf'] = output_file
                
            elif fmt == 'notebook':
                output_file = os.path.join(output_path, f"{backtest_id}_analysis.ipynb")
                report.export_notebook(output_file)
                output_files['notebook'] = output_file
        
        return output_files
    
    def generate_optimization_report(
        self,
        optimization_id: str,
        formats: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate an optimization results report.
        
        Args:
            optimization_id: ID of the optimization run to report on
            formats: List of output formats ('html', 'pdf', 'notebook')
            output_path: Custom output path
            
        Returns:
            Dictionary mapping formats to file paths
        """
        # Set default formats if not provided
        if formats is None:
            formats = self.default_formats
            
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = os.path.join(self.output_dir, 'optimization_reports', f"{optimization_id}-{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load optimization data
        optimization_data = self.model_data_loader.load_optimization_results(optimization_id)
        
        if optimization_data is None:
            raise ValueError(f"No data found for optimization ID: {optimization_id}")
        
        # Create report object
        report = OptimizationReport(
            optimization_id=optimization_id,
            optimization_data=optimization_data,
            config=self.config,
            template_dir=self.template_dir
        )
        
        # Generate visualizations
        hp_plots = self.hyperparameter_visualizer.create_optimization_plots(optimization_data)
        
        # Add plots to report
        report.add_visualizations(hp_plots)
        
        # Generate reports in requested formats
        output_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                output_file = os.path.join(output_path, f"{optimization_id}_report.html")
                report.export_html(output_file)
                output_files['html'] = output_file
                
            elif fmt == 'pdf':
                output_file = os.path.join(output_path, f"{optimization_id}_report.pdf")
                report.export_pdf(output_file)
                output_files['pdf'] = output_file
                
            elif fmt == 'notebook':
                output_file = os.path.join(output_path, f"{optimization_id}_analysis.ipynb")
                report.export_notebook(output_file)
                output_files['notebook'] = output_file
        
        return output_files
    
    def generate_comparison_report(
        self,
        model_ids: List[str] = None,
        strategy_ids: List[str] = None,
        formats: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        comparison_name: str = "comparison"
    ) -> Dict[str, str]:
        """
        Generate a comparison report between multiple models and/or strategies.
        
        Args:
            model_ids: List of model IDs to compare (optional)
            strategy_ids: List of strategy IDs to compare (optional)
            formats: List of output formats ('html', 'pdf', 'notebook')
            output_path: Custom output path
            comparison_name: Name for the comparison report
            
        Returns:
            Dictionary mapping formats to file paths
        """
        if not model_ids and not strategy_ids:
            raise ValueError("Must provide at least one model_id or strategy_id for comparison")
            
        # Set default formats if not provided
        if formats is None:
            formats = self.default_formats
            
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = os.path.join(self.output_dir, 'comparison_reports', f"{comparison_name}-{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load model data if provided
        model_data_list = []
        if model_ids:
            for model_id in model_ids:
                model_data = self.model_data_loader.load_model_evaluation_results(model_id)
                if model_data:
                    model_data_list.append(model_data)
        
        # Load strategy data if provided
        strategy_data_list = []
        if strategy_ids:
            for strategy_id in strategy_ids:
                strategy_data = self.strategy_data_loader.load_strategy_results(strategy_id)
                if strategy_data:
                    strategy_data_list.append(strategy_data)
        
        # Create report object
        report = ComparisonReport(
            model_data_list=model_data_list,
            strategy_data_list=strategy_data_list,
            comparison_name=comparison_name,
            config=self.config,
            template_dir=self.template_dir
        )
        
        # Generate comparison visualizations
        comparison_plots = {}
        
        # Model comparison visualizations if we have models
        if model_data_list:
            model_comparison_plots = self.comparison_visualizer.create_model_comparison_plots(model_data_list)
            comparison_plots.update(model_comparison_plots)
            
        # Strategy comparison visualizations if we have strategies
        if strategy_data_list:
            strategy_comparison_plots = self.comparison_visualizer.create_strategy_comparison_plots(strategy_data_list)
            comparison_plots.update(strategy_comparison_plots)
            
        # Add plots to report
        report.add_visualizations(comparison_plots)
        
        # Generate reports in requested formats
        output_files = {}
        
        for fmt in formats:
            if fmt == 'html':
                output_file = os.path.join(output_path, f"{comparison_name}_report.html")
                report.export_html(output_file)
                output_files['html'] = output_file
                
            elif fmt == 'pdf':
                output_file = os.path.join(output_path, f"{comparison_name}_report.pdf")
                report.export_pdf(output_file)
                output_files['pdf'] = output_file
                
            elif fmt == 'notebook':
                output_file = os.path.join(output_path, f"{comparison_name}_analysis.ipynb")
                report.export_notebook(output_file)
                output_files['notebook'] = output_file
        
        return output_files
    
    def create_model_dashboard(
        self,
        model_id: str,
        port: int = 8050, 
        host: str = '0.0.0.0',
        mode: str = 'inline'
    ) -> Any:
        """
        Create and serve an interactive model dashboard.
        
        Args:
            model_id: Model ID to create dashboard for
            port: Port to serve dashboard on
            host: Host to serve dashboard on
            mode: Dashboard mode ('serve', 'inline', 'jupyter')
            
        Returns:
            Dashboard object
        """
        # Load model data
        model_data = self.model_data_loader.load_model_evaluation_results(model_id)
        
        if model_data is None:
            raise ValueError(f"No data found for model ID: {model_id}")
        
        # Create dashboard object
        dashboard = ModelDashboard(
            model_id=model_id,
            model_data=model_data,
            config=self.config
        )
        
        # Serve, display, or return dashboard based on mode
        if mode == 'serve':
            dashboard.serve(host=host, port=port)
            return dashboard
        elif mode == 'inline':
            return dashboard.display_inline()
        elif mode == 'jupyter':
            return dashboard.display_jupyter()
        else:
            return dashboard

    def create_strategy_dashboard(
        self,
        strategy_id: str,
        port: int = 8051, 
        host: str = '0.0.0.0',
        mode: str = 'inline'
    ) -> Any:
        """
        Create and serve an interactive strategy dashboard.
        
        Args:
            strategy_id: Strategy ID to create dashboard for
            port: Port to serve dashboard on
            host: Host to serve dashboard on
            mode: Dashboard mode ('serve', 'inline', 'jupyter')
            
        Returns:
            Dashboard object
        """
        # Load strategy data
        strategy_data = self.strategy_data_loader.load_strategy_results(strategy_id)
        
        if strategy_data is None:
            raise ValueError(f"No data found for strategy ID: {strategy_id}")
        
        # Create dashboard object
        dashboard = StrategyDashboard(
            strategy_id=strategy_id,
            strategy_data=strategy_data,
            config=self.config
        )
        
        # Serve, display, or return dashboard based on mode
        if mode == 'serve':
            dashboard.serve(host=host, port=port)
            return dashboard
        elif mode == 'inline':
            return dashboard.display_inline()
        elif mode == 'jupyter':
            return dashboard.display_jupyter()
        else:
            return dashboard
    
    def create_backtest_dashboard(
        self,
        backtest_id: str,
        port: int = 8052, 
        host: str = '0.0.0.0',
        mode: str = 'inline'
    ) -> Any:
        """
        Create and serve an interactive backtest dashboard.
        
        Args:
            backtest_id: Backtest ID to create dashboard for
            port: Port to serve dashboard on
            host: Host to serve dashboard on
            mode: Dashboard mode ('serve', 'inline', 'jupyter')
            
        Returns:
            Dashboard object
        """
        # Load backtest data
        backtest_data = self.strategy_data_loader.load_backtest_results(backtest_id)
        
        if backtest_data is None:
            raise ValueError(f"No data found for backtest ID: {backtest_id}")
        
        # Create dashboard object
        dashboard = BacktestDashboard(
            backtest_id=backtest_id,
            backtest_data=backtest_data,
            config=self.config
        )
        
        # Serve, display, or return dashboard based on mode
        if mode == 'serve':
            dashboard.serve(host=host, port=port)
            return dashboard
        elif mode == 'inline':
            return dashboard.display_inline()
        elif mode == 'jupyter':
            return dashboard.display_jupyter()
        else:
            return dashboard
    
    def create_optimization_dashboard(
        self,
        optimization_id: str,
        port: int = 8053, 
        host: str = '0.0.0.0',
        mode: str = 'inline'
    ) -> Any:
        """
        Create and serve an interactive hyperparameter optimization dashboard.
        
        Args:
            optimization_id: Optimization run ID to create dashboard for
            port: Port to serve dashboard on
            host: Host to serve dashboard on
            mode: Dashboard mode ('serve', 'inline', 'jupyter')
            
        Returns:
            Dashboard object
        """
        # Load optimization data
        optimization_data = self.model_data_loader.load_optimization_results(optimization_id)
        
        if optimization_data is None:
            raise ValueError(f"No data found for optimization ID: {optimization_id}")
        
        # Create dashboard object
        dashboard = OptimizationDashboard(
            optimization_id=optimization_id,
            optimization_data=optimization_data,
            config=self.config
        )
        
        # Serve, display, or return dashboard based on mode
        if mode == 'serve':
            dashboard.serve(host=host, port=port)
            return dashboard
        elif mode == 'inline':
            return dashboard.display_inline()
        elif mode == 'jupyter':
            return dashboard.display_jupyter()
        else:
            return dashboard
            
    def export_visualization(
        self,
        visualization_func: callable,
        output_path: str,
        format: str = 'png',
        **viz_params
    ) -> str:
        """
        Export a single visualization from a visualizer function.
        
        Args:
            visualization_func: Function that generates a visualization
            output_path: Path to save visualization to
            format: Export format ('png', 'svg', 'pdf')
            **viz_params: Parameters to pass to visualization function
            
        Returns:
            Path to saved visualization
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate visualization
        fig = visualization_func(**viz_params)
        
        # Export to requested format
        if format == 'png':
            fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        elif format == 'svg':
            fig.savefig(output_path, format='svg', bbox_inches='tight')
        elif format == 'pdf':
            fig.savefig(output_path, format='pdf', bbox_inches='tight')
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        return output_path
        
    def schedule_report_generation(
        self,
        report_type: str,
        report_id: str,
        schedule: str,
        formats: Optional[List[str]] = None, 
        output_path: Optional[str] = None,
        notification_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule a report to be generated on a recurring basis.
        Note: Requires a task scheduler to be set up.
        
        Args:
            report_type: Type of report to generate ('model', 'strategy', 'backtest', 'optimization')
            report_id: ID to generate report for
            schedule: Cron-style schedule string (e.g., '0 0 * * *' for daily at midnight)
            formats: List of output formats
            output_path: Custom output path
            notification_email: Email to send notification to
            
        Returns:
            Dictionary with scheduled task details
        """
        # This is a placeholder that would integrate with a task scheduler
        # Implementation depends on the specific task scheduling system used
        
        scheduled_task = {
            'id': f"report_schedule_{report_type}_{report_id}",
            'report_type': report_type,
            'report_id': report_id,
            'schedule': schedule,
            'formats': formats,
            'output_path': output_path,
            'notification_email': notification_email,
            'created_at': datetime.now().isoformat()
        }
        
        # In a real implementation, we would save this to a database or task scheduler
        
        return scheduled_task
```

### 4.2 Model Performance Visualizer

Creates visualizations for model performance metrics:

```python
# visualizers/model_performance.py
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from trading_optimization.visualization.visualizers.base import BaseVisualizer
from trading_optimization.visualization.utils.color_schemes import get_color_scheme
from trading_optimization.visualization.utils.plotting_utils import (
    add_plot_styling, 
    create_subplot_grid,
    plot_confusion_matrix
)

class ModelPerformanceVisualizer(BaseVisualizer):
    """
    Creates visualizations for model performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer.
        
        Args:
            config: Visualizer configuration
        """
        super().__init__(config)
        self.fig_size = config.get('fig_size', (10, 6))
        self.dpi = config.get('dpi', 100)
    
    def create_performance_plots(
        self, 
        model_data: Dict[str, Any]
    ) -> Dict[str, Figure]:
        """
        Create a set of performance plots for a model.
        
        Args:
            model_data: Model evaluation data
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        plots = {}
        
        # Extract metrics
        metrics = model_data.get('metrics', {})
        
        # Create prediction vs actual plot if time series data present
        if 'predictions' in model_data and 'actual' in model_data:
            predictions = model_data['predictions']
            actual = model_data['actual']
            
            # Check if we have timestamps
            if 'timestamps' in model_data:
                timestamps = model_data['timestamps']
                
                # Create time series prediction vs actual plot
                plots['prediction_vs_actual'] = self.plot_predictions_vs_actual_timeseries(
                    timestamps=timestamps,
                    actual=actual,
                    predictions=predictions,
                    title=f"Model {model_data.get('model_id', 'Unknown')} - Predictions vs Actual"
                )
            else:
                # Create scatter prediction vs actual plot
                plots['prediction_vs_actual'] = self.plot_predictions_vs_actual_scatter(
                    actual=actual,
                    predictions=predictions,
                    title=f"Model {model_data.get('model_id', 'Unknown')} - Predictions vs Actual"
                )
        
        # Create regression metrics plot if available
        if 'mae' in metrics or 'mse' in metrics or 'rmse' in metrics or 'r2' in metrics:
            plots['regression_metrics'] = self.plot_regression_metrics(metrics)
        
        # Create classification metrics plot if available
        if 'accuracy' in metrics or 'precision' in metrics or 'recall' in metrics or 'f1' in metrics:
            plots['classification_metrics'] = self.plot_classification_metrics(metrics)
        
        # Create confusion matrix if available
        if 'confusion_matrix' in model_data:
            plots['confusion_matrix'] = self.plot_confusion_matrix(
                model_data['confusion_matrix'],
                model_data.get('class_labels', None)
            )
        elif 'y_true' in model_data and 'y_pred' in model_data:
            # Compute confusion matrix if we have true values and predictions
            cm = confusion_matrix(model_data['y_true'], model_data['y_pred'])
            plots['confusion_matrix'] = self.plot_confusion_matrix(
                cm,
                model_data.get('class_labels', None)
            )
        
        # Create ROC curve if available
        if 'fpr' in model_data and 'tpr' in model_data and 'roc_auc' in metrics:
            plots['roc_curve'] = self.plot_roc_curve(
                fpr=model_data['fpr'],
                tpr=model_data['tpr'],
                roc_auc=metrics['roc_auc']
            )
        
        # Create residual plot if regression model
        if 'predictions' in model_data and 'actual' in model_data:
            plots['residuals'] = self.plot_residuals(
                actual=model_data['actual'],
                predictions=model_data['predictions']
            )
        
        # Create error distribution plot if regression model
        if 'predictions' in model_data and 'actual' in model_data:
            plots['error_distribution'] = self.plot_error_distribution(
                actual=model_data['actual'],
                predictions=model_data['predictions']
            )
        
        return plots
    
    def plot_predictions_vs_actual_timeseries(
        self,
        timestamps: List[Union[str, pd.Timestamp]],
        actual: List[float],
        predictions: List[float],
        title: str = 'Predictions vs Actual (Time Series)'
    ) -> Figure:
        """
        Create a time series plot of predictions vs actual values.
        
        Args:
            timestamps: List of timestamps
            actual: List of actual values
            predictions: List of predicted values
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Convert timestamps to pandas datetime if they are strings
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot actual values
        ax.plot(timestamps, actual, label='Actual', color=self.color_scheme['actual'], linewidth=2)
        
        # Plot predicted values
        ax.plot(timestamps, predictions, label='Predicted', color=self.color_scheme['predicted'], 
                linewidth=2, linestyle='--')
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_predictions_vs_actual_scatter(
        self,
        actual: List[float],
        predictions: List[float],
        title: str = 'Predictions vs Actual (Scatter)'
    ) -> Figure:
        """
        Create a scatter plot of predictions vs actual values.
        
        Args:
            actual: List of actual values
            predictions: List of predicted values
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create scatter plot
        ax.scatter(actual, predictions, alpha=0.5, color=self.color_scheme['scatter'])
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predictions))
        max_val = max(max(actual), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add R² value
        r2 = np.corrcoef(actual, predictions)[0, 1] ** 2
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_regression_metrics(
        self, 
        metrics: Dict[str, float]
    ) -> Figure:
        """
        Create a bar plot of regression metrics.
        
        Args:
            metrics: Dictionary of regression metrics
            
        Returns:
            Matplotlib Figure object
        """
        # Extract regression metrics
        regression_metrics = {}
        
        for metric in ['mae', 'mse', 'rmse', 'r2', 'mape']:
            if metric in metrics:
                # Special case for r2 to display as percentage
                if metric == 'r2':
                    regression_metrics[metric.upper()] = metrics[metric]
                else:
                    regression_metrics[metric.upper()] = metrics[metric]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create bar colors
        colors = [self.color_scheme['metrics_1'], self.color_scheme['metrics_2'], 
                  self.color_scheme['metrics_3'], self.color_scheme['metrics_4']]
        
        # Create bar plot
        bars = ax.bar(regression_metrics.keys(), regression_metrics.values(), color=colors)
        
        # Add styling
        ax.set_title('Regression Metrics', fontsize=16)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height < 0.01:  # For small values
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_classification_metrics(
        self, 
        metrics: Dict[str, float]
    ) -> Figure:
        """
        Create a bar plot of classification metrics.
        
        Args:
            metrics: Dictionary of classification metrics
            
        Returns:
            Matplotlib Figure object
        """
        # Extract classification metrics
        classification_metrics = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
            if metric in metrics:
                classification_metrics[metric.capitalize()] = metrics[metric]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create bar colors
        colors = [self.color_scheme['metrics_1'], self.color_scheme['metrics_2'], 
                  self.color_scheme['metrics_3'], self.color_scheme['metrics_4']]
        
        # Create bar plot
        bars = ax.bar(classification_metrics.keys(), classification_metrics.values(), color=colors)
        
        # Add styling
        ax.set_title('Classification Metrics', fontsize=16)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.15)  # Fixed scale from 0 to 1 with space for labels
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_labels: Optional[List[str]] = None
    ) -> Figure:
        """
        Create a confusion matrix plot.
        
        Args:
            cm: Confusion matrix array
            class_labels: List of class labels
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Set default class labels if not provided
        if class_labels is None:
            class_labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Plot confusion matrix using utility function
        plot_confusion_matrix(ax, cm, class_labels, normalize=True,
                              cmap=self.color_scheme.get('cmap', 'Blues'))
        
        # Add styling
        ax.set_title('Confusion Matrix', fontsize=16)
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        roc_auc: float
    ) -> Figure:
        """
        Create a ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: ROC AUC score
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=self.color_scheme['metrics_1'], lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot random guessing line
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        # Add styling
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_residuals(
        self,
        actual: List[float],
        predictions: List[float]
    ) -> Figure:
        """
        Create a residual plot.
        
        Args:
            actual: List of actual values
            predictions: List of predicted values
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Calculate residuals
        residuals = np.array(actual) - np.array(predictions)
        
        # Create scatter plot of predictions vs residuals
        ax.scatter(predictions, residuals, alpha=0.5, color=self.color_scheme['scatter'])
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='k', linestyle='--')
        
        # Add styling
        ax.set_title('Residual Plot', fontsize=16)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_error_distribution(
        self,
        actual: List[float],
        predictions: List[float]
    ) -> Figure:
        """
        Create an error distribution plot.
        
        Args:
            actual: List of actual values
            predictions: List of predicted values
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Calculate errors
        errors = np.array(actual) - np.array(predictions)
        
        # Create distribution plot
        sns.histplot(errors, kde=True, color=self.color_scheme['metrics_1'], ax=ax)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='k', linestyle='--')
        
        # Add styling
        ax.set_title('Error Distribution', fontsize=16)
        ax.set_xlabel('Error (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        stats_text = f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=12, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
```

### 4.3 Strategy Performance Visualizer

Creates visualizations for trading strategy performance:

```python
# visualizers/strategy_performance.py
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from trading_optimization.visualization.visualizers.base import BaseVisualizer
from trading_optimization.visualization.utils.color_schemes import get_color_scheme
from trading_optimization.visualization.utils.plotting_utils import (
    add_plot_styling, 
    create_subplot_grid
)

class StrategyPerformanceVisualizer(BaseVisualizer):
    """
    Creates visualizations for trading strategy performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy performance visualizer.
        
        Args:
            config: Visualizer configuration
        """
        super().__init__(config)
        self.fig_size = config.get('fig_size', (10, 6))
        self.dpi = config.get('dpi', 100)
    
    def create_performance_plots(
        self, 
        strategy_data: Dict[str, Any]
    ) -> Dict[str, Figure]:
        """
        Create a set of performance plots for a trading strategy.
        
        Args:
            strategy_data: Strategy performance data
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        plots = {}
        
        # Extract performance metrics
        metrics = strategy_data.get('metrics', {})
        
        # Create equity curve plot
        if 'equity_curve' in strategy_data:
            timestamps = strategy_data.get('timestamps', None)  # Optional timestamps
            plots['equity_curve'] = self.plot_equity_curve(
                equity_curve=strategy_data['equity_curve'],
                timestamps=timestamps,
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Equity Curve"
            )
        
        # Create drawdown plot
        if 'drawdowns' in strategy_data:
            timestamps = strategy_data.get('timestamps', None)  # Optional timestamps
            plots['drawdown'] = self.plot_drawdown(
                drawdowns=strategy_data['drawdowns'],
                timestamps=timestamps,
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Drawdown"
            )
        
        # Create returns distribution plot
        if 'returns' in strategy_data:
            plots['returns_distribution'] = self.plot_returns_distribution(
                returns=strategy_data['returns'],
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Returns Distribution"
            )
        
        # Create combined performance metrics plot
        if metrics:
            plots['performance_metrics'] = self.plot_performance_metrics(
                metrics=metrics,
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Performance Metrics"
            )
        
        # Create trade analysis plots if we have trade data
        if 'trades' in strategy_data and strategy_data['trades']:
            plots['trade_analysis'] = self.plot_trade_analysis(
                trades=strategy_data['trades'],
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Trade Analysis"
            )
            
            plots['trade_pnl_distribution'] = self.plot_trade_pnl_distribution(
                trades=strategy_data['trades'],
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Trade P&L Distribution"
            )
        
        # Create position analysis plot if available
        if 'positions' in strategy_data and strategy_data['positions']:
            plots['position_analysis'] = self.plot_position_analysis(
                positions=strategy_data['positions'],
                title=f"Strategy {strategy_data.get('strategy_id', 'Unknown')} - Position Analysis"
            )
        
        return plots
    
    def create_backtest_plots(
        self, 
        backtest_data: Dict[str, Any]
    ) -> Dict[str, Figure]:
        """
        Create a set of plots for a backtest.
        
        Args:
            backtest_data: Backtest results data
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        plots = {}
        
        # Create basic performance plots first
        performance_plots = self.create_performance_plots(backtest_data)
        plots.update(performance_plots)
        
        # Create additional backtest-specific plots
        
        # Create trade timing plot if we have trades and prices
        if ('trades' in backtest_data and backtest_data['trades'] and
            'prices' in backtest_data and 'timestamps' in backtest_data):
            plots['trade_timing'] = self.plot_trade_timing(
                trades=backtest_data['trades'],
                prices=backtest_data['prices'],
                timestamps=backtest_data['timestamps'],
                title=f"Backtest {backtest_data.get('backtest_id', 'Unknown')} - Trade Timing"
            )
        
        # Create comparison to benchmark if available
        if 'equity_curve' in backtest_data and 'benchmark_returns' in backtest_data:
            plots['benchmark_comparison'] = self.plot_benchmark_comparison(
                strategy_equity=backtest_data['equity_curve'],
                benchmark_returns=backtest_data['benchmark_returns'],
                timestamps=backtest_data.get('timestamps', None),
                title=f"Backtest {backtest_data.get('backtest_id', 'Unknown')} - Benchmark Comparison"
            )
        
        # Create rolling metrics plot if available
        if 'rolling_metrics' in backtest_data:
            plots['rolling_metrics'] = self.plot_rolling_metrics(
                rolling_metrics=backtest_data['rolling_metrics'],
                timestamps=backtest_data.get('timestamps', None),
                title=f"Backtest {backtest_data.get('backtest_id', 'Unknown')} - Rolling Metrics"
            )
        
        return plots
    
    def plot_equity_curve(
        self,
        equity_curve: List[float],
        timestamps: Optional[List[Union[str, pd.Timestamp]]] = None,
        title: str = 'Strategy Equity Curve'
    ) -> Figure:
        """
        Create an equity curve plot.
        
        Args:
            equity_curve: List of equity values
            timestamps: Optional list of timestamps
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create x-axis values
        if timestamps is not None:
            # Convert timestamps to pandas datetime if they are strings
            if isinstance(timestamps[0], str):
                x_values = pd.to_datetime(timestamps)
            else:
                x_values = timestamps
                
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        else:
            # Use simple indices if no timestamps
            x_values = range(len(equity_curve))
        
        # Plot equity curve
        ax.plot(x_values, equity_curve, color=self.color_scheme['equity'], linewidth=2)
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time' if timestamps else 'Trading Period', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add key statistics
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]
        total_return_pct = ((final_equity / initial_equity) - 1) * 100
        max_equity = max(equity_curve)
        
        stats_text = (f'Initial: ${initial_equity:,.2f}\n'
                     f'Final: ${final_equity:,.2f}\n'
                     f'Return: {total_return_pct:.2f}%')
        
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=12, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_drawdown(
        self,
        drawdowns: List[float],
        timestamps: Optional[List[Union[str, pd.Timestamp]]] = None,
        title: str = 'Strategy Drawdown'
    ) -> Figure:
        """
        Create a drawdown plot.
        
        Args:
            drawdowns: List of drawdown values (as decimals, not percentages)
            timestamps: Optional list of timestamps
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create x-axis values
        if timestamps is not None:
            # Convert timestamps to pandas datetime if they are strings
            if isinstance(timestamps[0], str):
                x_values = pd.to_datetime(timestamps)
            else:
                x_values = timestamps
                
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        else:
            # Use simple indices if no timestamps
            x_values = range(len(drawdowns))
        
        # Plot drawdown
        ax.fill_between(x_values, 0, [-d * 100 for d in drawdowns], color=self.color_scheme['drawdown'], alpha=0.7)
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time' if timestamps else 'Trading Period', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis for better visualization (more negative is down)
        ax.invert_yaxis()
        
        # Add key statistics
        max_drawdown = max(drawdowns) * 100
        
        stats_text = f'Max Drawdown: {max_drawdown:.2f}%'
        
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=12, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_returns_distribution(
        self,
        returns: List[float],
        title: str = 'Returns Distribution'
    ) -> Figure:
        """
        Create a returns distribution plot.
        
        Args:
            returns: List of return values
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Create histogram with kernel density estimate
        ax.hist(returns, bins=50, alpha=0.7, color=self.color_scheme['histogram'])
        
        # Add normal distribution for comparison
        import scipy.stats as stats
        if len(returns) > 1:  # Need at least 2 points for stats
            mu, std = stats.norm.fit(returns)
            x = np.linspace(min(returns), max(returns), 100)
            p = stats.norm.pdf(x, mu, std)
            # Scale the normal distribution to match histogram height
            hist_height = np.histogram(returns, bins=50)[0].max()
            dist_height = p.max()
            scale_factor = hist_height / dist_height if dist_height > 0 else 1
            ax.plot(x, p * scale_factor, 'k--', linewidth=1.5)
        
        # Add vertical line at zero
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add key statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        skew = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 2 else 0
        
        stats_text = (f'Mean: {mean_return:.4f}\n'
                     f'Median: {median_return:.4f}\n'
                     f'Std Dev: {std_return:.4f}\n'
                     f'Skew: {skew:.4f}\n'
                     f'Kurtosis: {kurtosis:.4f}')
        
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_performance_metrics(
        self,
        metrics: Dict[str, float],
        title: str = 'Strategy Performance Metrics'
    ) -> Figure:
        """
        Create a performance metrics plot.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        axs = axs.flatten()
        
        # Define metric groups
        return_metrics = ['total_return', 'annualized_return', 'cagr', 'volatility']
        ratio_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']
        drawdown_metrics = ['max_drawdown', 'avg_drawdown', 'drawdown_duration']
        trade_metrics = ['win_rate', 'profit_factor', 'avg_trade_pnl', 'avg_win', 'avg_loss']
        
        # Filter metrics to only include those that are present
        return_metrics = [m for m in return_metrics if m in metrics]
        ratio_metrics = [m for m in ratio_metrics if m in metrics]
        drawdown_metrics = [m for m in drawdown_metrics if m in metrics]
        trade_metrics = [m for m in trade_metrics if m in metrics]
        
        # Plot Return Metrics
        if return_metrics:
            self._plot_metric_group(
                ax=axs[0],
                metrics={k: metrics[k] for k in return_metrics},
                title='Return Metrics',
                color=self.color_scheme['metrics_1']
            )
        else:
            axs[0].text(0.5, 0.5, 'No return metrics available', 
                       ha='center', va='center', transform=axs[0].transAxes)
            axs[0].set_title('Return Metrics')
        
        # Plot Ratio Metrics
        if ratio_metrics:
            self._plot_metric_group(
                ax=axs[1],
                metrics={k: metrics[k] for k in ratio_metrics},
                title='Ratio Metrics',
                color=self.color_scheme['metrics_2']
            )
        else:
            axs[1].text(0.5, 0.5, 'No ratio metrics available', 
                       ha='center', va='center', transform=axs[1].transAxes)
            axs[1].set_title('Ratio Metrics')
        
        # Plot Drawdown Metrics
        if drawdown_metrics:
            self._plot_metric_group(
                ax=axs[2],
                metrics={k: metrics[k] for k in drawdown_metrics},
                title='Drawdown Metrics',
                color=self.color_scheme['metrics_3']
            )
        else:
            axs[2].text(0.5, 0.5, 'No drawdown metrics available', 
                       ha='center', va='center', transform=axs[2].transAxes)
            axs[2].set_title('Drawdown Metrics')
        
        # Plot Trade Metrics
        if trade_metrics:
            self._plot_metric_group(
                ax=axs[3],
                metrics={k: metrics[k] for k in trade_metrics},
                title='Trade Metrics',
                color=self.color_scheme['metrics_4']
            )
        else:
            axs[3].text(0.5, 0.5, 'No trade metrics available', 
                       ha='center', va='center', transform=axs[3].transAxes)
            axs[3].set_title('Trade Metrics')
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add spacing between subplots
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Add plot styling for each subplot
        for ax in axs:
            add_plot_styling(None, ax)
        
        return fig
    
    def _plot_metric_group(
        self,
        ax: plt.Axes,
        metrics: Dict[str, float],
        title: str,
        color: str
    ):
        """
        Helper method to plot a group of metrics.
        
        Args:
            ax: Matplotlib Axes object
            metrics: Dictionary of metrics to plot
            title: Plot title
            color: Bar color
        """
        # Format metric names for better display
        formatted_names = [k.replace('_', ' ').title() for k in metrics.keys()]
        
        # Create bar plot
        bars = ax.bar(formatted_names, list(metrics.values()), color=color)
        
        # Add styling
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            value_text = f'{height:.2f}'
            
            # Handle percentages
            if height > 0.1 and height < 1:
                value_text = f'{height:.2f}'
            elif abs(height) < 0.01:
                value_text = f'{height:.4f}'
                
            # Position text based on bar height
            va = 'bottom' if height >= 0 else 'top'
            y_offset = 0.01 if height >= 0 else -0.01
            
            ax.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
                   value_text, ha='center', va=va, fontsize=10)
        
        # Rotate x-axis labels for better readability if there are many metrics
        if len(metrics) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_trade_analysis(
        self,
        trades: List[Dict[str, Any]],
        title: str = 'Trade Analysis'
    ) -> Figure:
        """
        Create a trade analysis plot.
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        axs = axs.flatten()
        
        # Extract trade data
        trade_pnls = [trade.get('pnl', 0) for trade in trades]
        trade_durations = [trade.get('duration_days', 0) for trade in trades 
                          if 'duration_days' in trade]
        trade_directions = [trade.get('direction', 0) for trade in trades]
        trade_assets = [trade.get('asset', 'unknown') for trade in trades]
        
        # Aggregate by asset
        asset_pnls = {}
        for trade in trades:
            asset = trade.get('asset', 'unknown')
            pnl = trade.get('pnl', 0)
            if asset not in asset_pnls:
                asset_pnls[asset] = []
            asset_pnls[asset].append(pnl)
        
        # Calculate wins vs losses
        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        total_count = len(trade_pnls)
        
        win_rate = win_count / total_count if total_count > 0 else 0
        
        # Plot 1: Win/Loss Distribution
        axs[0].bar(['Wins', 'Losses'], [win_count, loss_count], 
                  color=[self.color_scheme['win'], self.color_scheme['loss']])
        axs[0].set_title('Win/Loss Distribution', fontsize=14)
        axs[0].text(0, win_count, f'{win_count} ({win_rate:.1%})', 
                   ha='center', va='bottom', fontsize=10)
        axs[0].text(1, loss_count, f'{loss_count} ({1-win_rate:.1%})', 
                   ha='center', va='bottom', fontsize=10)
        axs[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average Win vs Average Loss
        if wins and losses:
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            axs[1].bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss],
                      color=[self.color_scheme['win'], self.color_scheme['loss']])
            axs[1].set_title('Average Win vs Average Loss', fontsize=14)
            axs[1].text(0, avg_win, f'${avg_win:.2f}', 
                       ha='center', va='bottom', fontsize=10)
            axs[1].text(1, avg_loss, f'${avg_loss:.2f}', 
                       ha='center', va='top' if avg_loss < 0 else 'bottom', fontsize=10)
            axs[1].grid(True, alpha=0.3, axis='y')
            
            # Add win/loss ratio
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            axs[1].text(0.5, 0.9, f'Win/Loss Ratio: {win_loss_ratio:.2f}', 
                       ha='center', va='center', transform=axs[1].transAxes, 
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axs[1].text(0.5, 0.5, 'Insufficient trade data', 
                       ha='center', va='center', transform=axs[1].transAxes)
            axs[1].set_title('Average Win vs Average Loss')
        
        # Plot 3: Trade Duration Histogram
        if trade_durations:
            axs[2].hist(trade_durations, bins=20, color=self.color_scheme['histogram'], alpha=0.7)
            axs[2].set_title('Trade Duration Distribution', fontsize=14)
            axs[2].set_xlabel('Duration (Days)')
            axs[2].set_ylabel('Frequency')
            axs[2].grid(True, alpha=0.3)
            
            # Add average duration
            avg_duration = sum(trade_durations) / len(trade_durations)
            axs[2].axvline(avg_duration, color='r', linestyle='--')
            axs[2].text(avg_duration, axs[2].get_ylim()[1]*0.9, 
                       f'Avg: {avg_duration:.1f} days', 
                       ha='right', va='top', color='r')
        else:
            axs[2].text(0.5, 0.5, 'No duration data available', 
                       ha='center', va='center', transform=axs[2].transAxes)
            axs[2].set_title('Trade Duration Distribution')
        
        # Plot 4: Performance by Asset
        if asset_pnls:
            # Calculate total P&L by asset
            total_pnls = {asset: sum(pnls) for asset, pnls in asset_pnls.items()}
            
            # Sort by total P&L
            sorted_assets = sorted(total_pnls.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top N assets for readability
            top_n = 10
            assets = [a for a, _ in sorted_assets[:top_n]]
            pnls = [p for _, p in sorted_assets[:top_n]]
            
            # Create colors based on P&L
            colors = [self.color_scheme['win'] if p > 0 else self.color_scheme['loss'] 
                     for p in pnls]
            
            # Create horizontal bar chart
            axs[3].barh(assets, pnls, color=colors)
            axs[3].set_title('Performance by Asset', fontsize=14)
            axs[3].set_xlabel('Total P&L')
            axs[3].grid(True, alpha=0.3, axis='x')
            
            # Add P&L values
            for i, p in enumerate(pnls):
                ha = 'left' if p > 0 else 'right'
                x_offset = 0.01 * axs[3].get_xlim()[1] if p > 0 else -0.01 * axs[3].get_xlim()[1]
                axs[3].text(p + x_offset, i, f'${p:.2f}', va='center', ha=ha, fontsize=9)
        else:
            axs[3].text(0.5, 0.5, 'No asset data available', 
                       ha='center', va='center', transform=axs[3].transAxes)
            axs[3].set_title('Performance by Asset')
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add spacing between subplots
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Add plot styling for each subplot
        for ax in axs:
            add_plot_styling(None, ax)
        
        return fig
    
    def plot_trade_pnl_distribution(
        self,
        trades: List[Dict[str, Any]],
        title: str = 'Trade P&L Distribution'
    ) -> Figure:
        """
        Create a trade P&L distribution plot.
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Extract trade P&Ls
        trade_pnls = [trade.get('pnl', 0) for trade in trades]
        
        if not trade_pnls:
            ax.text(0.5, 0.5, 'No trade data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=16)
            return fig
        
        # Create histogram with kernel density estimate
        ax.hist(trade_pnls, bins=30, alpha=0.7, color=self.color_scheme['histogram'])
        
        # Add vertical line at zero
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        
        # Add styling
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Trade P&L', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add key statistics
        mean_pnl = np.mean(trade_pnls)
        median_pnl = np.median(trade_pnls)
        std_pnl = np.std(trade_pnls)
        max_win = max(trade_pnls) if trade_pnls else 0
        max_loss = min(trade_pnls) if trade_pnls else 0
        
        stats_text = (f'Mean: ${mean_pnl:.2f}\n'
                     f'Median: ${median_pnl:.2f}\n'
                     f'Std Dev: ${std_pnl:.2f}\n'
                     f'Max Win: ${max_win:.2f}\n'
                     f'Max Loss: ${max_loss:.2f}')
        
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_position_analysis(
        self,
        positions: Dict[str, Dict[str, Any]],
        title: str = 'Position Analysis'
    ) -> Figure:
        """
        Create a position analysis plot.
        
        Args:
            positions: Dictionary mapping assets to position details
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=self.dpi)
        
        # Extract position data
        assets = list(positions.keys())
        
        if not assets:
            axs[0].text(0.5, 0.5, 'No position data available', 
                       ha='center', va='center', transform=axs[0].transAxes)
            axs[1].text(0.5, 0.5, 'No position data available', 
                       ha='center', va='center', transform=axs[1].transAxes)
            fig.suptitle(title, fontsize=16, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            return fig
        
        # Calculate position values and P&Ls
        position_values = [abs(pos.get('current_value', 0)) for pos in positions.values()]
        total_value = sum(position_values)
        
        position_pnls = [pos.get('unrealized_pnl', 0) + pos.get('realized_pnl', 0) 
                        for pos in positions.values()]
        
        # Plot 1: Position Allocation (Pie Chart)
        if total_value > 0:
            # Filter to only include significant positions (e.g., > 1% of total)
            significant_threshold = total_value * 0.01
            significant_positions = [(asset, abs(positions[asset].get('current_value', 0))) 
                                   for asset in assets 
                                   if abs(positions[asset].get('current_value', 0)) > significant_threshold]
            
            # Sort by size
            significant_positions.sort(key=lambda x: x[1], reverse=True)
            
            if len(significant_positions) > 10:
                # If too many positions, group the smallest ones into "Other"
                top_assets = [asset for asset, _ in significant_positions[:9]]
                top_values = [value for _, value in significant_positions[:9]]
                
                other_value = sum(value for _, value in significant_positions[9:])
                top_assets.append('Other')
                top_values.append(other_value)
            else:
                top_assets = [asset for asset, _ in significant_positions]
                top_values = [value for _, value in significant_positions]
            
            # Create pie chart
            axs[0].pie(top_values, labels=top_assets, autopct='%1.1f%%', 
                      startangle=90, colors=plt.cm.tab20.colors[:len(top_values)])
            axs[0].set_title('Position Allocation', fontsize=14)
        else:
            axs[0].text(0.5, 0.5, 'No position value data', 
                       ha='center', va='center', transform=axs[0].transAxes)
            axs[0].set_title('Position Allocation')
        
        # Plot 2: Position P&Ls (Bar Chart)
        # Sort positions by P&L
        pnl_positions = list(zip(assets, position_pnls))
        pnl_positions.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top/bottom N positions for readability
        top_n = 5
        bottom_n = 5
        
        if len(pnl_positions) > (top_n + bottom_n):
            position_display = pnl_positions[:top_n] + pnl_positions[-bottom_n:]
        else:
            position_display = pnl_positions
        
        display_assets = [asset for asset, _ in position_display]
        display_pnls = [pnl for _, pnl in position_display]
        
        # Create colors based on P&L
        colors = [self.color_scheme['win'] if p > 0 else self.color_scheme['loss'] 
                 for p in display_pnls]
        
        # Create horizontal bar chart
        bars = axs[1].barh(display_assets, display_pnls, color=colors)
        axs[1].set_title('Position P&Ls', fontsize=14)
        axs[1].set_xlabel('P&L')
        axs[1].grid(True, alpha=0.3, axis='x')
        
        # Add P&L values
        for i, p in enumerate(display_pnls):
            ha = 'left' if p > 0 else 'right'
            x_offset = 0.01 * axs[1].get_xlim()[1] if p > 0 else -0.01 * axs[1].get_xlim()[1]
            axs[1].text(p + x_offset, i, f'${p:.2f}', va='center', ha=ha, fontsize=9)
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add spacing between subplots
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Add plot styling for each subplot
        for ax in axs:
            add_plot_styling(None, ax)
        
        return fig
    
    def plot_trade_timing(
        self,
        trades: List[Dict[str, Any]],
        prices: Dict[str, List[float]],
        timestamps: List[Union[str, pd.Timestamp]],
        title: str = 'Trade Timing Analysis'
    ) -> Figure:
        """
        Create a trade timing analysis plot.
        
        Args:
            trades: List of trade dictionaries
            prices: Dictionary mapping assets to price series
            timestamps: List of timestamps
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        # Extract trades for a single asset (take the most traded one)
        asset_counts = {}
        for trade in trades:
            asset = trade.get('asset', 'unknown')
            if asset not in asset_counts:
                asset_counts[asset] = 0
            asset_counts[asset] += 1
        
        if not asset_counts:
            ax.text(0.5, 0.5, 'No trade data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=16)
            return fig
        
        # Get most traded asset
        most_traded_asset = max(asset_counts.items(), key=lambda x: x[1])[0]
        
        # Filter trades for this asset
        asset_trades = [t for t in trades if t.get('asset', 'unknown') == most_traded_asset]
        
        if most_traded_asset not in prices:
            ax.text(0.5, 0.5, f'No price data available for {most_traded_asset}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=16)
            return fig
        
        # Convert timestamps to pandas datetime if they are strings
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Plot price series
        ax.plot(timestamps, prices[most_traded_asset], color=self.color_scheme['price'], linewidth=1.5)
        
        # Plot trade entries and exits
        for trade in asset_trades:
            # Extract trade timestamps
            entry_time = pd.to_datetime(trade.get('entry_time', None))
            exit_time = pd.to_datetime(trade.get('exit_time', None))
            
            if entry_time is None or exit_time is None:
                continue
            
            # Find closest indices
            entry_idx = np.argmin([abs((t - entry_time).total_seconds()) for t in timestamps])
            exit_idx = np.argmin([abs((t - exit_time).total_seconds()) for t in timestamps])
            
            # Get prices
            entry_price = trade.get('entry_price', prices[most_traded_asset][entry_idx])
            exit_price = trade.get('exit_price', prices[most_traded_asset][exit_idx])
            
            # Plot entry point
            marker = '^' if trade.get('direction', 1) > 0 else 'v'
            color = 'g' if trade.get('direction', 1) > 0 else 'r'
            ax.scatter(timestamps[entry_idx], entry_price, marker=marker, color=color, s=100)
            
            # Plot exit point
            ax.scatter(timestamps[exit_idx], exit_price, marker='o', color='b', s=80)
            
            # Connect entry and exit with line
            ax.plot([timestamps[entry_idx], timestamps[exit_idx]], 
                   [entry_price, exit_price], 'b--', alpha=0.5)
        
        # Add styling
        ax.set_title(f"{title} - {most_traded_asset}", fontsize=16)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.color_scheme['price'], lw=2, label='Price'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Long Entry'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10, label='Short Entry'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Exit')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add plot styling
        add_plot_styling(fig, ax)
        
        return fig
    
    def plot_benchmark_comparison(
        self,
        strategy_equity: List[float],
        benchmark_returns: List[float],
        timestamps: Optional[List[Union[str, pd.Timestamp]]] = None,
        title: str = 'Strategy vs Benchmark'
    ) -> Figure:
        """
        Create a benchmark comparison plot.
        
        Args:
            strategy_equity: List of strategy equity values
            benchmark_returns: List of benchmark returns
            timestamps: Optional list of timestamps
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), dpi=self.dpi, 
                              gridspec_kw={'height_ratios': [2, 1]})
        
        # Create x-axis values
        if timestamps is not None:
            # Convert timestamps to pandas datetime if they are strings
            if isinstance(timestamps[0], str):
                x_values = pd.to_datetime(timestamps)
            else:
                x_values = timestamps
                
            # Format x-axis as dates
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        else:
            # Use simple indices if no timestamps
            x_values = range(len(strategy_equity))
        
        # Calculate normalized equity curves (starting at 100)
        benchmark_equity = [100]
        for ret in benchmark_returns:
            benchmark_equity.append(benchmark_equity[-1] * (1 + ret))
            
        norm_strategy = [e * 100 / strategy_equity[0] for e in strategy_equity]
        norm_benchmark = benchmark_equity[1:] if len(benchmark_equity) > len(norm_strategy) else benchmark_equity
        
        # Plot 1: Normalized equity curves
        axs[0].plot(x_values, norm_strategy, label='Strategy', color=self.color_scheme['strategy'], linewidth=2)
        axs[0].plot(x_values, norm_benchmark, label='Benchmark', color=self.color_scheme['benchmark'], linewidth=2)
        axs[0].set_title('Normalized Performance (Starting at 100)', fontsize=14)
        axs[0].set_ylabel('Value', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=12)
        
        # Calculate strategy outperformance
        outperformance = [s - b for s, b in zip(norm_strategy, norm_benchmark)]
        
        # Plot 2: Strategy outperformance
        axs[1].fill_between(x_values, outperformance, 0, 
                          where=[o > 0 for o in outperformance], 
                          color=self.color_scheme['win'], alpha=0.7)
        axs[1].fill_between(x_values, outperformance, 0, 
                          where=[o <= 0 for o in outperformance], 
                          color=self.color_scheme['loss'], alpha=0.7)
        axs[1].axhline(0, color='k', linestyle='-', linewidth=1)
        axs[1].set_title('Strategy Outperformance', fontsize=14)
        axs[1].set_xlabel('Time' if timestamps else 'Trading Period', fontsize=12)
        axs[1].set_ylabel('Difference', fontsize=12)
        axs[1].grid(True, alpha=0.3)
        
        # Add key statistics
        final_outperformance = outperformance[-1]
        avg_outperformance = sum(outperformance) / len(outperformance)
        
        stats_text = (f'Final Outperformance: {final_outperformance:.2f}%\n'
                     f'Average Outperformance: {avg_outperformance:.2f}%')
        
        axs[1].text(0.02, 0.95, stats_text, transform=axs[1].transAxes, 
                   fontsize=10, va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add spacing between subplots
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Add plot styling for each subplot
        for ax in axs:
            add_plot_styling(None, ax)
        
        return fig
    
    def plot_rolling_metrics(
        self,
        rolling_metrics: Dict[str, List[float]],
        timestamps: Optional[List[Union[str, pd.Timestamp]]] = None,
        title: str = 'Rolling Performance Metrics'
    ) -> Figure:
        """
        Create a rolling performance metrics plot.
        
        Args:
            rolling_metrics: Dictionary mapping metric names to lists of values
            timestamps: Optional list of timestamps
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with multiple subplots based on number of metrics
        metric_count = len(rolling_metrics)
        
        if metric_count == 0:
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No rolling metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=16)
            return fig
        
        # Create subplot grid
        fig, axs = create_subplot_grid(metric_count, figsize=(12, 10), dpi=self.dpi)
        
        # Create x-axis values
        if timestamps is not None:
            # Convert timestamps to pandas datetime if they are strings
            if isinstance(timestamps[0], str):
                x_values = pd.to_datetime(timestamps)
            else:
                x_values = timestamps
        else:
            # Use simple indices if no timestamps
            first_metric = list(rolling_metrics.values())[0]
            x_values = range(len(first_metric))
        
        # Plot each metric
        colors = plt.cm.tab10.colors
        for i, (metric_name, metric_values) in enumerate(rolling_metrics.items()):
            # Get current subplot
            if isinstance(axs, np.ndarray):
                ax = axs.flat[i] if metric_count > 1 else axs
            else:
                ax = axs
                
            # Plot metric
            ax.plot(x_values, metric_values, color=colors[i % len(colors)], linewidth=2)
            
            # Add horizontal line at zero for reference
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # Add styling
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=14)
            if i == metric_count - 1:  # Only add x-label to bottom plot
                ax.set_xlabel('Time' if timestamps else 'Trading Period', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis as dates if we have timestamps
            if timestamps is not None:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                
            # Add basic statistics
            mean_value = np.mean(metric_values)
            ax.axhline(mean_value, color='r', linestyle='-', alpha=0.5)
            ax.text(0.02, 0.95, f'Mean: {mean_value:.4f}', transform=ax.transAxes, 
                   fontsize=10, va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add spacing between subplots
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Format dates if we have timestamps
        if timestamps is not None:
            fig.autofmt_xdate()
        
        # Add plot styling for each subplot
        if isinstance(axs, np.ndarray):
            for ax in axs.flat:
                add_plot_styling(None, ax)
        else:
            add_plot_styling(None, axs)
        
        return fig
```

### 4.4 Base Report Class

Base class for report generation:

```python
# reports/base.py
from typing import Dict, List, Any, Optional, Union
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class BaseReport:
    """
    Base class for generating reports.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        template_dir: str = './templates'
    ):
        """
        Initialize base report.
        
        Args:
            config: Report configuration
            template_dir: Directory containing templates
        """
        self.config = config
        self.template_dir = template_dir
        self.visualizations = {}
        self.additional_data = {}
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'TradingOptimizationPipeline'
        }
    
    def add_visualizations(self, visualizations: Dict[str, Figure]):
        """
        Add visualizations to the report.
        
        Args:
            visualizations: Dictionary mapping names to visualization figures
        """
        self.visualizations.update(visualizations)
    
    def add_additional_data(self, additional_data: Dict[str, Any]):
        """
        Add additional data to the report.
        
        Args:
            additional_data: Dictionary of additional data
        """
        self.additional_data.update(additional_data)
    
    def export_html(self, output_file: str) -> str:
        """
        Export report as HTML.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("export_html must be implemented by subclasses")
    
    def export_pdf(self, output_file: str) -> str:
        """
        Export report as PDF.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("export_pdf must be implemented by subclasses")
    
    def export_notebook(self, output_file: str) -> str:
        """
        Export report as Jupyter notebook.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("export_notebook must be implemented by subclasses")
    
    def _save_figures_as_images(self, output_dir: str, format: str = 'png') -> Dict[str, str]:
        """
        Save all visualization figures as images.
        
        Args:
            output_dir: Directory to save images in
            format: Image format ('png', 'svg', 'pdf')
            
        Returns:
            Dictionary mapping visualization names to image file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = {}
        
        # Save each figure as an image
        for name, fig in self.visualizations.items():
            # Create a valid filename from the name
            filename = f"{name.replace(' ', '_').lower()}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save figure
            dpi = self.config.get('export_dpi', 300)
            fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
            
            # Add to image paths
            image_paths[name] = filepath
        
        return image_paths
```

### 4.5 Interactive Dashboard

Implementation of a model performance dashboard:

```python
# dashboards/model_dashboard.py
from typing import Dict, List, Any, Optional, Union
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from trading_optimization.visualization.dashboards.base import BaseDashboard

class ModelDashboard(BaseDashboard):
    """
    Interactive dashboard for model performance visualization.
    """
    
    def __init__(
        self,
        model_id: str,
        model_data: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """
        Initialize model dashboard.
        
        Args:
            model_id: Model ID
            model_data: Model evaluation data
            config: Dashboard configuration
        """
        super().__init__(config)
        self.model_id = model_id
        self.model_data = model_data
        self.title = f"Model Performance Dashboard: {model_id}"
        
        # Extract key data
        self.metrics = model_data.get('metrics', {})
        self.predictions = model_data.get('predictions', [])
        self.actual = model_data.get('actual', [])
        self.timestamps = model_data.get('timestamps', None)
        self.feature_importance = model_data.get('feature_importance', {})
        self.training_history = model_data.get('training_history', {})
    
    def create_layout(self):
        """
        Create the dashboard layout.
        
        Returns:
            Dash layout object
        """
        # Create tabs for different sections
        tabs = dcc.Tabs([
            # Overview Tab
            dcc.Tab(label='Overview', children=[
                html.Div([
                    html.H3('Model Metrics'),
                    self._create_metrics_cards(),
                    
                    html.H3('Predictions vs Actual'),
                    dcc.Graph(id='predictions-chart', figure=self._create_predictions_chart()),
                    
                    html.H3('Error Analysis'),
                    dcc.Graph(id='error-distribution', figure=self._create_error_distribution())
                ])
            ]),
            
            # Detailed Metrics Tab
            dcc.Tab(label='Detailed Metrics', children=[
                html.Div([
                    html.H3('Performance Metrics'),
                    self._create_detailed_metrics_table(),
                    
                    html.H3('Confusion Matrix'),
                    dcc.Graph(id='confusion-matrix', figure=self._create_confusion_matrix())
                    if 'confusion_matrix' in self.model_data else html.P('No confusion matrix available')
                ])
            ]),
            
            # Feature Importance Tab
            dcc.Tab(label='Feature Importance', children=[
                html.Div([
                    html.H3('Feature Importance'),
                    dcc.Graph(id='feature-importance', figure=self._create_feature_importance())
                    if self.feature_importance else html.P('No feature importance data available')
                ])
            ]),
            
            # Training History Tab
            dcc.Tab(label='Training History', children=[
                html.Div([
                    html.H3('Training History'),
                    dcc.Graph(id='training-history', figure=self._create_training_history())
                    if self.training_history else html.P('No training history data available')
                ])
            ])
        ])
        
        # Create overall layout
        layout = html.Div([
            html.H1(self.title),
            html.Div([
                html.P(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P(f"Model ID: {self.model_id}")
            ]),
            html.Hr(),
            tabs
        ])
        
        return layout
    
    def _create_metrics_cards(self):
        """
        Create metric cards for the overview tab.
        
        Returns:
            Dash component with metric cards
        """
        # Define key metrics to display
        key_metrics = [
            ('R²', self.metrics.get('r2', None), 'Higher is better'),
            ('RMSE', self.metrics.get('rmse', None), 'Lower is better'),
            ('MAE', self.metrics.get('mae', None), 'Lower is better'),
            ('Accuracy', self.metrics.get('accuracy', None), 'Higher is better')
        ]
        
        # Create card style
        card_style = {
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'padding': '16px',
            'margin': '10px',
            'borderRadius': '5px',
            'width': '200px',
            'display': 'inline-block',
            'textAlign': 'center'
        }
        
        # Create cards
        cards = []
        for name, value, description in key_metrics:
            if value is not None:
                # Format value based on metric type
                if name in ['R²', 'Accuracy']:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.6f}"
                
                card = html.Div([
                    html.H4(name),
                    html.H2(formatted_value),
                    html.P(description)
                ], style=card_style)
                
                cards.append(card)
        
        # Return div with cards
        return html.Div(cards, style={'display': 'flex', 'flexWrap': 'wrap'})
    
    def _create_predictions_chart(self):
        """
        Create predictions vs actual chart.
        
        Returns:
            Plotly figure
        """
        if not self.predictions or not self.actual:
            return go.Figure()
        
        if len(self.predictions) != len(self.actual):
            return go.Figure()
        
        # Create time series chart if timestamps available
        if self.timestamps and len(self.timestamps) == len(self.predictions):
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=self.timestamps,
                y=self.actual,
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Add predicted values
            fig.add_trace(go.Scatter(
                x=self.timestamps,
                y=self.predictions,
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add layout
            fig.update_layout(
                title='Predictions vs Actual (Time Series)',
                xaxis_title='Time',
                yaxis_title='Value',
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode='closest'
            )
        else:
            # Create scatter plot
            predictions = np.array(self.predictions)
            actual = np.array(self.actual)
            
            # Calculate regression line
            min_val = min(np.min(predictions), np.min(actual))
            max_val = max(np.max(predictions), np.max(actual))
            line_x = np.linspace(min_val, max_val, 100)
            
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=actual,
                y=predictions,
                mode='markers',
                name='Prediction vs Actual',
                marker=dict(
                    color='rgba(0, 123, 255, 0.5)',
                    size=8
                )
            ))
            
            # Add perfect prediction line
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_x,
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', width=2, dash='dash')
            ))
            
            # Calculate R²
            r2 = self.metrics.get('r2', np.corrcoef(actual, predictions)[0, 1] ** 2)
            
            # Add layout
            fig.update_layout(
                title=f'Predictions vs Actual (Scatter) - R² = {r2:.4f}',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode='closest'
            )
        
        return fig
    
    def _create_error_distribution(self):
        """
        Create error distribution chart.
        
        Returns:
            Plotly figure
        """
        if not self.predictions or not self.actual:
            return go.Figure()
        
        if len(self.predictions) != len(self.actual):
            return go.Figure()
        
        # Calculate errors
        errors = np.array(self.actual) - np.array(self.predictions)
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=errors,
            name='Error Distribution',
            marker_color='rgba(0, 123, 255, 0.7)',
            nbinsx=30
        ))
        
        # Add vertical line at zero
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            yref='paper',
            line=dict(color='black', width=2, dash='dash')
        )
        
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Add layout
        fig.update_layout(
            title=f'Error Distribution (Mean: {mean_error:.4f}, Std: {std_error:.4f})',
            xaxis_title='Error (Actual - Predicted)',
            yaxis_title='Frequency',
            hovermode='closest'
        )
        
        return fig
    
    def _create_detailed_metrics_table(self):
        """
        Create detailed metrics table.
        
        Returns:
            Dash DataTable component
        """
        if not self.metrics:
            return html.P('No metrics available')
        
        # Create metrics table
        metrics_df = pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
        
        # Format values
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, (float, np.float64)) else str(x))
        
        # Create DataTable
        return dash.dash_table.DataTable(
            data=metrics_df.to_dict('records'),
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'}
            ],
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_table={'overflowX': 'auto'}
        )
    
    def _create_confusion_matrix(self):
        """
        Create confusion matrix chart.
        
        Returns:
            Plotly figure
        """
        if 'confusion_matrix' not in self.model_data:
            return go.Figure()
        
        cm = self.model_data['confusion_matrix']
        class_labels = self.model_data.get('class_labels', 
                                        [f'Class {i}' for i in range(len(cm))])
        
        # Create heatmap
        fig = px.imshow(
            cm,
            labels=dict(x='Predicted Class', y='True Class', color='Count'),
            x=class_labels,
            y=class_labels,
            text_auto=True,
            color_continuous_scale='Blues'
        )
        
        # Update layout
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='True Class'
        )
        
        return fig
    
    def _create_feature_importance(self):
        """
        Create feature importance chart.
        
        Returns:
            Plotly figure
        """
        if not self.feature_importance:
            return go.Figure()
        
        # Check format of feature importance
        if isinstance(self.feature_importance, dict):
            # Dict format - {feature_name: importance_value}
            feature_names = list(self.feature_importance.keys())
            importance_values = list(self.feature_importance.values())
        elif (isinstance(self.feature_importance, list) and 
              len(self.feature_importance) > 0 and 
              isinstance(self.feature_importance[0], tuple)):
            # List of tuples format - [(feature_name, importance_value), ...]
            feature_names = [f[0] for f in self.feature_importance]
            importance_values = [f[1] for f in self.feature_importance]
        else:
            # Unknown format
            return go.Figure()
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)
        feature_names = [feature_names[i] for i in sorted_indices]
        importance_values = [importance_values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=feature_names,
            x=importance_values,
            orientation='h',
            marker_color='rgba(0, 123, 255, 0.7)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'},
            hovermode='closest'
        )
        
        return fig
    
    def _create_training_history(self):
        """
        Create training history chart.
        
        Returns:
            Plotly figure
        """
        if not self.training_history:
            return go.Figure()
        
        # Check if we have epochs
        if 'epoch' in self.training_history:
            epochs = self.training_history['epoch']
        else:
            epochs = list(range(1, len(list(self.training_history.values())[0]) + 1))
        
        # Create subplots for different metrics
        metrics_to_plot = [k for k in self.training_history.keys() if k != 'epoch']
        
        if not metrics_to_plot:
            return go.Figure()
        
        # Group validation metrics with their training counterparts
        metric_groups = {}
        for metric in metrics_to_plot:
            # Check if it's a validation metric
            if metric.startswith('val_'):
                base_metric = metric[4:]  # Remove 'val_' prefix
                if base_metric in metrics_to_plot:
                    if base_metric not in metric_groups:
                        metric_groups[base_metric] = []
                    metric_groups[base_metric].append(metric)
                else:
                    if metric not in metric_groups:
                        metric_groups[metric] = []
                    metric_groups[metric].append(metric)
            else:
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append(metric)
        
        # Create subplots
        n_plots = len(metric_groups)
        fig = make_subplots(rows=n_plots, cols=1, 
                           subplot_titles=list(metric_groups.keys()),
                           vertical_spacing=0.1)
        
        # Add traces for each metric
        for i, (metric_name, metrics) in enumerate(metric_groups.items(), 1):
            for m in metrics:
                line_style = dict(dash='solid') if not m.startswith('val_') else dict(dash='dash')
                display_name = m if not m.startswith('val_') else 'Validation ' + m[4:]
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=self.training_history[m],
                        mode='lines',
                        name=display_name,
                        line=line_style
                    ),
                    row=i, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=300 * n_plots,
            title='Training History',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            hovermode='closest'
        )
        
        # Update y-axis titles
        for i, metric_name in enumerate(metric_groups.keys(), 1):
            fig.update_yaxes(title_text=metric_name.capitalize(), row=i, col=1)
        
        # Update x-axis titles
        fig.update_xaxes(title_text='Epoch', row=n_plots, col=1)
        
        return fig
```

## 5. Data Access Layer

The data access layer provides adapters to access the results database:

```python
# data/results_db_adapter.py
from typing import Dict, List, Any, Optional, Union
import os
import json
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_optimization.visualization.data.data_adapter import DataAdapter

class ResultsDBAdapter(DataAdapter):
    """
    Adapter for accessing the results database.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize results database adapter.
        
        Args:
            config: Database configuration
        """
        super().__init__()
        self.config = config
        
        # Extract database connection info
        self.db_uri = config.get('db_uri', None)
        self.db_file = config.get('db_file', None)
        
        # Set up database connection
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up database connection."""
        if self.db_uri:
            # SQLAlchemy connection for PostgreSQL
            self.engine = create_engine(self.db_uri)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        elif self.db_file:
            # SQLite connection
            self.conn = sqlite3.connect(self.db_file)
            self.conn.row_factory = sqlite3.Row
        else:
            raise ValueError("No database connection information provided")
    
    def get_model_evaluation_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get evaluation results for a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary of model evaluation results or None if not found
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT m.id, m.name, m.description, m.created_at, m.hyperparameters, 
                       e.metrics, e.predictions, e.actual, e.timestamps, e.feature_importance,
                       e.training_history, e.confusion_matrix, e.roc_data,
                       e.created_at as evaluation_date
                FROM models m
                LEFT JOIN model_evaluations e ON m.id = e.model_id
                WHERE m.id = :model_id
                ORDER BY e.created_at DESC
                LIMIT 1
            """
            result = self.session.execute(query, {'model_id': model_id}).fetchone()
            
            if result:
                row_dict = dict(result._mapping)
                
                # Parse JSON fields
                for field in ['hyperparameters', 'metrics', 'feature_importance', 
                              'training_history', 'confusion_matrix', 'roc_data']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['predictions', 'actual', 'timestamps']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT m.id, m.name, m.description, m.created_at, m.hyperparameters, 
                       e.metrics, e.predictions, e.actual, e.timestamps, e.feature_importance,
                       e.training_history, e.confusion_matrix, e.roc_data,
                       e.created_at as evaluation_date
                FROM models m
                LEFT JOIN model_evaluations e ON m.id = e.model_id
                WHERE m.id = ?
                ORDER BY e.created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (model_id,))
            row = cursor.fetchone()
            
            if row:
                row_dict = dict(row)
                
                # Parse JSON fields
                for field in ['hyperparameters', 'metrics', 'feature_importance', 
                              'training_history', 'confusion_matrix', 'roc_data']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['predictions', 'actual', 'timestamps']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        
        return None
    
    def get_strategy_results(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific trading strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary of strategy results or None if not found
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT s.id, s.name, s.description, s.created_at, s.config,
                       r.metrics, r.equity_curve, r.returns, r.drawdowns, r.trades,
                       r.positions, r.timestamps, r.created_at as result_date
                FROM strategies s
                LEFT JOIN strategy_results r ON s.id = r.strategy_id
                WHERE s.id = :strategy_id
                ORDER BY r.created_at DESC
                LIMIT 1
            """
            result = self.session.execute(query, {'strategy_id': strategy_id}).fetchone()
            
            if result:
                row_dict = dict(result._mapping)
                
                # Parse JSON fields
                for field in ['config', 'metrics', 'trades', 'positions']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['equity_curve', 'returns', 'drawdowns', 'timestamps']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT s.id, s.name, s.description, s.created_at, s.config,
                       r.metrics, r.equity_curve, r.returns, r.drawdowns, r.trades,
                       r.positions, r.timestamps, r.created_at as result_date
                FROM strategies s
                LEFT JOIN strategy_results r ON s.id = r.strategy_id
                WHERE s.id = ?
                ORDER BY r.created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (strategy_id,))
            row = cursor.fetchone()
            
            if row:
                row_dict = dict(row)
                
                # Parse JSON fields
                for field in ['config', 'metrics', 'trades', 'positions']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['equity_curve', 'returns', 'drawdowns', 'timestamps']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        
        return None
    
    def get_backtest_results(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific backtest.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Dictionary of backtest results or None if not found
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT b.id, b.strategy_id, b.created_at, b.config, b.start_date, b.end_date,
                       b.metrics, b.equity_curve, b.returns, b.drawdowns, b.trades,
                       b.positions, b.timestamps, b.prices, b.benchmark_returns
                FROM backtests b
                WHERE b.id = :backtest_id
            """
            result = self.session.execute(query, {'backtest_id': backtest_id}).fetchone()
            
            if result:
                row_dict = dict(result._mapping)
                
                # Parse JSON fields
                for field in ['config', 'metrics', 'trades', 'positions', 'prices']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['equity_curve', 'returns', 'drawdowns', 'timestamps', 'benchmark_returns']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT b.id, b.strategy_id, b.created_at, b.config, b.start_date, b.end_date,
                       b.metrics, b.equity_curve, b.returns, b.drawdowns, b.trades,
                       b.positions, b.timestamps, b.prices, b.benchmark_returns
                FROM backtests b
                WHERE b.id = ?
            """
            cursor.execute(query, (backtest_id,))
            row = cursor.fetchone()
            
            if row:
                row_dict = dict(row)
                
                # Parse JSON fields
                for field in ['config', 'metrics', 'trades', 'positions', 'prices']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                # Parse array fields
                for field in ['equity_curve', 'returns', 'drawdowns', 'timestamps', 'benchmark_returns']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        
        return None
    
    def get_optimization_results(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a hyperparameter optimization run.
        
        Args:
            optimization_id: Optimization run ID
            
        Returns:
            Dictionary of optimization results or None if not found
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT o.id, o.name, o.created_at, o.config, o.search_space,
                       o.best_params, o.best_score, o.trials, o.duration_seconds
                FROM optimization_runs o
                WHERE o.id = :optimization_id
            """
            result = self.session.execute(query, {'optimization_id': optimization_id}).fetchone()
            
            if result:
                row_dict = dict(result._mapping)
                
                # Parse JSON fields
                for field in ['config', 'search_space', 'best_params', 'trials']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT o.id, o.name, o.created_at, o.config, o.search_space,
                       o.best_params, o.best_score, o.trials, o.duration_seconds
                FROM optimization_runs o
                WHERE o.id = ?
            """
            cursor.execute(query, (optimization_id,))
            row = cursor.fetchone()
            
            if row:
                row_dict = dict(row)
                
                # Parse JSON fields
                for field in ['config', 'search_space', 'best_params', 'trials']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except:
                            pass
                
                return row_dict
        
        return None
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models.
        
        Returns:
            List of model dictionaries
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT m.id, m.name, m.description, m.created_at, m.model_type,
                       (SELECT MAX(score) FROM model_evaluations WHERE model_id = m.id) as best_score
                FROM models m
                ORDER BY m.created_at DESC
            """
            results = self.session.execute(query).fetchall()
            
            return [dict(row._mapping) for row in results]
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT m.id, m.name, m.description, m.created_at, m.model_type,
                       (SELECT MAX(json_extract(metrics, '$.r2')) FROM model_evaluations WHERE model_id = m.id) as best_score
                FROM models m
                ORDER BY m.created_at DESC
            """
            cursor.execute(query)
            
            return [dict(row) for row in cursor.fetchall()]
        
        return []
    
    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """
        Get a list of all strategies.
        
        Returns:
            List of strategy dictionaries
        """
        if hasattr(self, 'session'):
            # PostgreSQL query using SQLAlchemy
            query = """
                SELECT s.id, s.name, s.description, s.created_at,
                       (SELECT MAX(json_extract(metrics::json, '$.sharpe_ratio')) 
                        FROM strategy_results 
                        WHERE strategy_id = s.id) as best_sharpe
                FROM strategies s
                ORDER BY s.created_at DESC
            """
            results = self.session.execute(query).fetchall()
            
            return [dict(row._mapping) for row in results]
        elif hasattr(self, 'conn'):
            # SQLite query
            cursor = self.conn.cursor()
            
            query = """
                SELECT s.id, s.name, s.description, s.created_at,
                       (SELECT MAX(json_extract(metrics, '$.sharpe_ratio')) 
                        FROM strategy_results 
                        WHERE strategy_id = s.id) as best_sharpe
                FROM strategies s
                ORDER BY s.created_at DESC
            """
            cursor.execute(query)
            
            return [dict(row) for row in cursor.fetchall()]
        
        return []
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'session'):
            self.session.close()
            self.engine.dispose()
        elif hasattr(self, 'conn'):
            self.conn.close()
```

## 6. Report Templates

Example HTML template for model reports:

```html
<!-- templates/html/model_report.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ model_id }} - Model Evaluation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        h1, h2, h3 {
            color: #007bff;
        }
        h1 {
            font-size: 24px;
        }
        h2 {
            font-size: 20px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 20px;
        }
        h3 {
            font-size: 18px;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 5px solid #007bff;
            margin-bottom: 20px;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            width: 200px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin-top: 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            margin: 10px 0;
        }
        .plot {
            margin: 20px 0;
            text-align: center;
        }
        .plot img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        footer {
            margin-top: 50px;
            text-align: center;
            color: #777;
            font-size: 14px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .container {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Model Evaluation Report</h1>
        <p>{{ model_name }} (ID: {{ model_id }})</p>
    </header>

    <div class="container">
        <h2>Model Overview</h2>
        <div class="metadata">
            <p><strong>Model ID:</strong> {{ model_id }}</p>
            <p><strong>Model Name:</strong> {{ model_name }}</p>
            <p><strong>Description:</strong> {{ model_description }}</p>
            <p><strong>Created Date:</strong> {{ model_created_at }}</p>
            <p><strong>Evaluation Date:</strong> {{ evaluation_date }}</p>
        </div>
    </div>

    <div class="container">
        <h2>Performance Metrics</h2>
        <div class="metrics-container">
            {% for metric in key_metrics %}
            <div class="metric-card">
                <h3>{{ metric.name }}</h3>
                <div class="metric-value">{{ metric.value }}</div>
                <p>{{ metric.description }}</p>
            </div>
            {% endfor %}
        </div>
        
        <h3>All Metrics</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for metric in all_metrics %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ metric.value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="container">
        <h2>Visualizations</h2>
        
        {% for plot in plots %}
        <div class="plot">
            <h3>{{ plot.title }}</h3>
            <img src="{{ plot.filepath }}" alt="{{ plot.title }}">
            {% if plot.description %}
            <p>{{ plot.description }}</p>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    {% if feature_importance %}
    <div class="container">
        <h2>Feature Importance</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
            </thead>
            <tbody>
                {% for feature in feature_importance %}
                <tr>
                    <td>{{ feature.name }}</td>
                    <td>{{ feature.value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    {% if hyperparameters %}
    <div class="container">
        <h2>Hyperparameters</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for param in hyperparameters %}
                <tr>
                    <td>{{ param.name }}</td>
                    <td>{{ param.value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <footer>
        <p>Generated by Trading Model Optimization Pipeline on {{ generation_date }}</p>
    </footer>
</body>
</html>
```

## 7. Configuration

Example configuration for the visualization module:

```yaml
# Configuration for visualization module
visualization:
  # Output directories
  output_dir: "./reports"
  template_dir: "./templates"
  
  # Default report formats
  default_formats: ["html", "pdf"]
  
  # Styling/theme
  theme: "default"
  color_scheme: "blue"
  
  # Figure settings
  fig_size: [10, 6]
  dpi: 100
  export_dpi: 300
  
  # Dashboard settings
  dash_debug: false
  dash_port_range_start: 8050
  jupyter_mode: "inline"
  
  # Database connection
  results_db:
    db_uri: "postgresql://username:password@localhost:5432/trading_optimization"
    # Alternative SQLite connection
    # db_file: "./data/results.db"
  
  # Visualization-specific settings
  model_viz:
    show_feature_importance: true
    show_training_progress: true
    confusion_matrix_normalize: true
    
  strategy_viz:
    show_drawdowns: true
    show_trade_analysis: true
    show_position_allocation: true
    
  training_viz:
    show_learning_rate: true
    show_validation_metrics: true
    
  hyperparameter_viz:
    show_parallel_coordinates: true
    show_importance: true
    show_correlation: true
    
  comparison_viz:
    metrics_to_compare:
      - "r2"
      - "rmse"
      - "sharpe_ratio"
      - "max_drawdown"
```

## 8. Usage Examples

### 8.1 Generating a Model Performance Report

```python
# Example of generating a model performance report

from trading_optimization.config import ConfigManager
from trading_optimization.visualization.manager import VisualizationManager

# Load configuration
config = ConfigManager.instance()
viz_config = config.get('visualization', {})

# Create visualization manager
viz_manager = VisualizationManager(viz_config)

# Generate model performance report
model_id = "lstm_price_predictor_v1"
report_files = viz_manager.generate_model_report(
    model_id=model_id,
    formats=["html", "pdf", "notebook"],
    include_plots=True
)

print(f"Generated reports:")
for fmt, file_path in report_files.items():
    print(f"- {fmt}: {file_path}")

# Open HTML report in browser
import webbrowser
webbrowser.open(f"file://{report_files['html']}")
```

### 8.2 Generating a Strategy Performance Report

```python
# Example of generating a trading strategy performance report

from trading_optimization.config import ConfigManager
from trading_optimization.visualization.manager import VisualizationManager

# Load configuration
config = ConfigManager.instance()
viz_config = config.get('visualization', {})

# Create visualization manager
viz_manager = VisualizationManager(viz_config)

# Generate strategy performance report with additional data
strategy_id = "lstm_trend_strategy_v2"
additional_data = {
    'benchmark': 'S&P 500',
    'trading_period': '2022-01-01 to 2022-12-31',
    'market_conditions': 'Volatile market with rising interest rates'
}

report_files = viz_manager.generate_strategy_report(
    strategy_id=strategy_id,
    formats=["html", "pdf"],
    include_plots=True,
    additional_data=additional_data
)

print(f"Generated reports:")
for fmt, file_path in report_files.items():
    print(f"- {fmt}: {file_path}")

# Open HTML report in browser
import webbrowser
webbrowser.open(f"file://{report_files['html']}")
```

### 8.3 Creating an Interactive Dashboard

```python
# Example of creating an interactive dashboard for model performance

from trading_optimization.config import ConfigManager
from trading_optimization.visualization.manager import VisualizationManager

# Load configuration
config = ConfigManager.instance()
viz_config = config.get('visualization', {})

# Create visualization manager
viz_manager = VisualizationManager(viz_config)

# Create and serve model dashboard
model_id = "lstm_price_predictor_v1"
dashboard = viz_manager.create_model_dashboard(
    model_id=model_id,
    port=8050,
    host='0.0.0.0',
    mode='serve'  # Serves the dashboard on the specified port
)

print(f"Dashboard running at http://localhost:8050")
```

### 8.4 Comparing Multiple Models

```python
# Example of generating a comparison report for multiple models

from trading_optimization.config import ConfigManager
from trading_optimization.visualization.manager import VisualizationManager

# Load configuration
config = ConfigManager.instance()
viz_config = config.get('visualization', {})

# Create visualization manager
viz_manager = VisualizationManager(viz_config)

# Generate comparison report for multiple models
model_ids = [
    "lstm_price_predictor_v1",
    "lstm_price_predictor_v2",
    "transformer_price_predictor_v1"
]

comparison_report = viz_manager.generate_comparison_report(
    model_ids=model_ids,
    formats=["html"],
    comparison_name="price_predictor_comparison"
)

print(f"Generated comparison report: {comparison_report['html']}")

# Open HTML report in browser
import webbrowser
webbrowser.open(f"file://{comparison_report['html']}")
```

### 8.5 Exporting Individual Visualizations

```python
# Example of exporting individual visualizations

import matplotlib.pyplot as plt
import numpy as np
from trading_optimization.config import ConfigManager
from trading_optimization.visualization.manager import VisualizationManager
from trading_optimization.visualization.visualizers.model_performance import ModelPerformanceVisualizer

# Load configuration
config = ConfigManager.instance()
viz_config = config.get('visualization', {})

# Create visualization manager
viz_manager = VisualizationManager(viz_config)

# Create a custom visualization
def create_custom_visualization(x, y, title="Custom Visualization"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(x, y, 'b-')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Export the visualization
output_path = viz_manager.export_visualization(
    visualization_func=create_custom_visualization,
    output_path="./reports/custom_visualization.png",
    format="png",
    x=x,
    y=y,
    title="Custom Sine Wave Visualization"
)

print(f"Exported visualization to: {output_path}")
```

## 9. Integration with Other Components

The Visualization and Reporting Module integrates with:

1. **Results Database Infrastructure**: Retrieves model evaluations, strategy results, and optimization data
2. **Model Training Module**: Visualizes model training progress and performance metrics
3. **Hyperparameter Tuning System**: Creates visualizations of optimization processes and results
4. **Trading Strategy Integration**: Generates reports on strategy performance and backtests
5. **Command Line Interface**: Provides visualization and reporting commands via CLI

Integration primarily occurs through:

1. **Data Adapters**: Connect to the results database to retrieve model and strategy data
2. **Standardized Metrics**: Common metrics format shared across modules
3. **Report Templates**: Templates that can be extended for different visualization needs
4. **Configuration System**: Shared configuration for visualization settings

## 10. Extension Points

The module is designed to be easily extended:

1. **Custom Visualizers**:
   - Inherit from BaseVisualizer to create domain-specific visualizations
   - Register them with the visualization factory

2. **Report Templates**:
   - Add new templates to the templates directory
   - Create new report classes that inherit from BaseReport

3. **Dashboard Components**:
   - Create reusable dashboard components in dashboards/components
   - Integrate them into existing or new dashboards

4. **Exporters**:
   - Add new export formats by implementing new exporters
   - Register them with the export factory

5. **Data Adapters**:
   - Add adapters for new data sources (beyond the results database)

## 11. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. Configuration management system is implemented
3. Results database infrastructure is implemented
4. Model training module is implemented with standardized metric output
5. Trading strategy integration is implemented and storing results
6. Required libraries are installed:
   - matplotlib
   - seaborn
   - pandas
   - numpy
   - plotly
   - dash
   - jinja2
   - pdfkit (with wkhtmltopdf)
   - nbformat

## 12. Implementation Sequence

1. Set up directory structure for visualization components
2. Implement the base visualizer classes
3. Create data adapters for retrieving data from results database
4. Implement core visualizers for model performance and strategy
5. Develop report generation capabilities with templates
6. Create interactive dashboard components
7. Implement exporters for different output formats
8. Develop high-level visualization manager
9. Add comprehensive examples and documentation
10. Add extension points for custom visualizations

## 13. Testing and Validation

### 13.1 Unit Tests

```python
# Example unit tests for visualization components

import unittest
import os
import tempfile
import numpy as np

from trading_optimization.visualization.visualizers.model_performance import ModelPerformanceVisualizer
from trading_optimization.visualization.visualizers.strategy_performance import StrategyPerformanceVisualizer

class TestModelVisualizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.config = {
            'fig_size': (8, 6),
            'dpi': 80
        }
        
        # Create sample model data
        self.model_data = {
            'model_id': 'test_model',
            'predictions': np.random.normal(0, 1, 100),
            'actual': np.random.normal(0, 1, 100),
            'timestamps': pd.date_range(start='2022-01-01', periods=100),
            'metrics': {
                'r2': 0.85,
                'rmse': 0.12,
                'mae': 0.09,
                'mape': 5.2
            },
            'confusion_matrix': np.array([
                [45, 5],
                [10, 40]
            ]),
            'feature_importance': {
                'feature1': 0.3,
                'feature2': 0.5,
                'feature3': 0.2
            }
        }
        
        self.visualizer = ModelPerformanceVisualizer(self.config)
    
    def test_create_performance_plots(self):
        """Test creating performance plots."""
        plots = self.visualizer.create_performance_plots(self.model_data)
        
        # Check that plots were created
        self.assertIn('prediction_vs_actual', plots)
        self.assertIn('regression_metrics', plots)
        self.assertIn('error_distribution', plots)
        
        # Check that plots are matplotlib figures
        for name, plot in plots.items():
            self.assertIsNotNone(plot)
            self.assertEqual(plot.__class__.__name__, 'Figure')
    
    def test_plot_predictions_vs_actual_timeseries(self):
        """Test plotting predictions vs actual as time series."""
        fig = self.visualizer.plot_predictions_vs_actual_timeseries(
            timestamps=self.model_data['timestamps'],
            actual=self.model_data['actual'],
            predictions=self.model_data['predictions']
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_plot_confusion_matrix(self):
        """Test plotting confusion matrix."""
        fig = self.visualizer.plot_confusion_matrix(
            cm=self.model_data['confusion_matrix'],
            class_labels=['Class 0', 'Class 1']
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')

class TestStrategyVisualizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.config = {
            'fig_size': (8, 6),
            'dpi': 80
        }
        
        # Create sample strategy data
        self.strategy_data = {
            'strategy_id': 'test_strategy',
            'equity_curve': np.linspace(10000, 12000, 100) + np.random.normal(0, 100, 100),
            'returns': np.random.normal(0.001, 0.01, 99),
            'drawdowns': np.abs(np.random.normal(0, 0.03, 100)),
            'timestamps': pd.date_range(start='2022-01-01', periods=100),
            'metrics': {
                'sharpe_ratio': 1.8,
                'sortino_ratio': 2.5,
                'max_drawdown': 0.15,
                'win_rate': 0.6,
                'profit_factor': 1.5
            },
            'trades': [
                {'asset': 'AAPL', 'entry_price': 150, 'exit_price': 160, 'pnl': 1000, 'direction': 1},
                {'asset': 'MSFT', 'entry_price': 250, 'exit_price': 240, 'pnl': -1000, 'direction': 1},
                {'asset': 'GOOG', 'entry_price': 2000, 'exit_price': 2100, 'pnl': 1000, 'direction': 1}
            ]
        }
        
        self.visualizer = StrategyPerformanceVisualizer(self.config)
    
    def test_create_performance_plots(self):
        """Test creating performance plots."""
        plots = self.visualizer.create_performance_plots(self.strategy_data)
        
        # Check that plots were created
        self.assertIn('equity_curve', plots)
        self.assertIn('drawdown', plots)
        self.assertIn('returns_distribution', plots)
        self.assertIn('performance_metrics', plots)
        
        # Check that plots are matplotlib figures
        for name, plot in plots.items():
            self.assertIsNotNone(plot)
            self.assertEqual(plot.__class__.__name__, 'Figure')
    
    def test_plot_equity_curve(self):
        """Test plotting equity curve."""
        fig = self.visualizer.plot_equity_curve(
            equity_curve=self.strategy_data['equity_curve'],
            timestamps=self.strategy_data['timestamps']
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_plot_trade_analysis(self):
        """Test plotting trade analysis."""
        fig = self.visualizer.plot_trade_analysis(
            trades=self.strategy_data['trades']
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
```

### 13.2 Integration Tests

```python
# Example integration tests for visualization reporting

import unittest
import os
import tempfile
import json
import pandas as pd
import numpy as np
from datetime import datetime

from trading_optimization.visualization.manager import VisualizationManager
from trading_optimization.config import ConfigManager

class MockResultsDBAdapter:
    """Mock adapter for testing."""
    
    def __init__(self):
        self.model_data = self._create_sample_model_data()
        self.strategy_data = self._create_sample_strategy_data()
        
    def _create_sample_model_data(self):
        """Create sample model data for testing."""
        return {
            'test_model': {
                'id': 'test_model',
                'name': 'Test LSTM Model',
                'description': 'A test model for visualization',
                'created_at': datetime.now().isoformat(),
                'hyperparameters': {'units': 64, 'layers': 2, 'dropout': 0.2},
                'metrics': {'r2': 0.85, 'rmse': 0.12, 'mae': 0.09},
                'predictions': list(np.random.normal(0, 1, 100)),
                'actual': list(np.random.normal(0, 1, 100)),
                'timestamps': [t.isoformat() for t in pd.date_range(start='2022-01-01', periods=100)],
                'feature_importance': {'feature1': 0.3, 'feature2': 0.5, 'feature3': 0.2},
                'training_history': {
                    'loss': list(np.linspace(0.5, 0.1, 50)),
                    'val_loss': list(np.linspace(0.55, 0.15, 50)),
                    'epoch': list(range(50))
                },
                'confusion_matrix': [[45, 5], [10, 40]],
                'evaluation_date': datetime.now().isoformat()
            }
        }
    
    def _create_sample_strategy_data(self):
        """Create sample strategy data for testing."""
        return {
            'test_strategy': {
                'id': 'test_strategy',
                'name': 'Test LSTM Strategy',
                'description': 'A test strategy for visualization',
                'created_at': datetime.now().isoformat(),
                'config': {'signal_threshold': 0.5, 'position_size': 0.1},
                'metrics': {
                    'sharpe_ratio': 1.8,
                    'sortino_ratio': 2.5,
                    'max_drawdown': 0.15,
                    'win_rate': 0.6
                },
                'equity_curve': list(np.linspace(10000, 12000, 100) + np.random.normal(0, 100, 100)),
                'returns': list(np.random.normal(0.001, 0.01, 99)),
                'drawdowns': list(np.abs(np.random.normal(0, 0.03, 100))),
                'timestamps': [t.isoformat() for t in pd.date_range(start='2022-01-01', periods=100)],
                'trades': [
                    {'asset': 'AAPL', 'entry_price': 150, 'exit_price': 160, 'pnl': 1000, 'direction': 1},
                    {'asset': 'MSFT', 'entry_price': 250, 'exit_price': 240, 'pnl': -1000, 'direction': 1}
                ],
                'positions': {
                    'AAPL': {'size': 100, 'avg_price': 150, 'current_value': 16000},
                    'MSFT': {'size': -50, 'avg_price': 250, 'current_value': -12500}
                },
                'result_date': datetime.now().isoformat()
            }
        }
    
    def get_model_evaluation_results(self, model_id):
        """Get model evaluation results."""
        return self.model_data.get(model_id)
    
    def get_strategy_results(self, strategy_id):
        """Get strategy results."""
        return self.strategy_data.get(strategy_id)

class TestVisualizationManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for reports
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock adapter
        self.mock_adapter = MockResultsDBAdapter()
        
        # Create configuration
        self.config = {
            'output_dir': self.test_dir,
            'template_dir': './trading_optimization/visualization/templates',
            'default_formats': ['html'],
            'theme': 'default',
            'fig_size': [8, 6],
            'dpi': 80
        }
        
        # Create visualization manager with mock adapter
        self.viz_manager = VisualizationManager(self.config)
        self.viz_manager.results_db_adapter = self.mock_adapter
        
        # Replace data loaders with mock versions
        class MockDataLoader:
            def __init__(self, mock_adapter, data_type):
                self.mock_adapter = mock_adapter
                self.data_type = data_type
                
            def load_model_evaluation_results(self, model_id):
                return self.mock_adapter.get_model_evaluation_results(model_id)
                
            def load_strategy_results(self, strategy_id):
                return self.mock_adapter.get_strategy_results(strategy_id)
        
        self.viz_manager.model_data_loader = MockDataLoader(self.mock_adapter, 'model')
        self.viz_manager.strategy_data_loader = MockDataLoader(self.mock_adapter, 'strategy')
    
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_generate_model_report(self):
        """Test generating a model report."""
        report_files = self.viz_manager.generate_model_report(
            model_id='test_model',
            formats=['html'],
            output_path=os.path.join(self.test_dir, 'model_report')
        )
        
        # Check that HTML report was created
        self.assertIn('html', report_files)
        self.assertTrue(os.path.exists(report_files['html']))
    
    def test_generate_strategy_report(self):
        """Test generating a strategy report."""
        report_files = self.viz_manager.generate_strategy_report(
            strategy_id='test_strategy',
            formats=['html'],
            output_path=os.path.join(self.test_dir, 'strategy_report')
        )
        
        # Check that HTML report was created
        self.assertIn('html', report_files)
        self.assertTrue(os.path.exists(report_files['html']))
    
    def test_generate_comparison_report(self):
        """Test generating a comparison report."""
        # Add another model for comparison
        self.mock_adapter.model_data['test_model_2'] = dict(self.mock_adapter.model_data['test_model'])
        self.mock_adapter.model_data['test_model_2']['id'] = 'test_model_2'
        self.mock_adapter.model_data['test_model_2']['metrics'] = {'r2': 0.75, 'rmse': 0.18, 'mae': 0.15}
        
        report_files = self.viz_manager.generate_comparison_report(
            model_ids=['test_model', 'test_model_2'],
            formats=['html'],
            output_path=os.path.join(self.test_dir, 'comparison_report'),
            comparison_name='model_comparison'
        )
        
        # Check that HTML report was created
        self.assertIn('html', report_files)
        self.assertTrue(os.path.exists(report_files['html']))
```

## 14. Security Considerations

1. **Output Sanitization**: All data from the database must be sanitized before inclusion in HTML reports to prevent XSS attacks
2. **Input Validation**: Query parameters for dashboards should be validated to prevent SQL injection
3. **Authentication**: Dashboard can implement authentication for sensitive data
4. **File Paths**: Ensure proper handling of file paths to prevent directory traversal attacks
5. **Exception Handling**: Errors should be caught and not exposed to end-users in production

## 15. Performance Considerations

1. **Data Caching**: Results database queries should be cached to reduce load
2. **Lazy Loading**: Dashboard components should load data lazily
3. **Image Optimization**: Exported visualizations should be optimized for web
4. **Pagination**: Large datasets should be paginated for dashboard displays
5. **Aggregation**: For very large datasets, consider pre-aggregation in the database or data access layer
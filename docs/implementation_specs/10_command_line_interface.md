# Command Line Interface: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Command Line Interface (CLI) of the Trading Model Optimization Pipeline. The CLI provides a unified interface for users to interact with all components of the pipeline, including data management, model training, hyperparameter tuning, evaluation, trading strategy execution, visualization, and reporting.

## 2. Component Responsibilities

The Command Line Interface is responsible for:

- Providing a consistent user experience for accessing all pipeline functionality
- Parsing command line arguments and options with proper validation
- Executing the appropriate business logic based on user commands
- Presenting results in a readable format in the terminal
- Supporting interactive and non-interactive (script-friendly) modes
- Offering help documentation and examples for all commands
- Managing user configuration profiles
- Supporting both development/research and production workflows

## 3. Architecture

### 3.1 Overall Architecture

The CLI follows a multi-layered architecture with clear separation between:
1. Command parsing and validation
2. Business logic orchestration
3. Output formatting and presentation

```
┌─────────────────────────────────────────────────┐
│                                                 │
│                   CLI Entry                     │ Main entry point and command registry
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────┐   ┌───────────────────────┐  │
│  │ Command       │   │ Command Groups        │  │ Command definition,
│  │ Parser        │   │ (data, model, etc.)   │  │ parsing, and validation
│  └───────────────┘   └───────────────────────┘  │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────┐   ┌───────────────────────┐  │
│  │ Command       │   │ Config/Profile        │  │ Command execution and
│  │ Handlers      │   │ Manager               │  │ configuration management
│  └───────────────┘   └───────────────────────┘  │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────┐   ┌───────────────────────┐  │
│  │ Output        │   │ Progress              │  │ Output formatting and
│  │ Formatter     │   │ Tracker               │  │ presentation
│  └───────────────┘   └───────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── cli/
    ├── __init__.py
    ├── main.py                   # Main CLI entry point
    ├── parser.py                 # CLI argument parser setup
    ├── config.py                 # CLI-specific configuration
    ├── profiles.py               # User profile management
    ├── exceptions.py             # CLI-specific exceptions
    ├── constants.py              # CLI-related constants
    ├── formatters/
    │   ├── __init__.py
    │   ├── base.py               # Base output formatter
    │   ├── table.py              # Tabular data formatter
    │   ├── json.py               # JSON output formatter
    │   ├── progress.py           # Progress bar formatter
    │   └── chart.py              # ASCII/Unicode chart formatter
    ├── commands/
    │   ├── __init__.py
    │   ├── base.py               # Base command class
    │   ├── data_commands.py      # Data management commands 
    │   ├── model_commands.py     # Model training commands
    │   ├── tuning_commands.py    # Hyperparameter tuning commands
    │   ├── evaluation_commands.py # Model evaluation commands
    │   ├── strategy_commands.py  # Trading strategy commands
    │   ├── visualization_commands.py # Visualization commands
    │   ├── system_commands.py    # System/utility commands
    │   └── workflow_commands.py  # End-to-end workflow commands
    └── utils/
        ├── __init__.py
        ├── validation.py         # Input validation utilities
        ├── io.py                 # Input/output utilities
        ├── terminal.py           # Terminal handling utilities
        └── completion.py         # Command completion utilities
```

## 4. Core Components Design

### 4.1 CLI Entry Point

The main entry point that initializes the CLI, registers all commands, and handles top-level exceptions:

```python
# main.py
import argparse
import sys
import logging
from typing import List, Optional, Dict, Any, Union

from trading_optimization.config import ConfigManager
from trading_optimization.cli.parser import create_main_parser
from trading_optimization.cli.commands import (
    DataCommands, ModelCommands, TuningCommands, EvaluationCommands,
    StrategyCommands, VisualizationCommands, SystemCommands, WorkflowCommands
)
from trading_optimization.cli.exceptions import CLIError, CommandError, ConfigurationError
from trading_optimization.cli.profiles import ProfileManager
from trading_optimization.cli.formatters import get_formatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_optimization.cli')

class TradingOptimizationCLI:
    """
    Main CLI class for Trading Model Optimization Pipeline.
    """
    
    def __init__(self):
        """Initialize CLI with command registry and configuration."""
        # Load configuration
        self.config_manager = ConfigManager.instance()
        
        # Initialize profile manager
        self.profile_manager = ProfileManager()
        
        # Create parser
        self.parser = create_main_parser()
        
        # Register commands
        self.data_commands = DataCommands()
        self.model_commands = ModelCommands()
        self.tuning_commands = TuningCommands()
        self.evaluation_commands = EvaluationCommands()
        self.strategy_commands = StrategyCommands()
        self.visualization_commands = VisualizationCommands()
        self.system_commands = SystemCommands()
        self.workflow_commands = WorkflowCommands()
        
        # Register all command parsers
        self._register_commands()
    
    def _register_commands(self):
        """Register all command parsers with the main parser."""
        subparsers = self.parser.add_subparsers(dest='command_group', help='Command groups')
        
        # Register command groups
        self.data_commands.register_commands(subparsers)
        self.model_commands.register_commands(subparsers)
        self.tuning_commands.register_commands(subparsers)
        self.evaluation_commands.register_commands(subparsers)
        self.strategy_commands.register_commands(subparsers)
        self.visualization_commands.register_commands(subparsers)
        self.system_commands.register_commands(subparsers)
        self.workflow_commands.register_commands(subparsers)
        
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with the given arguments.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
            
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        # Parse arguments
        if args is None:
            args = sys.argv[1:]
            
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)
            
            # Show help if no command specified
            if not hasattr(parsed_args, 'func'):
                self.parser.print_help()
                return 0
            
            # Configure output formatter
            formatter_type = getattr(parsed_args, 'output_format', 'text')
            formatter = get_formatter(formatter_type)
            
            # Load profile if specified
            if hasattr(parsed_args, 'profile') and parsed_args.profile:
                try:
                    profile_config = self.profile_manager.load_profile(parsed_args.profile)
                    # Override config with profile settings
                    # This updates the global configuration manager
                    self.config_manager.update_config(profile_config)
                except ConfigurationError as e:
                    logger.error(f"Error loading profile: {e}")
                    return 1
            
            # Execute command
            result = parsed_args.func(parsed_args, formatter)
            
            return 0 if result is None else (0 if result else 1)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130  # Standard exit code for Ctrl+C
        except CommandError as e:
            logger.error(f"Command error: {e}")
            return 1
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            return 1
        except CLIError as e:
            logger.error(f"CLI error: {e}")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return 1

def main():
    """Main entry point for the CLI."""
    cli = TradingOptimizationCLI()
    exit_code = cli.run()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
```

### 4.2 Command Parser

Sets up the CLI argument parser:

```python
# parser.py
import argparse
import os
from typing import Any

def create_main_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Main argument parser
    """
    # Create parser
    parser = argparse.ArgumentParser(
        prog='trading_opt',
        description='Trading Model Optimization Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trading_opt data list-datasets
  trading_opt model train --config configs/lstm_model.yaml
  trading_opt tune run --config configs/optimization.yaml
  trading_opt evaluate model --model-id model_1 --dataset validation
  trading_opt strategy backtest --strategy-id strategy_1 --period "2022-01-01:2022-12-31"
  trading_opt visualize model-report --model-id model_1 --output report.html
        """
    )
    
    # Global options
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity (can be used multiple times)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output except for errors')
    parser.add_argument('-p', '--profile', type=str,
                       help='Use specific configuration profile')
    parser.add_argument('-c', '--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('-f', '--output-format', type=str, choices=['text', 'json', 'csv'],
                       default='text', help='Output format')
    
    return parser

def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common dataset arguments to a parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset ID or path')
    parser.add_argument('--start-date', type=str,
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for data (YYYY-MM-DD)')

def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common model arguments to a parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument('--model-id', type=str,
                       help='Model ID')
    parser.add_argument('--model-type', type=str,
                       help='Model type (e.g., lstm, mlp, transformer)')

def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common output arguments to a parser.
    
    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument('--output', type=str,
                       help='Output file or directory path')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file/directory')
```

### 4.3 Base Command Class

Base class for all commands:

```python
# commands/base.py
import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from trading_optimization.cli.exceptions import CommandError
from trading_optimization.cli.formatters.base import BaseFormatter
from trading_optimization.config import ConfigManager

class BaseCommand(ABC):
    """
    Base class for CLI commands.
    """
    
    def __init__(self):
        """Initialize command."""
        self.logger = logging.getLogger(f'trading_optimization.cli.{self.__class__.__name__}')
        self.config = ConfigManager.instance()
        
    @abstractmethod
    def register_commands(self, subparsers) -> None:
        """
        Register commands with the given subparser.
        
        Args:
            subparsers: Subparser to register commands with
        """
        pass
    
    def validate_args(self, args: argparse.Namespace) -> bool:
        """
        Validate command arguments.
        
        Args:
            args: Command arguments
            
        Returns:
            True if arguments are valid, False otherwise
        """
        return True
    
    def execute(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Execute the command.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        # Validate arguments
        if not self.validate_args(args):
            return False
        
        # Configure logging based on verbosity
        self._configure_logging(args)
        
        try:
            # Execute command-specific logic
            return self._execute_command(args, formatter)
        except Exception as e:
            self.logger.exception(f"Error executing command: {e}")
            raise CommandError(f"Command execution failed: {e}")
    
    @abstractmethod
    def _execute_command(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Execute command-specific logic.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        pass
    
    def _configure_logging(self, args: argparse.Namespace) -> None:
        """
        Configure logging based on verbosity.
        
        Args:
            args: Command arguments
        """
        if hasattr(args, 'quiet') and args.quiet:
            logging.getLogger('trading_optimization').setLevel(logging.ERROR)
        elif hasattr(args, 'verbose'):
            if args.verbose == 0:
                logging.getLogger('trading_optimization').setLevel(logging.INFO)
            elif args.verbose == 1:
                logging.getLogger('trading_optimization').setLevel(logging.DEBUG)
            else:
                # Enable debug for all loggers
                logging.getLogger().setLevel(logging.DEBUG)
```

### 4.4 Data Commands

Implementation of data management commands:

```python
# commands/data_commands.py
import argparse
import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

from trading_optimization.cli.commands.base import BaseCommand
from trading_optimization.cli.formatters.base import BaseFormatter
from trading_optimization.cli.parser import add_common_dataset_args, add_common_output_args
from trading_optimization.cli.exceptions import CommandError
from trading_optimization.data.dataset import DatasetManager
from trading_optimization.data.processor import DataProcessor

class DataCommands(BaseCommand):
    """
    Commands for data management.
    """
    
    def register_commands(self, subparsers) -> None:
        """
        Register data commands.
        
        Args:
            subparsers: Subparser to register commands with
        """
        # Create data command parser
        data_parser = subparsers.add_parser('data', help='Data management commands')
        data_subparsers = data_parser.add_subparsers(dest='data_command')
        
        # List datasets command
        list_parser = data_subparsers.add_parser('list', help='List available datasets')
        list_parser.add_argument('--raw', action='store_true', help='List raw datasets')
        list_parser.add_argument('--processed', action='store_true', help='List processed datasets')
        list_parser.add_argument('--filter', type=str, help='Filter datasets by name')
        list_parser.set_defaults(func=self.list_datasets)
        
        # Import dataset command
        import_parser = data_subparsers.add_parser('import', help='Import data from external source')
        import_parser.add_argument('--source', type=str, required=True, 
                                 help='Data source (e.g., csv, sql, api)')
        import_parser.add_argument('--path', type=str, required=True,
                                 help='Path or URL to data source')
        import_parser.add_argument('--name', type=str, required=True,
                                 help='Name to assign to the dataset')
        import_parser.add_argument('--format', type=str, choices=['csv', 'json', 'excel', 'sql', 'api'],
                                 default='csv', help='Source format')
        import_parser.add_argument('--options', type=str,
                                 help='Additional options in JSON format')
        import_parser.set_defaults(func=self.import_dataset)
        
        # Process dataset command
        process_parser = data_subparsers.add_parser('process', help='Process raw dataset')
        process_parser.add_argument('--input', type=str, required=True,
                                  help='Input dataset name')
        process_parser.add_argument('--output', type=str, required=True,
                                  help='Output dataset name')
        process_parser.add_argument('--pipeline', type=str, required=True,
                                  help='Processing pipeline configuration or name')
        process_parser.add_argument('--force', action='store_true',
                                  help='Force overwrite if output exists')
        process_parser.set_defaults(func=self.process_dataset)
        
        # Describe dataset command
        describe_parser = data_subparsers.add_parser('describe', help='Describe dataset')
        describe_parser.add_argument('--dataset', type=str, required=True,
                                   help='Dataset name to describe')
        describe_parser.add_argument('--stats', action='store_true',
                                   help='Include statistical information')
        describe_parser.set_defaults(func=self.describe_dataset)
        
        # Export dataset command
        export_parser = data_subparsers.add_parser('export', help='Export dataset')
        export_parser.add_argument('--dataset', type=str, required=True,
                                 help='Dataset name to export')
        export_parser.add_argument('--output', type=str, required=True,
                                 help='Output file path')
        export_parser.add_argument('--format', type=str, choices=['csv', 'json', 'pickle', 'parquet'],
                                 default='csv', help='Export format')
        export_parser.add_argument('--sample', type=int,
                                 help='Export only a sample of N rows')
        export_parser.set_defaults(func=self.export_dataset)
        
        # Create split command
        split_parser = data_subparsers.add_parser('split', help='Split dataset into train/val/test')
        split_parser.add_argument('--dataset', type=str, required=True,
                                help='Input dataset name')
        split_parser.add_argument('--train-size', type=float, default=0.7,
                                help='Train set size (proportion or absolute count)')
        split_parser.add_argument('--val-size', type=float, default=0.15,
                                help='Validation set size (proportion or absolute count)')
        split_parser.add_argument('--test-size', type=float, default=0.15,
                                help='Test set size (proportion or absolute count)')
        split_parser.add_argument('--method', type=str, choices=['random', 'time', 'stratified'],
                                default='random', help='Splitting method')
        split_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducible splits')
        split_parser.add_argument('--output-prefix', type=str, required=True,
                                help='Prefix for output dataset names')
        split_parser.set_defaults(func=self.split_dataset)
        
        # Generate features command
        features_parser = data_subparsers.add_parser('generate-features', help='Generate features')
        features_parser.add_argument('--dataset', type=str, required=True,
                                   help='Input dataset name')
        features_parser.add_argument('--output', type=str, required=True,
                                   help='Output dataset name')
        features_parser.add_argument('--config', type=str, required=True,
                                   help='Feature generation config file')
        features_parser.add_argument('--force', action='store_true',
                                   help='Force overwrite if output exists')
        features_parser.set_defaults(func=self.generate_features)
    
    def list_datasets(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        List available datasets.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            
            # Determine which types to list
            list_raw = args.raw if hasattr(args, 'raw') and args.raw else False
            list_processed = args.processed if hasattr(args, 'processed') and args.processed else False
            
            # If neither is specified, list both
            if not list_raw and not list_processed:
                list_raw = list_processed = True
                
            # Get datasets
            datasets = []
            if list_raw:
                raw_datasets = dataset_manager.list_raw_datasets()
                for ds in raw_datasets:
                    ds['type'] = 'raw'
                    datasets.append(ds)
                    
            if list_processed:
                processed_datasets = dataset_manager.list_processed_datasets()
                for ds in processed_datasets:
                    ds['type'] = 'processed'
                    datasets.append(ds)
            
            # Apply filter if specified
            if hasattr(args, 'filter') and args.filter:
                datasets = [ds for ds in datasets if args.filter.lower() in ds['name'].lower()]
            
            # Format output
            headers = ['Name', 'Type', 'Size', 'Last Modified', 'Description']
            rows = [
                [
                    ds['name'],
                    ds['type'],
                    ds.get('size', 'N/A'),
                    ds.get('last_modified', 'N/A'),
                    ds.get('description', '')
                ]
                for ds in datasets
            ]
            
            formatter.output_table(headers, rows, title='Available Datasets')
            return True
            
        except Exception as e:
            self.logger.exception(f"Error listing datasets: {e}")
            raise CommandError(f"Failed to list datasets: {e}")
    
    def import_dataset(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Import dataset from external source.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            
            # Parse options if provided
            import_options = {}
            if hasattr(args, 'options') and args.options:
                import json
                import_options = json.loads(args.options)
            
            # Import dataset
            dataset_info = dataset_manager.import_dataset(
                source=args.source,
                path=args.path,
                name=args.name,
                format=args.format,
                options=import_options
            )
            
            # Format output
            formatter.output_success(f"Dataset '{args.name}' imported successfully")
            formatter.output_dict({
                'name': dataset_info['name'],
                'rows': dataset_info.get('rows', 'N/A'),
                'columns': dataset_info.get('columns', 'N/A'),
                'size': dataset_info.get('size', 'N/A'),
                'path': dataset_info.get('path', 'N/A')
            }, title=f"Dataset Import: {args.name}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error importing dataset: {e}")
            raise CommandError(f"Failed to import dataset: {e}")
    
    def process_dataset(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Process raw dataset.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            data_processor = DataProcessor(self.config)
            
            # Check if input dataset exists
            if not dataset_manager.dataset_exists(args.input, raw=True):
                raise CommandError(f"Input dataset '{args.input}' does not exist")
            
            # Check if output dataset exists and handle force flag
            if dataset_manager.dataset_exists(args.output, processed=True) and not args.force:
                raise CommandError(f"Output dataset '{args.output}' already exists. Use --force to overwrite")
            
            # Load pipeline configuration
            pipeline_config = {}
            if os.path.isfile(args.pipeline):
                # Load from file
                import yaml
                with open(args.pipeline, 'r') as f:
                    pipeline_config = yaml.safe_load(f)
            else:
                # Load named pipeline
                pipeline_config = self.config.get_pipeline(args.pipeline)
                if not pipeline_config:
                    raise CommandError(f"Pipeline '{args.pipeline}' not found")
            
            # Process dataset with progress tracking
            formatter.output_info(f"Processing dataset '{args.input}' with pipeline '{args.pipeline}'...")
            with formatter.progress_bar() as progress:
                def update_progress(percent):
                    progress.update(percent)
                
                result = data_processor.process_dataset(
                    input_name=args.input,
                    output_name=args.output,
                    pipeline_config=pipeline_config,
                    progress_callback=update_progress,
                    overwrite=args.force
                )
            
            # Format output
            formatter.output_success(f"Dataset '{args.input}' processed successfully")
            formatter.output_dict({
                'input_name': args.input,
                'output_name': args.output,
                'input_rows': result.get('input_rows', 'N/A'),
                'output_rows': result.get('output_rows', 'N/A'),
                'input_columns': result.get('input_columns', 'N/A'),
                'output_columns': result.get('output_columns', 'N/A'),
                'processing_time': f"{result.get('processing_time', 0):.2f} seconds"
            }, title=f"Dataset Processing: {args.output}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error processing dataset: {e}")
            raise CommandError(f"Failed to process dataset: {e}")
    
    def describe_dataset(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Describe dataset.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            
            # Determine dataset type (raw or processed)
            if dataset_manager.dataset_exists(args.dataset, raw=True):
                dataset_type = 'raw'
            elif dataset_manager.dataset_exists(args.dataset, processed=True):
                dataset_type = 'processed'
            else:
                raise CommandError(f"Dataset '{args.dataset}' does not exist")
            
            # Load dataset metadata
            metadata = dataset_manager.get_dataset_metadata(args.dataset, dataset_type == 'raw')
            
            # Load dataset for stats if requested
            if args.stats:
                df = dataset_manager.load_dataset(args.dataset, raw=(dataset_type == 'raw'))
                
                # Calculate statistics
                statistics = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'missing_values': {col: int(df[col].isna().sum()) for col in df.columns},
                    'numeric_stats': {
                        col: {
                            'min': float(df[col].min()),
                            'max': float(df[col].max()),
                            'mean': float(df[col].mean()),
                            'std': float(df[col].std())
                        }
                        for col in df.select_dtypes(include=['number']).columns
                    }
                }
                
                # Add statistics to metadata
                metadata['statistics'] = statistics
            
            # Format output
            formatter.output_dict(metadata, title=f"Dataset: {args.dataset} ({dataset_type})")
            
            # Output column information if statistics were calculated
            if args.stats and 'statistics' in metadata:
                column_stats = []
                for col, dtype in metadata['statistics']['column_types'].items():
                    col_info = {
                        'name': col,
                        'type': dtype,
                        'missing': metadata['statistics']['missing_values'].get(col, 0)
                    }
                    
                    # Add numeric stats if available
                    if col in metadata['statistics'].get('numeric_stats', {}):
                        stats = metadata['statistics']['numeric_stats'][col]
                        col_info.update({
                            'min': f"{stats['min']:.4f}",
                            'max': f"{stats['max']:.4f}",
                            'mean': f"{stats['mean']:.4f}",
                            'std': f"{stats['std']:.4f}"
                        })
                    
                    column_stats.append(col_info)
                
                # Create table for column stats
                if column_stats:
                    # Determine headers based on whether we have numeric columns
                    headers = ['Column', 'Type', 'Missing']
                    if any('min' in col for col in column_stats):
                        headers.extend(['Min', 'Max', 'Mean', 'Std'])
                    
                    rows = []
                    for col in column_stats:
                        row = [col['name'], col['type'], col['missing']]
                        if 'min' in col:
                            row.extend([col['min'], col['max'], col['mean'], col['std']])
                        elif len(headers) > 3:
                            row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
                        rows.append(row)
                    
                    formatter.output_table(headers, rows, title='Column Statistics')
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error describing dataset: {e}")
            raise CommandError(f"Failed to describe dataset: {e}")
    
    def export_dataset(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Export dataset to file.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            
            # Determine dataset type (raw or processed)
            if dataset_manager.dataset_exists(args.dataset, raw=True):
                dataset_type = 'raw'
            elif dataset_manager.dataset_exists(args.dataset, processed=True):
                dataset_type = 'processed'
            else:
                raise CommandError(f"Dataset '{args.dataset}' does not exist")
            
            # Load dataset
            df = dataset_manager.load_dataset(args.dataset, raw=(dataset_type == 'raw'))
            
            # Sample if requested
            if hasattr(args, 'sample') and args.sample:
                if args.sample < len(df):
                    df = df.sample(n=args.sample, random_state=42)
                    formatter.output_info(f"Sampled {args.sample} rows from dataset")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            # Export based on format
            if args.format == 'csv':
                df.to_csv(args.output, index=False)
            elif args.format == 'json':
                df.to_json(args.output, orient='records', lines=True)
            elif args.format == 'pickle':
                df.to_pickle(args.output)
            elif args.format == 'parquet':
                df.to_parquet(args.output, index=False)
            else:
                raise CommandError(f"Unsupported export format: {args.format}")
            
            # Format output
            formatter.output_success(
                f"Exported {len(df)} rows from dataset '{args.dataset}' to {args.output}"
            )
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error exporting dataset: {e}")
            raise CommandError(f"Failed to export dataset: {e}")
    
    def split_dataset(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Split dataset into train/val/test.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            
            # Check if input dataset exists
            if not dataset_manager.dataset_exists(args.dataset, processed=True):
                raise CommandError(f"Dataset '{args.dataset}' does not exist or is not processed")
            
            # Check that output datasets don't exist
            train_name = f"{args.output_prefix}_train"
            val_name = f"{args.output_prefix}_val"
            test_name = f"{args.output_prefix}_test"
            
            for name in [train_name, val_name, test_name]:
                if dataset_manager.dataset_exists(name, processed=True):
                    raise CommandError(f"Output dataset '{name}' already exists")
            
            # Load dataset
            df = dataset_manager.load_dataset(args.dataset, raw=False)
            
            # Perform split
            formatter.output_info(f"Splitting dataset '{args.dataset}' into train/val/test sets...")
            
            result = dataset_manager.split_dataset(
                dataset_name=args.dataset,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                method=args.method,
                seed=args.seed,
                output_prefix=args.output_prefix
            )
            
            # Format output
            formatter.output_success(f"Dataset split successfully")
            formatter.output_dict({
                'input_dataset': args.dataset,
                'train_dataset': train_name,
                'val_dataset': val_name,
                'test_dataset': test_name,
                'train_rows': result['train_rows'],
                'val_rows': result['val_rows'],
                'test_rows': result['test_rows'],
                'train_pct': f"{result['train_pct']:.1%}",
                'val_pct': f"{result['val_pct']:.1%}",
                'test_pct': f"{result['test_pct']:.1%}"
            }, title="Dataset Split Results")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error splitting dataset: {e}")
            raise CommandError(f"Failed to split dataset: {e}")
    
    def generate_features(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Generate features from dataset.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            dataset_manager = DatasetManager(self.config)
            data_processor = DataProcessor(self.config)
            
            # Check if input dataset exists
            if not dataset_manager.dataset_exists(args.dataset, processed=True):
                raise CommandError(f"Dataset '{args.dataset}' does not exist or is not processed")
            
            # Check if output dataset exists and handle force flag
            if dataset_manager.dataset_exists(args.output, processed=True) and not args.force:
                raise CommandError(f"Output dataset '{args.output}' already exists. Use --force to overwrite")
            
            # Load feature generation config
            import yaml
            with open(args.config, 'r') as f:
                feature_config = yaml.safe_load(f)
            
            # Generate features with progress tracking
            formatter.output_info(f"Generating features for dataset '{args.dataset}'...")
            with formatter.progress_bar() as progress:
                def update_progress(percent):
                    progress.update(percent)
                
                result = data_processor.generate_features(
                    dataset_name=args.dataset,
                    output_name=args.output,
                    feature_config=feature_config,
                    progress_callback=update_progress,
                    overwrite=args.force
                )
            
            # Format output
            formatter.output_success(f"Features generated successfully")
            formatter.output_dict({
                'input_dataset': args.dataset,
                'output_dataset': args.output,
                'input_features': result.get('input_features', 'N/A'),
                'output_features': result.get('output_features', 'N/A'),
                'rows_processed': result.get('rows_processed', 'N/A'),
                'processing_time': f"{result.get('processing_time', 0):.2f} seconds"
            }, title="Feature Generation Results")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error generating features: {e}")
            raise CommandError(f"Failed to generate features: {e}")
```

### 4.5 Model Commands

Implementation of model training commands:

```python
# commands/model_commands.py
import argparse
import os
import yaml
import logging
from typing import List, Dict, Any, Optional

from trading_optimization.cli.commands.base import BaseCommand
from trading_optimization.cli.formatters.base import BaseFormatter
from trading_optimization.cli.parser import add_common_dataset_args, add_common_model_args
from trading_optimization.cli.exceptions import CommandError
from trading_optimization.data.dataset import DatasetManager
from trading_optimization.model.trainer import ModelTrainer
from trading_optimization.model.registry import ModelRegistry

class ModelCommands(BaseCommand):
    """
    Commands for model training and management.
    """
    
    def register_commands(self, subparsers) -> None:
        """
        Register model commands.
        
        Args:
            subparsers: Subparser to register commands with
        """
        # Create model command parser
        model_parser = subparsers.add_parser('model', help='Model training and management commands')
        model_subparsers = model_parser.add_subparsers(dest='model_command')
        
        # List models command
        list_parser = model_subparsers.add_parser('list', help='List trained models')
        list_parser.add_argument('--filter', type=str, help='Filter models by name')
        list_parser.add_argument('--type', type=str, help='Filter models by type')
        list_parser.add_argument('--sort-by', type=str, choices=['name', 'created', 'score'], 
                              default='created', help='Sort models by field')
        list_parser.set_defaults(func=self.list_models)
        
        # Train model command
        train_parser = model_subparsers.add_parser('train', help='Train a model')
        train_parser.add_argument('--config', type=str, required=True,
                               help='Model configuration file')
        train_parser.add_argument('--dataset', type=str, required=True,
                               help='Training dataset name')
        train_parser.add_argument('--val-dataset', type=str,
                               help='Validation dataset name')
        train_parser.add_argument('--name', type=str,
                               help='Model name (defaults to auto-generated)')
        train_parser.add_argument('--save-checkpoints', action='store_true',
                               help='Save model checkpoints during training')
        train_parser.add_argument('--checkpoint-dir', type=str,
                               help='Directory to save checkpoints')
        train_parser.add_argument('--no-cache', action='store_true',
                               help='Disable caching during training')
        train_parser.set_defaults(func=self.train_model)
        
        # Show model command
        show_parser = model_subparsers.add_parser('show', help='Show model details')
        show_parser.add_argument('--model-id', type=str, required=True,
                              help='Model ID to show')
        show_parser.add_argument('--include-config', action='store_true',
                              help='Include full model configuration')
        show_parser.add_argument('--include-metrics', action='store_true',
                              help='Include detailed performance metrics')
        show_parser.set_defaults(func=self.show_model)
        
        # Export model command
        export_parser = model_subparsers.add_parser('export', help='Export model to file')
        export_parser.add_argument('--model-id', type=str, required=True,
                                help='Model ID to export')
        export_parser.add_argument('--output', type=str, required=True,
                                help='Output directory for model files')
        export_parser.add_argument('--format', type=str, choices=['pytorch', 'onnx', 'torchscript'],
                                default='pytorch', help='Export format')
        export_parser.add_argument('--include-config', action='store_true',
                                help='Include model configuration in export')
        export_parser.add_argument('--include-metadata', action='store_true',
                                help='Include model metadata in export')
        export_parser.set_defaults(func=self.export_model)
        
        # Delete model command
        delete_parser = model_subparsers.add_parser('delete', help='Delete a model')
        delete_parser.add_argument('--model-id', type=str, required=True,
                                help='Model ID to delete')
        delete_parser.add_argument('--force', action='store_true',
                                help='Force deletion without confirmation')
        delete_parser.set_defaults(func=self.delete_model)
        
        # Load model command
        load_parser = model_subparsers.add_parser('load', help='Load model from file')
        load_parser.add_argument('--path', type=str, required=True,
                              help='Path to model file or directory')
        load_parser.add_argument('--name', type=str,
                              help='Name to assign to the model')
        load_parser.add_argument('--config', type=str,
                              help='Path to model configuration file')
        load_parser.set_defaults(func=self.load_model)
        
        # Copy model command
        copy_parser = model_subparsers.add_parser('copy', help='Copy a model')
        copy_parser.add_argument('--model-id', type=str, required=True,
                              help='Model ID to copy')
        copy_parser.add_argument('--name', type=str, required=True,
                              help='Name for the copied model')
        copy_parser.set_defaults(func=self.copy_model)
    
    def list_models(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        List trained models.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Get models from registry
            models = model_registry.list_models()
            
            # Apply filters
            if hasattr(args, 'filter') and args.filter:
                models = [m for m in models if args.filter.lower() in m['name'].lower()]
                
            if hasattr(args, 'type') and args.type:
                models = [m for m in models if m.get('type', '').lower() == args.type.lower()]
                
            # Sort models
            sort_key = args.sort_by if hasattr(args, 'sort_by') else 'created'
            if sort_key == 'name':
                models.sort(key=lambda m: m['name'])
            elif sort_key == 'created':
                models.sort(key=lambda m: m['created_at'], reverse=True)
            elif sort_key == 'score':
                # Sort by score, handling missing scores
                models.sort(key=lambda m: m.get('best_score', float('-inf')), reverse=True)
            
            # Format output
            headers = ['ID', 'Name', 'Type', 'Created', 'Best Score']
            rows = [
                [
                    m['id'],
                    m['name'],
                    m.get('type', 'N/A'),
                    m.get('created_at', 'N/A'),
                    f"{m.get('best_score', 'N/A'):.4f}" if isinstance(m.get('best_score'), (int, float)) else m.get('best_score', 'N/A')
                ]
                for m in models
            ]
            
            formatter.output_table(headers, rows, title='Trained Models')
            return True
            
        except Exception as e:
            self.logger.exception(f"Error listing models: {e}")
            raise CommandError(f"Failed to list models: {e}")
    
    def train_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Train a model.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_trainer = ModelTrainer(self.config)
            dataset_manager = DatasetManager(self.config)
            model_registry = ModelRegistry(self.config)
            
            # Check if datasets exist
            if not dataset_manager.dataset_exists(args.dataset, processed=True):
                raise CommandError(f"Training dataset '{args.dataset}' does not exist or is not processed")
                
            if hasattr(args, 'val_dataset') and args.val_dataset:
                if not dataset_manager.dataset_exists(args.val_dataset, processed=True):
                    raise CommandError(f"Validation dataset '{args.val_dataset}' does not exist or is not processed")
            
            # Load model configuration
            with open(args.config, 'r') as f:
                model_config = yaml.safe_load(f)
            
            # Prepare training options
            training_options = {
                'save_checkpoints': args.save_checkpoints if hasattr(args, 'save_checkpoints') else False,
                'no_cache': args.no_cache if hasattr(args, 'no_cache') else False
            }
            
            if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
                training_options['checkpoint_dir'] = args.checkpoint_dir
                
            if hasattr(args, 'name') and args.name:
                training_options['name'] = args.name
            
            # Start training with progress tracking
            formatter.output_info(f"Training model using configuration: {args.config}")
            formatter.output_info(f"Training dataset: {args.dataset}")
            if hasattr(args, 'val_dataset') and args.val_dataset:
                formatter.output_info(f"Validation dataset: {args.val_dataset}")
            
            with formatter.progress_bar(total=100) as progress:
                def progress_callback(epoch, epochs, metrics, stage="training"):
                    # Update progress bar
                    progress.update((epoch / epochs) * 100)
                    
                    # Log metrics periodically
                    if epoch % 10 == 0 or epoch == epochs - 1:
                        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                        formatter.output_info(f"Epoch {epoch+1}/{epochs} - {metric_str}")
                
                # Start training
                training_result = model_trainer.train_model(
                    config=model_config,
                    train_dataset=args.dataset,
                    val_dataset=args.val_dataset if hasattr(args, 'val_dataset') else None,
                    options=training_options,
                    progress_callback=progress_callback
                )
            
            # Format output
            formatter.output_success(f"Model trained successfully")
            formatter.output_dict({
                'model_id': training_result['model_id'],
                'model_name': training_result['model_name'],
                'model_type': training_result.get('model_type', 'N/A'),
                'training_time': f"{training_result.get('training_time', 0):.2f} seconds",
                'epochs': training_result.get('epochs', 'N/A'),
                'best_epoch': training_result.get('best_epoch', 'N/A'),
                'train_loss': f"{training_result.get('train_loss', 0):.4f}",
                'val_loss': f"{training_result.get('val_loss', 0):.4f}" if 'val_loss' in training_result else 'N/A',
                'metrics': {k: f"{v:.4f}" for k, v in training_result.get('metrics', {}).items()}
            }, title=f"Training Results: {training_result['model_name']}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error training model: {e}")
            raise CommandError(f"Failed to train model: {e}")
    
    def show_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Show model details.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Get model details
            model_info = model_registry.get_model(args.model_id)
            if not model_info:
                raise CommandError(f"Model with ID '{args.model_id}' not found")
            
            # Format basic output
            output = {
                'id': model_info['id'],
                'name': model_info['name'],
                'type': model_info.get('type', 'N/A'),
                'created_at': model_info.get('created_at', 'N/A'),
                'size': model_info.get('size', 'N/A'),
                'description': model_info.get('description', '')
            }
            
            # Add config if requested
            if hasattr(args, 'include_config') and args.include_config:
                output['config'] = model_info.get('config', {})
            
            # Add metrics if requested
            if hasattr(args, 'include_metrics') and args.include_metrics:
                output['metrics'] = model_info.get('metrics', {})
                
                # Add training history if available
                if 'training_history' in model_info:
                    output['training_history'] = {
                        'epochs': len(model_info['training_history'].get('loss', [])),
                        'final_loss': model_info['training_history'].get('loss', [])[-1] if model_info['training_history'].get('loss', []) else None,
                        'best_epoch': model_info.get('best_epoch', 'N/A')
                    }
            
            formatter.output_dict(output, title=f"Model: {model_info['name']}")
            
            # Display architecture as table if available
            if 'architecture' in model_info and isinstance(model_info['architecture'], list):
                headers = ['Layer', 'Type', 'Parameters', 'Shape']
                rows = [
                    [
                        layer.get('name', 'N/A'),
                        layer.get('type', 'N/A'),
                        layer.get('parameters', 'N/A'),
                        layer.get('shape', 'N/A')
                    ]
                    for layer in model_info['architecture']
                ]
                
                formatter.output_table(headers, rows, title='Model Architecture')
            
            # Display feature importance if available
            if 'feature_importance' in model_info and isinstance(model_info['feature_importance'], dict):
                # Convert to list of tuples and sort by importance
                features = [(k, v) for k, v in model_info['feature_importance'].items()]
                features.sort(key=lambda x: x[1], reverse=True)
                
                headers = ['Feature', 'Importance']
                rows = [[feature, f"{importance:.4f}"] for feature, importance in features]
                
                formatter.output_table(headers, rows, title='Feature Importance')
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error showing model details: {e}")
            raise CommandError(f"Failed to show model details: {e}")
    
    def export_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Export model to file.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Check if model exists
            if not model_registry.model_exists(args.model_id):
                raise CommandError(f"Model with ID '{args.model_id}' not found")
            
            # Prepare export options
            export_options = {
                'format': args.format,
                'include_config': args.include_config if hasattr(args, 'include_config') else False,
                'include_metadata': args.include_metadata if hasattr(args, 'include_metadata') else False
            }
            
            # Export model
            formatter.output_info(f"Exporting model '{args.model_id}' to {args.output}")
            
            export_result = model_registry.export_model(
                model_id=args.model_id,
                output_path=args.output,
                options=export_options
            )
            
            # Format output
            formatter.output_success(f"Model exported successfully")
            formatter.output_dict({
                'model_id': args.model_id,
                'export_format': args.format,
                'output_path': export_result['path'],
                'files': export_result.get('files', []),
                'size': export_result.get('size', 'N/A')
            }, title=f"Model Export: {args.model_id}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error exporting model: {e}")
            raise CommandError(f"Failed to export model: {e}")
    
    def delete_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Delete a model.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Check if model exists
            if not model_registry.model_exists(args.model_id):
                raise CommandError(f"Model with ID '{args.model_id}' not found")
            
            # Get model info for display
            model_info = model_registry.get_model(args.model_id)
            
            # Confirm deletion if not forced
            if not args.force:
                formatter.output_warning(f"You are about to delete model '{model_info['name']}' (ID: {args.model_id})")
                confirmation = input("Are you sure? [y/N]: ")
                if confirmation.lower() != 'y':
                    formatter.output_info("Model deletion cancelled")
                    return True
            
            # Delete model
            model_registry.delete_model(args.model_id)
            
            # Format output
            formatter.output_success(f"Model '{model_info['name']}' (ID: {args.model_id}) deleted successfully")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error deleting model: {e}")
            raise CommandError(f"Failed to delete model: {e}")
    
    def load_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Load model from file.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Check if path exists
            if not os.path.exists(args.path):
                raise CommandError(f"Path '{args.path}' does not exist")
            
            # Prepare load options
            load_options = {}
            
            if hasattr(args, 'name') and args.name:
                load_options['name'] = args.name
                
            if hasattr(args, 'config') and args.config:
                # Load config file
                with open(args.config, 'r') as f:
                    load_options['config'] = yaml.safe_load(f)
            
            # Load model
            formatter.output_info(f"Loading model from {args.path}")
            
            load_result = model_registry.load_model(
                path=args.path,
                options=load_options
            )
            
            # Format output
            formatter.output_success(f"Model loaded successfully")
            formatter.output_dict({
                'model_id': load_result['model_id'],
                'model_name': load_result['model_name'],
                'model_type': load_result.get('model_type', 'N/A')
            }, title=f"Model Loading: {load_result['model_name']}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error loading model: {e}")
            raise CommandError(f"Failed to load model: {e}")
    
    def copy_model(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
        """
        Copy a model.
        
        Args:
            args: Command arguments
            formatter: Output formatter
            
        Returns:
            True on success, False on failure
        """
        try:
            model_registry = ModelRegistry(self.config)
            
            # Check if model exists
            if not model_registry.model_exists(args.model_id):
                raise CommandError(f"Model with ID '{args.model_id}' not found")
            
            # Copy model
            copy_result = model_registry.copy_model(
                model_id=args.model_id,
                new_name=args.name
            )
            
            # Format output
            formatter.output_success(f"Model copied successfully")
            formatter.output_dict({
                'original_id': args.model_id,
                'new_id': copy_result['model_id'],
                'new_name': copy_result['model_name']
            }, title=f"Model Copy: {args.name}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error copying model: {e}")
            raise CommandError(f"Failed to copy model: {e}")
```

### 4.6 Text Output Formatter

Implementation of a text-based output formatter:

```python
# formatters/table.py
import os
import sys
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
from contextlib import contextmanager

from trading_optimization.cli.formatters.base import BaseFormatter
from trading_optimization.cli.exceptions import FormatterError

class TableFormatter(BaseFormatter):
    """
    Formats output as text tables.
    """
    
    def __init__(self, **kwargs):
        """Initialize formatter."""
        super().__init__(**kwargs)
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bright_red': '\033[91m',
            'bright_green': '\033[92m',
            'bright_yellow': '\033[93m',
            'bright_blue': '\033[94m',
            'bright_magenta': '\033[95m',
            'bright_cyan': '\033[96m',
            'bright_white': '\033[97m',
        }
        
        # Disable colors if not supported by terminal
        if not sys.stdout.isatty():
            for key in self.colors:
                self.colors[key] = ''
    
    def output_info(self, message: str) -> None:
        """
        Output informational message.
        
        Args:
            message: Message to output
        """
        print(f"{self.colors['blue']}[INFO]{self.colors['reset']} {message}")
    
    def output_success(self, message: str) -> None:
        """
        Output success message.
        
        Args:
            message: Message to output
        """
        print(f"{self.colors['green']}[SUCCESS]{self.colors['reset']} {message}")
    
    def output_warning(self, message: str) -> None:
        """
        Output warning message.
        
        Args:
            message: Message to output
        """
        print(f"{self.colors['yellow']}[WARNING]{self.colors['reset']} {message}")
    
    def output_error(self, message: str) -> None:
        """
        Output error message.
        
        Args:
            message: Message to output
        """
        print(f"{self.colors['red']}[ERROR]{self.colors['reset']} {message}", file=sys.stderr)
    
    def output_table(self, headers: List[str], rows: List[List[Any]], title: Optional[str] = None) -> None:
        """
        Output data as a table.
        
        Args:
            headers: Table headers
            rows: Table rows
            title: Optional table title
        """
        if not rows:
            print("No data available")
            return
        
        # Determine column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                cell_str = str(cell)
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell_str))
        
        # Calculate table width
        table_width = sum(col_widths) + len(headers) * 3 - 1
        
        # Output title if provided
        if title:
            print(f"\n{self.colors['bold']}{title}{self.colors['reset']}")
        
        # Output header row
        header_line = '┌' + '┬'.join('─' * (width + 2) for width in col_widths) + '┐'
        print(header_line)
        
        header_cells = [f"{self.colors['bold']}{header.ljust(width)}{self.colors['reset']}" for header, width in zip(headers, col_widths)]
        print('│ ' + ' │ '.join(header_cells) + ' │')
        
        # Output separator line
        separator = '├' + '┼'.join('─' * (width + 2) for width in col_widths) + '┤'
        print(separator)
        
        # Output data rows
        for row in rows:
            row_data = []
            for i, cell in enumerate(row):
                cell_str = str(cell)
                width = col_widths[i] if i < len(col_widths) else len(cell_str)
                row_data.append(cell_str.ljust(width))
            
            print('│ ' + ' │ '.join(row_data) + ' │')
        
        # Output bottom line
        bottom_line = '└' + '┴'.join('─' * (width + 2) for width in col_widths) + '┘'
        print(bottom_line)
    
    def output_dict(self, data: Dict[str, Any], title: Optional[str] = None) -> None:
        """
        Output dictionary data.
        
        Args:
            data: Dictionary data
            title: Optional title
        """
        if title:
            print(f"\n{self.colors['bold']}{title}{self.colors['reset']}")
        
        # Find the maximum key length for alignment
        key_width = max(len(str(k)) for k in data.keys()) if data else 0
        
        # Process and print each key-value pair
        for key, value in data.items():
            key_str = f"{self.colors['cyan']}{str(key).ljust(key_width)}{self.colors['reset']}"
            
            # Handle complex values
            if isinstance(value, dict):
                # Nested dictionary
                print(f"{key_str}: ")
                for sub_key, sub_value in value.items():
                    print(f"  {self.colors['dim']}{sub_key}{self.colors['reset']}: {sub_value}")
            elif isinstance(value, list):
                # List value
                print(f"{key_str}: ")
                for item in value:
                    print(f"  - {item}")
            else:
                # Simple value
                print(f"{key_str}: {value}")
    
    @contextmanager
    def progress_bar(self, total: int = 100):
        """
        Context manager for progress bar.
        
        Args:
            total: Total value for 100% progress
            
        Yields:
            Progress bar object with update method
        """
        terminal_width = shutil.get_terminal_size().columns
        bar_width = min(50, terminal_width - 30)
        
        class ProgressBar:
            def __init__(self, total, bar_width):
                self.total = total
                self.bar_width = bar_width
                self.current = 0
                self._print_progress(0)
            
            def update(self, progress):
                value = min(progress, self.total)
                self.current = value
                self._print_progress(value)
            
            def _print_progress(self, value):
                percent = value / self.total * 100
                filled_width = int(self.bar_width * percent / 100)
                bar = '█' * filled_width + '░' * (self.bar_width - filled_width)
                sys.stdout.write(f"\r{percent:6.2f}% |{bar}| {value}/{self.total}")
                sys.stdout.flush()
        
        progress = ProgressBar(total, bar_width)
        try:
            yield progress
        finally:
            # Ensure we end with a newline
            sys.stdout.write('\n')
            sys.stdout.flush()
```

## 5. Command Structure

The CLI provides several command groups, each with specific commands:

### 5.1 Command Groups

1. **Data Commands**
   - `list`: List available datasets
   - `import`: Import data from external source
   - `process`: Process raw dataset
   - `describe`: Describe dataset
   - `export`: Export dataset
   - `split`: Split dataset into train/val/test
   - `generate-features`: Generate features

2. **Model Commands**
   - `list`: List trained models
   - `train`: Train a model
   - `show`: Show model details
   - `export`: Export model to file
   - `delete`: Delete a model
   - `load`: Load model from file
   - `copy`: Copy a model

3. **Tuning Commands**
   - `list`: List optimization runs
   - `run`: Run hyperparameter optimization
   - `show`: Show optimization results
   - `best-params`: Extract best parameters
   - `trials`: List optimization trials
   - `export`: Export optimization results

4. **Evaluation Commands**
   - `model`: Evaluate a model
   - `compare`: Compare multiple models
   - `list`: List evaluation runs
   - `show`: Show evaluation details
   - `export`: Export evaluation results

5. **Strategy Commands**
   - `list`: List strategies
   - `create`: Create trading strategy
   - `backtest`: Backtest strategy
   - `show`: Show strategy details
   - `optimize`: Optimize strategy parameters
   - `delete`: Delete strategy
   - `export`: Export strategy

6. **Visualization Commands**
   - `model-report`: Generate model report
   - `strategy-report`: Generate strategy report
   - `backtest-report`: Generate backtest report
   - `comparison-report`: Generate comparison report
   - `export`: Export visualizations
   - `dashboard`: Launch interactive dashboard

7. **System Commands**
   - `info`: Show system information
   - `cleanup`: Clean temporary files
   - `setup`: Setup environment
   - `config`: View/edit configuration
   - `version`: Show version info

8. **Workflow Commands**
   - `train-evaluate`: Train and evaluate in one step
   - `optimize-train`: Optimize hyperparameters and train
   - `full-pipeline`: Run complete pipeline
   - `custom`: Run custom workflow from YAML

## 6. Configuration Management

### 6.1 User Profiles

The CLI supports user profiles for different configurations:

```python
# profiles.py
import os
import yaml
import json
from typing import Dict, Any, List, Optional

from trading_optimization.cli.exceptions import ConfigurationError

class ProfileManager:
    """
    Manages user configuration profiles.
    """
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory containing profiles
        """
        # Use provided directory or default
        if profiles_dir:
            self.profiles_dir = profiles_dir
        else:
            # Default to ~/.trading_optimization/profiles
            home_dir = os.path.expanduser("~")
            self.profiles_dir = os.path.join(home_dir, ".trading_optimization", "profiles")
            
        # Create directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List available profiles.
        
        Returns:
            List of profile info dictionaries
        """
        profiles = []
        
        # Check all files in profiles directory
        if os.path.exists(self.profiles_dir):
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml') or filename.endswith('.json'):
                    profile_path = os.path.join(self.profiles_dir, filename)
                    profile_name = os.path.splitext(filename)[0]
                    
                    # Get modified time
                    modified = os.path.getmtime(profile_path)
                    
                    # Add to list
                    profiles.append({
                        'name': profile_name,
                        'path': profile_path,
                        'modified': modified
                    })
        
        return sorted(profiles, key=lambda x: x['name'])
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load a profile by name.
        
        Args:
            profile_name: Profile name
            
        Returns:
            Profile configuration dictionary
            
        Raises:
            ConfigurationError: If profile not found
        """
        # Check for profile with various extensions
        for ext in ['.yaml', '.yml', '.json']:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}{ext}")
            if os.path.exists(profile_path):
                try:
                    if profile_path.endswith('.json'):
                        with open(profile_path, 'r') as f:
                            return json.load(f)
                    else:
                        with open(profile_path, 'r') as f:
                            return yaml.safe_load(f)
                except Exception as e:
                    raise ConfigurationError(f"Error loading profile '{profile_name}': {e}")
        
        # Profile not found
        raise ConfigurationError(f"Profile '{profile_name}' not found")
    
    def save_profile(self, profile_name: str, config: Dict[str, Any], format: str = 'yaml') -> str:
        """
        Save a profile.
        
        Args:
            profile_name: Profile name
            config: Configuration dictionary
            format: File format ('yaml' or 'json')
            
        Returns:
            Path to saved profile
            
        Raises:
            ConfigurationError: If error saving profile
        """
        # Determine file extension
        ext = '.json' if format == 'json' else '.yaml'
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}{ext}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            
            # Save profile
            if format == 'json':
                with open(profile_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                with open(profile_path, 'w') as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
            
            return profile_path
        except Exception as e:
            raise ConfigurationError(f"Error saving profile '{profile_name}': {e}")
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a profile.
        
        Args:
            profile_name: Profile name
            
        Returns:
            True if profile was deleted, False if not found
            
        Raises:
            ConfigurationError: If error deleting profile
        """
        # Check for profile with various extensions
        for ext in ['.yaml', '.yml', '.json']:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}{ext}")
            if os.path.exists(profile_path):
                try:
                    os.remove(profile_path)
                    return True
                except Exception as e:
                    raise ConfigurationError(f"Error deleting profile '{profile_name}': {e}")
        
        return False
```

### 6.2 Handling Configuration Files

Example of the system command that manages configuration:

```python
# commands/system_commands.py (partial)
def _execute_config_command(self, args: argparse.Namespace, formatter: BaseFormatter) -> bool:
    """
    Execute config command.
    
    Args:
        args: Command arguments
        formatter: Output formatter
        
    Returns:
        True on success, False on failure
    """
    if args.subcommand == 'view':
        # View configuration
        if args.path:
            # View specific configuration path
            value = self.config.get(args.path)
            if value is None:
                formatter.output_error(f"Configuration path '{args.path}' not found")
                return False
                
            formatter.output_dict({args.path: value}, title=f"Configuration: {args.path}")
        else:
            # View full configuration
            formatter.output_dict(self.config.get_all(), title="Full Configuration")
            
        return True
        
    elif args.subcommand == 'set':
        # Set configuration value
        try:
            # Parse value
            value = self._parse_config_value(args.value)
            
            # Update configuration
            self.config.set(args.path, value)
            
            # Save configuration if requested
            if args.save:
                self.config.save()
                formatter.output_success(f"Configuration saved")
            
            formatter.output_success(f"Configuration '{args.path}' set to '{value}'")
            return True
        except Exception as e:
            formatter.output_error(f"Error setting configuration: {e}")
            return False
            
    elif args.subcommand == 'save-profile':
        # Save current configuration as profile
        try:
            profile_manager = ProfileManager()
            
            # Get current configuration
            config_dict = self.config.get_all()
            
            # Save as profile
            profile_path = profile_manager.save_profile(args.name, config_dict, args.format)
            
            formatter.output_success(f"Configuration saved as profile '{args.name}' at {profile_path}")
            return True
        except Exception as e:
            formatter.output_error(f"Error saving profile: {e}")
            return False
            
    elif args.subcommand == 'list-profiles':
        # List available profiles
        try:
            profile_manager = ProfileManager()
            profiles = profile_manager.list_profiles()
            
            if not profiles:
                formatter.output_info("No profiles found")
                return True
            
            # Format as table
            headers = ['Name', 'Last Modified']
            rows = [
                [
                    p['name'],
                    datetime.fromtimestamp(p['modified']).strftime('%Y-%m-%d %H:%M:%S')
                ]
                for p in profiles
            ]
            
            formatter.output_table(headers, rows, title="Available Profiles")
            return True
        except Exception as e:
            formatter.output_error(f"Error listing profiles: {e}")
            return False
            
    elif args.subcommand == 'load-profile':
        # Load configuration from profile
        try:
            profile_manager = ProfileManager()
            
            # Load profile
            profile_config = profile_manager.load_profile(args.name)
            
            # Update configuration
            self.config.update_config(profile_config)
            
            # Save configuration if requested
            if args.save:
                self.config.save()
                formatter.output_success(f"Configuration saved")
            
            formatter.output_success(f"Configuration loaded from profile '{args.name}'")
            return True
        except Exception as e:
            formatter.output_error(f"Error loading profile: {e}")
            return False
```

## 7. Integration with Other Components

The CLI integrates with all other components of the Trading Model Optimization Pipeline:

### 7.1 Integration with Data Management

```python
# Example from data_commands.py (process_dataset method)
dataset_manager = DatasetManager(self.config)
data_processor = DataProcessor(self.config)

# Check if input dataset exists
if not dataset_manager.dataset_exists(args.input, raw=True):
    raise CommandError(f"Input dataset '{args.input}' does not exist")

# Process dataset with progress tracking
formatter.output_info(f"Processing dataset '{args.input}' with pipeline '{args.pipeline}'...")
with formatter.progress_bar() as progress:
    def update_progress(percent):
        progress.update(percent)
    
    result = data_processor.process_dataset(
        input_name=args.input,
        output_name=args.output,
        pipeline_config=pipeline_config,
        progress_callback=update_progress,
        overwrite=args.force
    )
```

### 7.2 Integration with Model Training

```python
# Example from model_commands.py (train_model method)
model_trainer = ModelTrainer(self.config)
dataset_manager = DatasetManager(self.config)
model_registry = ModelRegistry(self.config)

# Start training with progress tracking
formatter.output_info(f"Training model using configuration: {args.config}")
formatter.output_info(f"Training dataset: {args.dataset}")

with formatter.progress_bar(total=100) as progress:
    def progress_callback(epoch, epochs, metrics, stage="training"):
        # Update progress bar
        progress.update((epoch / epochs) * 100)
        
        # Log metrics periodically
        if epoch % 10 == 0 or epoch == epochs - 1:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            formatter.output_info(f"Epoch {epoch+1}/{epochs} - {metric_str}")
    
    # Start training
    training_result = model_trainer.train_model(
        config=model_config,
        train_dataset=args.dataset,
        val_dataset=args.val_dataset if hasattr(args, 'val_dataset') else None,
        options=training_options,
        progress_callback=progress_callback
    )
```

### 7.3 Integration with Visualization Module

```python
# Example from visualization_commands.py (model_report method)
visualization_manager = VisualizationManager(self.config)
model_registry = ModelRegistry(self.config)

# Check if model exists
if not model_registry.model_exists(args.model_id):
    raise CommandError(f"Model with ID '{args.model_id}' not found")

# Generate report
formatter.output_info(f"Generating report for model '{args.model_id}'...")
report_files = visualization_manager.generate_model_report(
    model_id=args.model_id,
    formats=args.formats.split(',') if args.formats else None,
    include_plots=not args.no_plots,
    output_path=args.output
)

# Format output
formatter.output_success(f"Report generated successfully")
formatter.output_dict({
    'model_id': args.model_id,
    'formats': list(report_files.keys()),
    'files': report_files
}, title=f"Model Report: {args.model_id}")

# Open report if requested
if args.open and 'html' in report_files:
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(report_files['html'])}")
```

## 8. Usage Examples

### 8.1 Basic Usage

```bash
# List available datasets
trading_opt data list

# Import data from CSV
trading_opt data import --source file --path data/btc_prices.csv --name btc_prices

# Process raw dataset
trading_opt data process --input btc_prices --output btc_features --pipeline feature_engineering.yaml

# Train a model
trading_opt model train --config models/lstm_config.yaml --dataset btc_features_train --val-dataset btc_features_val --name btc_price_predictor

# Evaluate a model
trading_opt evaluate model --model-id model_001 --dataset btc_features_test

# Create trading strategy
trading_opt strategy create --model-id model_001 --config strategies/trend_following.yaml --name btc_trend_strategy

# Backtest strategy
trading_opt strategy backtest --strategy-id strategy_001 --period "2022-01-01:2022-12-31" --initial-capital 10000

# Generate visualization report
trading_opt visualize model-report --model-id model_001 --output ./reports/model_001_report.html --open
```

### 8.2 Advanced Usage

```bash
# Run hyperparameter optimization
trading_opt tune run --config tuning/bayesian_optimization.yaml --dataset btc_features_train --val-dataset btc_features_val

# Extract best parameters from optimization
trading_opt tune best-params --optimization-id opt_001 --output ./configs/best_params.yaml

# Train with optimized parameters
trading_opt model train --config ./configs/best_params.yaml --dataset btc_features_train --val-dataset btc_features_val

# Run full pipeline in one go
trading_opt workflow full-pipeline --config pipelines/btc_prediction.yaml --data-source data/btc_prices.csv
```

## 9. Extension Points

The CLI is designed to be easily extended with new commands or functionality:

### 9.1 Adding New Commands

1. Create a new command class in `commands/` inheriting from `BaseCommand`
2. Implement `register_commands` and `_execute_command` methods
3. Register the new command class in `main.py`

### 9.2 Adding New Formatters

1. Create a new formatter class in `formatters/` inheriting from `BaseFormatter`
2. Implement the required formatter methods
3. Register the new formatter in `formatters/__init__.py`

### 9.3 Adding Custom Workflows

1. Create a new workflow command in `commands/workflow_commands.py`
2. Define the workflow as a sequence of operations
3. Implement progress tracking and error handling

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# Example tests for data commands
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import argparse

from trading_optimization.cli.commands.data_commands import DataCommands
from trading_optimization.cli.formatters.base import BaseFormatter
from trading_optimization.cli.exceptions import CommandError

class MockFormatter(BaseFormatter):
    def __init__(self):
        self.outputs = []
    
    def output_info(self, message):
        self.outputs.append(('info', message))
    
    def output_success(self, message):
        self.outputs.append(('success', message))
    
    def output_warning(self, message):
        self.outputs.append(('warning', message))
    
    def output_error(self, message):
        self.outputs.append(('error', message))
    
    def output_table(self, headers, rows, title=None):
        self.outputs.append(('table', headers, rows, title))
    
    def output_dict(self, data, title=None):
        self.outputs.append(('dict', data, title))
    
    @contextmanager
    def progress_bar(self, total=100):
        class MockProgress:
            def update(self, value):
                pass
        yield MockProgress()

class TestDataCommands(unittest.TestCase):
    
    def setUp(self):
        # Create command instance
        self.command = DataCommands()
        
        # Mock config
        self.command.config = MagicMock()
        
        # Create formatter
        self.formatter = MockFormatter()
    
    @patch('trading_optimization.data.dataset.DatasetManager')
    def test_list_datasets(self, mock_dataset_manager):
        # Setup mock
        manager_instance = MagicMock()
        manager_instance.list_raw_datasets.return_value = [
            {'name': 'dataset1', 'size': '100MB', 'last_modified': '2022-01-01'}
        ]
        manager_instance.list_processed_datasets.return_value = [
            {'name': 'dataset2', 'size': '50MB', 'last_modified': '2022-01-02'}
        ]
        mock_dataset_manager.return_value = manager_instance
        
        # Create args
        args = argparse.Namespace()
        
        # Execute command
        result = self.command.list_datasets(args, self.formatter)
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(len(self.formatter.outputs), 1)
        self.assertEqual(self.formatter.outputs[0][0], 'table')
        
        # Check that both manager methods were called
        manager_instance.list_raw_datasets.assert_called_once()
        manager_instance.list_processed_datasets.assert_called_once()
    
    @patch('trading_optimization.data.dataset.DatasetManager')
    def test_describe_dataset_not_found(self, mock_dataset_manager):
        # Setup mock
        manager_instance = MagicMock()
        manager_instance.dataset_exists.return_value = False
        mock_dataset_manager.return_value = manager_instance
        
        # Create args
        args = argparse.Namespace()
        args.dataset = 'unknown_dataset'
        args.stats = False
        
        # Execute command and verify exception
        with self.assertRaises(CommandError):
            self.command.describe_dataset(args, self.formatter)
            
        # Verify dataset_exists was called with correct arguments
        manager_instance.dataset_exists.assert_any_call('unknown_dataset', raw=True)
        manager_instance.dataset_exists.assert_any_call('unknown_dataset', processed=True)
```

### 10.2 Integration Tests

```python
# Example integration test for CLI
import unittest
import subprocess
import os
import shutil
import tempfile
import json

class CLIIntegrationTests(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data_path = os.path.join(self.temp_dir, 'test_data.csv')
        with open(self.test_data_path, 'w') as f:
            f.write('timestamp,price,volume\n')
            f.write('2022-01-01,50000,100\n')
            f.write('2022-01-02,51000,150\n')
            f.write('2022-01-03,49000,200\n')
        
        # Set environment variables for testing
        os.environ['TRADING_OPT_CONFIG_DIR'] = self.temp_dir
        
        # Create minimal config file
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write('data_dir: {}\n'.format(self.temp_dir))
            f.write('model_dir: {}\n'.format(os.path.join(self.temp_dir, 'models')))
            f.write('results_dir: {}\n'.format(os.path.join(self.temp_dir, 'results')))
        
        # Create directories
        os.makedirs(os.path.join(self.temp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'results'), exist_ok=True)
        
        # Set config environment variable
        os.environ['TRADING_OPT_CONFIG'] = config_path
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Remove environment variables
        if 'TRADING_OPT_CONFIG_DIR' in os.environ:
            del os.environ['TRADING_OPT_CONFIG_DIR']
        if 'TRADING_OPT_CONFIG' in os.environ:
            del os.environ['TRADING_OPT_CONFIG']
    
    def run_command(self, args, expected_exit_code=0):
        """Run CLI command and return output."""
        cmd = ['python', '-m', 'trading_optimization.cli.main'] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check exit code
        self.assertEqual(result.returncode, expected_exit_code, 
                        f"Command exited with {result.returncode}, output: {result.stderr}")
        
        return result.stdout, result.stderr
    
    def test_version(self):
        """Test version command."""
        stdout, stderr = self.run_command(['system', 'version'])
        self.assertIn('Trading Model Optimization Pipeline', stdout)
        self.assertIn('version', stdout)
    
    def test_data_import_export(self):
        """Test importing and exporting data."""
        # Import data
        stdout, stderr = self.run_command([
            'data', 'import',
            '--source', 'file',
            '--path', self.test_data_path,
            '--name', 'test_dataset',
            '--format', 'csv'
        ])
        
        self.assertIn('SUCCESS', stdout)
        self.assertIn('test_dataset', stdout)
        
        # List datasets
        stdout, stderr = self.run_command(['data', 'list'])
        self.assertIn('test_dataset', stdout)
        
        # Export dataset
        export_path = os.path.join(self.temp_dir, 'exported.json')
        stdout, stderr = self.run_command([
            'data', 'export',
            '--dataset', 'test_dataset',
            '--output', export_path,
            '--format', 'json'
        ])
        
        self.assertIn('SUCCESS', stdout)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIsInstance(exported_data, list)
        self.assertEqual(len(exported_data), 3)  # 3 rows in test data
```

## 11. Deployment Considerations

### 11.1 Installation

The CLI is installed as part of the main package installation:

```
pip install trading-optimization[cli]
```

### 11.2 Shell Completion

Add support for shell completion:

```python
# completion.py
import os
import json

def generate_bash_completion():
    """Generate Bash completion script."""
    completion_script = """
# Trading optimization CLI completion
_trading_opt_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Top-level commands
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        opts="data model tune evaluate strategy visualize system workflow"
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    # Subcommands
    if [[ ${COMP_CWORD} -eq 2 ]]; then
        case "${prev}" in
            data)
                opts="list import process describe export split generate-features"
                ;;
            model)
                opts="list train show export delete load copy"
                ;;
            tune)
                opts="list run show best-params trials export"
                ;;
            evaluate)
                opts="model compare list show export"
                ;;
            strategy)
                opts="list create backtest show optimize delete export"
                ;;
            visualize)
                opts="model-report strategy-report backtest-report comparison-report export dashboard"
                ;;
            system)
                opts="info cleanup setup config version"
                ;;
            workflow)
                opts="train-evaluate optimize-train full-pipeline custom"
                ;;
            *)
                opts=""
                ;;
        esac
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    # Options for common flags
    local common_opts="-v --verbose -q --quiet -p --profile -c --config -f --output-format"
    
    # Handle special options
    case "${prev}" in
        --dataset)
            # List available datasets
            local datasets=$(trading_opt data list --output-format json 2>/dev/null | \
                            python -c "import json,sys; data=json.load(sys.stdin); print(' '.join([d['name'] for d in data]))" 2>/dev/null)
            COMPREPLY=( $(compgen -W "${datasets}" -- ${cur}) )
            return 0
            ;;
        --model-id)
            # List available models
            local models=$(trading_opt model list --output-format json 2>/dev/null | \
                         python -c "import json,sys; data=json.load(sys.stdin); print(' '.join([m['id'] for m in data]))" 2>/dev/null)
            COMPREPLY=( $(compgen -W "${models}" -- ${cur}) )
            return 0
            ;;
        --profile)
            # List available profiles
            local profiles=$(trading_opt system config list-profiles --output-format json 2>/dev/null | \
                           python -c "import json,sys; data=json.load(sys.stdin); print(' '.join([p['name'] for p in data]))" 2>/dev/null)
            COMPREPLY=( $(compgen -W "${profiles}" -- ${cur}) )
            return 0
            ;;
        --output-format)
            COMPREPLY=( $(compgen -W "text json csv" -- ${cur}) )
            return 0
            ;;
        --config|--output)
            # File completion
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
    esac
    
    # Handle flag completion
    if [[ ${cur} == -* ]]; then
        case "${COMP_WORDS[1]}" in
            data)
                case "${COMP_WORDS[2]}" in
                    import)
                        opts="--source --path --name --format --options ${common_opts}"
                        ;;
                    process)
                        opts="--input --output --pipeline --force ${common_opts}"
                        ;;
                    # Add other data subcommands...
                esac
                ;;
            model)
                case "${COMP_WORDS[2]}" in
                    train)
                        opts="--config --dataset --val-dataset --name --save-checkpoints --checkpoint-dir --no-cache ${common_opts}"
                        ;;
                    # Add other model subcommands...
                esac
                ;;
            # Add other commands...
        esac
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    return 0
}

complete -F _trading_opt_completion trading_opt
    """
    return completion_script

def setup_completion():
    """Setup shell completion."""
    # Create completion directory
    home_dir = os.path.expanduser("~")
    completion_dir = os.path.join(home_dir, ".trading_optimization", "completion")
    os.makedirs(completion_dir, exist_ok=True)
    
    # Generate and save bash completion
    bash_script = generate_bash_completion()
    with open(os.path.join(completion_dir, "trading_opt-completion.bash"), "w") as f:
        f.write(bash_script)
    
    # Print instructions
    print("Shell completion scripts generated")
    print(f"To enable bash completion, add the following line to your .bashrc or .bash_profile:")
    print(f"source {os.path.join(completion_dir, 'trading_opt-completion.bash')}")
    
    # Check if ZSH is being used
    if os.environ.get("SHELL", "").endswith("zsh"):
        print("\nFor zsh users, add this line to your .zshrc:")
        print("autoload -U +X bashcompinit && bashcompinit")
        print(f"source {os.path.join(completion_dir, 'trading_opt-completion.bash')}")
```

### 11.3 Docker Deployment

Example Dockerfile for CLI deployment:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["trading_opt"]
```

## 12. Implementation Prerequisites

Before implementing the Command Line Interface, ensure that the following components are completed:

1. Configuration Management System
2. Data Management Module
3. Model Training Module
4. Hyperparameter Tuning System
5. Model Evaluation Infrastructure
6. Trading Strategy Integration
7. Visualization and Reporting Module

Dependencies to be installed:

```
click>=8.0.0
colorama>=0.4.4
tabulate>=0.8.9
pyyaml>=6.0.0
tqdm>=4.62.3
questionary>=1.10.0
rich>=10.0.0
```

## 13. Implementation Sequence

1. Create directory structure for CLI component
2. Implement the base formatter classes
3. Implement base command class
4. Implement data commands
5. Implement model commands
6. Implement other command groups
7. Implement main CLI entry point
8. Add configuration management
9. Add progress tracking and terminal utilities
10. Add unit and integration tests
11. Add shell completion support
12. Add documentation and examples
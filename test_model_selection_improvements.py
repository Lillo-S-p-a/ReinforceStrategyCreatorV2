#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to demonstrate improvements in model selection and training.
Compares the original approach (Sharpe-only) with our enhanced multi-metric selection
and advanced training techniques (transfer learning and ensemble models).
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import datetime
import logging
from copy import deepcopy
import re # Added for Datadog metric cleaning
from datadog import initialize, statsd # Added for Datadog
import os # Already present, but ensure it's used for env vars

from reinforcestrategycreator.backtesting.workflow import BacktestingWorkflow
from reinforcestrategycreator.backtesting.cross_validation import CrossValidator
from reinforcestrategycreator.backtesting.model import ModelTrainer

# Setup logging
def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"model_selection_test_{timestamp}.log"
logger = setup_logger(__name__, log_file)

class ModelSelectionTester:
    """Class to test and compare different model selection and training approaches."""
    
    def __init__(self, config_path, data_path, use_hpo=False):
        """
        Initialize with configuration and data paths.
        
        Args:
            config_path: Path to the configuration file
            data_path: Path to the data file
            use_hpo: Whether to use hyperparameter optimization
        """
        self.config_path = config_path
        self.data_path = data_path
        self.use_hpo = use_hpo
        self.results_dir = Path(f"test_results_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Check if config exists, if not create a default one for testing
        if not os.path.exists(config_path):
            logger.info(f"Config file not found at {config_path}, creating a default configuration")
            self._create_default_config(config_path)
        
        # Create base workflow
        try:
            # Load configuration from file
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Initialize the workflow with proper parameters
            self.base_workflow = BacktestingWorkflow(
                config=config,
                results_dir=str(self.results_dir),
                asset="SPY",
                start_date="2020-01-01",
                end_date="2023-01-01",
                cv_folds=5,
                test_ratio=0.2,
                use_hpo=self.use_hpo,
                hpo_num_samples=10,
                hpo_max_concurrent_trials=4
            )
            
            logger.info(f"Initialized model selection tester with config: {config_path}")
            logger.info(f"Results will be saved to: {self.results_dir}")
        except Exception as e:
            logger.error(f"Error initializing workflow: {str(e)}")
            # Create a minimal error report
            error_report_path = self.results_dir / "initialization_error.json"
            with open(error_report_path, 'w') as f:
                json.dump({
                    'error': str(e),
                    'config_path': config_path,
                    'data_path': data_path,
                    'timestamp': timestamp
                }, f, indent=4)
            logger.info(f"Error report saved to {error_report_path}")
            raise # Re-raise the exception to stop further execution in __init__
        # Datadog Initialization
        self.test_run_id = timestamp # Use the existing timestamp

        dd_options = {
            'api_key': os.environ.get('DATADOG_API_KEY'),
            'app_key': os.environ.get('DATADOG_APP_KEY'),
            'statsd_host': os.environ.get('DATADOG_AGENT_HOST', '127.0.0.1'), # Default to localhost
            'statsd_port': int(os.environ.get('DATADOG_AGENT_PORT', 8125))
        }
        self.datadog_enabled = bool(dd_options['api_key'] and dd_options['app_key'])

        if self.datadog_enabled:
            try:
                initialize(**dd_options)
                logger.info(f"Datadog client initialized. Metrics will be sent via agent at {dd_options['statsd_host']}:{dd_options['statsd_port']}.")
            except Exception as e:
                logger.error(f"Failed to initialize Datadog: {e}. Metrics will NOT be sent.")
                self.datadog_enabled = False # Ensure it's disabled on init error
        else:
            logger.warning("DATADOG_API_KEY or DATADOG_APP_KEY not set in environment. Datadog metrics will NOT be sent.")

        # Send initial global metrics
        if hasattr(self, 'base_workflow') and self.base_workflow: # Check if base_workflow was successfully initialized
            if hasattr(self.base_workflow, 'data') and self.base_workflow.data is not None:
                self._send_metric("data.total_rows", self.base_workflow.data.shape[0])
                if 'features' in self.base_workflow.config.get('data', {}):
                     self._send_metric("data.total_features", len(self.base_workflow.config['data']['features']))
            if hasattr(self.base_workflow, 'start_date') and self.base_workflow.start_date:
                try:
                    # Assuming start_date is a string 'YYYY-MM-DD' or datetime object
                    start_ts = pd.Timestamp(self.base_workflow.start_date).timestamp()
                    self._send_metric("data.start_date", start_ts)
                except Exception as e:
                    logger.debug(f"Could not parse start_date for Datadog: {self.base_workflow.start_date} - {e}")
            if hasattr(self.base_workflow, 'end_date') and self.base_workflow.end_date:
                try:
                    end_ts = pd.Timestamp(self.base_workflow.end_date).timestamp()
                    self._send_metric("data.end_date", end_ts)
                except Exception as e:
                    logger.debug(f"Could not parse end_date for Datadog: {self.base_workflow.end_date} - {e}")

            if hasattr(self.base_workflow, 'test_ratio'):
                self._send_metric("data.split.test_ratio", self.base_workflow.test_ratio)
            if hasattr(self.base_workflow, 'cv_folds'):
                 self._send_metric("data.split.cv_folds", self.base_workflow.cv_folds)
            
            self._send_metric("config.overall_hpo_enabled", 1 if self.use_hpo else 0)

            metric_weights = self.base_workflow.config.get('cross_validation', {}).get('metric_weights', {})
            if metric_weights:
                self._send_metrics_dict(metric_weights, prefix="config.metric_weights")
    
    def _clean_datadog_name(self, name_str):
        if not isinstance(name_str, str):
            name_str = str(name_str)
        s = name_str.lower()
        s = re.sub(r'[^\w_.]+', '_', s) # Allow word chars, underscore, period
        s = re.sub(r'_+', '_', s) # Replace multiple underscores with single
        s = s.strip('_')
        if not s: return "unknown"
        # Datadog metric names must start with a letter.
        if not re.match(r'^[a-zA-Z]', s):
            s = "metric_" + s
        return s

    def _clean_datadog_tag_value(self, value_str):
        if not isinstance(value_str, str):
            value_str = str(value_str)
        # Replace characters not allowed or problematic in tag values.
        # Allowed: letters, numbers, underscore, minus, colon, period, slash.
        # Keep it simple: convert to string, replace common problematic chars, ensure not empty.
        s = re.sub(r'[^\w\-.:\/]', '_', value_str)
        s = s.strip('_')
        if not s:
            return "unknown_tag_value"
        # Datadog tags should not start with a hyphen
        if s.startswith('-'):
            s = "tag_" + s
        return s[:200] # Enforce max length

    def _send_metric(self, metric_name, value, metric_type='gauge', tags=None):
        if not self.datadog_enabled:
            return

        if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value))):
            logger.debug(f"Skipping Datadog metric {metric_name} due to None/NaN/Inf value.")
            return

        clean_metric_name = self._clean_datadog_name(metric_name)

        # Convert win_rate to percentage if applicable
        # Check original metric_name as well as cleaned one
        if (isinstance(metric_name, str) and "win_rate" in metric_name.lower()) or \
           ("win_rate" in clean_metric_name.lower()):
            if isinstance(value, (int, float, np.number)) and \
               value is not None and not np.isnan(value) and not np.isinf(value):
                # Win_rate is provided as a decimal (e.g., 0.52).
                # Dashboard handles multiplying by 100 for display.
                # Do not convert here.
                logger.debug(f"Sending win_rate metric '{clean_metric_name}' with value {value} (dashboard will multiply by 100).")
        
        base_tags = [
            f"test_run_id:{self.test_run_id}",
            f"asset:{self._clean_datadog_name(self.base_workflow.asset if hasattr(self.base_workflow, 'asset') and self.base_workflow.asset else 'unknown')}",
            f"env:local_test"
        ]
        
        final_tags = base_tags[:]
        if tags: # Expect tags to be a list of "key:value" strings
            for tag_str in tags:
                if isinstance(tag_str, str) and ':' in tag_str:
                    k, v_raw = tag_str.split(':', 1)
                    k_clean = self._clean_datadog_name(k) # Clean the key
                    v_clean_tag_val = self._clean_datadog_tag_value(v_raw) # Use new cleaner for value
                    if v_clean_tag_val and v_clean_tag_val != "unknown_tag_value":
                       final_tags.append(f"{k_clean}:{v_clean_tag_val}")
                    else:
                        logger.debug(f"Skipping Datadog tag {k_clean} due to empty or unknown cleaned value from raw value: {v_raw}")
                else:
                     logger.warning(f"Invalid tag format for Datadog: '{tag_str}', skipping.")
        
        try:
            if metric_type == 'gauge':
                statsd.gauge(clean_metric_name, float(value), tags=final_tags)
            elif metric_type == 'count':
                statsd.increment(clean_metric_name, int(value), tags=final_tags)
            logger.debug(f"Sent Datadog metric: {clean_metric_name}={value}, Type: {metric_type}, Tags: {final_tags}")
        except Exception as e:
            logger.warning(f"Failed to send Datadog metric {clean_metric_name}: {e}")

    def _send_metrics_dict(self, metrics_dict, prefix='', tags=None, metric_type='gauge'):
        if not self.datadog_enabled or not isinstance(metrics_dict, dict):
            return
        
        clean_prefix = self._clean_datadog_name(prefix)
        if clean_prefix and not clean_prefix.endswith('.'): # Ensure prefix ends with a dot if it's not empty
            if prefix: # only add dot if original prefix was not empty
                 clean_prefix += "."
        
        for key, value in metrics_dict.items():
            # Value check is handled by _send_metric
            self._send_metric(f"{clean_prefix}{self._clean_datadog_name(key)}", value, metric_type=metric_type, tags=tags)

    def _send_event(self, title, text, alert_type='info', tags=None):
        if not self.datadog_enabled:
            return

        clean_title = str(title)[:100] # Max title length for Datadog events
        clean_text = str(text)[:4000] # Max text length

        base_tags = [
            f"test_run_id:{self.test_run_id}",
            f"asset:{self._clean_datadog_name(self.base_workflow.asset if hasattr(self.base_workflow, 'asset') and self.base_workflow.asset else 'unknown')}",
            f"env:local_test"
        ]
        
        final_tags = base_tags[:]
        if tags: # Expect tags to be a list of "key:value" strings
            for tag_str in tags:
                if isinstance(tag_str, str) and ':' in tag_str:
                    k, v_raw = tag_str.split(':', 1)
                    k_clean = self._clean_datadog_name(k)
                    v_clean_tag_val = self._clean_datadog_tag_value(v_raw)
                    if v_clean_tag_val and v_clean_tag_val != "unknown_tag_value":
                       final_tags.append(f"{k_clean}:{v_clean_tag_val}")
                    else:
                        logger.debug(f"Skipping Datadog event tag {k_clean} due to empty or unknown cleaned value from raw value: {v_raw}")
                else:
                     logger.warning(f"Invalid event tag format for Datadog: '{tag_str}', skipping.")
        
        try:
            statsd.event(
                clean_title,          # Pass title positionally
                clean_text,           # Pass text positionally
                alert_type=alert_type, # 'info', 'warning', 'error', 'success'
                tags=final_tags
            )
            logger.debug(f"Sent Datadog event: {clean_title}, Type: {alert_type}, Tags: {final_tags}")
        except Exception as e:
            logger.warning(f"Failed to send Datadog event {clean_title}: {e}")

    def _process_model_config_for_datadog(self, model_config, base_tags=None, prefix="model.config"):
        """
        Processes model_config for Datadog: sends numeric values as metrics
        and returns a list of tags derived from non-numeric values.
        """
        if not isinstance(model_config, dict):
            return []

        derived_tags = []
        # Ensure base_tags is a list to avoid modifying a potential None object
        current_tags_for_metrics = list(base_tags) if base_tags else []
        # Extract model_type to add to current_tags_for_metrics and for specific 'type' key handling
        model_type_value = model_config.get('type', 'unknown')
        # Create a mutable copy for current_tags_for_metrics to avoid modifying base_tags if it's passed around
        current_tags_for_metrics_local = list(current_tags_for_metrics) 

        if model_type_value != 'unknown':
            # Add model_type to tags used for numeric metrics sent within this function
            current_tags_for_metrics_local.append(f"model_type:{self._clean_datadog_name(model_type_value)}")

        for key, value in model_config.items():
            clean_key_for_metric_or_tag_name = self._clean_datadog_name(key)
            metric_name = f"{prefix}.{clean_key_for_metric_or_tag_name}"

            if isinstance(value, (int, float)):
                self._send_metric(metric_name, float(value), tags=current_tags_for_metrics_local)
            elif isinstance(value, bool):
                self._send_metric(metric_name, 1 if value else 0, tags=current_tags_for_metrics_local)
            elif isinstance(value, str):
                tag_value = self._clean_datadog_name(value)
                if tag_value and tag_value != "unknown":
                    # Special handling for 'type' key to rename to 'model_type' tag
                    if key == 'type':
                        derived_tags.append(f"model_type:{tag_value}")
                    else:
                        derived_tags.append(f"{clean_key_for_metric_or_tag_name}:{tag_value}")
            elif isinstance(value, list) and key == 'layers': # Specific handling for 'layers'
                if value: # Ensure list is not empty
                    derived_tags.append(f"num_layers:{len(value)}")
                    for i, layer_size in enumerate(value):
                        if isinstance(layer_size, int):
                            derived_tags.append(f"layer_{i}_size:{layer_size}")
                        else:
                            logger.debug(f"Layer size at index {i} is not an integer: {layer_size}")
                else:
                    derived_tags.append("num_layers:0")
            # Other types are ignored for direct metrics/tag generation from model_config
            # logger.debug(f"Skipping model_config key '{key}' with type {type(value)} for Datadog processing in _process_model_config_for_datadog")

        return derived_tags

    def _serialize_hpo_trials(self, hpo_data):
        """
        Serialize HPO trial results to JSON-compatible format.
        
        Args:
            hpo_data: HPO results data from Ray Tune, which may contain DataFrames
            
        Returns:
            dict: JSON-serializable representation of HPO data
        """
        if not hpo_data:
            return {}
            
        try:
            # Handle different types of HPO data structures
            if isinstance(hpo_data, dict):
                serialized = {}
                for key, value in hpo_data.items():
                    if hasattr(value, 'to_dict'):  # DataFrame or Series
                        if hasattr(value, 'orient'):  # DataFrame
                            serialized[key] = value.to_dict(orient='records')
                        else:  # Series
                            serialized[key] = value.to_dict()
                    elif hasattr(value, 'to_json'):  # Other pandas objects
                        import json
                        serialized[key] = json.loads(value.to_json())
                    elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                        serialized[key] = value
                    else:
                        # Convert other objects to string representation
                        serialized[key] = str(value)
                return serialized
            elif hasattr(hpo_data, 'to_dict'):  # DataFrame or Series
                if hasattr(hpo_data, 'orient'):  # DataFrame
                    return hpo_data.to_dict(orient='records')
                else:  # Series
                    return hpo_data.to_dict()
            elif hasattr(hpo_data, 'to_json'):  # Other pandas objects
                import json
                return json.loads(hpo_data.to_json())
            else:
                # For other types, try to convert to string
                return str(hpo_data)
                
        except Exception as e:
            logger.warning(f"Error serializing HPO data: {e}")
            return {"serialization_error": str(e), "original_type": str(type(hpo_data))}

    def _ensure_json_serializable(self, data):
        """
        Recursively ensure all data is JSON serializable.
        
        Args:
            data: Any data structure that needs to be JSON serializable
            
        Returns:
            JSON-serializable version of the data
        """
        try:
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    if key == 'hpo_full_results':
                        # Use specialized HPO serialization for this key
                        result[key] = self._serialize_hpo_trials(value)
                    else:
                        result[key] = self._ensure_json_serializable(value)
                return result
            elif isinstance(data, (list, tuple)):
                return [self._ensure_json_serializable(item) for item in data]
            elif hasattr(data, 'to_dict'):  # DataFrame or Series
                if hasattr(data, 'orient'):  # DataFrame
                    return data.to_dict(orient='records')
                else:  # Series
                    return data.to_dict()
            elif hasattr(data, 'to_json'):  # Other pandas objects
                import json
                return json.loads(data.to_json())
            elif isinstance(data, (str, int, float, bool, type(None))):
                return data
            else:
                # Convert other objects to string representation
                return str(data)
        except Exception as e:
            logger.warning(f"Error ensuring JSON serialization for {type(data)}: {e}")
            return {"serialization_error": str(e), "original_type": str(type(data))}

    def _create_default_config(self, config_path: str):
        """Create a default configuration file for testing."""
        import json
        
        default_config = {
            "model": {
                "type": "dqn",
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "batch_size": 32,
                "memory_size": 10000,
                "layers": [64, 32]
            },
            "training": {
                "episodes": 100,
                "steps_per_episode": 1000,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "hyperparameters": {
                "learning_rate": [0.001, 0.0001],
                "batch_size": [32, 64],
                "layers": [[64, 32], [128, 64]]
            },
            "cross_validation": {
                "folds": 5,
                "metric_weights": {
                    "sharpe_ratio": 0.4,
                    "pnl": 0.3,
                    "win_rate": 0.2,
                    "max_drawdown": 0.1
                }
            },
            "data": {
                "features": ["price", "volume", "ma_20", "ma_50", "rsi"],
                "target": "returns"
            },
            "random_seed": 42
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
            
        logger.info(f"Created default configuration at {config_path}")
        
    def run_original_approach(self):
        """Run the original model selection approach (Sharpe ratio only)."""
        logger.info("===== Running Original Approach (Sharpe ratio only) =====")
        approach_name = "original"
        start_time = datetime.datetime.now()
        error_count = 0
        execution_status = 0 # 0 for success, 1 for error

        try:
            # Create a copy of the workflow for original approach
            workflow = deepcopy(self.base_workflow)
            
            # Ensure the cross-validator is initialized
            if not hasattr(workflow, 'cross_validator'):
                workflow.cross_validator = CrossValidator(
                    train_data=workflow.train_data,
                    config=workflow.config,
                    cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                    models_dir=workflow.models_dir,
                    random_seed=workflow.random_seed
                )
            
            # Modify the cross-validator to use only Sharpe ratio (original behavior)
            workflow.cross_validator.use_multi_metric = False
            logger.info("Configured cross-validator to use only Sharpe ratio (original approach)")

            # Send model and training config metrics
            if hasattr(workflow, 'config'):
                model_config = workflow.config.get('model', {})
                approach_specific_tags = [f"approach_name:{approach_name}"]
                
                # Process model_config: sends numeric metrics directly and gets tags from non-numeric parts
                model_config_derived_tags = self._process_model_config_for_datadog(
                    model_config,
                    base_tags=approach_specific_tags, # Pass current approach tag for metrics sent within
                    prefix="model.config"
                )
                # These tags (e.g., model_type:dqn, num_layers:2) will be added to other metrics
                all_model_related_tags = approach_specific_tags + model_config_derived_tags
                
                # training_config is typically all numeric, can still use _send_metrics_dict
                training_config = workflow.config.get('training', {})
                self._send_metrics_dict(training_config, prefix="training.config", tags=all_model_related_tags) # Add model tags here too

                # Send approach-specific configuration metrics that the dashboard queries
                if hasattr(self.base_workflow, 'test_ratio'):
                    self._send_metric("data.split.test_ratio", self.base_workflow.test_ratio, tags=approach_specific_tags)
                if hasattr(self.base_workflow, 'cv_folds'):
                    self._send_metric("data.split.cv_folds", self.base_workflow.cv_folds, tags=approach_specific_tags)
                
                # Send metric weights with approach-specific tags
                metric_weights = self.base_workflow.config.get('cross_validation', {}).get('metric_weights', {})
                if metric_weights:
                    self._send_metrics_dict(metric_weights, prefix="config.metric_weights", tags=approach_specific_tags)

                # Send combined config as an event for text widgets
                try:
                    config_event_text = f"**Model Config:**\n```json\n{json.dumps(model_config, indent=2)}\n```\n\n" \
                                        f"**Training Config:**\n```json\n{json.dumps(training_config, indent=2)}\n```"
                    self._send_event(
                        title=f"Configuration for {approach_name} approach",
                        text=config_event_text,
                        tags=approach_specific_tags # Just approach_name tag for the event itself
                    )
                except Exception as e:
                    logger.warning(f"Could not send configuration event for {approach_name}: {e}")

            # Check if we have training data
            if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                logger.info("Fetching training data...")
                try:
                    workflow.fetch_data()
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}")
                    # Create sample data directly in the workflow
                    logger.info("Attempting to create sample data directly")
                    from reinforcestrategycreator.backtesting.data import DataManager
                    workflow.data_manager = DataManager(
                        asset="SPY",
                        start_date="2020-01-01",
                        end_date="2023-01-01",
                        test_ratio=0.2
                    )
                    # Create minimal sample data
                    sample_data = self._create_minimal_sample_data()
                    workflow.data = sample_data
                    workflow.train_data = sample_data
                    workflow.test_data = sample_data.iloc[-50:].copy()

            # Send data split metrics
            if hasattr(workflow, 'train_data') and workflow.train_data is not None and not workflow.train_data.empty:
                self._send_metric("data.train.rows", workflow.train_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.train_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.train.start_date", workflow.train_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.train.end_date", workflow.train_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send train data date metrics for {approach_name}: {e}")
            if hasattr(workflow, 'test_data') and workflow.test_data is not None and not workflow.test_data.empty:
                self._send_metric("data.test.rows", workflow.test_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.test_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.test.start_date", workflow.test_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.test.end_date", workflow.test_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send test data date metrics for {approach_name}: {e}")
            
            # Run cross-validation
            logger.info("Running cross-validation with original Sharpe-only approach")
            try:
                workflow.run_cross_validation()
            except AttributeError:
                # If run_cross_validation doesn't exist, try perform_cross_validation instead
                logger.info("Using alternative cross-validation method")
                workflow.perform_cross_validation()
            
            # Select best model (original method)
            logger.info("Selecting best model using original approach")
            best_model_info = workflow.select_best_model()
            
            # Send CV best model metrics
            if best_model_info and isinstance(best_model_info.get('metrics'), dict):
                # Use all_model_related_tags which now includes model_type, layers_info etc.
                self._send_metrics_dict(best_model_info['metrics'], prefix="cv.best_model", tags=all_model_related_tags)
                if 'fold' in best_model_info: # fold_number is a more descriptive name
                     self._send_metric("cv.best_model.fold_number", best_model_info['fold'], tags=all_model_related_tags)

            # Train final model without advanced techniques
            logger.info("Training final model without advanced techniques")
            # Send training config for final model
            self._send_metric("training.final.use_transfer_learning", 0, tags=[f"approach_name:{approach_name}"])
            self._send_metric("training.final.use_ensemble", 0, tags=[f"approach_name:{approach_name}"])
            workflow.train_final_model(use_transfer_learning=False, use_ensemble=False)
            
            # Run final backtest
            logger.info("Running final backtest")
            final_results = workflow.evaluate_final_model()
            logger.info(f"Original approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")

            # Send final metrics
            if final_results and isinstance(final_results, dict):
                # Use all_model_related_tags which now includes model_type, layers_info etc.
                self._send_metrics_dict(final_results, prefix="final", tags=all_model_related_tags)
            
            # Store results
            self.original_results = {
                'best_model_info': best_model_info,
                'final_backtest': final_results
            }
            
            # Save results
            with open(self.results_dir / "original_approach_results.json", 'w') as f:
                json.dump({
                    'best_model_info': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in best_model_info.items() if k != 'model'
                    },
                    'final_metrics': final_results
                }, f, indent=4)
                
            logger.info(f"Original approach selected model from fold {best_model_info.get('fold', -1)}")
            metrics = best_model_info.get('metrics', {})
            
            # Ensure metrics are not None and have default values if keys are missing
            if metrics is None:
                metrics = {}
                
            logger.info(f"Metrics - Sharpe: {metrics.get('sharpe_ratio', 0.0):.4f}, "
                       f"PnL: ${metrics.get('pnl', 0.0):.2f}, "
                       f"Win Rate: {metrics.get('win_rate', 0.0)*100:.2f}%, "
                       f"Max Drawdown: {metrics.get('max_drawdown', 0.0)*100:.2f}%")
            
            # No explicit return here, will be handled by finally
            
        except Exception as e:
            logger.error(f"Error in original approach: {str(e)}")
            logger.exception("Exception details:")
            error_count += 1
            execution_status = 1
            self.original_results = { # Ensure original_results is set even on error
                'error': str(e)
            }
            # No explicit return here, will be handled by finally
        finally:
            duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self._send_metric("execution.duration_seconds", duration_seconds, tags=[f"approach_name:{approach_name}"])
            self._send_metric("execution.status", execution_status, tags=[f"approach_name:{approach_name}"])
            self._send_metric("execution.error.count", error_count, metric_type='count', tags=[f"approach_name:{approach_name}"])

            # Send final backtest metrics if available
            if hasattr(self, 'original_results') and self.original_results and 'final_backtest' in self.original_results:
                final_backtest_metrics = self.original_results.get('final_backtest')
                if isinstance(final_backtest_metrics, dict) and final_backtest_metrics:
                    # Add model_type tag if available from config
                    model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown')
                    final_tags = [f"approach_name:{approach_name}"]
                    if model_type_val != 'unknown':
                        final_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")
                    
                    self._send_metrics_dict(
                        final_backtest_metrics,
                        prefix="final",
                        tags=final_tags
                    )
                    logger.info(f"Sent final backtest metrics for {approach_name} to Datadog.")
                else:
                    logger.info(f"No final backtest metrics data to send for {approach_name}.")
            else:
                logger.info(f"original_results or final_backtest not found for {approach_name}, skipping final metrics sending.")
            
            # Save results or error report
            if hasattr(self, 'original_results'):
                if 'error' in self.original_results:
                    error_file_path = self.results_dir / f"{approach_name}_approach_error.json"
                    with open(error_file_path, 'w') as f:
                        json.dump(self.original_results, f, indent=4)
                    logger.info(f"Saved error report for {approach_name} to {error_file_path}")
                else:
                    results_file_path = self.results_dir / f"{approach_name}_approach_results.json"
                    with open(results_file_path, 'w') as f:
                        # Prepare data for JSON serialization
                        best_model_info_to_save = {}
                        if 'best_model_info' in self.original_results and self.original_results['best_model_info']:
                            best_model_info_to_save = {
                                k: v if not isinstance(v, np.ndarray) else v.tolist()
                                for k, v in self.original_results['best_model_info'].items() if k != 'model'
                            }
                        
                        final_metrics_to_save = {}
                        if 'final_backtest' in self.original_results and self.original_results['final_backtest']:
                             final_metrics_to_save = self.original_results['final_backtest']

                        json.dump({
                            'best_model_info': best_model_info_to_save,
                            'final_metrics': final_metrics_to_save
                        }, f, indent=4)
                    logger.info(f"Saved results for {approach_name} to {results_file_path}")

        # The logging of best_model_info metrics is already done before the try-except-finally,
        # and within the finally block, results (or errors) are saved.
        # The specific logging here is redundant and potentially error-prone if best_model_info is not set.

        return self.original_results # This should be at the same indentation level as the try-except-finally
    
    def run_enhanced_approach(self):
        """Run the enhanced model selection approach with advanced training."""
        logger.info("===== Running Enhanced Approach (Multi-metric + Advanced Training) =====")
        approach_name = "enhanced"
        start_time = datetime.datetime.now()
        error_count = 0
        execution_status = 0 # 0 for success, 1 for error
        
        try:
            # Create a copy of the workflow for enhanced approach
            workflow = deepcopy(self.base_workflow)
            
            # Ensure the cross-validator is initialized
            if not hasattr(workflow, 'cross_validator'):
                workflow.cross_validator = CrossValidator(
                    train_data=workflow.train_data,
                    config=workflow.config,
                    cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                    models_dir=workflow.models_dir,
                    random_seed=workflow.random_seed
                )
            
            # Ensure multi-metric selection is enabled
            workflow.cross_validator.use_multi_metric = True
            logger.info("Configured cross-validator to use multi-metric selection (enhanced approach)")

            # Send model and training config metrics
            if hasattr(workflow, 'config'):
                model_config = workflow.config.get('model', {})
                approach_specific_tags = [f"approach_name:{approach_name}"]

                # Process model_config: sends numeric metrics directly and gets tags from non-numeric parts
                model_config_derived_tags = self._process_model_config_for_datadog(
                    model_config,
                    base_tags=approach_specific_tags, # Pass current approach tag
                    prefix="model.config"
                )
                all_model_related_tags = approach_specific_tags + model_config_derived_tags
                
                training_config = workflow.config.get('training', {})
                self._send_metrics_dict(training_config, prefix="training.config", tags=all_model_related_tags) # Add model tags here too

                # Send approach-specific configuration metrics that the dashboard queries
                if hasattr(self.base_workflow, 'test_ratio'):
                    self._send_metric("data.split.test_ratio", self.base_workflow.test_ratio, tags=approach_specific_tags)
                if hasattr(self.base_workflow, 'cv_folds'):
                    self._send_metric("data.split.cv_folds", self.base_workflow.cv_folds, tags=approach_specific_tags)
                
                # Send metric weights with approach-specific tags
                metric_weights = self.base_workflow.config.get('cross_validation', {}).get('metric_weights', {})
                if metric_weights:
                    self._send_metrics_dict(metric_weights, prefix="config.metric_weights", tags=approach_specific_tags)

                # Send combined config as an event for text widgets
                try:
                    config_event_text = f"**Model Config:**\n```json\n{json.dumps(model_config, indent=2)}\n```\n\n" \
                                        f"**Training Config:**\n```json\n{json.dumps(training_config, indent=2)}\n```"
                    self._send_event(
                        title=f"Configuration for {approach_name} approach",
                        text=config_event_text,
                        tags=approach_specific_tags # Just approach_name tag for the event itself
                    )
                except Exception as e:
                    logger.warning(f"Could not send configuration event for {approach_name}: {e}")

            # Check if we have training data
            if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                logger.info("Fetching training data...")
                workflow.fetch_data()

            # Send data split metrics
            if hasattr(workflow, 'train_data') and workflow.train_data is not None and not workflow.train_data.empty:
                self._send_metric("data.train.rows", workflow.train_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.train_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.train.start_date", workflow.train_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.train.end_date", workflow.train_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send train data date metrics for {approach_name}: {e}")
            if hasattr(workflow, 'test_data') and workflow.test_data is not None and not workflow.test_data.empty:
                self._send_metric("data.test.rows", workflow.test_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.test_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.test.start_date", workflow.test_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.test.end_date", workflow.test_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send test data date metrics for {approach_name}: {e}")
            
            # Run cross-validation
            logger.info("Running cross-validation with enhanced multi-metric approach")
            try:
                workflow.run_cross_validation()
            except AttributeError:
                # If run_cross_validation doesn't exist, try perform_cross_validation instead
                logger.info("Using alternative cross-validation method")
                workflow.perform_cross_validation() # This populates workflow.cv_results (BacktestingWorkflow's list)

            # Ensure the workflow's cross_validator instance (CrossValidator type) has the results
            # before generating reports from it.
            if hasattr(workflow, 'cross_validator') and hasattr(workflow, 'cv_results'):
                if hasattr(workflow.cross_validator, 'cv_results'):
                    workflow.cross_validator.cv_results = workflow.cv_results
                    logger.info(f"Copied {len(workflow.cv_results)} CV results to workflow.cross_validator.cv_results for report generation.")
                else:
                    logger.warning("workflow.cross_validator does not have a cv_results attribute to copy to.")
            elif not hasattr(workflow, 'cross_validator'):
                logger.warning("workflow.cross_validator not found, cannot copy CV results for report generation.")
            elif not hasattr(workflow, 'cv_results'):
                logger.warning("workflow.cv_results not found, cannot copy CV results for report generation.")
            
            # Generate comprehensive CV report (text format)
            logger.info("Generating comprehensive cross-validation report")
            try:
                # Get text report
                cv_report_text = workflow.cross_validator.generate_cv_report()
                
                # Save text report
                cv_report_path = self.results_dir / "enhanced_cv_report.txt"
                with open(cv_report_path, 'w') as f:
                    f.write(str(cv_report_text))
                logger.info(f"Saved CV text report to {cv_report_path}")
                
                # Get DataFrame representation for visualization
                cv_report_df = workflow.cross_validator.generate_cv_dataframe()
                
                # Save DataFrame as CSV
                cv_df_path = self.results_dir / "enhanced_cv_report.csv"
                cv_report_df.to_csv(cv_df_path)
                logger.info(f"Saved CV DataFrame to {cv_df_path}")
                
                # Store both formats
                cv_report = {
                    'text': cv_report_text,
                    'dataframe': cv_report_df # cv_report_df is used for metrics
                }

                # Send CV report DataFrame metrics
                if cv_report_df is not None and not cv_report_df.empty:
                    # Iterate rows for per-fold/per-config metrics
                    for idx, row in cv_report_df.iterrows():
                        fold_id_val = row.get('Fold', row.get('fold', idx)) # Try to get Fold or fold column, else use index
                        model_config_val = row.get('Model Config', row.get('model_config', 'unknown')) # Try to get Model Config or model_config
                        
                        model_type_val = workflow.config.get('model', {}).get('type', 'unknown')
                        row_tags = [
                            f"approach_name:{approach_name}",
                            f"cv_fold_id:{self._clean_datadog_name(str(fold_id_val))}",
                            f"cv_model_config:{self._clean_datadog_name(str(model_config_val))}"
                        ]
                        if model_type_val != 'unknown':
                            row_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")
                        
                        # Send all numeric columns as metrics
                        for col_name, col_value in row.items():
                            if isinstance(col_value, (int, float, np.number)) and not pd.isna(col_value):
                                self._send_metric(f"cv.fold_detail.{self._clean_datadog_name(col_name)}", col_value, tags=row_tags)
                    
                    # Calculate and send aggregates for numeric columns
                    numeric_cols = cv_report_df.select_dtypes(include=np.number).columns
                    for col in numeric_cols:
                        # Skip columns that might be identifiers if they are numeric by chance
                        if col.lower() in ['fold', 'model_id', 'config_id']:
                            continue
                        model_type_val = workflow.config.get('model', {}).get('type', 'unknown')
                        agg_tags = [f"approach_name:{approach_name}"]
                        if model_type_val != 'unknown':
                            agg_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")
                                
                        self._send_metric(f"cv.all_folds.{self._clean_datadog_name(col)}.mean", cv_report_df[col].mean(), tags=agg_tags)
                        self._send_metric(f"cv.all_folds.{self._clean_datadog_name(col)}.std", cv_report_df[col].std(), tags=agg_tags)
                        self._send_metric(f"cv.all_folds.{self._clean_datadog_name(col)}.min", cv_report_df[col].min(), tags=agg_tags)
                        self._send_metric(f"cv.all_folds.{self._clean_datadog_name(col)}.max", cv_report_df[col].max(), tags=agg_tags)

            except Exception as e:
                logger.error(f"Error generating CV report: {str(e)}")
                cv_report = {
                    'text': str(e),
                    'dataframe': pd.DataFrame()  # Empty DataFrame
                }
            
            # Select best model using multi-metric approach
            logger.info("Selecting best model using enhanced multi-metric approach")
            best_model_info = workflow.select_best_model()
            
            # Send CV best model metrics
            if best_model_info and isinstance(best_model_info.get('metrics'), dict):
                # Use all_model_related_tags which now includes model_type, layers_info etc.
                self._send_metrics_dict(best_model_info['metrics'], prefix="cv.best_model", tags=all_model_related_tags)
                if 'fold' in best_model_info:
                     self._send_metric("cv.best_model.fold_number", best_model_info['fold'], tags=all_model_related_tags)

            # Train final model with advanced techniques
            logger.info("Training final model with transfer learning and ensemble techniques")
            # Send training config for final model
            self._send_metric("training.final.use_transfer_learning", 1, tags=[f"approach_name:{approach_name}"])
            self._send_metric("training.final.use_ensemble", 1, tags=[f"approach_name:{approach_name}"])
            workflow.train_final_model(use_transfer_learning=True, use_ensemble=True)
            
            # Run final backtest
            logger.info("Running final backtest with enhanced model")
            final_results = workflow.evaluate_final_model()
            logger.info(f"Enhanced approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")

            # Send final metrics
            if final_results and isinstance(final_results, dict):
                # Use all_model_related_tags which now includes model_type, layers_info etc.
                self._send_metrics_dict(final_results, prefix="final", tags=all_model_related_tags)
            
            # Store results
            self.enhanced_results = {
                'best_model_info': best_model_info,
                'final_backtest': final_results,
                'cv_report': cv_report
            }
            
            # Save results
            with open(self.results_dir / "enhanced_approach_results.json", 'w') as f:
                json.dump({
                    'best_model_info': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in best_model_info.items() if k != 'model'
                    },
                    'final_metrics': final_results
                }, f, indent=4)
                
            logger.info(f"Enhanced approach selected model from fold {best_model_info.get('fold', -1)}")
            metrics = best_model_info.get('metrics', {})
            
            # Ensure metrics are not None and have default values if keys are missing
            if metrics is None:
                metrics = {}
                
            logger.info(f"Metrics - Sharpe: {metrics.get('sharpe_ratio', 0.0):.4f}, "
                       f"PnL: ${metrics.get('pnl', 0.0):.2f}, "
                       f"Win Rate: {metrics.get('win_rate', 0.0)*100:.2f}%, "
                       f"Max Drawdown: {metrics.get('max_drawdown', 0.0)*100:.2f}%")
            
            # No explicit return here, will be handled by finally

        except Exception as e:
            logger.error(f"Error in enhanced approach: {str(e)}")
            logger.exception("Exception details:")
            error_count += 1
            execution_status = 1
            self.enhanced_results = { # Ensure enhanced_results is set even on error
                'error': str(e),
                'cv_report': cv_report if 'cv_report' in locals() else {'text': str(e), 'dataframe': pd.DataFrame()} # Store CV report if available
            }
            # No explicit return here, will be handled by finally
        finally:
            duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self._send_metric("execution.duration_seconds", duration_seconds, tags=[f"approach_name:{approach_name}"])
            self._send_metric("execution.status", execution_status, tags=[f"approach_name:{approach_name}"])
            self._send_metric("execution.error.count", error_count, metric_type='count', tags=[f"approach_name:{approach_name}"])

            # Send final backtest metrics if available
            if hasattr(self, 'enhanced_results') and self.enhanced_results and 'final_backtest' in self.enhanced_results:
                final_backtest_metrics = self.enhanced_results.get('final_backtest')
                if isinstance(final_backtest_metrics, dict) and final_backtest_metrics:
                    logger.info(f"DEBUG: Enhanced approach final_backtest_metrics before sending: {final_backtest_metrics}")
                    # Add model_type tag if available from config
                    model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown')
                    final_tags = [f"approach_name:{approach_name}"]
                    if model_type_val != 'unknown':
                        final_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")

                    self._send_metrics_dict(
                        final_backtest_metrics,
                        prefix="final",
                        tags=final_tags
                    )
                    logger.info(f"Sent final backtest metrics for {approach_name} to Datadog.")
                else:
                    logger.info(f"No final backtest metrics data to send for {approach_name}.")
            else:
                logger.info(f"enhanced_results or final_backtest not found for {approach_name}, skipping final metrics sending.")
            
            # Save results or error report
            if hasattr(self, 'enhanced_results'):
                if 'error' in self.enhanced_results:
                    error_file_path = self.results_dir / f"{approach_name}_approach_error.json"
                    with open(error_file_path, 'w') as f:
                        # Prepare data for JSON serialization, especially the DataFrame in cv_report
                        error_data_to_save = self.enhanced_results.copy()
                        if 'cv_report' in error_data_to_save and isinstance(error_data_to_save['cv_report'].get('dataframe'), pd.DataFrame):
                            error_data_to_save['cv_report']['dataframe'] = error_data_to_save['cv_report']['dataframe'].to_dict(orient='records')
                        json.dump(error_data_to_save, f, indent=4)
                    logger.info(f"Saved error report for {approach_name} to {error_file_path}")
                else:
                    results_file_path = self.results_dir / f"{approach_name}_approach_results.json"
                    with open(results_file_path, 'w') as f:
                        # Prepare data for JSON serialization
                        best_model_info_to_save = {}
                        if 'best_model_info' in self.enhanced_results and self.enhanced_results['best_model_info']:
                            best_model_info_to_save = {
                                k: v if not isinstance(v, np.ndarray) else v.tolist()
                                for k, v in self.enhanced_results['best_model_info'].items() if k != 'model'
                            }
                        
                        final_metrics_to_save = {}
                        if 'final_backtest' in self.enhanced_results and self.enhanced_results['final_backtest']:
                             final_metrics_to_save = self.enhanced_results['final_backtest']
                        
                        # CV report text is already string, DataFrame needs conversion
                        cv_report_to_save = {}
                        if 'cv_report' in self.enhanced_results and self.enhanced_results['cv_report']:
                            cv_report_to_save['text'] = self.enhanced_results['cv_report'].get('text', '')
                            if isinstance(self.enhanced_results['cv_report'].get('dataframe'), pd.DataFrame):
                                cv_report_to_save['dataframe_csv_path'] = str(self.results_dir / "enhanced_cv_report.csv") # Already saved
                            else:
                                cv_report_to_save['dataframe'] = {}


                        json.dump({
                            'best_model_info': best_model_info_to_save,
                            'final_metrics': final_metrics_to_save
                            # CV report text and df are saved separately already by the main logic
                        }, f, indent=4)
                    logger.info(f"Saved results for {approach_name} to {results_file_path}")
        
        return self.enhanced_results
    
    def run_ablation_study(self):
        """Run ablation study to test the impact of individual enhancements."""
        logger.info("===== Running Ablation Study =====")
        
        configurations = [
            {
                'name': 'multi_metric_only',
                'description': 'Multi-metric selection only',
                'use_multi_metric': True,
                'use_transfer_learning': False,
                'use_ensemble': False,
            },
            {
                'name': 'transfer_learning_only',
                'description': 'Transfer learning only',
                'use_multi_metric': False,
                'use_transfer_learning': True,
                'use_ensemble': False,
            },
            {
                'name': 'ensemble_only',
                'description': 'Ensemble model only',
                'use_multi_metric': False,
                'use_transfer_learning': False,
                'use_ensemble': True,
            }
        ]
        # Note: HPO specific ablation is not included here as it's better handled by run_hpo_approach
        
        ablation_results = {} # This will store results for each config_item
        
        for config_item in configurations:
            approach_name = f"ablation_{config_item['name']}"
            logger.info(f"===== Running Ablation: {config_item['description']} (Datadog approach_name: {approach_name}) =====")
            start_time = datetime.datetime.now()
            # Reset error count and status for each ablation configuration
            error_count_for_this_ablation = 0
            execution_status_for_this_ablation = 0 # 0 for success, 1 for error
            
            try:
                workflow = deepcopy(self.base_workflow)
                
                if not hasattr(workflow, 'cross_validator'):
                    workflow.cross_validator = CrossValidator(
                        train_data=workflow.train_data,
                        config=workflow.config,
                        cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                        models_dir=workflow.models_dir,
                        random_seed=workflow.random_seed
                    )
                
                workflow.cross_validator.use_multi_metric = config_item['use_multi_metric']
                logger.info(f"Configured cross-validator with multi-metric={config_item['use_multi_metric']} for {approach_name}")

                if hasattr(workflow, 'config'):
                    model_config_abl = workflow.config.get('model', {})
                    approach_specific_tags_abl = [f"approach_name:{approach_name}"]
                    
                    # Process model_config_abl: sends numeric metrics directly and gets tags from non-numeric parts
                    model_config_derived_tags_abl = self._process_model_config_for_datadog(
                        model_config_abl,
                        base_tags=approach_specific_tags_abl, # Pass current approach tag for metrics sent within
                        prefix="model.config"
                    )
                    # These tags (e.g., model_type:dqn, num_layers:2) will be added to other metrics
                    all_model_related_tags_abl = approach_specific_tags_abl + model_config_derived_tags_abl
                    
                    training_config_abl = workflow.config.get('training', {})
                    # Add model tags (derived from model_config_abl) to training_config metrics
                    self._send_metrics_dict(training_config_abl, prefix="training.config", tags=all_model_related_tags_abl)

                    # Send approach-specific configuration metrics that the dashboard queries
                    if hasattr(self.base_workflow, 'test_ratio'):
                        self._send_metric("data.split.test_ratio", self.base_workflow.test_ratio, tags=approach_specific_tags_abl)
                    if hasattr(self.base_workflow, 'cv_folds'):
                        self._send_metric("data.split.cv_folds", self.base_workflow.cv_folds, tags=approach_specific_tags_abl)
                    
                    # Send metric weights with approach-specific tags
                    metric_weights = self.base_workflow.config.get('cross_validation', {}).get('metric_weights', {})
                    if metric_weights:
                        self._send_metrics_dict(metric_weights, prefix="config.metric_weights", tags=approach_specific_tags_abl)

                if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                    logger.info(f"Fetching training data for {approach_name}...")
                    workflow.fetch_data()

                if hasattr(workflow, 'train_data') and workflow.train_data is not None and not workflow.train_data.empty:
                    self._send_metric("data.train.rows", workflow.train_data.shape[0], tags=[f"approach_name:{approach_name}"])
                    if isinstance(workflow.train_data.index, pd.DatetimeIndex):
                        try:
                            self._send_metric("data.train.start_date", workflow.train_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                            self._send_metric("data.train.end_date", workflow.train_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                        except Exception as e:
                            logger.debug(f"Could not send train data date metrics for {approach_name}: {e}")
                if hasattr(workflow, 'test_data') and workflow.test_data is not None and not workflow.test_data.empty:
                    self._send_metric("data.test.rows", workflow.test_data.shape[0], tags=[f"approach_name:{approach_name}"])
                    if isinstance(workflow.test_data.index, pd.DatetimeIndex):
                        try:
                            self._send_metric("data.test.start_date", workflow.test_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                            self._send_metric("data.test.end_date", workflow.test_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                        except Exception as e:
                            logger.debug(f"Could not send test data date metrics for {approach_name}: {e}")
                
                logger.info(f"Running cross-validation with {config_item['description']} configuration")
                try:
                    workflow.run_cross_validation()
                except AttributeError:
                    logger.info(f"Using alternative cross-validation method for {approach_name}")
                    workflow.perform_cross_validation()
                
                logger.info(f"Selecting best model for {approach_name}")
                best_model_info = workflow.select_best_model()

                if best_model_info and isinstance(best_model_info.get('metrics'), dict):
                    self._send_metrics_dict(best_model_info['metrics'], prefix="cv.best_model", tags=all_model_related_tags_abl)
                    if 'fold' in best_model_info:
                        self._send_metric("cv.best_model.fold_number", best_model_info['fold'], tags=all_model_related_tags_abl)

                logger.info(f"Training final model for {approach_name} with transfer_learning={config_item['use_transfer_learning']}, ensemble={config_item['use_ensemble']}")
                self._send_metric("training.final.use_transfer_learning", 1 if config_item['use_transfer_learning'] else 0, tags=[f"approach_name:{approach_name}"])
                self._send_metric("training.final.use_ensemble", 1 if config_item['use_ensemble'] else 0, tags=[f"approach_name:{approach_name}"])
                workflow.train_final_model(
                    use_transfer_learning=config_item['use_transfer_learning'],
                    use_ensemble=config_item['use_ensemble']
                )
                
                logger.info(f"Running final backtest for {approach_name}")
                final_results = workflow.evaluate_final_model()
                logger.info(f"{config_item['description']} final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")

                if final_results and isinstance(final_results, dict):
                    self._send_metrics_dict(final_results, prefix="final", tags=all_model_related_tags_abl)
                
                ablation_results[config_item['name']] = {
                    'best_model_info': best_model_info,
                    'final_backtest': final_results,
                    'configuration_details': config_item
                }
                # execution_status remains 0 (success)
                
            except Exception as e:
                logger.error(f"Error in {config_item['description']} configuration ({approach_name}): {str(e)}")
                logger.exception("Exception details:")
                error_count_for_this_ablation = 1
                execution_status_for_this_ablation = 1
                ablation_results[config_item['name']] = {'error': str(e), 'configuration_details': config_item}
            
            finally:
                duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
                self._send_metric("execution.duration_seconds", duration_seconds, tags=[f"approach_name:{approach_name}"])
                self._send_metric("execution.status", execution_status_for_this_ablation, tags=[f"approach_name:{approach_name}"])
                self._send_metric("execution.error.count", error_count_for_this_ablation, metric_type='count', tags=[f"approach_name:{approach_name}"])

                current_run_result = ablation_results.get(config_item['name'], {})
                if 'error' in current_run_result:
                    error_report_path = self.results_dir / f"ablation_{config_item['name']}_error.json"
                    with open(error_report_path, 'w') as f:
                        json.dump(current_run_result, f, indent=4)
                    logger.info(f"Saved error report for {config_item['name']} to {error_report_path}")
                elif current_run_result:
                    results_path = self.results_dir / f"ablation_{config_item['name']}_results.json"
                    with open(results_path, 'w') as f:
                        best_model_info_to_save = {}
                        if 'best_model_info' in current_run_result and isinstance(current_run_result['best_model_info'], dict):
                             best_model_info_to_save = {
                                k: v if not isinstance(v, np.ndarray) else v.tolist()
                                for k, v in current_run_result['best_model_info'].items() if k != 'model'
                            }
                        final_metrics_to_save = current_run_result.get('final_backtest', {})
                        if not isinstance(final_metrics_to_save, dict):
                            final_metrics_to_save = {}
                        
                        json.dump({
                            'best_model_info': best_model_info_to_save,
                            'final_metrics': final_metrics_to_save,
                            'configuration_details': current_run_result.get('configuration_details', {})
                        }, f, indent=4)
                    logger.info(f"Saved results for {config_item['name']} to {results_path}")

                    if 'best_model_info' in current_run_result and isinstance(current_run_result.get('best_model_info'), dict):
                        log_best_model_info = current_run_result['best_model_info']
                        logger.info(f"Configuration '{config_item['name']}' selected model from fold {log_best_model_info.get('fold', -1)}")
                        log_metrics = log_best_model_info.get('metrics', {})
                        if log_metrics is None: log_metrics = {}
                        logger.info(f"Metrics - Sharpe: {log_metrics.get('sharpe_ratio', 0.0):.4f}, "
                                   f"PnL: ${log_metrics.get('pnl', 0.0):.2f}, "
                                   f"Win Rate: {log_metrics.get('win_rate', 0.0)*100:.2f}%, "
                                   f"Max Drawdown: {log_metrics.get('max_drawdown', 0.0)*100:.2f}%")
        
        self.ablation_results = ablation_results
        return ablation_results
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report of all tested approaches."""
        logger.info("Generating comparison report")
        
        # Ensure we have results to compare
        if not hasattr(self, 'original_results') or not hasattr(self, 'enhanced_results'):
            logger.error("Missing results for comparison. Run both approaches first.")
            return None
        
        # Create DataFrame for metrics comparison
        approaches = ['Original', 'Enhanced']
        metrics_data = []
        
        # Helper function to safely extract metrics
        def extract_metrics(result_dict, is_final=False):
            try:
                if 'error' in result_dict:
                    logger.warning(f"Error found in results: {result_dict['error']}")
                    return [0.0, 0.0, 0.0, 0.0]  # Default values for error case
                
                if is_final:
                    metrics = result_dict['final_backtest']
                    return [
                        metrics.get('sharpe_ratio', 0.0),
                        metrics.get('pnl', 0.0),
                        metrics.get('win_rate', 0.0) * 100,
                        metrics.get('max_drawdown', 0.0) * 100
                    ]
                else:
                    # Check if we have CV report in the new format (dictionary with dataframe)
                    if 'cv_report' in result_dict and isinstance(result_dict['cv_report'], dict) and 'dataframe' in result_dict['cv_report']:
                        cv_df = result_dict['cv_report']['dataframe']
                        if not cv_df.empty:
                            # Calculate average metrics across all folds
                            return [
                                cv_df['sharpe_ratio'].mean(),
                                cv_df['pnl'].mean(),
                                cv_df['win_rate'].mean() * 100,
                                cv_df['max_drawdown'].mean() * 100
                            ]
                    
                    # Fall back to best_model_info if CV report is not available or empty
                    if 'best_model_info' in result_dict and 'metrics' in result_dict['best_model_info']:
                        metrics = result_dict['best_model_info']['metrics']
                        return [
                            metrics.get('sharpe_ratio', 0.0),
                            metrics.get('pnl', 0.0),
                            metrics.get('win_rate', 0.0) * 100,
                            metrics.get('max_drawdown', 0.0) * 100
                        ]
                
                # If we get here, we couldn't find metrics
                logger.warning(f"Could not find metrics in result_dict: {result_dict.keys()}")
                return [0.0, 0.0, 0.0, 0.0]  # Default values
                
            except Exception as e:
                logger.error(f"Error extracting metrics: {str(e)}")
                return [0.0, 0.0, 0.0, 0.0]  # Default values for error case
        
        # Add original approach metrics
        try:
            cv_metrics = extract_metrics(self.original_results, is_final=False)
            final_metrics = extract_metrics(self.original_results, is_final=True)
            metrics_data.append(cv_metrics + final_metrics)
        except Exception as e:
            logger.error(f"Error processing original approach metrics: {str(e)}")
            metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add enhanced approach metrics
        try:
            cv_metrics = extract_metrics(self.enhanced_results, is_final=False)
            final_metrics = extract_metrics(self.enhanced_results, is_final=True)
            metrics_data.append(cv_metrics + final_metrics)
        except Exception as e:
            logger.error(f"Error processing enhanced approach metrics: {str(e)}")
            metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add ablation study metrics if available
        if hasattr(self, 'ablation_results'):
            for config_name, results in self.ablation_results.items():
                approaches.append(config_name.replace('_', ' ').title())
                try:
                    if 'error' in results:
                        logger.warning(f"Error in ablation study {config_name}: {results['error']}")
                        metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        continue
                        
                    cv_metrics = extract_metrics(results, is_final=False)
                    final_metrics = extract_metrics(results, is_final=True)
                    metrics_data.append(cv_metrics + final_metrics)
                except Exception as e:
                    logger.error(f"Error processing ablation metrics for {config_name}: {str(e)}")
                    metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Create DataFrame
        columns = [
            'CV Sharpe', 'CV PnL ($)', 'CV Win Rate (%)', 'CV Max Drawdown (%)',
            'Final Sharpe', 'Final PnL ($)', 'Final Win Rate (%)', 'Final Max Drawdown (%)'
        ]
        df_comparison = pd.DataFrame(metrics_data, index=approaches, columns=columns)
        
        # Calculate improvement percentages
        improvement_row = pd.Series()
        try:
            for col in columns:
                original_val = df_comparison.loc['Original', col]
                enhanced_val = df_comparison.loc['Enhanced', col]
                
                # Avoid division by zero
                if original_val != 0:
                    improvement = ((enhanced_val - original_val) / abs(original_val)) * 100
                else:
                    # If original is zero, use absolute improvement or 100% if enhanced is positive
                    improvement = 100.0 if enhanced_val > 0 else 0.0
                    
                improvement_row[col] = improvement
                # Send this improvement metric for the 'Enhanced' vs 'Original'
                model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown')
                comp_tags = []
                if model_type_val != 'unknown':
                    comp_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")
                
                # Standardize metric name for "Enhanced vs Original" comparison as per dashboard
                # The dashboard query is `avg:comparison.improvement.final_sharpe_ratio` etc.
                # col is like 'Final Sharpe', 'CV PnL ($)'
                clean_col_name_for_metric = self._clean_datadog_name(col).replace('_$_', '_dollars') # e.g. final_sharpe, cv_pnl_dollars
                
                # Construct the metric name to match dashboard: comparison.improvement. + (final or cv) + _metric_name_ratio_pnl_etc
                # Example: col = "Final Sharpe" -> metric_name_part = "final_sharpe_ratio"
                # Example: col = "CV PnL ($)" -> metric_name_part = "cv_pnl" (assuming dashboard uses this)
                # Let's try to match the dashboard's `comparison.improvement.final_sharpe_ratio`
                if "final" in clean_col_name_for_metric.lower():
                    metric_name_part = "final_" + clean_col_name_for_metric.lower().split("final_")[-1]
                    if "sharpe" in metric_name_part and not metric_name_part.endswith("_ratio"):
                        metric_name_part += "_ratio"
                    # For other "final" metrics like PnL, Win Rate, Max Drawdown, the dashboard query is comparison.improvement.final_pnl etc.
                    # So, if clean_col_name_for_metric is "final_pnl_dollars", metric_name_part becomes "final_pnl_dollars"
                    # We need to ensure the metric name sent is exactly `comparison.improvement.final_pnl` or `comparison.improvement.final_sharpe_ratio`
                    if col == 'Final Sharpe':
                        target_metric_name = "comparison.improvement.final_sharpe_ratio"
                    elif col == 'Final PnL ($)':
                        target_metric_name = "comparison.improvement.final_pnl" # Dashboard might be `final_pnl`
                    elif col == 'Final Win Rate (%)':
                        target_metric_name = "comparison.improvement.final_win_rate"
                    elif col == 'Final Max Drawdown (%)':
                        target_metric_name = "comparison.improvement.final_max_drawdown"
                    else: # Skip if not one of the key comparison metrics for this widget
                        target_metric_name = None
                    
                    if target_metric_name:
                        self._send_metric(target_metric_name, improvement, tags=comp_tags)

            # Handle other approaches if they exist in df_comparison for improvement metrics
            # For example, if HPO results are added and we want its improvement over Original
            if 'HPO Final' in df_comparison.index and 'Original' in df_comparison.index:
                hpo_metrics_comp = df_comparison.loc['HPO Final']
                for col in columns: # Iterate through same metric columns
                    original_val_for_hpo = df_comparison.loc['Original', col]
                    hpo_val = hpo_metrics_comp[col]
                    if original_val_for_hpo != 0:
                        improvement_hpo = ((hpo_val - original_val_for_hpo) / abs(original_val_for_hpo)) * 100
                    else:
                        improvement_hpo = 100.0 if hpo_val > 0 else 0.0
                    # For HPO, we can use a tag to differentiate if dashboard is adapted, or send specific metric name
                    # For now, let's assume dashboard might add widgets for HPO comparison later
                    # To avoid conflict, we keep the specific HPO comparison name for now, or add a tag
                    model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown')
                    hpo_comp_tags = [f"comparison_pair:hpo_vs_original"] # Differentiate this comparison
                    if model_type_val != 'unknown':
                        hpo_comp_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")
                    self._send_metric(f"comparison.improvement.hpo_vs_original.{self._clean_datadog_name(col)}", improvement_hpo, tags=hpo_comp_tags)

        except Exception as e:
            logger.error(f"Error calculating improvement percentages: {str(e)}")
            # Create default improvement row
            improvement_row = pd.Series({col: 0.0 for col in columns})
        
        # For max drawdown, a negative percentage means improvement (less drawdown)
        for col in ['CV Max Drawdown (%)', 'Final Max Drawdown (%)']:
            improvement_row[col] = -improvement_row[col]
            
        # Add improvement row
        df_comparison.loc['% Improvement'] = improvement_row
        
        # Save to CSV
        report_path = self.results_dir / "approach_comparison.csv"
        df_comparison.to_csv(report_path)
        logger.info(f"Saved comparison report to {report_path}")
        
        # Generate visualizations
        self._generate_visualizations(df_comparison)
        
        return df_comparison
    
    def _generate_visualizations(self, df_comparison):
        """Generate visualizations comparing the different approaches."""
        logger.info("Generating visualization charts")
        
        # Bar chart comparing key metrics
        plt.figure(figsize=(14, 8))
        
        # Select key metrics for visualization
        key_metrics = ['Final Sharpe', 'Final PnL ($)', 'Final Win Rate (%)']
        
        # Select approaches (exclude % Improvement row)
        approaches = df_comparison.index[:-1]
        
        # Create grouped bar chart
        bar_width = 0.25
        x = np.arange(len(approaches))
        
        for i, metric in enumerate(key_metrics):
            plt.bar(x + i*bar_width, df_comparison.loc[approaches, metric], 
                    width=bar_width, label=metric)
        
        plt.xlabel('Approach')
        plt.ylabel('Value')
        plt.title('Comparison of Key Metrics Across Approaches')
        plt.xticks(x + bar_width, approaches)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        chart_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(chart_path)
        logger.info(f"Saved metrics comparison chart to {chart_path}")
        
        # Create improvement chart
        plt.figure(figsize=(10, 6))
        
        # Get improvement percentages (excluding max drawdown which is inverted)
        improvements = df_comparison.loc['% Improvement', 
                                         [c for c in df_comparison.columns if 'Max Drawdown' not in c]]
        
        # Create horizontal bar chart
        bars = plt.barh(improvements.index, improvements.values)
        
        # Color bars based on positive/negative values
        for i, bar in enumerate(bars):
            if improvements.values[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.xlabel('Improvement (%)')
        plt.title('Enhanced Approach % Improvement Over Original')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values to the end of each bar
        for i, v in enumerate(improvements.values):
            plt.text(v + 1, i, f"{v:.1f}%", va='center')
        
        # Save figure
        plt.tight_layout()
        chart_path = self.results_dir / "improvement_chart.png"
        plt.savefig(chart_path)
        logger.info(f"Saved improvement chart to {chart_path}")
        
        # If we have CV report, visualize fold performance
        if 'cv_report' in self.enhanced_results:
            # Pass the CV report to the visualization method
            # The method now handles different input types
            self._visualize_cv_performance(self.enhanced_results['cv_report'])
    
    def run_hpo_approach(self):
        """Run the hyperparameter optimization approach."""
        approach_name = "hpo_final"
        if not self.use_hpo:
            logger.info(f"===== Skipping HPO Approach ({approach_name}, use_hpo is False) =====")
            self.hpo_results = {'status': 'skipped', 'reason': 'use_hpo was False'}
            self._send_metric("execution.status", 2, tags=[f"approach_name:{approach_name}"]) # 2 for skipped
            return self.hpo_results
            
        logger.info(f"===== Running HPO Approach ({approach_name}) =====")
        start_time = datetime.datetime.now()
        error_count = 0
        execution_status = 0
        hpo_step_start_time = None

        try:
            workflow = deepcopy(self.base_workflow)
            workflow.use_hpo = True

            if hasattr(workflow, 'config'):
                model_config_hpo = workflow.config.get('model', {})
                approach_specific_tags_hpo = [f"approach_name:{approach_name}"]

                # Process initial model_config for HPO
                initial_model_config_derived_tags = self._process_model_config_for_datadog(
                    model_config_hpo,
                    base_tags=approach_specific_tags_hpo,
                    prefix="model.config.initial"
                )
                all_initial_model_tags_hpo = approach_specific_tags_hpo + initial_model_config_derived_tags
                
                training_config_hpo = workflow.config.get('training', {})
                # Send training config with initial model tags
                self._send_metrics_dict(training_config_hpo, prefix="training.config.initial", tags=all_initial_model_tags_hpo)

                # Send approach-specific configuration metrics that the dashboard queries
                if hasattr(self.base_workflow, 'test_ratio'):
                    self._send_metric("data.split.test_ratio", self.base_workflow.test_ratio, tags=approach_specific_tags_hpo)
                if hasattr(self.base_workflow, 'cv_folds'):
                    self._send_metric("data.split.cv_folds", self.base_workflow.cv_folds, tags=approach_specific_tags_hpo)
                
                # Send metric weights with approach-specific tags
                metric_weights = self.base_workflow.config.get('cross_validation', {}).get('metric_weights', {})
                if metric_weights:
                    self._send_metrics_dict(metric_weights, prefix="config.metric_weights", tags=approach_specific_tags_hpo)
            
            if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                logger.info(f"Fetching training data for {approach_name}...")
                workflow.fetch_data()

            if hasattr(workflow, 'train_data') and workflow.train_data is not None and not workflow.train_data.empty:
                self._send_metric("data.train.rows", workflow.train_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.train_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.train.start_date", workflow.train_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.train.end_date", workflow.train_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send train data date metrics for {approach_name}: {e}")
            if hasattr(workflow, 'test_data') and workflow.test_data is not None and not workflow.test_data.empty:
                self._send_metric("data.test.rows", workflow.test_data.shape[0], tags=[f"approach_name:{approach_name}"])
                if isinstance(workflow.test_data.index, pd.DatetimeIndex):
                    try:
                        self._send_metric("data.test.start_date", workflow.test_data.index.min().timestamp(), tags=[f"approach_name:{approach_name}"])
                        self._send_metric("data.test.end_date", workflow.test_data.index.max().timestamp(), tags=[f"approach_name:{approach_name}"])
                    except Exception as e:
                        logger.debug(f"Could not send test data date metrics for {approach_name}: {e}")
            
            logger.info(f"Performing hyperparameter optimization for {approach_name}...")
            hpo_step_start_time = datetime.datetime.now()
            hpo_results_data = workflow.perform_hyperparameter_optimization()
            hpo_duration_seconds = (datetime.datetime.now() - hpo_step_start_time).total_seconds()
            # Use the full tag set that includes model_type for HPO duration metric
            hpo_duration_tags = all_initial_model_tags_hpo  # This includes approach_name + model_type + other model config tags
            self._send_metric("hpo.duration_seconds", hpo_duration_seconds, tags=hpo_duration_tags)
            
            # HPO returns parameters directly, not nested under 'best_params'
            if not hpo_results_data or not isinstance(hpo_results_data, dict):
                logger.error(f"HPO ({approach_name}) did not return valid parameters. Aborting HPO approach.")
                self.hpo_results = {'error': 'HPO failed to find best parameters', 'hpo_full_results': hpo_results_data or {}}
                execution_status = 1
                
                # Even if HPO failed, try to send available metrics to Datadog
                if hpo_results_data:
                    logger.info(f"Sending available HPO data to Datadog despite failure: {hpo_results_data}")
                    
                    # Try to extract and send best parameters if available in any form
                    # Use the full tag set that includes model_type for failure case metrics too
                    failure_metric_tags = all_initial_model_tags_hpo
                    if isinstance(hpo_results_data, dict):
                        # Send individual parameters as metrics if they exist
                        for param_name, param_value in hpo_results_data.items():
                            if isinstance(param_value, (int, float, bool)):
                                self._send_metric(f"hpo.best_params.{param_name}", param_value, tags=failure_metric_tags)
                            elif param_name == 'layers' and isinstance(param_value, list):
                                # Handle layers specially - send as individual layer metrics
                                for i, layer_size in enumerate(param_value):
                                    if isinstance(layer_size, int):
                                        self._send_metric(f"hpo.best_params.layer_{i}_size", layer_size, tags=failure_metric_tags)
                    
                    # Try to send number of trials if available
                    num_trials_to_send = hpo_results_data.get('num_trials')
                    if num_trials_to_send is None and 'trials' in hpo_results_data and isinstance(hpo_results_data['trials'], list):
                        num_trials_to_send = len(hpo_results_data['trials'])
                    if num_trials_to_send is not None:
                        self._send_metric("hpo.num_trials", num_trials_to_send, tags=failure_metric_tags)
                    else:
                        # If no explicit trial count, assume at least 1 trial was attempted
                        logger.info("No trial count available, sending default value of 1")
                        self._send_metric("hpo.num_trials", 1, tags=failure_metric_tags)
                
                # Fall through to finally
            else:
                # HPO succeeded - hpo_results_data contains the parameters directly
                best_hpo_params = hpo_results_data
                logger.info(f"Best HPO parameters for {approach_name}: {best_hpo_params}")
                
                # Use the full tag set that includes model_type for HPO metrics
                hpo_metric_tags = all_initial_model_tags_hpo  # This includes approach_name + model_type + other model config tags
                self._send_metrics_dict(best_hpo_params, prefix="hpo.best_params", tags=hpo_metric_tags)
                
                # Get trial count from HPO optimizer
                num_trials_to_send = hpo_results_data.get('num_trials')
                if num_trials_to_send is None and 'trials' in hpo_results_data and isinstance(hpo_results_data['trials'], list):
                    num_trials_to_send = len(hpo_results_data['trials'])
                
                # If still no trial count, try to get it from the workflow's HPO optimizer
                if num_trials_to_send is None and hasattr(workflow, 'cross_validator') and hasattr(workflow.cross_validator, 'hyperparameter_optimizer'):
                    try:
                        hpo_results_list = workflow.cross_validator.hyperparameter_optimizer.get_hpo_results()
                        if isinstance(hpo_results_list, list):
                            num_trials_to_send = len(hpo_results_list)
                            logger.info(f"Retrieved trial count from HPO optimizer: {num_trials_to_send}")
                    except Exception as e:
                        logger.debug(f"Could not get trial count from HPO optimizer: {e}")
                
                if num_trials_to_send is not None:
                    self._send_metric("hpo.num_trials", num_trials_to_send, tags=hpo_metric_tags)
                else:
                    # Default fallback - assume at least 1 trial was completed since HPO succeeded
                    logger.info("No trial count available, sending default value of 1 for successful HPO")
                    self._send_metric("hpo.num_trials", 1, tags=hpo_metric_tags)

                workflow.update_config_with_hpo_params(best_hpo_params)
                logger.info(f"Workflow config updated with best HPO parameters for {approach_name}.")
                
                if hasattr(workflow, 'config'): # Send updated config
                    model_config_final = workflow.config.get('model', {})
                    # approach_specific_tags_hpo is already defined
                    
                    # Process final HPO model_config
                    final_model_config_derived_tags = self._process_model_config_for_datadog(
                        model_config_final,
                        base_tags=approach_specific_tags_hpo,
                        prefix="model.config.final_hpo"
                    )
                    all_final_model_tags_hpo = approach_specific_tags_hpo + final_model_config_derived_tags

                    training_config_final = workflow.config.get('training', {})
                    # Send final training config with final model tags
                    self._send_metrics_dict(training_config_final, prefix="training.config.final_hpo", tags=all_final_model_tags_hpo)

                # Optional: Re-run CV with optimized params if desired, then send cv.best_model.*
                # For now, directly training final model as per original logic.

                logger.info(f"Training final model with HPO-optimized parameters for {approach_name}...")
                self._send_metric("training.final.use_transfer_learning", 0, tags=hpo_metric_tags)
                self._send_metric("training.final.use_ensemble", 0, tags=hpo_metric_tags)
                workflow.train_final_model(use_transfer_learning=False, use_ensemble=False)
                
                logger.info(f"Running final backtest with HPO-optimized model for {approach_name}...")
                final_results = workflow.evaluate_final_model()
                logger.info(f"HPO approach ({approach_name}) final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")

                if final_results and isinstance(final_results, dict):
                    self._send_metrics_dict(final_results, prefix="final", tags=hpo_metric_tags)

                self.hpo_results = {
                    'best_hpo_params': best_hpo_params,
                    'final_backtest': final_results,
                    'hpo_full_results': hpo_results_data
                }
            
        except Exception as e:
            logger.error(f"Error in HPO approach ({approach_name}): {str(e)}")
            logger.exception("Exception details:")
            error_count +=1 # This should be error_count for the whole HPO approach
            execution_status = 1
            self.hpo_results = {'error': str(e), 'hpo_full_results': hpo_results_data if 'hpo_results_data' in locals() else {}}
        finally:
            total_duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
            model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown') # Get model_type from base_workflow config
            global_run_tags = []
            if model_type_val != 'unknown':
                global_run_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")

            self._send_metric("execution.duration_seconds", total_duration_seconds, tags=[f"approach_name:{approach_name}"] + global_run_tags) # Add global_run_tags here
            self._send_metric("execution.status", execution_status, tags=[f"approach_name:{approach_name}"] + global_run_tags) # Add global_run_tags here
            self._send_metric("execution.error.count", error_count, metric_type='count', tags=[f"approach_name:{approach_name}"] + global_run_tags) # Add global_run_tags here

            if hasattr(self, 'hpo_results'):
                if 'error' in self.hpo_results:
                    error_file_path = self.results_dir / f"{approach_name}_approach_error.json"
                    with open(error_file_path, 'w') as f:
                        # Serialize full HPO results carefully, even in error case if available
                        error_data_to_save = self.hpo_results.copy()
                        if 'hpo_full_results' in error_data_to_save and error_data_to_save['hpo_full_results']:
                             error_data_to_save['hpo_full_results'] = self._serialize_hpo_trials(error_data_to_save['hpo_full_results'])
                        json.dump(error_data_to_save, f, indent=4)
                    logger.info(f"Saved error report for {approach_name} to {error_file_path}")
                else: # Success case
                    results_file_path = self.results_dir / f"{approach_name}_approach_results.json"
                    with open(results_file_path, 'w') as f:
                        best_hpo_params_to_save = self.hpo_results.get('best_hpo_params', {})
                        final_metrics_to_save = self.hpo_results.get('final_backtest', {})
                        serializable_hpo_full_results = self._serialize_hpo_trials(self.hpo_results.get('hpo_full_results', {}))
                        
                        json.dump({
                            'best_hpo_params': best_hpo_params_to_save,
                            'final_metrics': final_metrics_to_save,
                            'hpo_summary': serializable_hpo_full_results
                        }, f, indent=4)
                    logger.info(f"Saved HPO results for {approach_name} to {results_file_path}")
            
        return self.hpo_results
    
    def _visualize_cv_performance(self, cv_report):
        """Create visualizations of cross-validation fold performance."""
        # Handle different input types
        if isinstance(cv_report, dict) and 'dataframe' in cv_report:
            # New format: dictionary with 'dataframe' key
            cv_df = cv_report['dataframe']
            logger.info(f"Using dataframe from cv_report dictionary, shape: {cv_df.shape if not cv_df.empty else 'empty'}")
        elif isinstance(cv_report, pd.DataFrame):
            # Direct DataFrame input
            cv_df = cv_report
            logger.info(f"Using direct DataFrame input, shape: {cv_df.shape if not cv_df.empty else 'empty'}")
        else:
            # If we have best_model_info, create a simple DataFrame from it
            if hasattr(self, 'enhanced_results') and 'best_model_info' in self.enhanced_results:
                logger.info("Creating DataFrame from best_model_info")
                best_info = self.enhanced_results['best_model_info']
                metrics = best_info.get('metrics', {})
                if metrics:
                    cv_df = pd.DataFrame([{
                        'fold': best_info.get('fold', 0),
                        'model_config': 'Config 0',
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                        'pnl': metrics.get('pnl', 0.0),
                        'win_rate': metrics.get('win_rate', 0.0),
                        'max_drawdown': metrics.get('max_drawdown', 0.0)
                    }])
                else:
                    cv_df = pd.DataFrame()
            else:
                # String or other type - create error visualization
                logger.error(f"Cannot visualize CV performance: cv_report is not a DataFrame but {type(cv_report)}")
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"CV Visualization Error:\nExpected DataFrame but got {type(cv_report)}",
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                
                chart_path = self.results_dir / "cv_visualization_error.png"
                plt.savefig(chart_path)
                logger.info(f"Saved error information to {chart_path}")
                return
        
        # Check if DataFrame is empty
        if cv_df is None or cv_df.empty:
            # Create a simple fallback visualization with the comparison data
            logger.warning("CV DataFrame is empty, creating fallback visualization from comparison data")
            
            # Try to create a synthetic DataFrame from the results we have
            try:
                # First check if we have any results from the approaches
                synthetic_rows = []
                
                # Check original results
                if hasattr(self, 'original_results') and 'best_model_info' in self.original_results:
                    metrics = self.original_results['best_model_info'].get('metrics', {})
                    if metrics and isinstance(metrics, dict):
                        synthetic_rows.append({
                            'fold': self.original_results['best_model_info'].get('fold', 0),
                            'model_config': 'Original',
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                            'pnl': metrics.get('pnl', 0.0),
                            'win_rate': metrics.get('win_rate', 0.0),
                            'max_drawdown': metrics.get('max_drawdown', 0.0)
                        })
                
                # Check enhanced results
                if hasattr(self, 'enhanced_results') and 'best_model_info' in self.enhanced_results:
                    metrics = self.enhanced_results['best_model_info'].get('metrics', {})
                    if metrics and isinstance(metrics, dict):
                        synthetic_rows.append({
                            'fold': self.enhanced_results['best_model_info'].get('fold', 0),
                            'model_config': 'Enhanced',
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                            'pnl': metrics.get('pnl', 0.0),
                            'win_rate': metrics.get('win_rate', 0.0),
                            'max_drawdown': metrics.get('max_drawdown', 0.0)
                        })
                
                # Check HPO results
                if hasattr(self, 'hpo_results') and 'best_model_info' in self.hpo_results: # Should be best_hpo_params or final_backtest
                    metrics = self.hpo_results.get('final_backtest', {}) # Use final_backtest for HPO comparison here
                    if metrics and isinstance(metrics, dict):
                        synthetic_rows.append({
                            'fold': 0, # HPO doesn't have folds in this context
                            'model_config': 'HPO',
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                            'pnl': metrics.get('pnl', 0.0),
                            'win_rate': metrics.get('win_rate', 0.0),
                            'max_drawdown': metrics.get('max_drawdown', 0.0)
                        })
                
                # Check ablation results
                if hasattr(self, 'ablation_results'):
                    for config_name, result in self.ablation_results.items():
                        if 'best_model_info' in result: # Ablation uses best_model_info from its CV
                            metrics = result['best_model_info'].get('metrics', {})
                            if metrics and isinstance(metrics, dict):
                                synthetic_rows.append({
                                    'fold': result['best_model_info'].get('fold', 0),
                                    'model_config': config_name.replace('_', ' ').title(),
                                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                                    'pnl': metrics.get('pnl', 0.0),
                                    'win_rate': metrics.get('win_rate', 0.0),
                                    'max_drawdown': metrics.get('max_drawdown', 0.0)
                                })
                
                # If we have synthetic data, use it
                if synthetic_rows:
                    logger.info(f"Created synthetic CV DataFrame with {len(synthetic_rows)} rows")
                    cv_df = pd.DataFrame(synthetic_rows)
                    
                    # Create a simple bar chart visualization
                    plt.figure(figsize=(14, 8))
                    
                    metrics_to_plot_synth = ['sharpe_ratio', 'pnl', 'win_rate', 'max_drawdown']
                    titles_synth = ['Sharpe Ratio (Best CV/Final)', 'PnL ($) (Best CV/Final)', 'Win Rate (Best CV/Final)', 'Max Drawdown (Best CV/Final)']
                    
                    for i, (metric, title) in enumerate(zip(metrics_to_plot_synth, titles_synth)):
                        plt.subplot(2, 2, i+1)
                        plt.bar(cv_df['model_config'], cv_df[metric])
                        plt.title(f'{title} by Approach')
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                    
                    chart_path = self.results_dir / "cv_performance_synthetic.png"
                    plt.savefig(chart_path)
                    logger.info(f"Saved synthetic CV visualization to {chart_path}")
                    
                    # Don't return here, let it proceed with the regular visualization if cv_df is now populated
                else:
                    # Try to use the comparison data if available
                    if hasattr(self, 'comparison_df') and not self.comparison_df.empty:
                        plt.figure(figsize=(14, 8))
                        
                        # Plot CV metrics from comparison data
                        metrics_comp = ['CV Sharpe', 'CV PnL ($)', 'CV Win Rate (%)', 'CV Max Drawdown (%)']
                        approaches_comp = self.comparison_df.index.tolist()
                        
                        for i, metric_c in enumerate(metrics_comp):
                            plt.subplot(2, 2, i+1)
                            values_c = self.comparison_df[metric_c].values
                            
                            # Skip % Improvement row if present
                            plot_approaches_c = [a for a in approaches_comp if 'Improvement' not in a]
                            if len(plot_approaches_c) < len(values_c): # if improvement row was there
                                values_c = values_c[:len(plot_approaches_c)]

                            plt.bar(plot_approaches_c, values_c)
                            plt.title(f'{metric_c} by Approach')
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                        
                        chart_path = self.results_dir / "cv_performance_fallback_from_comparison.png"
                        plt.savefig(chart_path)
                        logger.info(f"Saved fallback CV visualization from comparison df to {chart_path}")
                        return # Successfully created a fallback
                    else:
                        # Try to create a visualization from the comparison report CSV
                        try:
                            # Read the comparison CSV if it exists
                            comparison_path = self.results_dir / "approach_comparison.csv"
                            if comparison_path.exists():
                                logger.info(f"Reading comparison data from {comparison_path}")
                                temp_comparison_df = pd.read_csv(comparison_path, index_col=0)
                                
                                # Create a simple bar chart visualization
                                plt.figure(figsize=(14, 8))
                                
                                # Plot CV metrics from comparison data
                                metrics_csv = ['CV Sharpe', 'CV PnL ($)', 'CV Win Rate (%)', 'CV Max Drawdown (%)']
                                approaches_csv = temp_comparison_df.index.tolist()
                                
                                for i, metric_csv_col in enumerate(metrics_csv):
                                    if metric_csv_col not in temp_comparison_df.columns:
                                        logger.warning(f"Metric {metric_csv_col} not in comparison CSV, skipping for fallback viz.")
                                        continue
                                    plt.subplot(2, 2, i+1)
                                    values_csv = temp_comparison_df[metric_csv_col].values
                                    
                                    plot_approaches_csv = [a for a in approaches_csv if 'Improvement' not in a]
                                    if len(plot_approaches_csv) < len(values_csv):
                                         values_csv = values_csv[:len(plot_approaches_csv)]

                                    plt.bar(plot_approaches_csv, values_csv)
                                    plt.title(f'{metric_csv_col} by Approach (from CSV)')
                                    plt.xticks(rotation=45, ha="right")
                                    plt.tight_layout()
                                
                                chart_path = self.results_dir / "cv_performance_fallback_from_csv.png"
                                plt.savefig(chart_path)
                                logger.info(f"Saved fallback CV visualization from comparison CSV to {chart_path}")
                                return # Successfully created a fallback
                            else:
                                logger.warning(f"Comparison file not found at {comparison_path}")
                        except Exception as e_csv_fallback:
                            logger.error(f"Error creating fallback visualization from CSV: {str(e_csv_fallback)}")
                        
                        # If all else fails, show error
                        logger.error("Cannot visualize CV performance: DataFrame is empty and no comparison data available")
                        plt.figure(figsize=(10, 6))
                        plt.text(0.5, 0.5, "CV Visualization Error:\nNo CV data available for visualization",
                                ha='center', va='center', fontsize=12, color='red')
                        plt.axis('off')
                        
                        chart_path = self.results_dir / "cv_visualization_error_final.png"
                        plt.savefig(chart_path)
                        logger.info(f"Saved final error information to {chart_path}")
                        return # Cannot proceed with visualization
            except Exception as e_synth:
                logger.error(f"Error creating synthetic visualization: {str(e_synth)}")
                # Fallback to error message if synthetic creation fails badly
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "CV Visualization Error:\nError during synthetic data creation.",
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                chart_path = self.results_dir / "cv_visualization_error_synth_creation.png"
                plt.savefig(chart_path)
                logger.info(f"Saved synth creation error to {chart_path}")
                return # Cannot proceed
        
        # If cv_df is now populated (either originally or synthetically), proceed with main visualizations
        if cv_df is None or cv_df.empty: # Double check after potential synthetic population
            logger.info("CV DataFrame still empty after synthetic attempts, cannot create detailed CV visualizations.")
            return

        # Filter for key metrics if cv_df is valid
        metrics_to_plot = ['sharpe_ratio', 'pnl', 'win_rate', 'max_drawdown']
        # Ensure these columns exist in cv_df
        metrics_to_plot = [m for m in metrics_to_plot if m in cv_df.columns]
        if not metrics_to_plot:
            logger.warning("None of the target metrics for CV visualization are present in the CV DataFrame.")
            return


        # Create heatmap visualization
        plt.figure(figsize=(16, 12))
        
        # Ensure 'fold' and 'model_config' columns exist for pivot
        if 'fold' not in cv_df.columns or 'model_config' not in cv_df.columns:
            logger.warning("CV DataFrame missing 'fold' or 'model_config' columns for heatmap.")
        else:
            for i, metric in enumerate(metrics_to_plot):
                plt.subplot(2, 2, i+1)
                try:
                    metric_data = cv_df.pivot(index='fold', columns='model_config', values=metric)
                    im = plt.imshow(metric_data.values, cmap='viridis', aspect='auto')
                    plt.colorbar(im, label=metric)
                    plt.title(f'Cross-Validation Performance: {metric}')
                    plt.xlabel('Model Configuration')
                    plt.ylabel('Fold')
                    plt.xticks(range(len(metric_data.columns)),
                              [str(col) for col in metric_data.columns], # Use actual config names
                              rotation=45, ha="right")
                    plt.yticks(range(len(metric_data.index)), metric_data.index)
                except Exception as e_hm:
                    logger.error(f"Error creating heatmap for {metric}: {str(e_hm)}")
                    plt.text(0.5, 0.5, f"Error visualizing {metric}:\n{str(e_hm)}",
                            ha='center', va='center', fontsize=10, color='red')
                    plt.title(f'Cross-Validation Performance: {metric} (ERROR)')
            
            plt.tight_layout()
            chart_path = self.results_dir / "cv_performance_heatmap.png"
            plt.savefig(chart_path)
            logger.info(f"Saved cross-validation performance heatmap to {chart_path}")
        
        # Create parallel coordinates plot for multi-dimensional analysis
        plt.figure(figsize=(12, 6))
        if 'model_config' not in cv_df.columns:
            logger.warning("CV DataFrame missing 'model_config' column for parallel plot.")
        else:
            try:
                normalized_data = pd.DataFrame()
                for metric in metrics_to_plot:
                    if cv_df[metric].max() == cv_df[metric].min():
                        normalized_data[metric] = 0.5
                    elif metric == 'max_drawdown': # Higher is "worse" for drawdown
                        normalized_data[metric] = 1 - (cv_df[metric] - cv_df[metric].min()) / \
                                                 (cv_df[metric].max() - cv_df[metric].min())
                    else: # Higher is better for sharpe, pnl, win_rate
                        normalized_data[metric] = (cv_df[metric] - cv_df[metric].min()) / \
                                                 (cv_df[metric].max() - cv_df[metric].min())
                
                normalized_data['config'] = cv_df['model_config'] # Use 'config' for parallel_coordinates
                
                pd.plotting.parallel_coordinates(normalized_data, 'config', colormap='viridis')
                plt.title('Multi-Metric Performance by Model Configuration (Normalized)')
                plt.grid(True)
                plt.tight_layout()
                chart_path = self.results_dir / "multi_metric_parallel_plot.png"
                plt.savefig(chart_path)
                logger.info(f"Saved multi-metric parallel coordinates plot to {chart_path}")
            except Exception as e_pc:
                logger.error(f"Error creating parallel coordinates plot: {str(e_pc)}")
                # Error already plotted in the figure by this point if it fails
                plt.title('Multi-Metric Performance (ERROR)') # Ensure title is set
                plt.tight_layout() # Apply layout before saving
                chart_path_err = self.results_dir / "multi_metric_parallel_plot_error.png"
                if not plt.gcf().get_axes(): # If no axes, create a text error
                    plt.figure(figsize=(12,6)) # ensure figure exists
                    plt.text(0.5, 0.5, f"Error creating parallel plot:\n{str(e_pc)}", ha='center', va='center')
                    plt.axis('off')
                plt.savefig(chart_path_err)
                logger.info(f"Saved parallel plot error/figure to {chart_path_err}")
        
    def _create_minimal_sample_data(self):
        """Create a minimal sample DataFrame for testing when data fetching fails."""
        logger.info("Creating minimal sample data for testing")
        
        # Create date range for sample data (smaller dataset for faster testing)
        dates = pd.date_range(start='2020-01-01', end='2020-03-01', freq='D')
        
        # Generate base price data
        base_price = 100
        price_volatility = 2
        
        # Create sample data with required columns
        opens = np.random.normal(base_price, price_volatility, len(dates))
        closes = np.random.normal(base_price, price_volatility, len(dates))
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(1, 0.5, len(dates)))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(1, 0.5, len(dates)))
        
        data = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'price': closes,
            'ma_20': np.random.normal(base_price, price_volatility/2, len(dates)),
            'ma_50': np.random.normal(base_price, price_volatility/2, len(dates)),
            'rsi': np.random.uniform(30, 70, len(dates)),
            'returns': np.random.normal(0, 0.01, len(dates))
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data, index=dates)
        logger.info(f"Created minimal sample data with {len(df)} rows and columns: {', '.join(df.columns)}")
        return df

    def run_complete_test(self):
        """Run the complete test suite and generate comprehensive report."""
        logger.info("========== Starting Complete Model Selection Test Suite ==========")
        all_results = {}
        overall_test_start_time = datetime.datetime.now()
        overall_error_count = 0 # Counts errors from individual approaches
        success = True # Assume success initially

        try:
            logger.info("Step 1: Running original approach")
            original_results = self.run_original_approach()
            all_results['original'] = original_results
            if 'error' in original_results:
                logger.error(f"Original approach failed: {original_results['error']}")
                overall_error_count +=1
                success = False
            else:
                logger.info("Original approach completed.")

            logger.info("\nStep 2: Running enhanced approach")
            enhanced_results = self.run_enhanced_approach()
            all_results['enhanced'] = enhanced_results
            if 'error' in enhanced_results:
                logger.error(f"Enhanced approach failed: {enhanced_results['error']}")
                overall_error_count +=1
                success = False
            else:
                logger.info("Enhanced approach completed.")

            current_step = 3
            if self.use_hpo:
                logger.info(f"\nStep {current_step}: Running HPO approach")
                hpo_results = self.run_hpo_approach()
                all_results['hpo'] = hpo_results
                if 'error' in hpo_results:
                    logger.error(f"HPO approach failed: {hpo_results['error']}")
                    overall_error_count +=1
                    success = False
                elif hpo_results.get('status') == 'skipped':
                    logger.info("HPO approach was skipped.")
                else:
                    logger.info("HPO approach completed.")
                current_step += 1
            else:
                logger.info(f"\nStep {current_step}: Skipping HPO approach as use_hpo is False.")
                all_results['hpo'] = {'status': 'skipped_config'}
                # No increment to current_step here if HPO is skipped, step numbering will adjust.
                # Or, if we want to maintain step numbers: current_step += 1

            logger.info(f"\nStep {current_step}: Running ablation study")
            ablation_results = self.run_ablation_study() 
            all_results['ablation'] = ablation_results
            if isinstance(ablation_results, dict):
                for res_key, res_val in ablation_results.items():
                    if isinstance(res_val, dict) and 'error' in res_val:
                        overall_error_count +=1
                        success = False # Mark overall success as false if any ablation fails
                        logger.warning(f"Ablation config '{res_key}' reported an error.")
            logger.info("Ablation study completed.")
            current_step += 1

            logger.info(f"\nStep {current_step}: Generating comparison report")
            comparison_report_data = None # Renamed to avoid conflict
            try:
                comparison_report_data = self.generate_comparison_report()
                if comparison_report_data is not None:
                    logger.info("Comparison report generated.")
                    self.comparison_df = comparison_report_data # Store for potential use in CV viz fallback
                else: 
                    logger.warning("Comparison report generation skipped or failed (returned None).")
                    # Not necessarily an error to increment overall_error_count unless an exception occurred
            except Exception as report_exc:
                logger.error(f"Failed to generate comparison report: {report_exc}")
                all_results['comparison_report_error'] = str(report_exc)
                overall_error_count +=1 
                success = False
            current_step += 1
            
            logger.info(f"\nStep {current_step}: Generating CV performance visualizations (if applicable)")
            if hasattr(self, 'enhanced_results') and 'cv_report' in self.enhanced_results and isinstance(self.enhanced_results['cv_report'], dict) and not self.enhanced_results['cv_report'].get('dataframe', pd.DataFrame()).empty:
                try:
                    self._visualize_cv_performance(self.enhanced_results['cv_report'])
                    logger.info("CV performance visualization generated for enhanced approach.")
                except Exception as viz_exc:
                    logger.error(f"Failed to generate CV visualization: {viz_exc}")
                    all_results['cv_visualization_error'] = str(viz_exc)
                    overall_error_count +=1 
                    success = False
            else:
                logger.info("Skipping CV visualization as enhanced results or CV report DataFrame not found/empty.")
            
            if success:
                logger.info("========== Model Selection Test Suite Completed Successfully ==========")
            else:
                logger.warning("========== Model Selection Test Suite Completed With Errors/Skipped Steps ==========")
            
        except Exception as e:
            logger.error(f"Critical error in test suite execution: {str(e)}")
            logger.exception("Exception details:")
            success = False # Suite itself failed
            error_report_path = self.results_dir / "test_execution_error.json"
            with open(error_report_path, 'w') as f:
                # Ensure all_results is JSON serializable before dumping
                serializable_results = self._ensure_json_serializable(all_results)
                json.dump({
                    'error': str(e),
                    'timestamp': timestamp, # Use the global timestamp for consistency
                    'current_step_results': serializable_results
                }, f, indent=4)
            logger.info(f"Test execution error report saved to {error_report_path}")
            all_results['suite_error'] = str(e) 
        
        finally:
            total_duration_seconds = (datetime.datetime.now() - overall_test_start_time).total_seconds()
            model_type_val = self.base_workflow.config.get('model', {}).get('type', 'unknown')
            
            # Send overall test run metrics
            run_status_metric = 0 if success else 1 # 0 for success, 1 for failure
            self._send_metric("test_run.status", run_status_metric, metric_type='gauge', tags=[f"model_type:{self._clean_datadog_name(model_type_val)}"])
            self._send_metric("test_run.duration_seconds", total_duration_seconds, metric_type='gauge', tags=[f"model_type:{self._clean_datadog_name(model_type_val)}"])
            self._send_metric("test_run.error_count", overall_error_count, metric_type='count', tags=[f"model_type:{self._clean_datadog_name(model_type_val)}"])
            
            logger.info(f"Overall test suite execution time: {total_duration_seconds:.2f} seconds. Status: {'Success' if success else 'Failed'}. Errors: {overall_error_count}")
            global_run_tags = []
            if model_type_val != 'unknown':
                global_run_tags.append(f"model_type:{self._clean_datadog_name(model_type_val)}")

            self._send_metric("test.run.duration_seconds", total_duration_seconds, tags=global_run_tags)
            
            overall_run_status = 0 if success else 1 
            self._send_metric("test.run.status", overall_run_status, tags=global_run_tags)
            self._send_metric("test.run.total_error.count", overall_error_count, metric_type='count', tags=global_run_tags) # Errors are global, model_type might be less relevant or '*'
            logger.info(f"Overall test run status: {'Success' if success else 'Failed'}. Total errors from approaches: {overall_error_count}.")

        return all_results
def create_sample_data(data_path):
    """Create a sample market data file for testing if it doesn't exist."""
    try:
        if os.path.exists(data_path):
            logger.info(f"Using existing data file at {data_path}")
            return True
            
        logger.info(f"Sample data file not found at {data_path}, creating test data")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create date range for sample data
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        
        # Generate base price data
        base_price = 100
        price_volatility = 2
        
        # Create sample data with required columns
        # Ensure high > open/close and low < open/close for each day
        opens = np.random.normal(base_price, price_volatility, len(dates))
        closes = np.random.normal(base_price, price_volatility, len(dates))
        
        # Generate highs and lows that are consistent with open/close
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(1, 0.5, len(dates)))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(1, 0.5, len(dates)))
        
        data = {
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'price': closes,  # Add price column (same as close for simplicity)
            'ma_20': np.random.normal(base_price, price_volatility/2, len(dates)),
            'ma_50': np.random.normal(base_price, price_volatility/2, len(dates)),
            'rsi': np.random.uniform(30, 70, len(dates))
        }
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Ensure all required columns are present and properly formatted
        required_columns = ['high', 'low', 'close', 'price', 'volume', 'ma_20', 'ma_50', 'rsi']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Adding missing column: {col}")
                df[col] = df['close'] if col in ['price', 'high', 'low'] else np.random.normal(100, 1, len(df))
        
        # Add returns column
        df['returns'] = df['close'].pct_change().fillna(0)
        
        df.to_csv(data_path)
        
        logger.info(f"Created sample data file at {data_path} with {len(df)} rows and columns: {', '.join(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        return False

def main():
    """Main function to run the test."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Test model selection improvements')
        parser.add_argument('--config', type=str, default='config/backtesting_config.json', help='Path to configuration file')
        parser.add_argument('--data', type=str, default='data/processed_market_data.csv', help='Path to market data file')
        parser.add_argument('--hpo', action='store_true', help='Enable hyperparameter optimization')
        args = parser.parse_args()
        
        # Create sample data if needed
        if not create_sample_data(args.data):
            logger.error("Failed to create or verify sample data, cannot proceed with test")
            return None
            
        # Create tester
        logger.info(f"Initializing tester with config: {args.config} and data: {args.data}, HPO: {args.hpo}")
        tester = ModelSelectionTester(args.config, args.data, use_hpo=args.hpo)
        
        # Run the complete test
        logger.info("Starting complete test...")
        report = tester.run_complete_test()
        
        if report is not None:
            print("\nTest completed successfully.")
            print(f"Results saved to: {tester.results_dir}")
            print("\nComparison Report:")
            print(report)
        else:
            print("\nTest completed with errors. Check the logs for details.")
            print(f"Partial results may be available in: {tester.results_dir}")
        
        return report
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        logger.exception("Exception details:")
        print(f"Critical error: {str(e)}")
        print("Check the logs for details.")
        return None

if __name__ == "__main__":
    main()
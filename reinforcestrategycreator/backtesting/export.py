"""
Export module for backtesting.

This module provides functionality for exporting trained models
for use in paper trading or live trading environments.
"""

import os
import json
import logging
import datetime
import torch
import numpy as np
from typing import Dict, Any, Optional

from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types by converting them to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure logging
logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Exports trained models for trading.
    
    This class provides methods for exporting trained models and their
    metadata for use in paper trading or live trading environments.
    """
    
    def __init__(self, export_dir: str = "production_models") -> None:
        """
        Initialize the model exporter.
        
        Args:
            export_dir: Directory to export models
        """
        self.export_dir = export_dir
        
        # Create export directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_model(self, 
                    model: RLAgent,  # Using RLAgent alias
                    asset: str,
                    start_date: str,
                    end_date: str,
                    params: Dict[str, Any],
                    test_metrics: Dict[str, float],
                    benchmark_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """
        Export the model for paper/live trading.
        
        Args:
            model: Trained RL agent
            asset: Asset symbol
            start_date: Start date of training data
            end_date: End date of training data
            params: Model parameters
            test_metrics: Test performance metrics
            benchmark_metrics: Benchmark comparison metrics
            
        Returns:
            str: Path to the exported model file
        """
        logger.info("Exporting model for trading")
        
        if model is None:
            raise ValueError("No model available for export")
            
        try:
            # Generate model ID
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"model_ep{params.get('episodes', 100)}_run{asset}_{timestamp}"
            
            # Export model weights using PyTorch's saving mechanism
            model_path = os.path.join(self.export_dir, f"{model_id}.pth")
            
            # Check if model has the right attributes
            if hasattr(model, 'model') and model.model is not None:
                torch.save(model.model.state_dict(), model_path)
                logger.info(f"Model state_dict saved to {model_path}")
            else:
                logger.error("Model doesn't have the expected structure for saving")
                raise ValueError("Model doesn't have the required structure for saving")
            
            # Export model metadata with added model structure information
            metadata = {
                "model_id": model_id,
                "asset": asset,
                "training_period": f"{start_date} to {end_date}",
                "created_date": timestamp,
                "parameters": {
                    **params,  # Include all original parameters
                    "state_size": model.state_size if hasattr(model, 'state_size') else None,
                    "action_size": model.action_size if hasattr(model, 'action_size') else None,
                },
                "test_metrics": test_metrics
            }
            
            # Add benchmark comparison if available
            if benchmark_metrics:
                metadata["benchmark_comparison"] = {
                    name: {
                        "pnl_difference": test_metrics["pnl"] - metrics["pnl"],
                        "sharpe_difference": test_metrics["sharpe_ratio"] - metrics["sharpe_ratio"]
                    } for name, metrics in benchmark_metrics.items()
                }
            
            # Save metadata
            metadata_path = os.path.join(self.export_dir, f"{model_id}.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4, cls=NumpyEncoder)
            
            logger.info(f"Model exported successfully to {model_path}")
            logger.info(f"Model metadata saved to {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}", exc_info=True)
            raise
    
    def list_exported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all exported models and their metadata.
        
        Returns:
            Dictionary mapping model IDs to their metadata
        """
        models = {}
        
        try:
            # Find all JSON metadata files
            for filename in os.listdir(self.export_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.export_dir, filename)
                    
                    with open(file_path, "r") as f:
                        metadata = json.load(f)
                        
                    model_id = metadata.get("model_id")
                    if model_id:
                        models[model_id] = metadata
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing exported models: {e}", exc_info=True)
            return {}
    
    def load_model(self, model_id: str) -> Optional[RLAgent]:  # Using RLAgent alias
        """
        Load an exported model.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded RL agent or None if loading failed
        """
        try:
            # Check if model exists - now using .pth extension for PyTorch models
            model_path = os.path.join(self.export_dir, f"{model_id}.pth")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Get metadata to determine model parameters
            metadata_path = os.path.join(self.export_dir, f"{model_id}.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Model metadata file not found: {metadata_path}")
                return None
                
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            # Create a new agent with parameters from metadata
            params = metadata.get("parameters", {})
            state_size = params.get("state_size", 10)  # Default if not found
            action_size = params.get("action_size", 3)  # Default if not found
            
            # Create agent instance
            agent = RLAgent(state_size=state_size, action_size=action_size)
            
            # Load model weights
            try:
                # Load the state_dict into the model
                agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
                agent.update_target_model()  # Update target model with loaded weights
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                return None
            
            logger.info(f"Model {model_id} loaded successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an exported model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary containing model metadata or None if not found
        """
        try:
            # Check if metadata exists
            metadata_path = os.path.join(self.export_dir, f"{model_id}.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Model metadata file not found: {metadata_path}")
                return None
                
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for model {model_id}: {e}", exc_info=True)
            return None
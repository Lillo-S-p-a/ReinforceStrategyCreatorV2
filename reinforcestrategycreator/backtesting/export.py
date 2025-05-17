"""
Export module for backtesting.

This module provides functionality for exporting trained models
for use in paper trading or live trading environments.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional

from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent

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
            
            # Export model weights
            model_path = os.path.join(self.export_dir, f"{model_id}.h5")
            model.save(model_path)
            
            # Export model metadata
            metadata = {
                "model_id": model_id,
                "asset": asset,
                "training_period": f"{start_date} to {end_date}",
                "created_date": timestamp,
                "parameters": params,
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
                json.dump(metadata, f, indent=4)
            
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
            # Check if model exists
            model_path = os.path.join(self.export_dir, f"{model_id}.h5")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
                
            # Load model
            agent = RLAgent.load(model_path)  # Using RLAgent alias
            
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
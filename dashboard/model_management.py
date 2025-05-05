import os
import json
import datetime
import logging
from typing import List, Dict, Any, Optional

# --- Configuration ---
MODELS_DIR = "models"
PRODUCTION_MODELS_DIR = "production_models"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRODUCTION_MODELS_DIR, exist_ok=True)

def save_model_to_production(episode_id: int, run_id: str, model_data: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Save a model to production directory.
    Returns the path where the model was saved.
    """
    # Create a unique model filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_ep{episode_id}_run{run_id}_{timestamp}.json"
    model_path = os.path.join(PRODUCTION_MODELS_DIR, model_filename)
    
    # Combine model data with metrics
    model_info = {
        "model_data": model_data,
        "metrics": metrics,
        "saved_at": timestamp,
        "episode_id": episode_id,
        "run_id": run_id
    }
    
    # Save the model info as JSON
    try:
        with open(model_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logging.info(f"Model saved successfully to {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {e}")
        return ""

def get_saved_production_models() -> List[Dict[str, Any]]:
    """Get list of saved production models with their metrics"""
    models = []
    try:
        for filename in os.listdir(PRODUCTION_MODELS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(PRODUCTION_MODELS_DIR, filename)
                with open(file_path, 'r') as f:
                    model_info = json.load(f)
                    model_info['filename'] = filename
                    models.append(model_info)
    except Exception as e:
        logging.error(f"Error reading production models: {e}")
        
    # Sort by saved timestamp, newest first
    return sorted(models, key=lambda x: x.get('saved_at', ''), reverse=True)

def load_model(model_filename: str) -> Optional[Dict[str, Any]]:
    """
    Load a specific model from the production directory.
    Returns the model data or None if not found.
    """
    file_path = os.path.join(PRODUCTION_MODELS_DIR, model_filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_info = json.load(f)
            return model_info
        else:
            logging.warning(f"Model file not found: {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {e}")
        return None

def delete_model(model_filename: str) -> bool:
    """
    Delete a model from the production directory.
    Returns True if successful, False otherwise.
    """
    file_path = os.path.join(PRODUCTION_MODELS_DIR, model_filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Model deleted successfully: {file_path}")
            return True
        else:
            logging.warning(f"Model file not found for deletion: {file_path}")
            return False
    except Exception as e:
        logging.error(f"Error deleting model {file_path}: {e}")
        return False
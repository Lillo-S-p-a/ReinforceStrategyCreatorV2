"""Example usage of the artifact store system."""

import tempfile
from pathlib import Path
import json
import numpy as np
import pickle

from src.artifact_store import LocalFileSystemStore, ArtifactType
from src.config.manager import ConfigManager


def main():
    """Demonstrate artifact store usage."""
    
    # Initialize config manager
    config_manager = ConfigManager()
    config_manager.load_config()  # Load the configuration first
    config = config_manager.get_config()
    
    # Initialize artifact store from config
    if hasattr(config.artifact_store, 'root_path'):
        artifact_root = config.artifact_store.root_path
    else:
        artifact_root = config.artifact_store
    
    store = LocalFileSystemStore(artifact_root)
    print(f"Initialized artifact store at: {artifact_root}")
    
    # Example 1: Save a model artifact
    print("\n1. Saving a model artifact...")
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        # Simulate saving a model
        model_data = {
            "weights": np.random.randn(100, 50).tolist(),
            "config": {"layers": [100, 50, 10], "activation": "relu"}
        }
        pickle.dump(model_data, f)
        model_path = f.name
    
    model_metadata = store.save_artifact(
        artifact_id="dqn-model-experiment-1",
        artifact_path=model_path,
        artifact_type=ArtifactType.MODEL,
        description="DQN model trained on AAPL data",
        tags=["dqn", "aapl", "experiment"],
        metadata={
            "training_episodes": 1000,
            "final_reward": 1.25,
            "sharpe_ratio": 1.8
        }
    )
    print(f"Saved model: {model_metadata.artifact_id} v{model_metadata.version}")
    Path(model_path).unlink()  # Clean up temp file
    
    # Example 2: Save a dataset artifact
    print("\n2. Saving a dataset artifact...")
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = Path(temp_dir) / "dataset"
        dataset_dir.mkdir()
        
        # Create sample dataset files
        (dataset_dir / "train.csv").write_text("date,price,volume\n2023-01-01,150.0,1000000")
        (dataset_dir / "test.csv").write_text("date,price,volume\n2023-06-01,155.0,1200000")
        (dataset_dir / "metadata.json").write_text(json.dumps({
            "symbols": ["AAPL"],
            "date_range": ["2023-01-01", "2023-12-31"],
            "features": ["price", "volume"]
        }))
        
        dataset_metadata = store.save_artifact(
            artifact_id="aapl-dataset-2023",
            artifact_path=dataset_dir,
            artifact_type=ArtifactType.DATASET,
            description="AAPL price and volume data for 2023",
            tags=["aapl", "2023", "price-volume"]
        )
        print(f"Saved dataset: {dataset_metadata.artifact_id} v{dataset_metadata.version}")
    
    # Example 3: Save an evaluation report
    print("\n3. Saving an evaluation report...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_data = {
            "model_id": "dqn-model-experiment-1",
            "dataset_id": "aapl-dataset-2023",
            "metrics": {
                "sharpe_ratio": 1.8,
                "total_return": 0.25,
                "max_drawdown": -0.08,
                "win_rate": 0.58
            },
            "evaluation_date": "2024-01-15"
        }
        json.dump(report_data, f, indent=2)
        report_path = f.name
    
    report_metadata = store.save_artifact(
        artifact_id="evaluation-report-exp1",
        artifact_path=report_path,
        artifact_type=ArtifactType.REPORT,
        description="Evaluation report for DQN model experiment 1",
        tags=["evaluation", "dqn", "experiment-1"],
        metadata=report_data["metrics"]
    )
    print(f"Saved report: {report_metadata.artifact_id} v{report_metadata.version}")
    Path(report_path).unlink()
    
    # Example 4: List artifacts
    print("\n4. Listing artifacts...")
    all_artifacts = store.list_artifacts()
    print(f"Total artifacts: {len(all_artifacts)}")
    for artifact in all_artifacts:
        print(f"  - {artifact.artifact_id} ({artifact.artifact_type.value}) v{artifact.version}")
    
    # Example 5: Filter artifacts by type
    print("\n5. Filtering by type...")
    models = store.list_artifacts(artifact_type=ArtifactType.MODEL)
    print(f"Model artifacts: {len(models)}")
    for model in models:
        print(f"  - {model.artifact_id}: {model.description}")
    
    # Example 6: Load an artifact
    print("\n6. Loading an artifact...")
    loaded_model_path = store.load_artifact("dqn-model-experiment-1")
    print(f"Loaded model to: {loaded_model_path}")
    
    with open(loaded_model_path, 'rb') as f:
        loaded_model = pickle.load(f)
        print(f"Model config: {loaded_model['config']}")
    
    # Example 7: Version management
    print("\n7. Version management...")
    
    # Save a new version of the model
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        updated_model_data = {
            "weights": np.random.randn(100, 50).tolist(),
            "config": {"layers": [100, 50, 10], "activation": "relu", "dropout": 0.2}
        }
        pickle.dump(updated_model_data, f)
        updated_model_path = f.name
    
    v2_metadata = store.save_artifact(
        artifact_id="dqn-model-experiment-1",
        artifact_path=updated_model_path,
        artifact_type=ArtifactType.MODEL,
        version="v2.0",
        description="DQN model with dropout - improved version",
        tags=["dqn", "aapl", "experiment", "v2"],
        metadata={
            "training_episodes": 2000,
            "final_reward": 1.45,
            "sharpe_ratio": 2.1
        }
    )
    print(f"Saved new version: {v2_metadata.version}")
    Path(updated_model_path).unlink()
    
    # List versions
    versions = store.list_versions("dqn-model-experiment-1")
    print(f"Available versions: {versions}")
    
    # Load specific version
    v1_path = store.load_artifact("dqn-model-experiment-1", version=versions[0])
    print(f"Loaded v1 from: {v1_path}")
    
    # Example 8: Metadata retrieval
    print("\n8. Retrieving metadata...")
    latest_metadata = store.get_artifact_metadata("dqn-model-experiment-1")
    print(f"Latest version metadata:")
    print(f"  - Version: {latest_metadata.version}")
    print(f"  - Created: {latest_metadata.created_at}")
    print(f"  - Tags: {latest_metadata.tags}")
    print(f"  - Properties: {latest_metadata.properties}")
    
    print("\nArtifact store example completed!")


if __name__ == "__main__":
    main()
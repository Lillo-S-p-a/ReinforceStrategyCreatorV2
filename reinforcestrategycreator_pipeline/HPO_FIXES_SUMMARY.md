# HPO Example Fixes Summary

This document summarizes the fixes applied to make the `hpo_example.py` script work correctly.

## Issues Fixed

### 1. ImportError for LocalArtifactStore
**Error:** `ImportError: cannot import name 'LocalArtifactStore'`
**Fix:** Changed import from:
```python
from src.artifact_store.local_adapter import LocalArtifactStore
```
to:
```python
from src.artifact_store.local_adapter import LocalFileSystemStore as LocalArtifactStore
```

### 2. TypeError for LocalFileSystemStore Constructor
**Error:** `TypeError: LocalFileSystemStore.__init__() got an unexpected keyword argument 'base_path'`
**Fix:** Changed the argument name from `base_path` to `root_path` in two places:
```python
# Before
artifact_store = LocalArtifactStore(base_path="./artifacts")
# After
artifact_store = LocalArtifactStore(root_path="./artifacts")
```

### 3. Ray Tune DeprecationWarning
**Error:** `DeprecationWarning: The local_dir argument is deprecated. You should set the storage_path instead.`
**Fix:** In `hpo_optimizer.py`, changed:
```python
local_dir=str(self.results_dir)
```
to:
```python
storage_path=str(self.results_dir)
```

### 4. Ray Tune ArrowInvalid Error
**Error:** `pyarrow.lib.ArrowInvalid: URI has empty scheme: 'hpo_results'`
**Fix:** Changed the storage_path to include the file:// scheme:
```python
storage_path=f"file://{self.results_dir.absolute()}"
```

### 5. Model Type Configuration Issues
**Error:** `ValueError: Configuration must contain 'model_type' key`
**Fix:** 
- Changed the key from `"type"` to `"model_type"` in the model_config
- Changed the value from lowercase `"ppo"` to uppercase `"PPO"` to match the model registry

### 6. Ray Tune Report TypeError
**Error:** `TypeError: report() got an unexpected keyword argument 'loss'`
**Fix:** Changed all `tune.report()` calls to use the `metrics` parameter:
```python
# Before
tune.report(loss=value, val_loss=value2)
tune.report(metrics={"loss": float('inf'), "error": error})

# After
tune.report(metrics={"loss": value, "val_loss": value2})
tune.report(metrics={"loss": float('inf'), "error": error})
```

### 7. LocalFileSystemStore save_artifact TypeError
**Error:** `TypeError: LocalFileSystemStore.save_artifact() got an unexpected keyword argument 'data'`
**Fix:** Changed the save_artifact call to use the correct parameters:
```python
# Before
artifact_id = self.artifact_store.save_artifact(
    artifact_type=ArtifactType.REPORT,
    data=results,
    metadata={...}
)

# After
artifact_metadata = self.artifact_store.save_artifact(
    artifact_id=f"hpo_results_{run_name}",
    artifact_path=results_file,  # Path to the saved JSON file
    artifact_type=ArtifactType.REPORT,
    metadata={...},
    tags=["hpo", "optimization", "results"]
)
```

## Result

After applying all these fixes, the HPO example runs successfully:
- Completes 3 trials with the PPO model
- Saves results to both local files and the artifact store
- Displays best parameters and scores
- Successfully demonstrates the resume functionality

The example now properly showcases the hyperparameter optimization capabilities of the pipeline.
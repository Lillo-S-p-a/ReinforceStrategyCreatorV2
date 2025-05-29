"""Unit tests for CrossValidator."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock

from src.evaluation.cross_validator import CrossValidator, CVResults, CVFoldResult
from src.models.base import ModelBase
from src.models.factory import ModelFactory
from src.training.engine import TrainingEngine


class MockModel(ModelBase):
    """Mock model for testing."""
    
    def __init__(self, model_type: str = "mock", **kwargs):
        super().__init__(model_type=model_type, **kwargs)
        self.train_call_count = 0
        self.evaluate_call_count = 0
    
    def train(self, data, **kwargs):
        """Mock train method."""
        self.train_call_count += 1
        # Return decreasing loss over epochs
        return {"loss": 0.5 / (self.train_call_count + 1)}
    
    def evaluate(self, data, **kwargs):
        """Mock evaluate method."""
        self.evaluate_call_count += 1
        # Return slightly higher loss for validation
        return {"loss": 0.6 / (self.evaluate_call_count + 1)}
    
    def predict(self, data):
        """Mock predict method."""
        return np.random.random(len(data))
    
    def save(self, path):
        """Mock save method."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "model.json", "w") as f:
            json.dump({"type": "mock"}, f)
    
    def load(self, path):
        """Mock load method."""
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_factory():
    """Create a mock model factory."""
    factory = Mock(spec=ModelFactory)
    factory.create_from_config.return_value = MockModel()
    return factory


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create DataFrame
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    data["target"] = np.random.randint(0, 2, n_samples)
    
    return data


@pytest.fixture
def cross_validator(mock_factory, temp_dir):
    """Create a CrossValidator instance."""
    return CrossValidator(
        model_factory=mock_factory,
        checkpoint_dir=temp_dir
    )


class TestCrossValidator:
    """Test cases for CrossValidator."""
    
    def test_initialization(self, temp_dir):
        """Test CrossValidator initialization."""
        cv = CrossValidator(checkpoint_dir=temp_dir)
        
        assert cv.checkpoint_dir == Path(temp_dir)
        assert cv.n_jobs == 1
        assert cv.use_multiprocessing is False
    
    def test_cross_validate_basic(self, cross_validator, sample_data):
        """Test basic cross-validation functionality."""
        model_config = {"type": "mock", "name": "test_model"}
        data_config = {"data": sample_data}
        cv_config = {"method": "kfold", "n_folds": 3}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config,
            training_config={"epochs": 2}
        )
        
        assert isinstance(results, CVResults)
        assert len(results.fold_results) == 3
        assert results.config["n_folds"] == 3
        assert "loss" in results.aggregated_metrics["train_loss"]
        assert "loss" in results.aggregated_metrics["val_loss"]
    
    def test_fold_results_structure(self, cross_validator, sample_data):
        """Test structure of fold results."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"n_folds": 2}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config
        )
        
        for fold_result in results.fold_results:
            assert isinstance(fold_result, CVFoldResult)
            assert isinstance(fold_result.fold_idx, int)
            assert isinstance(fold_result.train_metrics, dict)
            assert isinstance(fold_result.val_metrics, dict)
            assert isinstance(fold_result.training_time, float)
            assert fold_result.training_time > 0
    
    def test_aggregated_metrics(self, cross_validator, sample_data):
        """Test metric aggregation."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"n_folds": 5}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config
        )
        
        # Check aggregated metrics structure
        for metric_name, stats in results.aggregated_metrics.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert stats["mean"] >= stats["min"]
            assert stats["mean"] <= stats["max"]
            assert stats["std"] >= 0
    
    def test_best_fold_selection(self, cross_validator, sample_data):
        """Test best fold selection."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"n_folds": 3}
        
        # Test with min mode (default)
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config,
            scoring_metric="loss",
            scoring_mode="min"
        )
        
        best_fold = results.fold_results[results.best_fold_idx]
        best_score = best_fold.val_metrics["loss"]
        
        # Verify it's actually the minimum
        for fold in results.fold_results:
            assert fold.val_metrics["loss"] >= best_score
    
    def test_time_series_cv(self, cross_validator, sample_data):
        """Test time series cross-validation."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"method": "time_series", "n_folds": 3}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config
        )
        
        assert len(results.fold_results) == 3
        assert results.config["cv_config"]["method"] == "time_series"
    
    def test_stratified_cv(self, cross_validator, sample_data):
        """Test stratified cross-validation."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {
            "method": "stratified",
            "n_folds": 3,
            "target_column": "target"
        }
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config
        )
        
        assert len(results.fold_results) == 3
        assert results.config["cv_config"]["method"] == "stratified"
    
    def test_multiple_metrics(self, cross_validator, sample_data):
        """Test evaluation with multiple metrics."""
        # Mock model that returns multiple metrics
        mock_model = MockModel()
        mock_model.train = Mock(return_value={"loss": 0.5, "accuracy": 0.8})
        mock_model.evaluate = Mock(return_value={"loss": 0.6, "accuracy": 0.75})
        
        cross_validator.model_factory.create_from_config.return_value = mock_model
        
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        metrics = ["loss", "accuracy"]
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            metrics=metrics
        )
        
        # Check that all metrics are present
        assert "train_loss" in results.aggregated_metrics
        assert "val_loss" in results.aggregated_metrics
        assert "train_accuracy" in results.aggregated_metrics
        assert "val_accuracy" in results.aggregated_metrics
    
    def test_save_models_option(self, cross_validator, sample_data, temp_dir):
        """Test model saving during CV."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"n_folds": 2}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config,
            save_models=True
        )
        
        # Check that model paths are set
        for fold_result in results.fold_results:
            assert fold_result.model_path is not None
            assert fold_result.model_path.exists()
    
    def test_cv_results_serialization(self, cross_validator, sample_data):
        """Test CVResults serialization."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config
        )
        
        # Test to_dict
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert "fold_results" in results_dict
        assert "aggregated_metrics" in results_dict
        assert "best_fold_idx" in results_dict
        assert "total_time" in results_dict
        assert "config" in results_dict
        assert "timestamp" in results_dict
        
        # Test get_summary_df
        summary_df = results.get_summary_df()
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(results.fold_results)
        assert "fold" in summary_df.columns
        assert "train_loss" in summary_df.columns
        assert "val_loss" in summary_df.columns
    
    def test_compare_models(self, cross_validator, sample_data):
        """Test model comparison functionality."""
        model_configs = [
            {"type": "mock", "name": "model_1"},
            {"type": "mock", "name": "model_2"}
        ]
        data_config = {"data": sample_data}
        
        results_dict = cross_validator.compare_models(
            model_configs=model_configs,
            data_config=data_config,
            cv_config={"n_folds": 2}
        )
        
        assert len(results_dict) == 2
        assert "model_1" in results_dict
        assert "model_2" in results_dict
        
        for model_name, cv_results in results_dict.items():
            assert isinstance(cv_results, CVResults)
    
    @patch('src.evaluation.cross_validator.ProcessPoolExecutor')
    def test_parallel_cv_multiprocessing(self, mock_executor, cross_validator, sample_data):
        """Test parallel CV with multiprocessing."""
        cross_validator.n_jobs = 2
        cross_validator.use_multiprocessing = True
        
        # Mock the executor
        mock_future = Mock()
        mock_future.result.return_value = CVFoldResult(
            fold_idx=0,
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.6},
            training_time=1.0
        )
        
        mock_executor_instance = Mock()
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=None)
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed
        with patch('src.evaluation.cross_validator.as_completed', return_value=[mock_future]):
            model_config = {"type": "mock"}
            data_config = {"data": sample_data}
            cv_config = {"n_folds": 2}
            
            results = cross_validator.cross_validate(
                model_config=model_config,
                data_config=data_config,
                cv_config=cv_config
            )
            
            assert mock_executor.called
            assert mock_executor.call_args[1]["max_workers"] == 2
    
    def test_error_handling(self, cross_validator):
        """Test error handling in cross-validation."""
        # Test with missing data
        model_config = {"type": "mock"}
        data_config = {}  # No data provided
        
        with pytest.raises(ValueError, match="No data provided"):
            cross_validator.cross_validate(
                model_config=model_config,
                data_config=data_config
            )
    
    def test_numpy_array_data(self, cross_validator):
        """Test CV with numpy array data."""
        # Create numpy array data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        data = np.column_stack([X, y])
        
        model_config = {"type": "mock"}
        data_config = {"data": data}
        cv_config = {"n_folds": 3}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config
        )
        
        assert len(results.fold_results) == 3
    
    def test_custom_callbacks(self, cross_validator, sample_data):
        """Test CV with custom callbacks."""
        # Create a mock callback
        mock_callback = Mock()
        mock_callback.on_epoch_end = Mock()
        
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        cv_config = {"n_folds": 2}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config,
            cv_config=cv_config,
            callbacks=[mock_callback]
        )
        
        assert len(results.fold_results) == 2
    
    def test_results_saving(self, cross_validator, sample_data, temp_dir):
        """Test that CV results are saved to disk."""
        model_config = {"type": "mock"}
        data_config = {"data": sample_data}
        
        results = cross_validator.cross_validate(
            model_config=model_config,
            data_config=data_config
        )
        
        # Check that results files were created
        results_files = list(Path(temp_dir).glob("cv_results_*.json"))
        summary_files = list(Path(temp_dir).glob("cv_summary_*.csv"))
        
        assert len(results_files) > 0
        assert len(summary_files) > 0
        
        # Load and verify saved results
        with open(results_files[0], "r") as f:
            saved_results = json.load(f)
        
        assert saved_results["best_fold_idx"] == results.best_fold_idx
        assert len(saved_results["fold_results"]) == len(results.fold_results)


class TestCVVisualization:
    """Test cases for CV visualization (basic structure tests)."""
    
    def test_visualizer_import(self):
        """Test that CVVisualizer can be imported."""
        from src.evaluation.cv_visualization import CVVisualizer
        
        viz = CVVisualizer()
        assert viz.style == "seaborn"
        assert viz.figsize == (10, 6)
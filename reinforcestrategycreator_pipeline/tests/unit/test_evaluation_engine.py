"""Unit tests for the evaluation engine."""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import numpy as np
import pandas as pd

from src.evaluation.engine import EvaluationEngine
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.benchmarks import BenchmarkEvaluator
from src.models.base import ModelBase
from src.artifact_store.base import ArtifactType


class TestEvaluationEngine:
    """Test cases for EvaluationEngine."""
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry."""
        registry = Mock()
        
        # Mock model
        mock_model = Mock(spec=ModelBase)
        mock_model.model_type = "test_model"
        mock_model.is_trained = True
        
        # Mock registry methods
        registry.load_model.return_value = mock_model
        registry.get_model_metadata.return_value = {
            "model_id": "test_model_123",
            "version": "v1",
            "model_type": "test_model",
            "hyperparameters": {"learning_rate": 0.001}
        }
        
        return registry
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        manager = Mock()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100)
        prices = 100 + np.random.randn(100).cumsum()
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        manager.load_data.return_value = data
        
        return manager
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create a mock artifact store."""
        store = Mock()
        store.save_artifact.return_value = Mock(
            artifact_id="eval_test_123",
            version="20240101_120000"
        )
        store.list_artifacts.return_value = []
        store.load_artifact.return_value = "/tmp/eval_test_123"
        
        return store
    
    @pytest.fixture
    def evaluation_engine(self, mock_model_registry, mock_data_manager, mock_artifact_store):
        """Create an evaluation engine instance."""
        return EvaluationEngine(
            model_registry=mock_model_registry,
            data_manager=mock_data_manager,
            artifact_store=mock_artifact_store,
            metrics_config={"risk_free_rate": 0.02},
            benchmark_config={"initial_balance": 10000}
        )
    
    def test_initialization(self, evaluation_engine):
        """Test evaluation engine initialization."""
        assert evaluation_engine.model_registry is not None
        assert evaluation_engine.data_manager is not None
        assert evaluation_engine.artifact_store is not None
        assert isinstance(evaluation_engine.metrics_calculator, MetricsCalculator)
        assert isinstance(evaluation_engine.benchmark_evaluator, BenchmarkEvaluator)
        assert evaluation_engine.report_formats == ["json", "markdown", "html"]
    
    def test_evaluate_basic(self, evaluation_engine, mock_model_registry, mock_data_manager):
        """Test basic evaluation functionality."""
        # Run evaluation
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            compare_benchmarks=False,
            save_results=False
        )
        
        # Verify structure
        assert "evaluation_id" in results
        assert results["evaluation_id"].startswith("eval_test_model_123_")
        assert "timestamp" in results
        assert "model" in results
        assert "data" in results
        assert "metrics" in results
        
        # Verify model info
        assert results["model"]["id"] == "test_model_123"
        assert results["model"]["type"] == "test_model"
        
        # Verify metrics exist
        metrics = results["metrics"]
        assert "pnl" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        
        # Verify method calls
        mock_model_registry.load_model.assert_called_once()
        mock_data_manager.load_data.assert_called_once()
    
    def test_evaluate_with_benchmarks(self, evaluation_engine):
        """Test evaluation with benchmark comparison."""
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            compare_benchmarks=True,
            save_results=False
        )
        
        # Verify benchmark results
        assert "benchmarks" in results
        assert "relative_performance" in results
        
        benchmarks = results["benchmarks"]
        assert "buy_and_hold" in benchmarks
        assert "simple_moving_average" in benchmarks
        assert "random" in benchmarks
        
        # Verify relative performance
        rel_perf = results["relative_performance"]
        for strategy in ["buy_and_hold", "simple_moving_average", "random"]:
            assert strategy in rel_perf
            assert "absolute_difference" in rel_perf[strategy]
            assert "percentage_difference" in rel_perf[strategy]
            assert "sharpe_ratio_difference" in rel_perf[strategy]
    
    def test_evaluate_with_specific_metrics(self, evaluation_engine):
        """Test evaluation with specific metrics requested."""
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            metrics=["pnl", "sharpe_ratio"],
            compare_benchmarks=False,
            save_results=False
        )
        
        metrics = results["metrics"]
        assert "pnl" in metrics
        assert "sharpe_ratio" in metrics
    
    def test_report_generation(self, evaluation_engine):
        """Test report generation in different formats."""
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            compare_benchmarks=False,
            save_results=False,
            report_formats=["json", "markdown", "html"]
        )
        
        assert "reports" in results
        reports = results["reports"]
        
        # Test JSON report
        assert "json" in reports
        json_report = json.loads(reports["json"])
        assert json_report["evaluation_id"] == results["evaluation_id"]
        
        # Test Markdown report
        assert "markdown" in reports
        md_report = reports["markdown"]
        assert "# Evaluation of test_model_123" in md_report
        assert "## Performance Metrics" in md_report
        
        # Test HTML report
        assert "html" in reports
        html_report = reports["html"]
        assert "<html>" in html_report
        assert "Performance Metrics" in html_report
    
    def test_save_results(self, evaluation_engine, mock_artifact_store):
        """Test saving evaluation results."""
        with patch('pathlib.Path.mkdir'), \
             patch('builtins.open', create=True) as mock_open, \
             patch('shutil.rmtree'):
            
            results = evaluation_engine.evaluate(
                model_id="test_model_123",
                data_source_id="test_data",
                save_results=True,
                tags=["test", "unit_test"]
            )
            
            # Verify artifact store was called
            mock_artifact_store.save_artifact.assert_called_once()
            call_args = mock_artifact_store.save_artifact.call_args
            
            assert call_args[1]["artifact_type"] == ArtifactType.EVALUATION
            assert call_args[1]["tags"] == ["test", "unit_test"]
            assert "model_id" in call_args[1]["metadata"]
    
    def test_list_evaluations(self, evaluation_engine, mock_artifact_store):
        """Test listing evaluation results."""
        # Mock artifact store response
        mock_metadata = Mock()
        mock_metadata.artifact_id = "eval_123"
        mock_metadata.version = "v1"
        mock_metadata.created_at = datetime.now()
        mock_metadata.properties = {
            "model_id": "test_model_123",
            "model_version": "v1",
            "metrics": {"pnl": 100}
        }
        mock_metadata.tags = ["test"]
        mock_metadata.description = "Test evaluation"
        
        mock_artifact_store.list_artifacts.return_value = [mock_metadata]
        
        # List evaluations
        evaluations = evaluation_engine.list_evaluations(model_id="test_model_123")
        
        assert len(evaluations) == 1
        assert evaluations[0]["evaluation_id"] == "eval_123"
        assert evaluations[0]["model_id"] == "test_model_123"
    
    def test_load_evaluation(self, evaluation_engine, mock_artifact_store):
        """Test loading evaluation results."""
        # Mock file reading
        mock_results = {
            "evaluation_id": "eval_123",
            "metrics": {"pnl": 100}
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.load', return_value=mock_results):
            
            loaded_results = evaluation_engine.load_evaluation("eval_123")
            
            assert loaded_results["evaluation_id"] == "eval_123"
            assert loaded_results["metrics"]["pnl"] == 100
            
            mock_artifact_store.load_artifact.assert_called_once_with(
                artifact_id="eval_123",
                version=None
            )
    
    def test_evaluate_error_handling(self, evaluation_engine, mock_model_registry):
        """Test error handling during evaluation."""
        # Make model loading fail
        mock_model_registry.load_model.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception) as exc_info:
            evaluation_engine.evaluate(
                model_id="invalid_model",
                data_source_id="test_data"
            )
        
        assert "Model not found" in str(exc_info.value)
    
    def test_markdown_report_with_benchmarks(self, evaluation_engine):
        """Test markdown report generation with benchmarks."""
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            compare_benchmarks=True,
            save_results=False,
            report_formats=["markdown"]
        )
        
        md_report = results["reports"]["markdown"]
        
        # Check for benchmark sections
        assert "## Benchmark Comparison" in md_report
        assert "### Benchmark Performance" in md_report
        assert "### Relative Performance vs Benchmarks" in md_report
        assert "Buy And Hold" in md_report
        assert "Simple Moving Average" in md_report
    
    def test_html_report_structure(self, evaluation_engine):
        """Test HTML report structure and content."""
        results = evaluation_engine.evaluate(
            model_id="test_model_123",
            data_source_id="test_data",
            compare_benchmarks=False,
            save_results=False,
            report_formats=["html"]
        )
        
        html_report = results["reports"]["html"]
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in html_report
        assert "<style>" in html_report
        assert '<div class="section">' in html_report
        assert '<table>' in html_report
        assert 'class="metric-value"' in html_report
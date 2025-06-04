"""
Unit tests for drift detection components.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from reinforcestrategycreator_pipeline.src.config.models import (
    DataDriftConfig, ModelDriftConfig,
    DataDriftDetectionMethod, ModelDriftDetectionMethod
)
from reinforcestrategycreator_pipeline.src.monitoring.drift_detection import (
    DataDriftDetector, ModelDriftDetector
)


class TestDataDriftDetector:
    """Test cases for DataDriftDetector."""
    
    def test_init(self):
        """Test DataDriftDetector initialization."""
        config = DataDriftConfig(
            enabled=True,
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2
        )
        detector = DataDriftDetector(config)
        assert detector.config == config
    
    def test_detect_psi_no_drift(self):
        """Test PSI detection when no drift is present."""
        # Create similar distributions
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        })
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert not result['drift_detected']
        assert result['score'] < 0.2
        assert result['method'] == 'psi'
        assert 'feature_scores' in result['details']
    
    def test_detect_psi_with_drift(self):
        """Test PSI detection when drift is present."""
        # Create different distributions
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        })
        current_data = pd.DataFrame({
            'feature1': np.random.normal(3, 1, 100),  # Shifted mean
            'feature2': np.random.normal(10, 2, 100)  # Shifted mean
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert result['drift_detected']
        assert result['score'] > 0.2
        assert result['method'] == 'psi'
    
    def test_detect_ks_no_drift(self):
        """Test KS detection when no drift is present."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(2, 1000)
        })
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.exponential(2, 100)
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.KS,
            threshold=0.05  # p-value threshold
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert not result['drift_detected']
        assert result['score'] > 0.05  # p-value should be high
        assert result['method'] == 'ks'
        assert 'feature_results' in result['details']
    
    def test_detect_ks_with_drift(self):
        """Test KS detection when drift is present."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
        })
        current_data = pd.DataFrame({
            'feature1': np.random.normal(5, 1, 100),  # Very different distribution
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.KS,
            threshold=0.05
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert result['drift_detected']
        assert result['score'] < 0.05  # p-value should be low
        assert result['method'] == 'ks'
    
    def test_detect_chi2_no_drift(self):
        """Test Chi-squared detection when no drift is present."""
        np.random.seed(42)
        categories = ['A', 'B', 'C', 'D']
        reference_data = pd.DataFrame({
            'category': np.random.choice(categories, 1000, p=[0.25, 0.25, 0.25, 0.25])
        })
        current_data = pd.DataFrame({
            'category': np.random.choice(categories, 100, p=[0.25, 0.25, 0.25, 0.25])
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.CHI2,
            threshold=0.05
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert not result['drift_detected']
        assert result['score'] > 0.05
        assert result['method'] == 'chi2'
    
    def test_detect_chi2_with_drift(self):
        """Test Chi-squared detection when drift is present."""
        np.random.seed(42)
        categories = ['A', 'B', 'C', 'D']
        reference_data = pd.DataFrame({
            'category': np.random.choice(categories, 1000, p=[0.25, 0.25, 0.25, 0.25])
        })
        current_data = pd.DataFrame({
            'category': np.random.choice(categories, 100, p=[0.7, 0.1, 0.1, 0.1])  # Very different distribution
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.CHI2,
            threshold=0.05
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert result['drift_detected']
        assert result['score'] < 0.05
        assert result['method'] == 'chi2'
    
    def test_detect_with_feature_filter(self):
        """Test detection with specific features to monitor."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.normal(10, 3, 1000)
        })
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),  # Drift in feature2
            'feature3': np.random.normal(10, 3, 100)
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2,
            features_to_monitor=['feature1', 'feature3']  # Exclude feature2
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert not result['drift_detected']  # Should not detect drift in monitored features
        assert 'feature2' not in result['details']['feature_scores']
    
    def test_detect_with_missing_features(self):
        """Test detection when specified features are missing."""
        reference_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        current_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2,
            features_to_monitor=['feature1', 'missing_feature']
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert not result['drift_detected']
        assert 'missing_feature' not in result['details'].get('feature_scores', {})
    
    def test_detect_with_non_dataframe_input(self):
        """Test detection with non-DataFrame input (should convert)."""
        reference_data = {
            'feature1': [1, 2, 3, 4, 5] * 20,
            'feature2': [5, 4, 3, 2, 1] * 20
        }
        current_data = {
            'feature1': [1, 2, 3, 4, 5] * 4,
            'feature2': [5, 4, 3, 2, 1] * 4
        }
        
        config = DataDriftConfig(
            method=DataDriftDetectionMethod.PSI,
            threshold=0.2
        )
        detector = DataDriftDetector(config)
        result = detector.detect(current_data, reference_data)
        
        assert isinstance(result, dict)
        assert 'drift_detected' in result


class TestModelDriftDetector:
    """Test cases for ModelDriftDetector."""
    
    def test_init(self):
        """Test ModelDriftDetector initialization."""
        config = ModelDriftConfig(
            enabled=True,
            method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
            performance_metric="accuracy",
            degradation_threshold=0.1
        )
        detector = ModelDriftDetector(config)
        assert detector.config == config
        assert detector.baseline_performance is None
    
    def test_set_baseline_performance(self):
        """Test setting baseline performance."""
        config = ModelDriftConfig()
        detector = ModelDriftDetector(config)
        detector.set_baseline_performance(0.85)
        assert detector.baseline_performance == 0.85
    
    def test_performance_degradation_no_drift(self):
        """Test performance degradation detection when no drift."""
        np.random.seed(42)
        # Simulate good predictions
        ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        predictions = ground_truth.copy()
        predictions[5] = 0  # One error for 99% accuracy
        
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
            performance_metric="accuracy",
            degradation_threshold=0.1
        )
        detector = ModelDriftDetector(config)
        detector.set_baseline_performance(0.95)
        
        result = detector.detect(predictions, ground_truth)
        
        assert not result['drift_detected']
        assert result['metric_value'] > 0.85  # Above threshold
        assert result['method'] == 'performance_degradation'
    
    def test_performance_degradation_with_drift(self):
        """Test performance degradation detection when drift present."""
        np.random.seed(42)
        # Simulate poor predictions
        ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        predictions = np.random.randint(0, 2, size=100)  # Random predictions
        
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
            performance_metric="accuracy",
            degradation_threshold=0.1
        )
        detector = ModelDriftDetector(config)
        detector.set_baseline_performance(0.95)
        
        result = detector.detect(predictions, ground_truth)
        
        assert result['drift_detected']
        assert result['metric_value'] < 0.855  # Below threshold (0.95 * 0.9)
        assert result['method'] == 'performance_degradation'
    
    def test_performance_metrics(self):
        """Test different performance metrics."""
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # 50% accuracy
        
        metrics = ["accuracy", "f1_score", "precision", "recall"]
        
        for metric in metrics:
            config = ModelDriftConfig(
                method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
                performance_metric=metric,
                degradation_threshold=0.1
            )
            detector = ModelDriftDetector(config)
            result = detector.detect(predictions, ground_truth, baseline_performance=0.9)
            
            assert result['drift_detected']  # All metrics should show degradation
            assert result['details']['metric_name'] == metric
    
    def test_prediction_confidence_no_drift(self):
        """Test prediction confidence when no drift."""
        # High confidence scores
        confidence_scores = np.random.uniform(0.8, 0.99, size=100)
        
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PREDICTION_CONFIDENCE,
            confidence_threshold=0.7
        )
        detector = ModelDriftDetector(config)
        
        result = detector.detect(None, model_confidence=confidence_scores)
        
        assert not result['drift_detected']
        assert result['metric_value'] > 0.7
        assert result['method'] == 'prediction_confidence'
    
    def test_prediction_confidence_with_drift(self):
        """Test prediction confidence when drift present."""
        # Low confidence scores
        confidence_scores = np.random.uniform(0.3, 0.6, size=100)
        
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PREDICTION_CONFIDENCE,
            confidence_threshold=0.7
        )
        detector = ModelDriftDetector(config)
        
        result = detector.detect(None, model_confidence=confidence_scores)
        
        assert result['drift_detected']
        assert result['metric_value'] < 0.7
        assert 'min_confidence' in result['details']
        assert 'max_confidence' in result['details']
        assert 'std_confidence' in result['details']
    
    def test_missing_ground_truth(self):
        """Test performance degradation with missing ground truth."""
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION
        )
        detector = ModelDriftDetector(config)
        
        result = detector.detect([1, 2, 3], ground_truth=None)
        
        assert not result['drift_detected']
        assert result['details']['error'] == 'Missing ground truth'
    
    def test_missing_confidence_scores(self):
        """Test prediction confidence with missing scores."""
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PREDICTION_CONFIDENCE
        )
        detector = ModelDriftDetector(config)
        
        result = detector.detect([1, 2, 3], model_confidence=None)
        
        assert not result['drift_detected']
        assert result['details']['error'] == 'Missing model confidence'
    
    def test_no_baseline_performance(self):
        """Test performance degradation without baseline (should use current as baseline)."""
        ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        predictions = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Perfect predictions
        
        config = ModelDriftConfig(
            method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
            performance_metric="accuracy",
            degradation_threshold=0.1
        )
        detector = ModelDriftDetector(config)
        
        result = detector.detect(predictions, ground_truth)
        
        assert not result['drift_detected']  # No drift when comparing to itself
        assert detector.baseline_performance == 1.0  # Should be set to current performance
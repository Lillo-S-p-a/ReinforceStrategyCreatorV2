"""
Mechanisms for detecting data and model drift.
"""
from typing import Any, Dict, Optional, List, Union
import numpy as np
import pandas as pd

# Handle scipy import gracefully for compatibility issues
SCIPY_AVAILABLE = False
stats = None

def _import_scipy():
    """Lazy import of scipy to handle compatibility issues."""
    global SCIPY_AVAILABLE, stats
    if not SCIPY_AVAILABLE:
        try:
            from scipy import stats as scipy_stats
            stats = scipy_stats
            SCIPY_AVAILABLE = True
        except Exception as e:
            print(f"Warning: scipy not available ({e}). Drift detection with KS and Chi2 tests will be disabled.")
            SCIPY_AVAILABLE = False
            stats = None
    return SCIPY_AVAILABLE

# Lazy import for sklearn to handle missing dependency gracefully
def _import_sklearn():
    """
    Lazy import function for sklearn to handle missing dependency gracefully.
    Returns tuple of (sklearn_available, sklearn_metrics_module)
    """
    try:
        from sklearn import metrics
        return True, metrics
    except ImportError as e:
        logger.warning(f"sklearn not available: {e}. Model drift detection will use fallback behavior.")
        return False, None

from ..config.models import DataDriftConfig, ModelDriftConfig, DataDriftDetectionMethod, ModelDriftDetectionMethod
from .logger import get_logger

logger = get_logger("monitoring.drift_detection")

class DataDriftDetector:
    """Detects drift in input data distributions."""

    def __init__(self, config: DataDriftConfig):
        """
        Initialize the data drift detector.

        Args:
            config: Configuration for data drift detection.
        """
        self.config = config
        logger.info(f"DataDriftDetector initialized with method: {config.method.value}, threshold: {config.threshold}")

    def detect(self, current_data: Any, reference_data: Any) -> Dict[str, Any]:
        """
        Detects data drift between current and reference data.

        Args:
            current_data: The current batch of data (e.g., pandas DataFrame).
            reference_data: The reference dataset (e.g., pandas DataFrame).

        Returns:
            A dictionary containing drift detection results:
            {
                "drift_detected": bool,
                "score": float, # Method-specific drift score
                "method": str,
                "details": Optional[Dict] # Additional details
            }
        """
        logger.debug(f"Performing data drift detection using method: {self.config.method.value}")
        
        # Convert to pandas DataFrames if not already
        if not isinstance(current_data, pd.DataFrame):
            current_data = pd.DataFrame(current_data)
        if not isinstance(reference_data, pd.DataFrame):
            reference_data = pd.DataFrame(reference_data)
        
        # Filter features if specified
        features = self.config.features_to_monitor
        if features:
            available_features = [f for f in features if f in current_data.columns and f in reference_data.columns]
            if not available_features:
                logger.warning(f"None of the specified features {features} found in data")
                return {
                    "drift_detected": False,
                    "score": 0.0,
                    "method": self.config.method.value,
                    "details": {"error": "No specified features found in data"}
                }
            current_data = current_data[available_features]
            reference_data = reference_data[available_features]
        
        try:
            if self.config.method == DataDriftDetectionMethod.PSI:
                return self._calculate_psi(current_data, reference_data)
            elif self.config.method == DataDriftDetectionMethod.KS:
                return self._calculate_ks(current_data, reference_data)
            elif self.config.method == DataDriftDetectionMethod.CHI2:
                return self._calculate_chi2(current_data, reference_data)
            else:
                logger.warning(f"Unsupported data drift detection method: {self.config.method.value}")
                return {
                    "drift_detected": False,
                    "score": 0.0,
                    "method": self.config.method.value,
                    "details": {"error": "Unsupported method"}
                }
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}")
            return {
                "drift_detected": False,
                "score": 0.0,
                "method": self.config.method.value,
                "details": {"error": str(e)}
            }
    
    def _calculate_psi(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) for numerical features."""
        psi_scores = {}
        total_psi = 0.0
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            try:
                # Create bins based on reference data
                _, bin_edges = pd.qcut(reference_data[column].dropna(), q=10, retbins=True, duplicates='drop')
                
                # Calculate distributions
                ref_counts, _ = np.histogram(reference_data[column].dropna(), bins=bin_edges)
                curr_counts, _ = np.histogram(current_data[column].dropna(), bins=bin_edges)
                
                # Convert to proportions
                ref_props = (ref_counts + 1) / (ref_counts.sum() + len(bin_edges) - 1)  # Add 1 to avoid log(0)
                curr_props = (curr_counts + 1) / (curr_counts.sum() + len(bin_edges) - 1)
                
                # Calculate PSI
                psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
                psi_scores[column] = psi
                total_psi += psi
                
            except Exception as e:
                logger.warning(f"Could not calculate PSI for column {column}: {str(e)}")
                continue
        
        avg_psi = total_psi / len(psi_scores) if psi_scores else 0.0
        drift_detected = avg_psi > self.config.threshold
        
        logger.info(f"PSI calculated: {avg_psi:.4f} (threshold: {self.config.threshold})")
        
        return {
            "drift_detected": drift_detected,
            "score": avg_psi,
            "method": self.config.method.value,
            "details": {
                "feature_scores": psi_scores,
                "message": f"Average PSI: {avg_psi:.4f}"
            }
        }
    
    def _calculate_ks(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Kolmogorov-Smirnov test for numerical features."""
        if not _import_scipy():
            logger.warning("KS test requires scipy which is not available. Returning no drift detected.")
            return {
                "drift_detected": False,
                "score": 1.0,
                "method": self.config.method.value,
                "details": {"error": "scipy not available for KS test"}
            }
        
        ks_results = {}
        min_p_value = 1.0
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            try:
                # Perform KS test
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                ks_results[column] = {"statistic": ks_stat, "p_value": p_value}
                min_p_value = min(min_p_value, p_value)
                
            except Exception as e:
                logger.warning(f"Could not calculate KS test for column {column}: {str(e)}")
                continue
        
        # For KS test, we use p-value: lower p-value indicates drift
        drift_detected = min_p_value < self.config.threshold
        
        logger.info(f"KS test minimum p-value: {min_p_value:.4f} (threshold: {self.config.threshold})")
        
        return {
            "drift_detected": drift_detected,
            "score": min_p_value,
            "method": self.config.method.value,
            "details": {
                "feature_results": ks_results,
                "message": f"Minimum p-value: {min_p_value:.4f}"
            }
        }
    
    def _calculate_chi2(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Chi-squared test for categorical features."""
        if not _import_scipy():
            logger.warning("Chi-squared test requires scipy which is not available. Returning no drift detected.")
            return {
                "drift_detected": False,
                "score": 1.0,
                "method": self.config.method.value,
                "details": {"error": "scipy not available for Chi-squared test"}
            }
        
        chi2_results = {}
        min_p_value = 1.0
        
        # Get categorical columns
        cat_columns = current_data.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_columns) == 0:
            logger.warning("No categorical features found for Chi-squared test")
            return {
                "drift_detected": False,
                "score": 1.0,
                "method": self.config.method.value,
                "details": {"error": "No categorical features found"}
            }
        
        for column in cat_columns:
            try:
                # Create contingency table
                ref_counts = reference_data[column].value_counts()
                curr_counts = current_data[column].value_counts()
                
                # Align categories
                all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                ref_aligned = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                curr_aligned = np.array([curr_counts.get(cat, 0) for cat in all_categories])
                
                # Normalize expected frequencies to match observed total
                ref_total = ref_aligned.sum()
                curr_total = curr_aligned.sum()
                if ref_total > 0 and curr_total > 0:
                    expected_freq = ref_aligned * (curr_total / ref_total)
                    
                    # Perform Chi-squared test
                    chi2_stat, p_value = stats.chisquare(curr_aligned, expected_freq)
                    chi2_results[column] = {"statistic": chi2_stat, "p_value": p_value}
                    min_p_value = min(min_p_value, p_value)
                else:
                    logger.warning(f"Empty data for column {column}")
                
            except Exception as e:
                logger.warning(f"Could not calculate Chi-squared test for column {column}: {str(e)}")
                continue
        
        # For Chi-squared test, lower p-value indicates drift
        drift_detected = min_p_value < self.config.threshold
        
        logger.info(f"Chi-squared test minimum p-value: {min_p_value:.4f} (threshold: {self.config.threshold})")
        
        return {
            "drift_detected": drift_detected,
            "score": min_p_value,
            "method": self.config.method.value,
            "details": {
                "feature_results": chi2_results,
                "message": f"Minimum p-value: {min_p_value:.4f}"
            }
        }


class ModelDriftDetector:
    """Detects drift in model performance or behavior."""

    def __init__(self, config: ModelDriftConfig):
        """
        Initialize the model drift detector.

        Args:
            config: Configuration for model drift detection.
        """
        self.config = config
        self.baseline_performance = None  # Will be set during initialization or from historical data
        logger.info(f"ModelDriftDetector initialized with method: {config.method.value}")
    
    def set_baseline_performance(self, baseline: float):
        """Set the baseline performance for comparison."""
        self.baseline_performance = baseline
        logger.info(f"Baseline performance set to: {baseline}")
    
    def detect(self, model_predictions: Any, ground_truth: Optional[Any] = None, 
               model_confidence: Optional[Any] = None, baseline_performance: Optional[float] = None) -> Dict[str, Any]:
        """
        Detects model drift.

        Args:
            model_predictions: Predictions from the model.
            ground_truth: Actual outcomes (required for performance_degradation).
            model_confidence: Model's confidence scores (required for prediction_confidence).
            baseline_performance: Optional baseline performance to compare against.

        Returns:
            A dictionary containing drift detection results:
            {
                "drift_detected": bool,
                "metric_value": float, # Value of the monitored metric
                "method": str,
                "details": Optional[Dict] # Additional details
            }
        """
        logger.debug(f"Performing model drift detection using method: {self.config.method.value}")
        
        try:
            if self.config.method == ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION:
                return self._check_performance_degradation(model_predictions, ground_truth, baseline_performance)
            elif self.config.method == ModelDriftDetectionMethod.PREDICTION_CONFIDENCE:
                return self._check_prediction_confidence(model_confidence)
            else:
                logger.warning(f"Unsupported model drift detection method: {self.config.method.value}")
                return {
                    "drift_detected": False,
                    "metric_value": 0.0,
                    "method": self.config.method.value,
                    "details": {"error": "Unsupported method"}
                }
        except Exception as e:
            logger.error(f"Error during model drift detection: {str(e)}")
            return {
                "drift_detected": False,
                "metric_value": 0.0,
                "method": self.config.method.value,
                "details": {"error": str(e)}
            }
    
    def _check_performance_degradation(self, model_predictions: Any, ground_truth: Any, 
                                     baseline_performance: Optional[float] = None) -> Dict[str, Any]:
        """Check for performance degradation."""
        if ground_truth is None:
            logger.error("Ground truth data is required for performance degradation check.")
            return {
                "drift_detected": False, 
                "metric_value": 0.0, 
                "method": self.config.method.value, 
                "details": {"error": "Missing ground truth"}
            }
        
        # Convert to numpy arrays if needed
        predictions = np.array(model_predictions)
        truth = np.array(ground_truth)
        
        # Calculate performance metric
        metric_name = self.config.performance_metric or "accuracy"
        metric_value = 0.0
        
        try:
            sklearn_available, sklearn_metrics = _import_sklearn()
            
            if not sklearn_available:
                logger.warning(f"sklearn not available. Cannot calculate {metric_name} metric. Returning default value 0.0.")
                metric_value = 0.0
            elif metric_name == "accuracy":
                metric_value = sklearn_metrics.accuracy_score(truth, predictions)
            elif metric_name == "f1_score":
                metric_value = sklearn_metrics.f1_score(truth, predictions, average='weighted')
            elif metric_name == "precision":
                metric_value = sklearn_metrics.precision_score(truth, predictions, average='weighted')
            elif metric_name == "recall":
                metric_value = sklearn_metrics.recall_score(truth, predictions, average='weighted')
            else:
                logger.warning(f"Unknown performance metric: {metric_name}, defaulting to accuracy")
                metric_value = sklearn_metrics.accuracy_score(truth, predictions)
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {str(e)}")
            return {
                "drift_detected": False,
                "metric_value": 0.0,
                "method": self.config.method.value,
                "details": {"error": f"Error calculating {metric_name}: {str(e)}"}
            }
        
        # Use provided baseline or instance baseline
        baseline = baseline_performance or self.baseline_performance
        if baseline is None:
            logger.warning("No baseline performance available, using metric value as baseline")
            self.baseline_performance = metric_value
            baseline = metric_value
        
        # Check for degradation
        threshold = self.config.degradation_threshold or 0.1
        min_acceptable = baseline * (1 - threshold)
        drift_detected = metric_value < min_acceptable
        
        logger.info(f"Performance check - Metric: {metric_name}, Value: {metric_value:.4f}, "
                   f"Baseline: {baseline:.4f}, Min acceptable: {min_acceptable:.4f}")
        
        return {
            "drift_detected": drift_detected,
            "metric_value": metric_value,
            "method": self.config.method.value,
            "details": {
                "metric_name": metric_name,
                "baseline": baseline,
                "threshold": threshold,
                "min_acceptable": min_acceptable,
                "message": f"{metric_name}: {metric_value:.4f} (baseline: {baseline:.4f})"
            }
        }
    
    def _check_prediction_confidence(self, model_confidence: Any) -> Dict[str, Any]:
        """Check for low prediction confidence."""
        if model_confidence is None:
            logger.error("Model confidence scores are required for prediction confidence check.")
            return {
                "drift_detected": False,
                "metric_value": 0.0,
                "method": self.config.method.value,
                "details": {"error": "Missing model confidence"}
            }
        
        # Convert to numpy array and calculate average confidence
        confidence_scores = np.array(model_confidence)
        avg_confidence = np.mean(confidence_scores)
        
        # Check against threshold
        threshold = self.config.confidence_threshold or 0.7
        drift_detected = avg_confidence < threshold
        
        # Calculate additional statistics
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
        std_confidence = np.std(confidence_scores)
        
        logger.info(f"Prediction confidence check - Average: {avg_confidence:.4f}, "
                   f"Threshold: {threshold}, Min: {min_confidence:.4f}, Max: {max_confidence:.4f}")
        
        return {
            "drift_detected": drift_detected,
            "metric_value": avg_confidence,
            "method": self.config.method.value,
            "details": {
                "threshold": threshold,
                "min_confidence": float(min_confidence),
                "max_confidence": float(max_confidence),
                "std_confidence": float(std_confidence),
                "message": f"Average confidence: {avg_confidence:.4f} (threshold: {threshold})"
            }
        }
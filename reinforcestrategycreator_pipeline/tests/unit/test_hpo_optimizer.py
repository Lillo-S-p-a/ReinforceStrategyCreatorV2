"""Unit tests for HPOptimizer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from src.training.hpo_optimizer import HPOptimizer
from src.training.engine import TrainingEngine
from src.artifact_store.base import ArtifactStore, ArtifactType


class TestHPOptimizer:
    """Test cases for HPOptimizer."""
    
    @pytest.fixture
    def mock_training_engine(self):
        """Create a mock training engine."""
        engine = Mock(spec=TrainingEngine)
        return engine
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create a mock artifact store."""
        store = Mock(spec=ArtifactStore)
        store.save_artifact.return_value = "artifact_123"
        return store
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def hpo_optimizer(self, mock_training_engine, mock_artifact_store, temp_dir):
        """Create an HPOptimizer instance."""
        with patch('src.training.hpo_optimizer.RAY_AVAILABLE', True):
            optimizer = HPOptimizer(
                training_engine=mock_training_engine,
                artifact_store=mock_artifact_store,
                results_dir=temp_dir
            )
            return optimizer
    
    def test_init(self, mock_training_engine, mock_artifact_store, temp_dir):
        """Test HPOptimizer initialization."""
        with patch('src.training.hpo_optimizer.RAY_AVAILABLE', True):
            optimizer = HPOptimizer(
                training_engine=mock_training_engine,
                artifact_store=mock_artifact_store,
                results_dir=temp_dir
            )
            
            assert optimizer.training_engine == mock_training_engine
            assert optimizer.artifact_store == mock_artifact_store
            assert optimizer.results_dir == temp_dir
            assert optimizer.best_params is None
            assert optimizer.best_score is None
            assert optimizer.all_trials == []
    
    def test_init_without_ray(self, mock_training_engine):
        """Test initialization fails without Ray."""
        with patch('src.training.hpo_optimizer.RAY_AVAILABLE', False):
            with pytest.raises(ImportError, match="Ray Tune is required"):
                HPOptimizer(training_engine=mock_training_engine)
    
    def test_define_search_space_uniform(self, hpo_optimizer):
        """Test defining uniform search space."""
        param_space = {
            "learning_rate": {
                "type": "uniform",
                "low": 0.001,
                "high": 0.1
            }
        }
        
        with patch('src.training.hpo_optimizer.tune') as mock_tune:
            mock_tune.uniform.return_value = "uniform_dist"
            
            processed_space, search_alg = hpo_optimizer.define_search_space(
                param_space, search_algorithm="random"
            )
            
            mock_tune.uniform.assert_called_once_with(0.001, 0.1)
            assert processed_space["learning_rate"] == "uniform_dist"
            assert search_alg is None  # Random search has no algorithm
    
    def test_define_search_space_loguniform(self, hpo_optimizer):
        """Test defining loguniform search space."""
        param_space = {
            "learning_rate": {
                "type": "loguniform",
                "low": 0.00001,
                "high": 0.01
            }
        }
        
        with patch('src.training.hpo_optimizer.tune') as mock_tune:
            mock_tune.loguniform.return_value = "loguniform_dist"
            
            processed_space, _ = hpo_optimizer.define_search_space(param_space)
            
            mock_tune.loguniform.assert_called_once_with(0.00001, 0.01)
            assert processed_space["learning_rate"] == "loguniform_dist"
    
    def test_define_search_space_choice(self, hpo_optimizer):
        """Test defining choice search space."""
        param_space = {
            "batch_size": {
                "type": "choice",
                "values": [32, 64, 128]
            }
        }
        
        with patch('src.training.hpo_optimizer.tune') as mock_tune:
            mock_tune.choice.return_value = "choice_dist"
            
            processed_space, _ = hpo_optimizer.define_search_space(param_space)
            
            mock_tune.choice.assert_called_once_with([32, 64, 128])
            assert processed_space["batch_size"] == "choice_dist"
    
    def test_define_search_space_with_optuna(self, hpo_optimizer):
        """Test defining search space with Optuna."""
        param_space = {"lr": {"type": "uniform", "low": 0.001, "high": 0.1}}
        
        with patch('src.training.hpo_optimizer.tune'), \
             patch('src.training.hpo_optimizer.OPTUNA_AVAILABLE', True), \
             patch('src.training.hpo_optimizer.OptunaSearch') as mock_optuna:
            
            mock_search = Mock()
            mock_optuna.return_value = mock_search
            
            _, search_alg = hpo_optimizer.define_search_space(
                param_space, 
                search_algorithm="optuna",
                metric="loss",
                mode="min"
            )
            
            mock_optuna.assert_called_once_with(metric="loss", mode="min")
            assert search_alg == mock_search
    
    def test_set_nested_config(self, hpo_optimizer):
        """Test setting nested configuration values."""
        config = {}
        
        # Test single level
        hpo_optimizer._set_nested_config(config, "learning_rate", 0.001)
        assert config["learning_rate"] == 0.001
        
        # Test nested level
        hpo_optimizer._set_nested_config(config, "hyperparameters.batch_size", 64)
        assert config["hyperparameters"]["batch_size"] == 64
        
        # Test deep nesting
        hpo_optimizer._set_nested_config(
            config, "model.layers.hidden.units", 128
        )
        assert config["model"]["layers"]["hidden"]["units"] == 128
    
    def test_create_trainable(self, hpo_optimizer, mock_training_engine):
        """Test creating a trainable function."""
        model_config = {"type": "ppo", "hyperparameters": {}}
        data_config = {"source": "test"}
        training_config = {"epochs": 10}
        
        # Mock training engine response
        mock_training_engine.train.return_value = {
            "success": True,
            "final_metrics": {"loss": 0.5},
            "history": {
                "loss": [0.8, 0.6, 0.5],
                "val_loss": [0.9, 0.7, 0.6],
                "epochs": [0, 1, 2]
            }
        }
        
        trainable = hpo_optimizer._create_trainable(
            model_config, data_config, training_config
        )
        
        # Test the trainable function
        with patch('src.training.hpo_optimizer.tune') as mock_tune:
            trainable({"learning_rate": 0.001})
            
            # Verify training was called
            mock_training_engine.train.assert_called_once()
            call_args = mock_training_engine.train.call_args[1]
            assert call_args["model_config"]["hyperparameters"]["learning_rate"] == 0.001
            
            # Verify metrics were reported
            assert mock_tune.report.call_count == 3  # One per epoch
    
    def test_create_trainable_with_param_mapping(self, hpo_optimizer, mock_training_engine):
        """Test creating trainable with parameter mapping."""
        model_config = {"type": "ppo", "hyperparameters": {}}
        param_mapping = {"lr": "hyperparameters.learning_rate"}
        
        mock_training_engine.train.return_value = {
            "success": True,
            "history": {"loss": [0.5], "epochs": [0]}
        }
        
        trainable = hpo_optimizer._create_trainable(
            model_config, {}, {}, param_mapping
        )
        
        with patch('src.training.hpo_optimizer.tune'):
            trainable({"lr": 0.001})
            
            call_args = mock_training_engine.train.call_args[1]
            assert call_args["model_config"]["hyperparameters"]["learning_rate"] == 0.001
    
    @patch('src.training.hpo_optimizer.ray')
    @patch('src.training.hpo_optimizer.tune')
    def test_optimize_basic(self, mock_tune, mock_ray, hpo_optimizer, mock_training_engine):
        """Test basic optimization run."""
        # Setup mocks
        mock_ray.is_initialized.return_value = False
        mock_analysis = Mock()
        mock_best_trial = Mock()
        mock_best_trial.config = {"learning_rate": 0.005}
        mock_best_trial.last_result = {"loss": 0.3}
        mock_analysis.get_best_trial.return_value = mock_best_trial
        mock_analysis.trials = [mock_best_trial]
        mock_tune.run.return_value = mock_analysis
        
        # Run optimization
        model_config = {"type": "ppo"}
        data_config = {"source": "test"}
        training_config = {"epochs": 10}
        param_space = {
            "learning_rate": {"type": "uniform", "low": 0.001, "high": 0.01}
        }
        
        results = hpo_optimizer.optimize(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            param_space=param_space,
            num_trials=5,
            metric="loss",
            mode="min"
        )
        
        # Verify Ray was initialized and shutdown
        mock_ray.init.assert_called_once()
        mock_ray.shutdown.assert_called_once()
        
        # Verify results
        assert results["best_params"] == {"learning_rate": 0.005}
        assert results["best_score"] == 0.3
        assert results["num_trials"] == 5
        assert len(results["all_trials"]) == 1
    
    def test_analyze_results(self, hpo_optimizer):
        """Test analyzing HPO results."""
        results = {
            "best_params": {"learning_rate": 0.005},
            "best_score": 0.3,
            "mode": "min",
            "all_trials": [
                {
                    "trial_id": "trial_1",
                    "params": {"learning_rate": 0.005},
                    "metric": 0.3
                },
                {
                    "trial_id": "trial_2",
                    "params": {"learning_rate": 0.008},
                    "metric": 0.4
                },
                {
                    "trial_id": "trial_3",
                    "params": {"learning_rate": 0.002},
                    "metric": 0.35
                }
            ]
        }
        
        analysis = hpo_optimizer.analyze_results(results, top_k=2)
        
        assert analysis["total_trials"] == 3
        assert analysis["successful_trials"] == 3
        assert analysis["best_trial"]["score"] == 0.3
        assert len(analysis["top_k_trials"]) == 2
        assert analysis["top_k_trials"][0]["metric"] == 0.3
        assert analysis["top_k_trials"][1]["metric"] == 0.35
        assert "parameter_importance" in analysis
        assert "metric_stats" in analysis
    
    def test_get_best_model_config(self, hpo_optimizer):
        """Test getting best model configuration."""
        hpo_optimizer.best_params = {"learning_rate": 0.005, "batch_size": 64}
        
        base_config = {
            "type": "ppo",
            "hyperparameters": {"learning_rate": 0.001}
        }
        
        best_config = hpo_optimizer.get_best_model_config(base_config)
        
        assert best_config["hyperparameters"]["learning_rate"] == 0.005
        assert best_config["hyperparameters"]["batch_size"] == 64
    
    def test_get_best_model_config_with_mapping(self, hpo_optimizer):
        """Test getting best model config with parameter mapping."""
        hpo_optimizer.best_params = {"lr": 0.005}
        
        base_config = {"type": "ppo", "hyperparameters": {}}
        param_mapping = {"lr": "hyperparameters.learning_rate"}
        
        best_config = hpo_optimizer.get_best_model_config(base_config, param_mapping)
        
        assert best_config["hyperparameters"]["learning_rate"] == 0.005
    
    def test_get_best_model_config_no_results(self, hpo_optimizer):
        """Test getting best config without results raises error."""
        with pytest.raises(ValueError, match="No optimization results"):
            hpo_optimizer.get_best_model_config({})
    
    def test_load_results(self, hpo_optimizer, temp_dir):
        """Test loading results from file."""
        results = {
            "best_params": {"learning_rate": 0.005},
            "best_score": 0.3,
            "all_trials": [{"trial_id": "trial_1"}]
        }
        
        results_file = temp_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)
        
        loaded_results = hpo_optimizer.load_results(results_file)
        
        assert loaded_results["best_params"] == {"learning_rate": 0.005}
        assert hpo_optimizer.best_params == {"learning_rate": 0.005}
        assert hpo_optimizer.best_score == 0.3
        assert len(hpo_optimizer.all_trials) == 1
    
    @patch('src.training.hpo_optimizer.ray')
    @patch('src.training.hpo_optimizer.tune')
    def test_optimize_with_scheduler(self, mock_tune, mock_ray, hpo_optimizer):
        """Test optimization with ASHA scheduler."""
        # Setup mocks
        mock_ray.is_initialized.return_value = True  # Already initialized
        mock_analysis = Mock()
        mock_analysis.get_best_trial.return_value = Mock(
            config={"lr": 0.005},
            last_result={"loss": 0.3}
        )
        mock_analysis.trials = []
        mock_tune.run.return_value = mock_analysis
        
        with patch('src.training.hpo_optimizer.ASHAScheduler') as mock_asha:
            mock_scheduler = Mock()
            mock_asha.return_value = mock_scheduler
            
            hpo_optimizer.optimize(
                model_config={},
                data_config={},
                training_config={"epochs": 100},
                param_space={"lr": {"type": "uniform", "low": 0.001, "high": 0.01}},
                scheduler="asha",
                metric="loss",
                mode="min"
            )
            
            # Verify scheduler was created
            mock_asha.assert_called_once_with(
                metric="loss",
                mode="min",
                max_t=100,
                grace_period=1,
                reduction_factor=2
            )
            
            # Verify tune.run was called with scheduler
            tune_call_args = mock_tune.run.call_args[1]
            assert tune_call_args["scheduler"] == mock_scheduler
    
    def test_analyze_results_empty(self, hpo_optimizer):
        """Test analyzing empty results."""
        analysis = hpo_optimizer.analyze_results()
        assert analysis["error"] == "No results to analyze"
        
        analysis = hpo_optimizer.analyze_results({"all_trials": []})
        assert analysis["error"] == "No successful trials to analyze"
    
    def test_optimize_saves_to_artifact_store(self, hpo_optimizer, mock_artifact_store):
        """Test that optimization results are saved to artifact store."""
        with patch('src.training.hpo_optimizer.ray'), \
             patch('src.training.hpo_optimizer.tune') as mock_tune:
            
            mock_analysis = Mock()
            mock_analysis.get_best_trial.return_value = Mock(
                config={"lr": 0.005},
                last_result={"loss": 0.3}
            )
            mock_analysis.trials = []
            mock_tune.run.return_value = mock_analysis
            
            results = hpo_optimizer.optimize(
                model_config={},
                data_config={},
                training_config={},
                param_space={"lr": {"type": "uniform", "low": 0.001, "high": 0.01}}
            )
            
            # Verify artifact store was called
            mock_artifact_store.save_artifact.assert_called_once()
            call_args = mock_artifact_store.save_artifact.call_args
            assert call_args[1]["artifact_type"] == ArtifactType.REPORT
            assert "artifact_id" in results
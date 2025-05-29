import unittest
from unittest.mock import MagicMock, patch, call, mock_open

from reinforcestrategycreator_pipeline.src.pipeline.orchestrator import ModelPipeline, ModelPipelineError
from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.pipeline.executor import PipelineExecutor, PipelineExecutionError
# from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager # To be mocked
# from reinforcestrategycreator_pipeline.src.monitoring.logger import AppLogger # To be mocked

# A concrete stage for testing dynamic loading
class SampleStageAlpha(PipelineStage):
    def setup(self, context: PipelineContext) -> None: context.set(f"{self.name}_setup", True)
    def run(self, context: PipelineContext) -> PipelineContext:
        context.set(f"{self.name}_run", True)
        return context
    def teardown(self, context: PipelineContext) -> None: context.set(f"{self.name}_teardown", True)

class SampleStageBeta(PipelineStage):
    def setup(self, context: PipelineContext) -> None: context.set(f"{self.name}_setup", True)
    def run(self, context: PipelineContext) -> PipelineContext:
        context.set(f"{self.name}_run", True)
        # Simulate failure
        if self.config.get("fail_in_run"):
            raise ValueError("Beta stage run failure")
        return context
    def teardown(self, context: PipelineContext) -> None: context.set(f"{self.name}_teardown", True)


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        # Mock ConfigManager
        self.mock_config_manager_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.orchestrator.ConfigManager')
        self.MockConfigManager = self.mock_config_manager_patcher.start()
        self.mock_config_manager_instance = self.MockConfigManager.return_value

        # Mock get_logger for ModelPipeline
        self.mock_orchestrator_get_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.orchestrator.get_logger')
        self.mock_orchestrator_get_logger = self.mock_orchestrator_get_logger_patcher.start()
        self.mock_orchestrator_logger_instance = MagicMock()
        self.mock_orchestrator_get_logger.return_value = self.mock_orchestrator_logger_instance
        
        # Mock PipelineExecutor
        self.mock_executor_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.orchestrator.PipelineExecutor')
        self.MockPipelineExecutor = self.mock_executor_patcher.start()
        self.mock_executor_instance = self.MockPipelineExecutor.return_value
        
        # Mock importlib.import_module and getattr for dynamic stage loading
        self.mock_importlib_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.orchestrator.importlib')
        self.mock_importlib = self.mock_importlib_patcher.start()

        # Patch get_logger for PipelineStage (used by SampleStageAlpha/Beta)
        self.mock_stage_get_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_stage_get_logger = self.mock_stage_get_logger_patcher.start()
        self.mock_stage_logger_instance = MagicMock()
        self.mock_stage_get_logger.return_value = self.mock_stage_logger_instance


    def tearDown(self):
        self.mock_config_manager_patcher.stop()
        self.mock_orchestrator_get_logger_patcher.stop()
        self.mock_executor_patcher.stop()
        self.mock_importlib_patcher.stop()
        self.mock_stage_get_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def _setup_mock_config(self, pipeline_name="test_pipeline", stages_config=None):
        if stages_config is None:
            stages_config = [
                {
                    "name": "AlphaStage",
                    "module": "test_module.alpha",
                    "class": "SampleStageAlpha",
                    "config": {"alpha_param": 1}
                },
                {
                    "name": "BetaStage",
                    "module": "test_module.beta",
                    "class": "SampleStageBeta",
                    "config": {"beta_param": "hello"}
                }
            ]
        
        pipeline_full_config = {
            pipeline_name: {
                "stages": stages_config
            }
        }
        self.mock_config_manager_instance.get_config.return_value = pipeline_full_config
        
        # Setup importlib mocks
        def mock_import_module(module_path):
            mock_mod = MagicMock()
            if module_path == "test_module.alpha":
                setattr(mock_mod, "SampleStageAlpha", SampleStageAlpha)
            elif module_path == "test_module.beta":
                setattr(mock_mod, "SampleStageBeta", SampleStageBeta)
            else:
                raise ImportError(f"Test mock cannot import {module_path}")
            return mock_mod
            
        self.mock_importlib.import_module.side_effect = mock_import_module


    def test_pipeline_initialization_successful_loading(self):
        self._setup_mock_config(pipeline_name="my_pipe")
        
        pipeline = ModelPipeline(pipeline_name="my_pipe", config_manager=self.mock_config_manager_instance)

        self.assertEqual(pipeline.pipeline_name, "my_pipe")
        self.assertEqual(len(pipeline.stages), 2)
        self.assertIsInstance(pipeline.stages[0], SampleStageAlpha)
        self.assertEqual(pipeline.stages[0].name, "AlphaStage")
        self.assertEqual(pipeline.stages[0].config, {"alpha_param": 1})
        self.assertIsInstance(pipeline.stages[1], SampleStageBeta)
        self.assertEqual(pipeline.stages[1].name, "BetaStage")
        
        self.mock_orchestrator_get_logger.assert_called_with('orchestrator.ModelPipeline.my_pipe')
        self.mock_config_manager_instance.get_config.assert_called_once_with('pipelines')
        self.MockPipelineExecutor.assert_called_once_with(pipeline.stages)
        self.mock_orchestrator_logger_instance.info.assert_any_call("Successfully loaded and instantiated stage: AlphaStage")
        self.mock_orchestrator_logger_instance.info.assert_any_call("Successfully loaded and instantiated stage: BetaStage")

    def test_pipeline_initialization_no_pipelines_config(self):
        self.mock_config_manager_instance.get_config.return_value = None # No 'pipelines' key
        with self.assertRaisesRegex(ModelPipelineError, "No 'pipelines' section found in the configuration."):
            ModelPipeline(pipeline_name="p1", config_manager=self.mock_config_manager_instance)

    def test_pipeline_initialization_pipeline_def_not_found(self):
        self.mock_config_manager_instance.get_config.return_value = {"other_pipe": {}} # 'pipelines' exists, but not 'p1'
        with self.assertRaisesRegex(ModelPipelineError, "Pipeline definition for 'p1' not found in configuration."):
            ModelPipeline(pipeline_name="p1", config_manager=self.mock_config_manager_instance)

    def test_pipeline_initialization_missing_stages_key(self):
        self._setup_mock_config(stages_config=None) # This sets up a valid config
        # Now, make get_config return a pipeline_def without 'stages'
        self.mock_config_manager_instance.get_config.return_value = {"test_pipeline": {"description": "no stages here"}}
        with self.assertRaisesRegex(ModelPipelineError, "Pipeline 'test_pipeline' definition is missing a 'stages' list or it's not a list."):
            ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)

    def test_pipeline_initialization_stage_def_missing_attrs(self):
        bad_stage_config = [{"name": "BadStage"}] # Missing module and class
        self._setup_mock_config(stages_config=bad_stage_config)
        with self.assertRaisesRegex(ModelPipelineError, "Stage definition in pipeline 'test_pipeline' is missing 'name', 'module', or 'class'."):
            ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)

    def test_pipeline_initialization_stage_module_import_error(self):
        self._setup_mock_config()
        self.mock_importlib.import_module.side_effect = ImportError("Module not found for testing")
        with self.assertRaisesRegex(ModelPipelineError, "Could not import module for stage AlphaStage: Module not found for testing"):
            ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)

    def test_pipeline_initialization_stage_class_attr_error(self):
        self._setup_mock_config()
        # Make getattr fail for the first stage
        def mock_import_module_attr_err(module_path):
            if module_path == "test_module.alpha":
                mock_mod = MagicMock()
                # Configure the mock_mod to raise AttributeError when 'SampleStageAlpha' is accessed
                mock_mod.configure_mock(**{'SampleStageAlpha.side_effect': AttributeError("Test: Class not found")})
                # More direct way:
                # type(mock_mod).SampleStageAlpha = PropertyMock(side_effect=AttributeError("Test: Class not found"))
                # Or even simpler if we know getattr will be called:
                del mock_mod.SampleStageAlpha # This makes getattr raise AttributeError
                return mock_mod
            # Fallback for other imports if any during this specific test setup
            # This part might need adjustment if other stages are loaded in the same pipeline init
            # For this test, we only care about the first stage failing.
            # If other stages (like BetaStage) are also processed in the same ModelPipeline init,
            # their modules also need to be mocked correctly by this side_effect function.
            
            # Let's make it more robust for other stages in the same test pipeline init
            original_importer = self.mock_importlib.import_module.__defaults__[0] if self.mock_importlib.import_module.__defaults__ else None # Get original side_effect if any
            if module_path == "test_module.beta": # Assuming BetaStage is next
                 mock_beta_mod = MagicMock()
                 setattr(mock_beta_mod, "SampleStageBeta", SampleStageBeta)
                 return mock_beta_mod
            if original_importer:
                 return original_importer(module_path)
            raise ImportError(f"Test mock (attr_err) cannot import {module_path}")

        # Temporarily change side_effect for this test
        original_side_effect = self.mock_importlib.import_module.side_effect
        self.mock_importlib.import_module.side_effect = mock_import_module_attr_err
        
        # The expected error message comes from the AttributeError catch block
        with self.assertRaisesRegex(ModelPipelineError, "Could not find class for stage AlphaStage: "): # Added colon and space
            ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)
        
        self.mock_importlib.import_module.side_effect = original_side_effect # Restore

    def test_pipeline_initialization_stage_not_subclass_of_pipelinestage(self):
        self._setup_mock_config()
        # Make SampleStageAlpha appear as not a subclass
        class NotAPipelineStage: pass
        
        def mock_import_module_wrong_type(module_path):
            mock_mod = MagicMock()
            if module_path == "test_module.alpha":
                setattr(mock_mod, "SampleStageAlpha", NotAPipelineStage) # Return wrong type
            elif module_path == "test_module.beta":
                 setattr(mock_mod, "SampleStageBeta", SampleStageBeta)
            return mock_mod
        self.mock_importlib.import_module.side_effect = mock_import_module_wrong_type
        
        with self.assertRaisesRegex(ModelPipelineError, "Class SampleStageAlpha for stage AlphaStage does not inherit from PipelineStage."):
            ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)


    def test_run_pipeline_successful(self):
        self._setup_mock_config(pipeline_name="run_pipe")
        pipeline = ModelPipeline(pipeline_name="run_pipe", config_manager=self.mock_config_manager_instance)
        
        # Mock executor's run_pipeline to return a successful context
        mock_final_context = PipelineContext.get_instance() # Use the same context instance
        mock_final_context.set_metadata("pipeline_status", "completed")
        self.mock_executor_instance.run_pipeline.return_value = mock_final_context
        
        result_context = pipeline.run()

        self.mock_executor_instance.run_pipeline.assert_called_once()
        self.assertEqual(result_context.get_metadata("pipeline_status"), "completed")
        self.assertEqual(result_context.get_metadata("pipeline_name"), "run_pipe")
        self.mock_orchestrator_logger_instance.info.assert_any_call("Pipeline 'run_pipe' executed successfully.")

    def test_run_pipeline_executor_fails(self):
        self._setup_mock_config(pipeline_name="fail_pipe")
        pipeline = ModelPipeline(pipeline_name="fail_pipe", config_manager=self.mock_config_manager_instance)

        # Mock executor's run_pipeline to raise an error and return a context with failure info
        mock_failed_context = PipelineContext.get_instance()
        mock_failed_context.set_metadata("pipeline_status", "failed")
        mock_failed_context.set_metadata("error_stage", "BetaStage")
        mock_failed_context.set_metadata("error_message", "Executor error")
        
        self.mock_executor_instance.run_pipeline.side_effect = PipelineExecutionError("Executor error")
        # When run_pipeline raises, the ModelPipeline's run method should catch it and return self.context
        # So, we need to ensure self.context (which is mock_failed_context here) has the error info.
        # The executor itself is responsible for setting this on the context it's using.
        # For this test, we can assume the executor sets it correctly before raising.
        # Let's make the mock executor return the context when it fails, which is what the real one does.
        # No, the real executor raises, and the ModelPipeline.run() returns its own self.context.
        # So, we need to ensure that self.context within ModelPipeline gets updated.
        # The executor uses PipelineContext.get_instance(), so it's the same instance.

        # To simulate the executor setting the context before raising:
        def mock_run_pipeline_that_fails_and_sets_context():
            ctx = PipelineContext.get_instance()
            ctx.set_metadata("pipeline_status", "failed")
            ctx.set_metadata("error_stage", "BetaStage")
            ctx.set_metadata("error_message", "Executor error from mock")
            raise PipelineExecutionError("Executor error from mock")

        self.mock_executor_instance.run_pipeline.side_effect = mock_run_pipeline_that_fails_and_sets_context
        
        result_context = pipeline.run() # Should not re-raise PipelineExecutionError

        self.mock_executor_instance.run_pipeline.assert_called_once()
        self.assertEqual(result_context.get_metadata("pipeline_status"), "failed")
        self.assertEqual(result_context.get_metadata("error_stage"), "BetaStage")
        self.assertEqual(result_context.get_metadata("error_message"), "Executor error from mock")
        # This log is from the except block
        self.mock_orchestrator_logger_instance.error.assert_any_call(
            "Pipeline 'fail_pipe' execution failed: Executor error from mock", exc_info=True
        )
        # The other log ("...failed at stage...") is NOT called if PipelineExecutionError is raised by executor.run_pipeline()


    def test_run_pipeline_critical_failure(self):
        self._setup_mock_config(pipeline_name="crit_fail_pipe")
        pipeline = ModelPipeline(pipeline_name="crit_fail_pipe", config_manager=self.mock_config_manager_instance)
        self.mock_executor_instance.run_pipeline.side_effect = Exception("Very bad thing happened")

        with self.assertRaisesRegex(ModelPipelineError, "Critical failure in pipeline 'crit_fail_pipe': Very bad thing happened"):
            pipeline.run()
        
        self.assertEqual(self.context.get_metadata("pipeline_status"), "critical_failure")
        self.assertEqual(self.context.get_metadata("critical_error_message"), "Very bad thing happened")

    def test_get_stage_found(self):
        self._setup_mock_config()
        pipeline = ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)
        stage = pipeline.get_stage("AlphaStage")
        self.assertIsNotNone(stage)
        self.assertEqual(stage.name, "AlphaStage")

    def test_get_stage_not_found(self):
        self._setup_mock_config()
        pipeline = ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)
        stage = pipeline.get_stage("NonExistentStage")
        self.assertIsNone(stage)

    def test_pipeline_repr(self):
        self._setup_mock_config()
        pipeline = ModelPipeline(pipeline_name="test_pipeline", config_manager=self.mock_config_manager_instance)
        self.assertEqual(repr(pipeline), "<ModelPipeline(name='test_pipeline', stages_count=2)>")


if __name__ == '__main__':
    unittest.main()
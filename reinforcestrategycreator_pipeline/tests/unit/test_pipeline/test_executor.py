import unittest
from unittest.mock import MagicMock, patch, call

from reinforcestrategycreator_pipeline.src.pipeline.executor import PipelineExecutor, PipelineExecutionError
from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
# from reinforcestrategycreator_pipeline.src.monitoring.logger import AppLogger # To be mocked

# Concrete stage for testing, can be reused or defined per test
class MockStage(PipelineStage):
    def __init__(self, name: str, config=None, should_fail_in=None, fail_message="Test Failure"):
        super().__init__(name, config or {})
        self.should_fail_in = should_fail_in # "setup", "run", "teardown"
        self.fail_message = fail_message
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False

    def setup(self, context: PipelineContext) -> None:
        self.setup_called = True
        context.set(f"{self.name}_setup_done", True)
        if self.should_fail_in == "setup":
            raise Exception(self.fail_message)

    def run(self, context: PipelineContext) -> PipelineContext:
        self.run_called = True
        context.set(f"{self.name}_run_done", True)
        if self.should_fail_in == "run":
            raise Exception(self.fail_message)
        context.set(f"{self.name}_output", f"output_from_{self.name}")
        return context

    def teardown(self, context: PipelineContext) -> None:
        self.teardown_called = True
        context.set(f"{self.name}_teardown_done", True)
        if self.should_fail_in == "teardown":
            raise Exception(self.fail_message)

class TestPipelineExecutor(unittest.TestCase):

    def setUp(self):
        # Reset context singleton for test isolation
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()
        
        # Patch get_logger for PipelineExecutor
        self.mock_executor_get_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.executor.get_logger')
        self.mock_executor_get_logger = self.mock_executor_get_logger_patcher.start()
        self.mock_executor_logger_instance = MagicMock()
        self.mock_executor_get_logger.return_value = self.mock_executor_logger_instance
        
        # Patch get_logger for PipelineStage (used by MockStage)
        self.mock_stage_get_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_stage_get_logger = self.mock_stage_get_logger_patcher.start()
        self.mock_stage_logger_instance = MagicMock()
        self.mock_stage_get_logger.return_value = self.mock_stage_logger_instance


    def tearDown(self):
        self.mock_executor_get_logger_patcher.stop()
        self.mock_stage_get_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_executor_initialization_no_stages(self):
        with self.assertRaisesRegex(ValueError, "PipelineExecutor must be initialized with at least one stage."):
            PipelineExecutor(stages=[])

    def test_executor_initialization_with_stages(self):
        stage1 = MockStage(name="Stage1")
        executor = PipelineExecutor(stages=[stage1])
        self.assertEqual(len(executor.stages), 1)
        self.assertIs(executor.stages[0], stage1)
        self.mock_executor_get_logger.assert_called_with('executor.PipelineExecutor')

    def test_run_pipeline_successful_single_stage(self):
        stage1 = MockStage(name="Stage1")
        executor = PipelineExecutor(stages=[stage1])
        
        final_context = executor.run_pipeline()

        self.assertTrue(stage1.setup_called)
        self.assertTrue(stage1.run_called)
        self.assertTrue(stage1.teardown_called)
        self.assertEqual(final_context.get("Stage1_output"), "output_from_Stage1")
        self.assertEqual(final_context.get_metadata("pipeline_status"), "completed")
        self.assertEqual(final_context.get_metadata("total_stages"), 1)
        self.mock_executor_logger_instance.info.assert_any_call("Starting pipeline execution...")
        self.mock_executor_logger_instance.info.assert_any_call("Stage completed successfully: Stage1")
        self.mock_executor_logger_instance.info.assert_any_call("Pipeline execution finished.")

    def test_run_pipeline_successful_multiple_stages(self):
        stage1 = MockStage(name="Stage1")
        stage2 = MockStage(name="Stage2")
        executor = PipelineExecutor(stages=[stage1, stage2])

        final_context = executor.run_pipeline()

        self.assertTrue(stage1.run_called)
        self.assertTrue(stage1.teardown_called)
        self.assertTrue(stage2.run_called)
        self.assertTrue(stage2.teardown_called)
        self.assertEqual(final_context.get("Stage2_output"), "output_from_Stage2")
        self.assertEqual(final_context.get_metadata("pipeline_status"), "completed")
        self.assertEqual(final_context.get_metadata("total_stages"), 2)

    def test_run_pipeline_stage_fails_in_setup(self):
        stage1 = MockStage(name="FailSetupStage", should_fail_in="setup", fail_message="Setup Failed Here")
        stage2 = MockStage(name="NeverRunStage")
        executor = PipelineExecutor(stages=[stage1, stage2])

        with self.assertRaisesRegex(PipelineExecutionError, "Stage 'FailSetupStage' failed: Setup Failed Here"):
            executor.run_pipeline()
        
        self.assertTrue(stage1.setup_called) # Setup was called and failed
        self.assertFalse(stage1.run_called)   # Run should not be called
        self.assertTrue(stage1.teardown_called) # Teardown for failed stage should be attempted

        self.assertFalse(stage2.setup_called) # Stage2 should not start

        self.assertEqual(self.context.get_metadata("pipeline_status"), "failed")
        self.assertEqual(self.context.get_metadata("error_stage"), "FailSetupStage")
        self.assertEqual(self.context.get_metadata("error_message"), "Setup Failed Here")
        self.mock_executor_logger_instance.error.assert_any_call(
            "Error during execution of stage 'FailSetupStage': Setup Failed Here", exc_info=True
        )

    def test_run_pipeline_stage_fails_in_run(self):
        stage1 = MockStage(name="FailRunStage", should_fail_in="run", fail_message="Run Failed Badly")
        stage2 = MockStage(name="NotReachedStage")
        executor = PipelineExecutor(stages=[stage1, stage2])

        with self.assertRaisesRegex(PipelineExecutionError, "Stage 'FailRunStage' failed: Run Failed Badly"):
            executor.run_pipeline()

        self.assertTrue(stage1.setup_called)
        self.assertTrue(stage1.run_called) # Run was called and failed
        self.assertTrue(stage1.teardown_called) # Teardown for failed stage attempted

        self.assertFalse(stage2.setup_called)

        self.assertEqual(self.context.get_metadata("pipeline_status"), "failed")
        self.assertEqual(self.context.get_metadata("error_stage"), "FailRunStage")

    def test_run_pipeline_stage_fails_in_teardown_after_success(self):
        # If teardown fails for a stage that otherwise succeeded, pipeline should still complete,
        # but an error should be logged and noted in context.
        stage1 = MockStage(name="SuccessfulRunStage")
        stage2 = MockStage(name="FailTeardownStage", should_fail_in="teardown", fail_message="Teardown Exploded")
        stage3 = MockStage(name="FinalStageAfterTeardownFail") # This should still run
        
        executor = PipelineExecutor(stages=[stage1, stage2, stage3])
        final_context = executor.run_pipeline()

        self.assertTrue(stage1.teardown_called)
        self.assertTrue(stage2.setup_called)
        self.assertTrue(stage2.run_called)
        self.assertTrue(stage2.teardown_called) # Teardown was called and failed
        self.assertTrue(stage3.run_called) # Stage3 should still run

        self.assertEqual(final_context.get_metadata("pipeline_status"), "completed") # Pipeline completes
        self.assertEqual(final_context.get_metadata(f"teardown_error_{stage2.name}"), "Teardown Exploded")
        self.mock_executor_logger_instance.error.assert_any_call(
            "Error during teardown of stage 'FailTeardownStage': Teardown Exploded", exc_info=True
        )

    def test_run_pipeline_stage_fails_in_teardown_after_run_failure(self):
        # If run fails, and then its teardown also fails.
        stage1 = MockStage(name="FailRunAndTeardown", should_fail_in="run", fail_message="Run Failed First")
        # Modify the mock stage instance to also fail in teardown
        original_teardown = stage1.teardown
        def new_teardown(context):
            original_teardown(context) # Call original to set teardown_called
            raise Exception("Teardown Also Failed")
        stage1.teardown = new_teardown
        
        stage2 = MockStage(name="WontRunStage")
        executor = PipelineExecutor(stages=[stage1, stage2])

        with self.assertRaisesRegex(PipelineExecutionError, "Stage 'FailRunAndTeardown' failed: Run Failed First"):
            executor.run_pipeline()

        self.assertTrue(stage1.setup_called)
        self.assertTrue(stage1.run_called)
        self.assertTrue(stage1.teardown_called) # Teardown was attempted

        self.assertEqual(self.context.get_metadata("pipeline_status"), "failed")
        self.assertEqual(self.context.get_metadata("error_stage"), "FailRunAndTeardown")
        self.assertEqual(self.context.get_metadata("error_message"), "Run Failed First")
        
        # Check that the teardown error for the failed stage was logged
        self.mock_executor_logger_instance.error.assert_any_call(
            "Error during teardown of failed stage 'FailRunAndTeardown': Teardown Also Failed", exc_info=True
        )

    def test_context_updates_during_execution(self):
        stage1 = MockStage(name="CtxStage1")
        stage2 = MockStage(name="CtxStage2")
        executor = PipelineExecutor(stages=[stage1, stage2])

        # Mock stage methods to check context at each point
        def setup1_check_ctx(context):
            self.assertEqual(context.get_metadata("current_stage_index"), 0)
            self.assertEqual(context.get_metadata("current_stage_name"), "CtxStage1")
            stage1.setup_called = True
        stage1.setup = setup1_check_ctx
        
        def run1_check_ctx(context):
            self.assertEqual(context.get_metadata("current_stage_name"), "CtxStage1")
            stage1.run_called = True
            return context
        stage1.run = run1_check_ctx

        def setup2_check_ctx(context):
            self.assertEqual(context.get_metadata("current_stage_index"), 1)
            self.assertEqual(context.get_metadata("current_stage_name"), "CtxStage2")
            stage2.setup_called = True
        stage2.setup = setup2_check_ctx

        executor.run_pipeline()
        self.assertTrue(stage1.setup_called)
        self.assertTrue(stage1.run_called)
        self.assertTrue(stage2.setup_called)
        self.assertEqual(self.context.get_metadata("pipeline_status"), "completed")


if __name__ == '__main__':
    unittest.main()
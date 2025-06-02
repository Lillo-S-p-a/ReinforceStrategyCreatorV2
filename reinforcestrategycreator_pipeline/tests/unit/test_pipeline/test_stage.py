import unittest
from unittest.mock import MagicMock, patch # Added patch

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
# Assuming AppLogger can be mocked or is available
# from reinforcestrategycreator_pipeline.src.monitoring.logger import AppLogger

# A concrete implementation for testing purposes
class ConcreteTestStage(PipelineStage):
    def setup(self, context: PipelineContext) -> None:
        context.set(f"{self.name}_setup_called", True)

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set(f"{self.name}_run_called", True)
        context.set(f"{self.name}_output", "dummy_output")
        return context

    def teardown(self, context: PipelineContext) -> None:
        context.set(f"{self.name}_teardown_called", True)

class TestPipelineStage(unittest.TestCase):

    def setUp(self):
        # Reset context singleton for test isolation
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()
        
        # Mock get_logger for this test module
        self.mock_get_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_get_logger = self.mock_get_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger_instance


    def tearDown(self):
        self.mock_get_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_stage_initialization(self):
        stage_name = "TestStage1"
        stage_config = {"param1": "value1", "retries": 3}
        
        stage = ConcreteTestStage(name=stage_name, config=stage_config)
        
        self.assertEqual(stage.name, stage_name)
        self.assertEqual(stage.config, stage_config)
        self.assertTrue(hasattr(stage, "logger")) # Check logger attribute exists
        # self.mock_logger.assert_called_once_with('ConcreteTestStage') # If AppLogger is mocked

    def test_stage_abstract_methods_existence(self):
        # Check that the abstract methods are defined
        self.assertTrue(hasattr(PipelineStage, "setup"))
        self.assertTrue(hasattr(PipelineStage, "run"))
        self.assertTrue(hasattr(PipelineStage, "teardown"))

    def test_stage_repr(self):
        stage = ConcreteTestStage(name="MyReprStage", config={})
        self.assertEqual(repr(stage), "<PipelineStage(name='MyReprStage')>")

    def test_concrete_stage_execution_flow(self):
        """
        Tests the basic flow of a concrete stage's methods being called.
        This indirectly tests that the ABC structure allows for such calls.
        """
        stage = ConcreteTestStage(name="FlowTestStage", config={"key": "val"})
        
        # Test setup
        stage.setup(self.context)
        self.assertTrue(self.context.get("FlowTestStage_setup_called"))

        # Test run
        updated_context = stage.run(self.context)
        self.assertIs(updated_context, self.context, "Run method should return the context instance.")
        self.assertTrue(self.context.get("FlowTestStage_run_called"))
        self.assertEqual(self.context.get("FlowTestStage_output"), "dummy_output")

        # Test teardown
        stage.teardown(self.context)
        self.assertTrue(self.context.get("FlowTestStage_teardown_called"))

if __name__ == '__main__':
    unittest.main()
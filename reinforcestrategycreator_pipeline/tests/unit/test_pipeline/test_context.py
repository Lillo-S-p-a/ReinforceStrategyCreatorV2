import unittest
import threading
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext, PipelineContextError

class TestPipelineContext(unittest.TestCase):

    def setUp(self):
        # Ensure a clean context for each test by resetting the singleton
        # This is a bit tricky with singletons. We might need to manually reset its internal state.
        # Or, for testing, allow creating new instances or provide a reset method.
        # The current PipelineContext has a reset() method.
        # And get_instance() will create one if _instance is None.
        # We need to ensure _instance is None before each test or that reset() is effective.
        
        # Hacky way to reset singleton for testing, not ideal for production code structure
        if PipelineContext._instance:
            PipelineContext._instance.reset() # Use the provided reset method
            PipelineContext._instance = None # Force re-creation for isolation
        
        self.context = PipelineContext.get_instance()

    def tearDown(self):
        # Clean up the instance after tests if necessary
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None


    def test_singleton_instance(self):
        instance1 = PipelineContext.get_instance()
        instance2 = PipelineContext.get_instance()
        self.assertIs(instance1, instance2, "PipelineContext should return the same instance.")
        # Test that direct instantiation raises an error after the first one is made via get_instance()
        # self.context is already one instance.
        with self.assertRaises(PipelineContextError):
             PipelineContext()


    def test_set_and_get_data(self):
        self.context.set("test_key", "test_value")
        self.assertEqual(self.context.get("test_key"), "test_value")
        self.assertIsNone(self.context.get("non_existent_key"))
        self.assertEqual(self.context.get("non_existent_key", "default"), "default")

    def test_delete_data(self):
        self.context.set("to_delete", "value")
        self.assertIsNotNone(self.context.get("to_delete"))
        self.context.delete("to_delete")
        self.assertIsNone(self.context.get("to_delete"))
        with self.assertRaises(KeyError):
            self.context.delete("non_existent_key_for_delete")

    def test_get_all_data(self):
        self.context.set("key1", "value1")
        self.context.set("key2", 123)
        all_data = self.context.get_all_data()
        self.assertEqual(all_data, {"key1": "value1", "key2": 123})
        # Ensure it's a copy
        all_data["key3"] = "value3"
        self.assertIsNone(self.context.get("key3"))


    def test_clear_data(self):
        self.context.set("key1", "value1")
        self.context.clear_data()
        self.assertEqual(self.context.get_all_data(), {})
        self.assertIsNone(self.context.get("key1"))

    def test_set_and_get_metadata(self):
        self.context.set_metadata("meta_key", "meta_value")
        self.assertEqual(self.context.get_metadata("meta_key"), "meta_value")
        self.assertIsNone(self.context.get_metadata("non_existent_meta"))
        self.assertEqual(self.context.get_metadata("non_existent_meta", "default_meta"), "default_meta")

    def test_get_all_metadata(self):
        self.context.set_metadata("m_key1", "m_value1")
        self.context.set_metadata("m_key2", True)
        all_metadata = self.context.get_all_metadata()
        self.assertEqual(all_metadata, {"m_key1": "m_value1", "m_key2": True})
        # Ensure it's a copy
        all_metadata["m_key3"] = "m_value3"
        self.assertIsNone(self.context.get_metadata("m_key3"))

    def test_clear_metadata(self):
        self.context.set_metadata("m_key1", "m_value1")
        self.context.clear_metadata()
        self.assertEqual(self.context.get_all_metadata(), {})
        self.assertIsNone(self.context.get_metadata("m_key1"))

    def test_reset_context(self):
        self.context.set("data_key", "data_val")
        self.context.set_metadata("meta_key", "meta_val")
        self.context.reset()
        self.assertEqual(self.context.get_all_data(), {})
        self.assertEqual(self.context.get_all_metadata(), {})
        self.assertIsNone(self.context.get("data_key"))
        self.assertIsNone(self.context.get_metadata("meta_key"))

    def test_thread_safety(self):
        # This is a basic test for thread safety. More rigorous testing might be needed.
        num_threads = 10
        iterations = 100
        
        # Reset context for this specific test
        PipelineContext._instance = None
        context_instance = PipelineContext.get_instance()

        def worker_set(key_prefix):
            for i in range(iterations):
                context_instance.set(f"{key_prefix}_{i}", i)

        def worker_get(key_prefix):
            for i in range(iterations):
                context_instance.get(f"{key_prefix}_{i}")
        
        threads = []
        for i in range(num_threads // 2):
            threads.append(threading.Thread(target=worker_set, args=(f"thread_set_{i}",)))
        for i in range(num_threads // 2):
            threads.append(threading.Thread(target=worker_get, args=(f"thread_set_{i}",))) # Getting keys set by other threads

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check if all expected keys are present (from set operations)
        for i in range(num_threads // 2):
            for j in range(iterations):
                self.assertEqual(context_instance.get(f"thread_set_{i}_{j}"), j)
        
        # Clean up after thread safety test
        PipelineContext._instance.reset()
        PipelineContext._instance = None


if __name__ == '__main__':
    unittest.main()
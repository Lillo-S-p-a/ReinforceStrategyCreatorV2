import sys
import os
from pathlib import Path

# Add project root to Python path to allow direct execution of script
# and ensure relative imports within the pipeline work correctly.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.pipeline.orchestrator import ModelPipeline
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger

# Initialize a basic logger for the script itself
script_logger = get_logger("run_main_pipeline_script")

def main():
    script_logger.info("Starting main pipeline execution script...")
    try:
        # Define paths
        pipeline_dir = Path(__file__).resolve().parent
        config_dir = pipeline_dir / "configs"
        config_base_path = config_dir / "base"

        script_logger.info(f"Using config directory: {config_dir}")

        # Instantiate ConfigManager
        config_manager = ConfigManager(config_dir=config_dir)
        script_logger.info("ConfigManager instantiated.")

        # Load base pipeline.yaml configuration
        script_logger.info(f"Loading base configuration from: {config_base_path / 'pipeline.yaml'}")
        config_manager.load_config(config_path="base/pipeline.yaml") # Relative to config_dir
        script_logger.info("Base configuration loaded.")

        # Load pipeline definitions
        from reinforcestrategycreator_pipeline.src.config.loader import ConfigLoader
        loader = ConfigLoader(base_path=config_base_path)
        pipelines_definition_file = "pipelines_definition.yaml"
        script_logger.info(f"Loading pipeline definitions from: {config_base_path / pipelines_definition_file}")
        pipelines_def_dict = loader.load_yaml(pipelines_definition_file)
        script_logger.info("Pipeline definitions loaded.")

        # Update config_manager with pipeline definitions
        config_manager.update_config(pipelines_def_dict)
        script_logger.info("ConfigManager updated with pipeline definitions.")

        # Define the pipeline name to run
        pipeline_name = "full_cycle_pipeline"
        script_logger.info(f"Initializing ModelPipeline for: {pipeline_name}")

        # Instantiate ModelPipeline
        pipeline = ModelPipeline(pipeline_name=pipeline_name, config_manager=config_manager)
        script_logger.info("ModelPipeline instantiated.")

        # Run the pipeline
        script_logger.info(f"Running pipeline: {pipeline_name}...")
        pipeline.run()
        script_logger.info(f"Pipeline '{pipeline_name}' execution finished.")

    except Exception as e:
        script_logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
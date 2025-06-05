import logging
import sys

def get_pipeline_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger instance for the pipeline.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a handler if not already present, to avoid duplicate logs on re-runs in same session
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to the root logger if it has default handlers
    # that might also output to stdout, causing duplicates.
    logger.propagate = False

    return logger

if __name__ == "__main__":
    # Example usage:
    logger1 = get_pipeline_logger("MyModule1")
    logger1.info("This is an info message from MyModule1.")
    logger1.warning("This is a warning message from MyModule1.")

    logger2 = get_pipeline_logger("MyModule2", level=logging.DEBUG)
    logger2.debug("This is a debug message from MyModule2.")
    logger2.info("This is an info message from MyModule2 that should also appear.")

    # Test no duplicate handlers
    logger1_again = get_pipeline_logger("MyModule1")
    logger1_again.info("Another info message from MyModule1, should not have double handlers.")
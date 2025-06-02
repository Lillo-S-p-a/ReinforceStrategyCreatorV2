"""Unit tests for the logging utilities."""

import pytest
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.monitoring.logger import (
    StructuredFormatter,
    PipelineLogger,
    get_logger,
    configure_logging,
    log_with_context,
    log_execution_time,
    log_step
)


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""
    
    def test_format_basic_log(self):
        """Test formatting a basic log record."""
        formatter = StructuredFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON
        log_data = json.loads(formatted)
        
        # Verify fields
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert log_data["line"] == 10
        assert "timestamp" in log_data
    
    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.extra_fields = {"user_id": 123, "action": "login"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == 123
        assert log_data["action"] == "login"
    
    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
        assert isinstance(log_data["exception"]["traceback"], list)


class TestPipelineLogger:
    """Test the PipelineLogger class."""
    
    def test_singleton_pattern(self):
        """Test that PipelineLogger follows singleton pattern."""
        logger1 = PipelineLogger()
        logger2 = PipelineLogger()
        assert logger1 is logger2
    
    def test_configure_with_console(self):
        """Test configuring logger with console output."""
        logger = PipelineLogger()
        logger.configure(
            log_level="DEBUG",
            enable_console=True,
            enable_json=False
        )
        
        # Check that console handler is added
        assert len(logger._logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in logger._logger.handlers)
        assert logger._logger.level == logging.DEBUG
    
    def test_configure_with_file(self):
        """Test configuring logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            logger = PipelineLogger()
            logger.configure(
                log_level="INFO",
                log_file=str(log_file),
                enable_console=False
            )
            
            # Check that file handler is added
            assert any(
                isinstance(h, logging.handlers.RotatingFileHandler) 
                for h in logger._logger.handlers
            )
            
            # Test logging
            test_logger = logger.get_logger("test")
            test_logger.info("Test message")
            
            # Verify file was created
            assert log_file.exists()
    
    def test_get_logger(self):
        """Test getting logger instances."""
        pipeline_logger = PipelineLogger()
        
        # Get root pipeline logger
        root_logger = pipeline_logger.get_logger()
        assert root_logger.name == "pipeline"
        
        # Get named logger
        named_logger = pipeline_logger.get_logger("test.module")
        assert named_logger.name == "pipeline.test.module"
    
    def test_log_with_context(self):
        """Test logging with context."""
        logger = PipelineLogger()
        
        with patch.object(logger._logger, 'info') as mock_info:
            logger.log_with_context(
                "info",
                "Test message",
                user_id=123,
                action="test"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Test message"
            assert "extra" in call_args[1]
            assert call_args[1]["extra"]["extra_fields"]["user_id"] == 123


class TestLoggingFunctions:
    """Test the module-level logging functions."""
    
    def test_get_logger_function(self):
        """Test the get_logger function."""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        
        assert logger1.name == "pipeline.test1"
        assert logger2.name == "pipeline.test2"
    
    def test_configure_logging_function(self):
        """Test the configure_logging function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            configure_logging(
                log_level="WARNING",
                log_file=str(log_file),
                enable_console=True,
                enable_json=True
            )
            
            # Test that configuration was applied
            logger = get_logger("test")
            assert logger.level <= logging.WARNING
    
    def test_log_with_context_function(self):
        """Test the log_with_context function."""
        with patch('src.monitoring.logger.logger_instance.log_with_context') as mock_log:
            log_with_context("info", "Test", key="value")
            mock_log.assert_called_once_with("info", "Test", key="value")


class TestDecorators:
    """Test the logging decorators."""
    
    def test_log_execution_time_success(self):
        """Test log_execution_time decorator with successful execution."""
        @log_execution_time
        def test_function():
            return "success"
        
        with patch('src.monitoring.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function()
            
            assert result == "success"
            assert mock_logger.info.called
            
            # Check that success was logged
            call_args = mock_logger.info.call_args
            assert "completed" in call_args[0][0]
    
    def test_log_execution_time_failure(self):
        """Test log_execution_time decorator with failed execution."""
        @log_execution_time
        def test_function():
            raise ValueError("Test error")
        
        with patch('src.monitoring.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                test_function()
            
            assert mock_logger.error.called
            
            # Check that error was logged
            call_args = mock_logger.error.call_args
            assert "failed" in call_args[0][0]
    
    def test_log_step_decorator(self):
        """Test log_step decorator."""
        @log_step("Test Step")
        def test_function(value):
            return value * 2
        
        with patch('src.monitoring.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function(5)
            
            assert result == 10
            
            # Check that step start and completion were logged
            assert mock_logger.info.call_count == 2
            
            # Check start log
            start_call = mock_logger.info.call_args_list[0]
            assert "Starting pipeline step: Test Step" in start_call[0][0]
            
            # Check completion log
            end_call = mock_logger.info.call_args_list[1]
            assert "Completed pipeline step: Test Step" in end_call[0][0]
    
    def test_log_step_decorator_with_error(self):
        """Test log_step decorator with error."""
        @log_step("Error Step")
        def test_function():
            raise RuntimeError("Step failed")
        
        with patch('src.monitoring.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(RuntimeError):
                test_function()
            
            # Check that error was logged
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "Failed pipeline step: Error Step" in error_call[0][0]
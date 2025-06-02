"""Centralized logging configuration and utilities for the pipeline."""

import logging
import logging.handlers
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
import traceback


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class PipelineLogger:
    """Centralized logger for the pipeline with structured logging support."""
    
    _instance: Optional['PipelineLogger'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'PipelineLogger':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger (only once)."""
        if not self._initialized:
            self._logger = logging.getLogger("pipeline")
            self._logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
            self._logger.propagate = False
            self._initialized = True
    
    def configure(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_json: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Configure the logger with specified settings.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            enable_console: Whether to enable console output
            enable_json: Whether to use JSON formatting
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self._logger.setLevel(level)
        
        # Configure formatter
        if enable_json:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (will be prefixed with 'pipeline.')
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f"pipeline.{name}")
        return self._logger
    
    def log_with_context(
        self,
        level: str,
        message: str,
        **context: Any
    ) -> None:
        """
        Log a message with additional context.
        
        Args:
            level: Log level
            message: Log message
            **context: Additional context to include in the log
        """
        logger = self._logger
        log_method = getattr(logger, level.lower())
        
        # Create a LogRecord with extra fields
        extra = {"extra_fields": context}
        log_method(message, extra=extra)


# Singleton instance
logger_instance = PipelineLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    return logger_instance.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = True
) -> None:
    """
    Configure the centralized logger.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        enable_console: Whether to enable console output
        enable_json: Whether to use JSON formatting
    """
    logger_instance.configure(
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_json=enable_json
    )


def log_with_context(level: str, message: str, **context: Any) -> None:
    """
    Log a message with additional context.
    
    Args:
        level: Log level
        message: Log message
        **context: Additional context
    """
    logger_instance.log_with_context(level, message, **context)


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now(timezone.utc)
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Function {func.__name__} completed",
                extra={
                    "extra_fields": {
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "status": "success"
                    }
                }
            )
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "extra_fields": {
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                },
                exc_info=True
            )
            raise
    
    return wrapper


def log_step(step_name: str):
    """
    Decorator to log pipeline steps.
    
    Args:
        step_name: Name of the pipeline step
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            logger.info(
                f"Starting pipeline step: {step_name}",
                extra={
                    "extra_fields": {
                        "step_name": step_name,
                        "function": func.__name__,
                        "status": "started"
                    }
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                logger.info(
                    f"Completed pipeline step: {step_name}",
                    extra={
                        "extra_fields": {
                            "step_name": step_name,
                            "function": func.__name__,
                            "status": "completed"
                        }
                    }
                )
                return result
                
            except Exception as e:
                logger.error(
                    f"Failed pipeline step: {step_name}",
                    extra={
                        "extra_fields": {
                            "step_name": step_name,
                            "function": func.__name__,
                            "status": "failed",
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    },
                    exc_info=True
                )
                raise
    
        return wrapper
    return decorator
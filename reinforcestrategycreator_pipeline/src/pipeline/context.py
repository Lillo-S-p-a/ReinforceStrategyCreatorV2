from typing import Any, Dict, Optional
import threading

class PipelineContextError(Exception):
    """Custom exception for PipelineContext errors.
    
    Raised when there are issues with PipelineContext operations,
    such as attempting to create multiple instances of the singleton.
    """
    pass

class PipelineContext:
    """Manages shared state and data between pipeline stages.

    This class implements a thread-safe singleton pattern to provide a centralized
    storage mechanism for data and metadata that needs to be shared across different
    stages of a machine learning pipeline. It maintains two separate dictionaries:
    one for pipeline data (models, datasets, results) and another for metadata
    (execution status, timestamps, configuration).
    
    The singleton pattern ensures that all pipeline stages access the same context
    instance, maintaining consistency throughout the pipeline execution.
    
    Attributes:
        _instance: The singleton instance of PipelineContext
        _lock: Threading lock for thread-safe operations
        _data: Dictionary storing pipeline data
        _metadata: Dictionary storing pipeline metadata
    
    Example:
        >>> context = PipelineContext.get_instance()
        >>> context.set('model', trained_model)
        >>> context.set_metadata('pipeline_status', 'training_complete')
        >>> model = context.get('model')
    """
    _instance: Optional['PipelineContext'] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        if PipelineContext._instance is not None:
            raise PipelineContextError("PipelineContext is a singleton and has already been instantiated.")
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {} # For storing pipeline run info, etc.
        PipelineContext._instance = self

    @classmethod
    def get_instance(cls) -> 'PipelineContext':
        """Provide access to the singleton instance of PipelineContext.
        
        This method implements thread-safe lazy initialization of the singleton
        instance using double-check locking pattern.
        
        :return: The singleton instance of PipelineContext
        :rtype: PipelineContext
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: # Double-check locking
                    cls._instance = cls()
        return cls._instance

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context data dictionary.

        :param key: The key to store the value under
        :type key: str
        :param value: The value to store (can be any type)
        :type value: Any
        """
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get a value from the context data dictionary.

        :param key: The key of the value to retrieve
        :type key: str
        :param default: The default value to return if the key is not found
        :type default: Optional[Any]
        
        :return: The value associated with the key, or the default value if not found
        :rtype: Optional[Any]
        """
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str) -> None:
        """Delete a key-value pair from the context data dictionary.

        :param key: The key to delete
        :type key: str
        
        :raises KeyError: If the key is not found in the context
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
            else:
                raise KeyError(f"Key '{key}' not found in PipelineContext.")

    def get_all_data(self) -> Dict[str, Any]:
        """Return a copy of all data stored in the context.
        
        :return: A dictionary containing all key-value pairs in the data store
        :rtype: Dict[str, Any]
        """
        with self._lock:
            return self._data.copy()

    def clear_data(self) -> None:
        """Clear all data from the context.
        
        This method removes all key-value pairs from the data dictionary
        but preserves the metadata.
        """
        with self._lock:
            self._data.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Metadata is used to store pipeline-level information such as
        execution status, timestamps, and configuration details.
        
        :param key: The metadata key
        :type key: str
        :param value: The metadata value
        :type value: Any
        """
        with self._lock:
            self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get a metadata value.
        
        :param key: The metadata key to retrieve
        :type key: str
        :param default: The default value if key is not found
        :type default: Optional[Any]
        
        :return: The metadata value or default if not found
        :rtype: Optional[Any]
        """
        with self._lock:
            return self._metadata.get(key, default)

    def get_all_metadata(self) -> Dict[str, Any]:
        """Return a copy of all metadata.
        
        :return: A dictionary containing all metadata key-value pairs
        :rtype: Dict[str, Any]
        """
        with self._lock:
            return self._metadata.copy()

    def clear_metadata(self) -> None:
        """Clear all metadata from the context.
        
        This method removes all key-value pairs from the metadata dictionary
        but preserves the data dictionary.
        """
        with self._lock:
            self._metadata.clear()
            
    def reset(self) -> None:
        """Reset the context by clearing both data and metadata.
        
        This method is useful for testing scenarios or when re-running
        pipelines with a clean state. It removes all stored data and
        metadata but maintains the singleton instance.
        """
        self.clear_data()
        self.clear_metadata()

    def __repr__(self) -> str:
        return f"<PipelineContext(data_keys={list(self._data.keys())}, metadata_keys={list(self._metadata.keys())})>"

# Ensure the placeholder import in stage.py can be resolved if context.py is created first.
# This is a common pattern to avoid circular dependencies at module load time,
# though direct usage in stage.py's __init__ might still need careful handling
# if type hinting `PipelineContext` directly.
# For now, the forward reference `from .context import PipelineContext` in stage.py
# should work once context.py exists.
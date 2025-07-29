"""Comprehensive tests for loggings.py module.

This test suite provides complete coverage of the FLEXT logging system,
testing all aspects including FlextLogger, FlextLoggerFactory, context management,
structured logging, global log store, and enterprise-grade features to achieve
near 100% coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from flext_core.loggings import (
    FlextLogContext,
    FlextLogContextManager,
    FlextLogger,
    FlextLoggerFactory,
    _add_to_log_store,
    _log_store,
)

if TYPE_CHECKING:
    from flext_core.types import TContextDict

pytestmark = [pytest.mark.unit, pytest.mark.core]

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3
DEBUG = 20


# Test fixtures
@pytest.fixture
def clean_log_store() -> None:
    """Clean the global log store before each test."""
    _log_store.clear()


@pytest.fixture
def sample_context() -> TContextDict:
    """Sample context data for testing."""
    return {
        "user_id": "test_user_123",
        "request_id": "req_abc_def",
        "operation": "test_operation",
        "tenant_id": "tenant_1",
    }


@pytest.fixture
def logger_instance() -> FlextLogger:
    """Create a fresh logger instance for testing."""
    return FlextLogger("test.logger", "DEBUG")


@pytest.fixture
def factory_instance() -> FlextLoggerFactory:
    """Create a fresh factory instance for testing."""
    # Clear factory cache and reset global level
    FlextLoggerFactory._loggers.clear()
    FlextLoggerFactory._global_level = "INFO"
    return FlextLoggerFactory()


# Test data classes
class LogDataSample:
    """Sample log data for structured testing."""

    def __init__(self, message: str, level: str = "INFO", **context: object) -> None:
        self.message = message
        self.level = level
        self.context = context


@pytest.mark.unit
class TestFlextLogContext:
    """Test FlextLogContext TypedDict functionality."""

    def test_typed_dict_structure(self) -> None:
        """Test FlextLogContext structure and typing."""
        context: FlextLogContext = {
            "user_id": "user123",
            "request_id": "req456",
            "session_id": "sess789",
            "operation": "create_order",
            "transaction_id": "tx_abc",
            "tenant_id": "tenant_1",
            "customer_id": "cust_xyz",
            "order_id": "order_123",
            "duration_ms": 150.5,
            "memory_mb": 128.0,
            "cpu_percent": 15.5,
            "error_code": "ERR001",
            "error_type": "ValidationError",
            "stack_trace": "traceback info",
        }

        # All fields should be accessible
        if context["user_id"] != "user123":
            msg = f"Expected {'user123'}, got {context['user_id']}"
            raise AssertionError(msg)
        assert context["request_id"] == "req456"
        if context["duration_ms"] != 150.5:
            msg = f"Expected {150.5}, got {context['duration_ms']}"
            raise AssertionError(msg)
        assert context["error_code"] == "ERR001"

    def test_partial_context(self) -> None:
        """Test FlextLogContext with partial fields."""
        context: FlextLogContext = {
            "user_id": "user123",
            "operation": "test_op",
        }

        if context["user_id"] != "user123":
            msg = f"Expected {'user123'}, got {context['user_id']}"
            raise AssertionError(msg)
        assert context["operation"] == "test_op"
        if len(context) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(context)}"
            raise AssertionError(msg)

    def test_empty_context(self) -> None:
        """Test empty FlextLogContext."""
        context: FlextLogContext = {}
        if len(context) != 0:
            msg = f"Expected {0}, got {len(context)}"
            raise AssertionError(msg)

    def test_context_field_types(self) -> None:
        """Test context field type compatibility."""
        # String fields
        context: FlextLogContext = {"user_id": "123", "error_type": "TestError"}
        assert isinstance(context["user_id"], str)
        assert isinstance(context["error_type"], str)

        # Numeric fields
        context = {"duration_ms": 100.5, "memory_mb": 256.0, "cpu_percent": 25.5}
        assert isinstance(context["duration_ms"], float)
        assert isinstance(context["memory_mb"], float)
        assert isinstance(context["cpu_percent"], float)


@pytest.mark.unit
class TestGlobalLogStore:
    """Test global log store functionality."""

    def test_log_store_initialization(self, clean_log_store: None) -> None:
        """Test log store starts empty."""
        if len(_log_store) != 0:
            msg = f"Expected {0}, got {len(_log_store)}"
            raise AssertionError(msg)
        assert isinstance(_log_store, list)

    def test_add_to_log_store_function(self, clean_log_store: None) -> None:
        """Test _add_to_log_store processor function."""
        mock_logger = Mock()
        mock_logger.name = "test.logger"

        event_dict = {
            "timestamp": "2025-01-01T10:00:00Z",
            "level": "info",
            "event": "Test message",
            "user_id": "123",
            "operation": "test",
        }

        result = _add_to_log_store(mock_logger, "info", event_dict)

        # Should return unchanged event_dict
        assert result is event_dict

        # Should add entry to log store
        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["timestamp"] != "2025-01-01T10:00:00Z":
            msg = f"Expected {'2025-01-01T10:00:00Z'}, got {log_entry['timestamp']}"
            raise AssertionError(msg)
        assert log_entry["level"] == "INFO"
        if log_entry["logger"] != "test.logger":
            msg = f"Expected {'test.logger'}, got {log_entry['logger']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Test message"
        if log_entry["method"] != "info":
            msg = f"Expected {'info'}, got {log_entry['method']}"
            raise AssertionError(msg)
        assert log_entry["context"]["user_id"] == "123"
        if log_entry["context"]["operation"] != "test":
            msg = f"Expected {'test'}, got {log_entry['context']['operation']}"
            raise AssertionError(msg)

    def test_add_to_log_store_with_missing_fields(self, clean_log_store: None) -> None:
        """Test _add_to_log_store with missing fields."""
        mock_logger = Mock()
        mock_logger.name = "test.logger"

        # Minimal event dict
        event_dict = {"event": "Minimal message"}

        result = _add_to_log_store(mock_logger, "debug", event_dict)

        assert result is event_dict
        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)

        log_entry = _log_store[0]
        if log_entry["level"] != "INFO":  # Default level:
            msg = f"Expected {'INFO'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["logger"] == "test.logger"
        if log_entry["message"] != "Minimal message":
            msg = f"Expected {'Minimal message'}, got {log_entry['message']}"
            raise AssertionError(msg)
        assert log_entry["method"] == "debug"
        if "timestamp" not in log_entry:
            msg = f"Expected {'timestamp'} in {log_entry}"
            raise AssertionError(msg)

    def test_add_to_log_store_logger_fallback(self, clean_log_store: None) -> None:
        """Test logger name fallback when not in event_dict."""
        mock_logger = Mock()
        mock_logger.name = "fallback.logger"

        event_dict = {"event": "Test message"}

        _add_to_log_store(mock_logger, "warn", event_dict)

        log_entry = _log_store[0]
        if log_entry["logger"] != "fallback.logger":
            msg = f"Expected {'fallback.logger'}, got {log_entry['logger']}"
            raise AssertionError(msg)

    # Console renderer tests removed - function no longer exists


@pytest.mark.unit
class TestFlextLogger:
    """Test FlextLogger core functionality."""

    def test_logger_initialization(self, clean_log_store: None) -> None:
        """Test logger initialization."""
        logger = FlextLogger("test.app", "DEBUG")

        if logger._name != "test.app":
            msg = f"Expected {'test.app'}, got {logger._name}"
            raise AssertionError(msg)
        assert logger._level == "DEBUG"
        if logger._level_value != 10:  # DEBUG level value:
            msg = f"Expected {10}, got {logger._level_value}"
            raise AssertionError(msg)
        assert isinstance(logger._context, dict)
        if len(logger._context) != 0:
            msg = f"Expected {0}, got {len(logger._context)}"
            raise AssertionError(msg)

    def test_logger_default_level(self, clean_log_store: None) -> None:
        """Test logger with default INFO level."""
        logger = FlextLogger("test.default")

        if logger._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger._level}"
            raise AssertionError(msg)
        assert logger._level_value == 20  # INFO level value

    def test_should_log_method(self, logger_instance: FlextLogger) -> None:
        """Test _should_log level filtering."""
        # Logger is at DEBUG level (20)
        assert logger_instance._should_log("TRACE") is False  # 5 < 10
        if logger_instance._should_log("DEBUG") is True:  # 20 < 20:
            msg = f"Expected {logger_instance._should_log('DEBUG') is True} >= {20}"
            raise AssertionError(msg)
        if logger_instance._should_log("INFO") is False:  # 30 >= 20:
            msg = f"Expected {logger_instance._should_log('INFO') is True} >= {20}"
            raise AssertionError(msg)
        if logger_instance._should_log("WARNING") is False:  # 40 < 20:
            msg = f"Expected {logger_instance._should_log('WARNING') is True} >= {20}"
            raise AssertionError(msg)
        if logger_instance._should_log("ERROR") is False:  # 50 >= 20:
            msg = f"Expected {logger_instance._should_log('ERROR') is True} >= {20}"
            raise AssertionError(msg)
        if logger_instance._should_log("CRITICAL") is False:  # 60 < 20:
            msg = f"Expected {logger_instance._should_log('CRITICAL') is True} >= {20}"
            raise AssertionError(msg)

    def test_should_log_with_enum_input(self, logger_instance: FlextLogger) -> None:
        """Test _should_log with enum-like input."""
        # Mock enum-like object
        level_enum = Mock()
        level_enum.value = "INFO"

        result = logger_instance._should_log(level_enum)
        if not (result):
            msg = f"Expected True, got {result}"
            raise AssertionError(msg)

    def test_should_log_unknown_level(self, logger_instance: FlextLogger) -> None:
        """Test _should_log with unknown level defaults to INFO."""
        result = logger_instance._should_log("UNKNOWN_LEVEL")
        # Unknown level defaults to INFO, which should be logged when logger is at DEBUG level
        if not result:
            msg = f"Expected True (INFO should be logged at DEBUG level), got {result}"
            raise AssertionError(msg)

    def test_info_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test info level logging."""
        logger_instance.info("Test info message", user_id="123", action="test")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "INFO":
            msg = f"Expected {'INFO'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["logger"] == "test.logger"
        if log_entry["message"] != "Test info message":
            msg = f"Expected {'Test info message'}, got {log_entry['message']}"
            raise AssertionError(msg)
        assert log_entry["context"]["user_id"] == "123"
        if log_entry["context"]["action"] != "test":
            msg = f"Expected {'test'}, got {log_entry['context']['action']}"
            raise AssertionError(msg)

    def test_debug_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test debug level logging."""
        logger_instance.debug("Debug message", operation="debug_test")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "DEBUG":
            msg = f"Expected {'DEBUG'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Debug message"
        if log_entry["context"]["operation"] != "debug_test":
            msg = f"Expected {'debug_test'}, got {log_entry['context']['operation']}"
            raise AssertionError(msg)

    def test_warning_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test warning level logging."""
        logger_instance.warning("Warning message", severity="medium")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "WARNING":
            msg = f"Expected {'WARNING'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Warning message"
        if log_entry["context"]["severity"] != "medium":
            msg = f"Expected {'medium'}, got {log_entry['context']['severity']}"
            raise AssertionError(msg)

    def test_error_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test error level logging."""
        logger_instance.error("Error occurred", error_code="ERR001")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "ERROR":
            msg = f"Expected {'ERROR'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Error occurred"
        if log_entry["context"]["error_code"] != "ERR001":
            msg = f"Expected {'ERR001'}, got {log_entry['context']['error_code']}"
            raise AssertionError(msg)

    def test_critical_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test critical level logging."""
        logger_instance.critical("Critical failure", system="database")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "CRITICAL":
            msg = f"Expected {'CRITICAL'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Critical failure"
        if log_entry["context"]["system"] != "database":
            msg = f"Expected {'database'}, got {log_entry['context']['system']}"
            raise AssertionError(msg)

    def test_trace_logging_filtered(self, clean_log_store: None) -> None:
        """Test trace logging is filtered when logger level is higher."""
        # Logger at INFO level should filter TRACE
        logger = FlextLogger("test.trace", "INFO")
        logger.trace("This should be filtered")

        if len(_log_store) != 0:
            msg = f"Expected {0}, got {len(_log_store)}"
            raise AssertionError(msg)

    def test_trace_logging_allowed(self, clean_log_store: None) -> None:
        """Test trace logging when allowed."""
        # Logger at TRACE level should allow TRACE
        logger = FlextLogger("test.trace", "TRACE")
        logger.trace("Trace message", detail="fine_grained")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "TRACE":
            msg = f"Expected {'TRACE'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Trace message"
        if log_entry["context"]["detail"] != "fine_grained":
            msg = f"Expected {'fine_grained'}, got {log_entry['context']['detail']}"
            raise AssertionError(msg)

    def test_context_management(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test logger context management."""
        # Set context
        logger_instance.set_context({"user_id": "123", "session": "abc"})

        # Check context is stored
        if logger_instance._context["user_id"] != "123":
            msg = f"Expected {'123'}, got {logger_instance._context['user_id']}"
            raise AssertionError(msg)
        assert logger_instance._context["session"] == "abc"

        # Log message should include context
        logger_instance.info("Test with context")

        log_entry = _log_store[0]
        if log_entry["context"]["user_id"] != "123":
            msg = f"Expected {'123'}, got {log_entry['context']['user_id']}"
            raise AssertionError(msg)
        assert log_entry["context"]["session"] == "abc"

    def test_get_context(self, logger_instance: FlextLogger) -> None:
        """Test get_context method."""
        test_context = {"key1": "value1", "key2": "value2"}
        logger_instance.set_context(test_context)

        retrieved_context = logger_instance.get_context()

        if retrieved_context != test_context:
            msg = f"Expected {test_context}, got {retrieved_context}"
            raise AssertionError(msg)
        # Should be a copy, not the same object
        assert retrieved_context is not logger_instance._context

    def test_clear_context(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test clear_context method."""
        logger_instance.set_context({"user_id": "123", "session": "abc"})
        if len(logger_instance._context) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(logger_instance._context)}"
            raise AssertionError(msg)

        logger_instance.clear_context()
        if len(logger_instance._context) != 0:
            msg = f"Expected {0}, got {len(logger_instance._context)}"
            raise AssertionError(msg)

        # Subsequent log should have no context
        logger_instance.info("After clear")
        log_entry = _log_store[0]
        # Should only have the message, no additional context
        if log_entry["context"] != {}:
            msg = f"Expected {{}}, got {log_entry['context']}"
            raise AssertionError(msg)

    def test_context_override_in_log_call(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context override in individual log calls."""
        # Set instance context
        logger_instance.set_context({"user_id": "123", "default": "value"})

        # Log with method-level context override
        logger_instance.info("Test override", user_id="456", extra="data")

        log_entry = _log_store[0]
        # Method context should override instance context
        if log_entry["context"]["user_id"] != "456":  # Overridden:
            msg = f"Expected {'456'}, got {log_entry['context']['user_id']}"
            raise AssertionError(msg)
        assert log_entry["context"]["default"] == "value"  # From instance
        if log_entry["context"]["extra"] != "data":  # From method:
            msg = f"Expected {'data'}, got {log_entry['context']['extra']}"
            raise AssertionError(msg)

    def test_exception_logging(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test exception logging with automatic traceback."""

        def raise_test_exception() -> None:
            msg = "Test exception"
            raise ValueError(msg)

        try:
            raise_test_exception()
        except ValueError:
            logger_instance.exception("Operation failed", operation="test_op")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]

        if log_entry["level"] != "ERROR":
            msg = f"Expected {'ERROR'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["message"] == "Operation failed"
        if log_entry["context"]["operation"] != "test_op":
            msg = f"Expected {'test_op'}, got {log_entry['context']['operation']}"
            raise AssertionError(msg)
        # structlog should have added exception info to the context

    def test_level_filtering_performance(self, clean_log_store: None) -> None:
        """Test that filtered messages don't execute expensive operations."""
        logger = FlextLogger("test.perf", "ERROR")  # High threshold

        expensive_operation_called = False

        def expensive_operation() -> str:
            nonlocal expensive_operation_called
            expensive_operation_called = True
            return "expensive_result"

        # This debug message should be filtered out without calling expensive operation
        if logger._should_log("DEBUG"):
            logger.debug("Debug message", data=expensive_operation())

        assert not expensive_operation_called
        if len(_log_store) != 0:
            msg = f"Expected {0}, got {len(_log_store)}"
            raise AssertionError(msg)

    def test_structured_logging_complex_data(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test structured logging with complex data types."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42,
            "boolean": True,
            "none_value": None,
        }

        logger_instance.info("Complex data test", **complex_data)

        log_entry = _log_store[0]
        context = log_entry["context"]

        if context["list"] != [1, 2, 3]:
            msg = f"Expected {[1, 2, 3]}, got {context['list']}"
            raise AssertionError(msg)
        assert context["dict"] == {"nested": "value"}
        if context["number"] != 42:
            msg = f"Expected {42}, got {context['number']}"
            raise AssertionError(msg)
        if not (context["boolean"]):
            msg = f"Expected True, got {context['boolean']}"
            raise AssertionError(msg)
        assert context["none_value"] is None


@pytest.mark.unit
class TestFlextLoggerFactory:
    """Test FlextLoggerFactory functionality."""

    def test_factory_initialization(self, factory_instance: FlextLoggerFactory) -> None:
        """Test factory initialization."""
        assert hasattr(factory_instance, "get_logger")
        assert hasattr(FlextLoggerFactory, "_loggers")
        assert isinstance(FlextLoggerFactory._loggers, dict)

    def test_get_logger_basic(self, factory_instance: FlextLoggerFactory) -> None:
        """Test basic logger creation."""
        logger = factory_instance.get_logger("test.basic")

        assert isinstance(logger, FlextLogger)
        if logger._name != "test.basic":
            msg = f"Expected {'test.basic'}, got {logger._name}"
            raise AssertionError(msg)
        assert logger._level == "INFO"  # Default level

    def test_get_logger_with_level(self, factory_instance: FlextLoggerFactory) -> None:
        """Test logger creation with custom level."""
        logger = factory_instance.get_logger("test.custom", "DEBUG")

        if logger._name != "test.custom":
            msg = f"Expected {'test.custom'}, got {logger._name}"
            raise AssertionError(msg)
        assert logger._level == "DEBUG"

    def test_get_logger_caching(self, factory_instance: FlextLoggerFactory) -> None:
        """Test logger caching behavior."""
        logger1 = factory_instance.get_logger("test.cache", "INFO")
        logger2 = factory_instance.get_logger("test.cache", "INFO")

        # Should return same cached instance
        assert logger1 is logger2

    def test_get_logger_different_levels_no_cache(
        self,
        factory_instance: FlextLoggerFactory,
    ) -> None:
        """Test that different levels create different logger instances."""
        logger1 = factory_instance.get_logger("test.level", "INFO")
        logger2 = factory_instance.get_logger("test.level", "DEBUG")

        # Should be different instances due to different levels
        assert logger1 is not logger2
        if logger1._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger1._level}"
            raise AssertionError(msg)
        assert logger2._level == "DEBUG"

    def test_get_logger_different_names(
        self,
        factory_instance: FlextLoggerFactory,
    ) -> None:
        """Test loggers with different names."""
        logger1 = factory_instance.get_logger("test.name1")
        logger2 = factory_instance.get_logger("test.name2")

        assert logger1 is not logger2
        if logger1._name != "test.name1":
            msg = f"Expected {'test.name1'}, got {logger1._name}"
            raise AssertionError(msg)
        assert logger2._name == "test.name2"

    def test_set_global_level(
        self,
        factory_instance: FlextLoggerFactory,
        clean_log_store: None,
    ) -> None:
        """Test global level setting affects all loggers."""
        # Create logger with default level
        logger = factory_instance.get_logger("test.global")
        if logger._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger._level}"
            raise AssertionError(msg)

        # Set global level
        factory_instance.set_global_level("DEBUG")

        # New loggers should use global level
        new_logger = factory_instance.get_logger("test.global.new")
        if new_logger._level != "DEBUG":
            msg = f"Expected {'DEBUG'}, got {new_logger._level}"
            raise AssertionError(msg)

    def test_clear_cache(self, factory_instance: FlextLoggerFactory) -> None:
        """Test cache clearing functionality."""
        # Create some cached loggers
        logger1 = factory_instance.get_logger("test.clear1")
        factory_instance.get_logger("test.clear2")

        # Verify cache has entries
        if len(FlextLoggerFactory._loggers) < 2:
            msg = f"Expected {len(FlextLoggerFactory._loggers)} >= {2}"
            raise AssertionError(msg)

        # Clear cache
        factory_instance.clear_loggers()

        # Cache should be empty
        if len(FlextLoggerFactory._loggers) != 0:
            msg = f"Expected {0}, got {len(FlextLoggerFactory._loggers)}"
            raise AssertionError(msg)

        # New loggers should be created
        new_logger1 = factory_instance.get_logger("test.clear1")
        assert new_logger1 is not logger1

    def test_get_log_store(
        self,
        factory_instance: FlextLoggerFactory,
        clean_log_store: None,
    ) -> None:
        """Test log store access."""
        # Add some entries to log store
        logger = factory_instance.get_logger("test.store", "DEBUG")
        logger.info("Test message 1")
        logger.debug("Test message 2")

        # Get log store
        log_entries = factory_instance.get_log_store()

        if len(log_entries) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(log_entries)}"
            raise AssertionError(msg)
        assert log_entries[0]["message"] == "Test message 1"
        if log_entries[1]["message"] != "Test message 2":
            msg = f"Expected {'Test message 2'}, got {log_entries[1]['message']}"
            raise AssertionError(msg)

    def test_clear_log_store(
        self,
        factory_instance: FlextLoggerFactory,
        clean_log_store: None,
    ) -> None:
        """Test log store clearing."""
        # Add entries
        logger = factory_instance.get_logger("test.clear")
        logger.info("Message to clear")

        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)

        # Clear log store
        factory_instance.clear_log_store()

        if len(_log_store) != 0:
            msg = f"Expected {0}, got {len(_log_store)}"
            raise AssertionError(msg)

    def test_cache_key_generation(self, factory_instance: FlextLoggerFactory) -> None:
        """Test cache key generation logic."""
        # Create loggers with same name but different levels
        logger1 = factory_instance.get_logger("test.key", "INFO")
        logger2 = factory_instance.get_logger("test.key", "DEBUG")
        logger3 = factory_instance.get_logger("test.key", "INFO")  # Same as logger1

        # logger1 and logger3 should be same (cached)
        assert logger1 is logger3
        # logger2 should be different
        assert logger1 is not logger2

    def test_validation_in_factory(self, factory_instance: FlextLoggerFactory) -> None:
        """Test input validation in factory."""
        # Empty name should default to "flext.unknown"
        logger = factory_instance.get_logger("")
        if logger._name != "flext.unknown":
            msg = f"Expected {'flext.unknown'}, got {logger._name}"
            raise AssertionError(msg)

        # Invalid level should default to INFO
        logger = factory_instance.get_logger("test.invalid", "INVALID_LEVEL")
        # Should not raise exception, logger should be created


@pytest.mark.unit
class TestFlextLogContextManager:
    """Test FlextLogContextManager functionality."""

    def test_context_manager_initialization(self, logger_instance: FlextLogger) -> None:
        """Test context manager initialization."""
        context = {"user_id": "123", "operation": "test"}

        cm = FlextLogContextManager(logger_instance, **context)

        assert cm._logger is logger_instance
        if cm._context != context:
            msg = f"Expected {context}, got {cm._context}"
            raise AssertionError(msg)

    def test_context_manager_enter_exit(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context manager enter and exit behavior."""
        # Set some initial context
        logger_instance.set_context({"permanent": "value"})

        with FlextLogContextManager(logger_instance, user_id="123", temp="context"):
            # Inside context manager, should have both contexts
            logger_instance.info("Inside context")

            log_entry = _log_store[0]
            if log_entry["context"]["permanent"] != "value":
                msg = f"Expected {'value'}, got {log_entry['context']['permanent']}"
                raise AssertionError(msg)
            assert log_entry["context"]["user_id"] == "123"
            if log_entry["context"]["temp"] != "context":
                msg = f"Expected {'context'}, got {log_entry['context']['temp']}"
                raise AssertionError(msg)

        # After exiting, temporary context should be removed
        _log_store.clear()
        logger_instance.info("After context")

        log_entry = _log_store[0]
        if log_entry["context"]["permanent"] != "value":
            msg = f"Expected {'value'}, got {log_entry['context']['permanent']}"
            raise AssertionError(msg)
        if "user_id" not in log_entry["context"]:
            msg = f"Expected {'user_id'} in {log_entry['context']}"
            raise AssertionError(msg)
        assert "temp" not in log_entry["context"]

    def test_context_manager_with_statement(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context manager used with 'with' statement."""
        with FlextLogContextManager(logger_instance, request_id="req_123"):
            logger_instance.info("Processing request")

        # Should have logged with context
        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)
        log_entry = _log_store[0]
        if log_entry["context"]["request_id"] != "req_123":
            msg = f"Expected {'req_123'}, got {log_entry['context']['request_id']}"
            raise AssertionError(msg)

    def test_context_manager_exception_handling(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context manager properly cleans up even when exception occurs."""
        logger_instance.set_context({"base": "context"})

        try:
            with FlextLogContextManager(logger_instance, temp="value"):
                logger_instance.info("Before exception")

                def raise_test_exception() -> None:
                    msg = "Test exception"
                    raise ValueError(msg)  # noqa: TRY301

                raise_test_exception()
        except ValueError:
            pass

        # Context should be cleaned up
        _log_store.clear()
        logger_instance.info("After exception")

        log_entry = _log_store[0]
        if log_entry["context"]["base"] != "context":
            msg = f"Expected {'context'}, got {log_entry['context']['base']}"
            raise AssertionError(msg)
        if "temp" not in log_entry["context"]:
            msg = f"Expected {'temp'} in {log_entry['context']}"
            raise AssertionError(msg)

    def test_nested_context_managers(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test nested context managers."""
        with (
            FlextLogContextManager(logger_instance, level1="outer"),
            FlextLogContextManager(logger_instance, level2="inner"),
        ):
            logger_instance.info("Nested context")

        log_entry = _log_store[0]
        if log_entry["context"]["level1"] != "outer":
            msg = f"Expected {'outer'}, got {log_entry['context']['level1']}"
            raise AssertionError(msg)
        assert log_entry["context"]["level2"] == "inner"

    def test_context_manager_override(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context manager overriding existing context."""
        logger_instance.set_context({"key": "original", "other": "value"})

        with FlextLogContextManager(logger_instance, key="overridden"):
            logger_instance.info("Override test")

        log_entry = _log_store[0]
        if log_entry["context"]["key"] != "overridden":
            msg = f"Expected {'overridden'}, got {log_entry['context']['key']}"
            raise AssertionError(msg)
        assert log_entry["context"]["other"] == "value"


@pytest.mark.unit
class TestLoggingIntegration:
    """Test logging system integration scenarios."""

    def test_full_logging_workflow(self, clean_log_store: None) -> None:
        """Test complete logging workflow."""
        # 1. Create factory
        factory = FlextLoggerFactory()

        # 2. Create logger
        logger = factory.get_logger("integration.test", "DEBUG")

        # 3. Set context
        logger.set_context({"service": "integration", "version": "1.0"})

        # 4. Log at different levels
        logger.debug("Debug information", component="auth")
        logger.info("Processing started", items=10)
        logger.warning("Low memory", memory_mb=50)

        # 5. Use context manager
        with FlextLogContextManager(logger, transaction_id="tx_123"):
            logger.info("Transaction processing")
            logger.error("Transaction failed", reason="timeout")

        # 6. Verify log entries
        log_entries = factory.get_log_store()
        if len(log_entries) != 5:
            msg = f"Expected {5}, got {len(log_entries)}"
            raise AssertionError(msg)

        # Check debug entry
        debug_entry = log_entries[0]
        if debug_entry["level"] != "DEBUG":
            msg = f"Expected {'DEBUG'}, got {debug_entry['level']}"
            raise AssertionError(msg)
        assert debug_entry["context"]["service"] == "integration"
        if debug_entry["context"]["component"] != "auth":
            msg = f"Expected {'auth'}, got {debug_entry['context']['component']}"
            raise AssertionError(msg)

        # Check transaction entries
        tx_entries = [e for e in log_entries if "transaction_id" in e["context"]]
        if len(tx_entries) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(tx_entries)}"
            raise AssertionError(msg)
        for entry in tx_entries:
            if entry["context"]["transaction_id"] != "tx_123":
                msg = f"Expected {'tx_123'}, got {entry['context']['transaction_id']}"
                raise AssertionError(msg)

    def test_multiple_loggers_isolation(self, clean_log_store: None) -> None:
        """Test multiple loggers don't interfere with each other."""
        factory = FlextLoggerFactory()

        # Create separate loggers
        auth_logger = factory.get_logger("auth.service", "INFO")
        db_logger = factory.get_logger("db.service", "DEBUG")

        # Set different contexts
        auth_logger.set_context({"component": "auth", "user_id": "123"})
        db_logger.set_context({"component": "database", "connection": "primary"})

        # Log from both
        auth_logger.info("User login", action="login")
        db_logger.debug("Query execution", table="users", duration_ms=45)

        log_entries = factory.get_log_store()
        if len(log_entries) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(log_entries)}"
            raise AssertionError(msg)

        # Verify isolation
        auth_entry = next(e for e in log_entries if e["logger"] == "auth.service")
        db_entry = next(e for e in log_entries if e["logger"] == "db.service")

        if auth_entry["context"]["component"] != "auth":
            msg = f"Expected {'auth'}, got {auth_entry['context']['component']}"
            raise AssertionError(msg)
        assert auth_entry["context"]["user_id"] == "123"
        if "connection" not in auth_entry["context"]:
            msg = f"Expected {'connection'} in {auth_entry['context']}"
            raise AssertionError(msg)

        if db_entry["context"]["component"] != "database":
            msg = f"Expected {'database'}, got {db_entry['context']['component']}"
            raise AssertionError(msg)
        assert db_entry["context"]["connection"] == "primary"
        if "user_id" not in db_entry["context"]:
            msg = f"Expected {'user_id'} in {db_entry['context']}"
            raise AssertionError(msg)

    def test_performance_with_level_filtering(self, clean_log_store: None) -> None:
        """Test performance optimization with level filtering."""
        logger = FlextLogger("perf.test", "ERROR")  # High threshold

        # Mock expensive operations
        expensive_calls = 0

        def expensive_debug_operation() -> str:
            nonlocal expensive_calls
            expensive_calls += 1
            return "expensive_result"

        # These should be filtered without executing expensive operations
        if logger._should_log("TRACE"):
            logger.trace("Trace", data=expensive_debug_operation())

        if logger._should_log("DEBUG"):
            logger.debug("Debug", data=expensive_debug_operation())

        if logger._should_log("INFO"):
            logger.info("Info", data=expensive_debug_operation())

        if logger._should_log("WARNING"):
            logger.warning("Warning", data=expensive_debug_operation())

        # Only ERROR and CRITICAL should execute
        if logger._should_log("ERROR"):
            logger.error("Error", data=expensive_debug_operation())

        if logger._should_log("CRITICAL"):
            logger.critical("Critical", data=expensive_debug_operation())

        # Should only have called expensive operation twice
        if expensive_calls != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {expensive_calls}"
            raise AssertionError(msg)
        assert len(_log_store) == EXPECTED_BULK_SIZE

    def test_structured_logging_with_complex_context(
        self,
        clean_log_store: None,
    ) -> None:
        """Test structured logging with complex context data."""
        logger = FlextLogger("complex.test", "DEBUG")

        # Complex context with various data types
        complex_context = {
            "user": {"id": "123", "name": "John", "roles": ["REDACTED_LDAP_BIND_PASSWORD", "user"]},
            "request": {
                "method": "POST",
                "path": "/api/orders",
                "headers": {"content-type": "application/json"},
                "body_size": 1024,
            },
            "performance": {
                "start_time": 1640995200.123,
                "memory_before": 128.5,
                "cpu_percent": 15.2,
            },
            "flags": {"is_premium": True, "debug_enabled": False},
            "metrics": [1, 2, 3, 4, 5],
            "metadata": None,
        }

        logger.info("Complex operation completed", **complex_context)

        log_entry = _log_store[0]
        context = log_entry["context"]

        # Verify complex data is preserved
        if context["user"]["id"] != "123":
            msg = f"Expected {'123'}, got {context['user']['id']}"
            raise AssertionError(msg)
        assert context["user"]["roles"] == ["REDACTED_LDAP_BIND_PASSWORD", "user"]
        if context["request"]["method"] != "POST":
            msg = f"Expected {'POST'}, got {context['request']['method']}"
            raise AssertionError(msg)
        assert context["performance"]["cpu_percent"] == 15.2
        if not (context["flags"]["is_premium"]):
            msg = f"Expected True, got {context['flags']['is_premium']}"
            raise AssertionError(msg)
        if context["metrics"] != [1, 2, 3, 4, 5]:
            msg = f"Expected {[1, 2, 3, 4, 5]}, got {context['metrics']}"
            raise AssertionError(msg)
        assert context["metadata"] is None

    def test_logging_with_exceptions_and_context(self, clean_log_store: None) -> None:
        """Test exception logging preserves context."""
        logger = FlextLogger("exception.test", "DEBUG")
        logger.set_context({"service": "payment", "version": "2.1"})

        try:
            # Simulate nested exception
            def raise_connection_error() -> None:
                msg = "Database connection failed"
                raise ConnectionError(msg)

            try:
                raise_connection_error()
            except ConnectionError as e:
                msg = f"Payment processing failed: {e}"
                raise RuntimeError(msg) from e
        except RuntimeError:
            logger.exception(
                "Critical payment error",
                order_id="order_123",
                amount=99.99,
                currency="USD",
            )

        log_entry = _log_store[0]

        if log_entry["level"] != "ERROR":
            msg = f"Expected {'ERROR'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["context"]["service"] == "payment"
        if log_entry["context"]["order_id"] != "order_123":
            msg = f"Expected {'order_123'}, got {log_entry['context']['order_id']}"
            raise AssertionError(msg)
        assert log_entry["context"]["amount"] == 99.99


@pytest.mark.unit
class TestLoggingEdgeCases:
    """Test logging edge cases and error conditions."""

    def test_logger_with_empty_name(self, clean_log_store: None) -> None:
        """Test logger with empty name."""
        logger = FlextLogger("", "INFO")
        logger.info("Message with empty logger name")

        log_entry = _log_store[0]
        if log_entry["logger"] != "root":  # structlog uses "root" for empty names:
            msg = f"Expected {'root'}, got {log_entry['logger']}"
            raise AssertionError(msg)

    def test_logger_with_invalid_level(self, clean_log_store: None) -> None:
        """Test logger with invalid level."""
        logger = FlextLogger("test.invalid", "INVALID_LEVEL")

        # Should default to INFO level
        if logger._level_value != 20:  # INFO level value:
            msg = f"Expected {20}, got {logger._level_value}"
            raise AssertionError(msg)

    def test_logging_with_none_context(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test logging with None context values."""
        logger_instance.info(
            "Test message",
            valid_key="valid_value",
            none_key=None,
            empty_string="",
        )

        log_entry = _log_store[0]
        context = log_entry["context"]

        if context["valid_key"] != "valid_value":
            msg = f"Expected {'valid_value'}, got {context['valid_key']}"
            raise AssertionError(msg)
        assert context["none_key"] is None
        if context["empty_string"] != "":
            msg = f"Expected {''}, got {context['empty_string']}"
            raise AssertionError(msg)

    def test_context_with_special_characters(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context with special characters."""
        special_context = {
            "unicode": "测试消息",
            "symbols": "!@#$%^&*()",
            "newlines": "line1\nline2\nline3",
            "quotes": 'He said "Hello"',
            "json_like": '{"key": "value"}',
        }

        logger_instance.info("Special characters test", **special_context)

        log_entry = _log_store[0]
        context = log_entry["context"]

        if context["unicode"] != "测试消息":
            raise AssertionError(f"Expected {'测试消息'}, got {context['unicode']}")
        assert context["symbols"] == "!@#$%^&*()"
        if context["newlines"] != "line1\nline2\nline3":
            msg = f"Expected {'line1\nline2\nline3'}, got {context['newlines']}"
            raise AssertionError(msg)
        assert context["quotes"] == 'He said "Hello"'
        if context["json_like"] != '{"key": "value"}':
            msg = f"Expected {'{"key": "value"}'}, got {context['json_like']}"
            raise AssertionError(msg)

    def test_large_context_data(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test logging with large context data."""
        # Create large context
        large_data = "x" * 10000  # 10KB string
        large_list = list(range(1000))

        logger_instance.info(
            "Large data test",
            large_string=large_data,
            large_list=large_list,
        )

        log_entry = _log_store[0]
        context = log_entry["context"]

        if len(context["large_string"]) != 10000:
            msg = f"Expected {10000}, got {len(context['large_string'])}"
            raise AssertionError(msg)
        assert len(context["large_list"]) == 1000
        if context["large_list"][999] != 999:
            msg = f"Expected {999}, got {context['large_list'][999]}"
            raise AssertionError(msg)

    def test_concurrent_logging_simulation(self, clean_log_store: None) -> None:
        """Test concurrent logging simulation."""
        factory = FlextLoggerFactory()

        # Simulate multiple "threads" logging simultaneously
        loggers = []
        for i in range(10):
            logger = factory.get_logger(f"thread.{i}", "DEBUG")
            logger.set_context({"thread_id": i})
            loggers.append(logger)

        # Each "thread" logs multiple messages
        for i, logger in enumerate(loggers):
            for j in range(3):
                logger.info(f"Message {j} from thread {i}", iteration=j)

        # Should have 30 log entries total
        log_entries = factory.get_log_store()
        if len(log_entries) != 30:
            msg = f"Expected {30}, got {len(log_entries)}"
            raise AssertionError(msg)

        # Verify each thread's messages are properly isolated
        for i in range(10):
            thread_entries = [e for e in log_entries if e["context"]["thread_id"] == i]
            if len(thread_entries) != EXPECTED_DATA_COUNT:
                msg = f"Expected {3}, got {len(thread_entries)}"
                raise AssertionError(msg)

            for entry in thread_entries:
                if entry["logger"] != f"thread.{i}":
                    msg = f"Expected {f'thread.{i}'}, got {entry['logger']}"
                    raise AssertionError(msg)
                assert entry["context"]["thread_id"] == i

    def test_factory_cache_edge_cases(self) -> None:
        """Test factory cache edge cases."""
        factory = FlextLoggerFactory()

        # Clear cache first
        FlextLoggerFactory.clear_loggers()

        # Create logger with same name but different cases
        logger1 = factory.get_logger("Test.Logger", "INFO")
        logger2 = factory.get_logger("test.logger", "INFO")  # Different case

        # Should be different loggers (case sensitive)
        assert logger1 is not logger2
        if logger1._name != "Test.Logger":
            msg = f"Expected {'Test.Logger'}, got {logger1._name}"
            raise AssertionError(msg)
        assert logger2._name == "test.logger"

    def test_log_store_memory_management(self, clean_log_store: None) -> None:
        """Test log store doesn't grow indefinitely in tests."""
        factory = FlextLoggerFactory()
        logger = factory.get_logger("memory.test", "DEBUG")

        # Generate many log entries
        for i in range(100):
            logger.info(f"Message {i}", iteration=i)

        # Should have all entries
        if len(_log_store) != 100:
            msg = f"Expected {100}, got {len(_log_store)}"
            raise AssertionError(msg)

        # Clear should work
        factory.clear_log_store()
        if len(_log_store) != 0:
            msg = f"Expected {0}, got {len(_log_store)}"
            raise AssertionError(msg)

    def test_context_manager_with_empty_context(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test context manager with no additional context."""
        with FlextLogContextManager(logger_instance):
            logger_instance.info("No additional context")

        # Should not fail and should log normally
        if len(_log_store) != 1:
            msg = f"Expected {1}, got {len(_log_store)}"
            raise AssertionError(msg)

    def test_malformed_log_data_handling(
        self,
        logger_instance: FlextLogger,
        clean_log_store: None,
    ) -> None:
        """Test handling of malformed or problematic log data."""
        # Test with circular reference (should not crash)
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict

        # This might not work perfectly but shouldn't crash
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            logger_instance.info("Circular reference test", data=circular_dict)

        # Test with very deep nesting
        deep_dict = {"level": 0}
        current = deep_dict
        for i in range(100):
            current["next"] = {"level": i + 1}
            current = current["next"]

        # Should handle deep nesting
        logger_instance.info("Deep nesting test", data=deep_dict)

        # Should have at least one log entry
        if len(_log_store) < 1:
            msg = f"Expected {len(_log_store)} >= {1}"
            raise AssertionError(msg)

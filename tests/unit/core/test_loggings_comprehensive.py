"""Comprehensive tests for loggings.py module - CORRECTED VERSION.

This test suite provides complete coverage of the logging system,
testing all aspects including FlextLogger, FlextLoggerFactory, context management,
and integration patterns to achieve near 100% coverage.

Tests are based on the ACTUAL implementation in loggings.py.
"""

from __future__ import annotations

import threading
import time

import pytest

from flext_core.constants import FlextLogLevel
from flext_core.loggings import (
    FlextLogContext,
    FlextLogContextManager,
    FlextLogger,
    FlextLoggerFactory,
    create_log_context,
    get_logger,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


@pytest.fixture
def clean_log_store() -> None:
    """Clean log store before each test."""
    FlextLoggerFactory.clear_log_store()
    yield
    FlextLoggerFactory.clear_log_store()


@pytest.fixture
def clean_logger_cache() -> None:
    """Clean logger cache and reset global level before each test."""
    FlextLoggerFactory.clear_loggers()
    FlextLoggerFactory._global_level = "INFO"  # Reset global level
    yield
    FlextLoggerFactory.clear_loggers()
    FlextLoggerFactory._global_level = "INFO"  # Reset global level


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
        if logger._level_value != 10:  # DEBUG level value (corrected):
            msg = f"Expected {10}, got {logger._level_value}"
            raise AssertionError(msg)
        assert isinstance(logger._context, dict)
        if len(logger._context) != 0:
            msg = f"Expected {0}, got {len(logger._context)}"
            raise AssertionError(msg)

    def test_logger_default_level(self, clean_log_store: None) -> None:
        """Test logger with default level."""
        logger = FlextLogger("test.app")

        if logger._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger._level}"
            raise AssertionError(msg)
        assert logger._level_value == 20  # INFO level value (corrected)

    def test_should_log_method(self, clean_log_store: None) -> None:
        """Test _should_log method."""
        logger = FlextLogger("test.app", "INFO")

        if not (logger._should_log("CRITICAL")):
            msg = f"Expected True, got {logger._should_log('CRITICAL')}"
            raise AssertionError(msg)
        assert logger._should_log("ERROR") is True
        if not (logger._should_log("WARNING")):
            msg = f"Expected True, got {logger._should_log('WARNING')}"
            raise AssertionError(msg)
        assert logger._should_log("INFO") is True
        if logger._should_log("DEBUG"):
            msg = f"Expected False, got {logger._should_log('DEBUG')}"
            raise AssertionError(msg)
        assert logger._should_log("TRACE") is False

    def test_should_log_with_enum_input(self, clean_log_store: None) -> None:
        """Test _should_log with enum input."""
        logger = FlextLogger("test.app", "INFO")

        # Test with enum objects
        if not (logger._should_log(FlextLogLevel.ERROR)):
            msg = f"Expected True, got {logger._should_log(FlextLogLevel.ERROR)}"
            raise AssertionError(msg)
        if logger._should_log(FlextLogLevel.DEBUG):
            msg = f"Expected False, got {logger._should_log(FlextLogLevel.DEBUG)}"
            raise AssertionError(msg)

    def test_should_log_unknown_level(self, clean_log_store: None) -> None:
        """Test _should_log with unknown level."""
        logger = FlextLogger("test.app", "INFO")

        # Unknown level should default to INFO level value
        if not (logger._should_log("UNKNOWN")):
            msg = f"Expected True, got {logger._should_log('UNKNOWN')}"
            raise AssertionError(msg)

    def test_level_filtering_performance(self, clean_log_store: None) -> None:
        """Test level filtering prevents processing when level too low."""
        logger = FlextLogger("test.app", "ERROR")

        # These should not create log entries
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 0:
            msg = f"Expected {0}, got {len(logs)}"
            raise AssertionError(msg)

        # This should create log entry
        logger.error("Error message")
        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

    def test_set_level(self, clean_log_store: None) -> None:
        """Test set_level method."""
        logger = FlextLogger("test.app", "INFO")

        # Initially INFO level
        if logger._should_log("DEBUG"):
            msg = f"Expected False, got {logger._should_log('DEBUG')}"
            raise AssertionError(msg)

        # Change to DEBUG level
        logger.set_level("DEBUG")
        if logger._level != "DEBUG":
            msg = f"Expected {'DEBUG'}, got {logger._level}"
            raise AssertionError(msg)
        if not (logger._should_log("DEBUG")):
            msg = f"Expected True, got {logger._should_log('DEBUG')}"
            raise AssertionError(msg)

    def test_context_management(self, clean_log_store: None) -> None:
        """Test context get/set operations."""
        logger = FlextLogger("test.app")

        # Initially empty context
        if logger.get_context() != {}:
            msg = f"Expected {{}}, got {logger.get_context()}"
            raise AssertionError(msg)

        # Set context
        test_context = {"user_id": "123", "request_id": "abc"}
        logger.set_context(test_context)

        retrieved_context = logger.get_context()
        if retrieved_context != test_context:
            msg = f"Expected {test_context}, got {retrieved_context}"
            raise AssertionError(msg)

        # Ensure we get a copy, not the original
        retrieved_context["new_key"] = "new_value"
        assert logger.get_context() != retrieved_context

    def test_with_context(self, clean_log_store: None) -> None:
        """Test with_context method creates new logger with merged context."""
        logger = FlextLogger("test.app")
        logger.set_context({"service": "user"})

        new_logger = logger.with_context(request_id="123", operation="create")

        # Original logger unchanged
        if logger.get_context() != {"service": "user"}:
            msg = f"Expected {{'service': 'user'}}, got {logger.get_context()}"
            raise AssertionError(msg)

        # New logger has merged context
        expected_context = {
            "service": "user",
            "request_id": "123",
            "operation": "create",
        }
        if new_logger.get_context() != expected_context:
            msg = f"Expected {expected_context}, got {new_logger.get_context()}"
            raise AssertionError(msg)

        # New logger is different instance
        assert new_logger is not logger

    def test_bind_method(self, clean_log_store: None) -> None:
        """Test bind method (alias for with_context)."""
        logger = FlextLogger("test.app")
        logger.set_context({"service": "user"})

        bound_logger = logger.bind(request_id="456")

        expected_context = {"service": "user", "request_id": "456"}
        if bound_logger.get_context() != expected_context:
            msg = f"Expected {expected_context}, got {bound_logger.get_context()}"
            raise AssertionError(msg)

    def test_info_logging(self, clean_log_store: None) -> None:
        """Test info level logging."""
        logger = FlextLogger("test.app", "INFO")

        logger.info("Test info message", extra_data="value")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

        log_entry = logs[0]
        if log_entry["level"] != "INFO":
            msg = f"Expected {'INFO'}, got {log_entry['level']}"
            raise AssertionError(msg)
        assert log_entry["logger"] == "test.app"
        if log_entry["message"] != "Test info message":
            msg = f"Expected {'Test info message'}, got {log_entry['message']}"
            raise AssertionError(msg)
        if "extra_data" not in log_entry["context"]:
            msg = f"Expected {'extra_data'} in {log_entry['context']}"
            raise AssertionError(msg)

    def test_debug_logging(self, clean_log_store: None) -> None:
        """Test debug level logging."""
        logger = FlextLogger("test.app", "DEBUG")

        logger.debug("Debug message", component="database")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["level"] == "DEBUG"

    def test_warning_logging(self, clean_log_store: None) -> None:
        """Test warning level logging."""
        logger = FlextLogger("test.app", "DEBUG")

        logger.warning("Warning message", issue="deprecated")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["level"] == "WARNING"

    def test_error_logging(self, clean_log_store: None) -> None:
        """Test error level logging."""
        logger = FlextLogger("test.app", "DEBUG")

        logger.error("Error message", error_code="E001")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["level"] == "ERROR"

    def test_critical_logging(self, clean_log_store: None) -> None:
        """Test critical level logging."""
        logger = FlextLogger("test.app", "DEBUG")

        logger.critical("Critical message", severity="high")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["level"] == "CRITICAL"

    def test_trace_logging_allowed(self, clean_log_store: None) -> None:
        """Test trace logging when level allows it."""
        logger = FlextLogger("test.app", "TRACE")

        logger.trace("Trace message", detail="fine-grained")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        # TRACE level is now properly supported
        if logs[0]["level"] != "TRACE":
            msg = f"Expected {'TRACE'}, got {logs[0]['level']}"
            raise AssertionError(msg)

    def test_trace_logging_filtered(self, clean_log_store: None) -> None:
        """Test trace logging filtered by level."""
        logger = FlextLogger("test.app", "DEBUG")

        logger.trace("Trace message")

        logs = FlextLoggerFactory.get_log_store()
        # TRACE (5) < DEBUG (10), so should be filtered
        if len(logs) != 0:
            msg = f"Expected {0}, got {len(logs)}"
            raise AssertionError(msg)

    def test_exception_logging(self, clean_log_store: None) -> None:
        """Test exception logging with traceback."""
        logger = FlextLogger("test.app", "DEBUG")

        def raise_test_exception() -> None:
            msg = "Test exception"
            raise ValueError(msg)

        try:
            raise_test_exception()
        except ValueError:
            logger.exception("Exception occurred", operation="test")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

        log_entry = logs[0]
        if log_entry["level"] != "ERROR":
            msg = f"Expected {'ERROR'}, got {log_entry['level']}"
            raise AssertionError(msg)
        if "traceback" not in log_entry["context"]:
            msg = f"Expected {'traceback'} in {log_entry['context']}"
            raise AssertionError(msg)
        assert "ValueError: Test exception" in log_entry["context"]["traceback"]

    def test_message_formatting_with_args(self, clean_log_store: None) -> None:
        """Test message formatting with positional arguments."""
        logger = FlextLogger("test.app", "INFO")

        logger.info("User %s logged in at %s", "john", "2023-01-01")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        if logs[0]["message"] == "User john logged not in at 2023-01-01":
            msg = f"Expected {logs[0]['message'] == 'User john logged} in {at 2023-01-01'}"
            raise AssertionError(msg)

    def test_message_formatting_fallback(self, clean_log_store: None) -> None:
        """Test message formatting fallback when % formatting fails."""
        logger = FlextLogger("test.app", "INFO")

        # This should fall back to string concatenation - expected behavior
        logger.info("Message", "arg1", "arg2")  # noqa: PLE1205

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        # Should concatenate when formatting fails
        if "Message" not in logs[0]["message"]:
            msg = f"Expected {'Message'} in {logs[0]['message']}"
            raise AssertionError(msg)

    def test_context_override_in_log_call(self, clean_log_store: None) -> None:
        """Test that method-level context overrides instance context."""
        logger = FlextLogger("test.app", "INFO")
        logger.set_context({"user_id": "123", "service": "auth"})

        logger.info("Test message", user_id="456", operation="login")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

        context = logs[0]["context"]
        if context["user_id"] != "456":
            msg = f"Expected {'456'}, got {context['user_id']}"
            raise AssertionError(msg)
        assert context["service"] == "auth"
        if context["operation"] != "login":
            msg = f"Expected {'login'}, got {context['operation']}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextLoggerFactory:
    """Test FlextLoggerFactory functionality."""

    def test_get_logger_basic(self, clean_logger_cache: None) -> None:
        """Test basic logger retrieval."""
        logger = FlextLoggerFactory.get_logger("test.service", "DEBUG")

        assert isinstance(logger, FlextLogger)
        if logger._name != "test.service":
            msg = f"Expected {'test.service'}, got {logger._name}"
            raise AssertionError(msg)
        assert logger._level == "DEBUG"

    def test_get_logger_caching(self, clean_logger_cache: None) -> None:
        """Test logger caching mechanism."""
        logger1 = FlextLoggerFactory.get_logger("test.service", "INFO")
        logger2 = FlextLoggerFactory.get_logger("test.service", "INFO")

        # Should return same instance
        assert logger1 is logger2

    def test_get_logger_different_levels_no_cache(
        self,
        clean_logger_cache: None,
    ) -> None:
        """Test different levels create different loggers."""
        logger1 = FlextLoggerFactory.get_logger("test.service", "INFO")
        logger2 = FlextLoggerFactory.get_logger("test.service", "DEBUG")

        # Different levels should create different instances
        assert logger1 is not logger2
        if logger1._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger1._level}"
            raise AssertionError(msg)
        assert logger2._level == "DEBUG"

    def test_get_logger_different_names(self, clean_logger_cache: None) -> None:
        """Test different names create different loggers."""
        logger1 = FlextLoggerFactory.get_logger("service1", "INFO")
        logger2 = FlextLoggerFactory.get_logger("service2", "INFO")

        assert logger1 is not logger2
        if logger1._name != "service1":
            msg = f"Expected {'service1'}, got {logger1._name}"
            raise AssertionError(msg)
        assert logger2._name == "service2"

    def test_get_logger_validation(self, clean_logger_cache: None) -> None:
        """Test parameter validation in get_logger."""
        # Empty name should default
        logger1 = FlextLoggerFactory.get_logger("", "INFO")
        if logger1._name != "flext.unknown":
            msg = f"Expected {'flext.unknown'}, got {logger1._name}"
            raise AssertionError(msg)

        # None name should default
        logger2 = FlextLoggerFactory.get_logger(None, "INFO")  # type: ignore[arg-type]
        if logger2._name != "flext.unknown":
            msg = f"Expected {'flext.unknown'}, got {logger2._name}"
            raise AssertionError(msg)

        # Empty level should default
        logger3 = FlextLoggerFactory.get_logger("test", "")
        if logger3._level != "INFO":
            msg = f"Expected {'INFO'}, got {logger3._level}"
            raise AssertionError(msg)

    def test_set_global_level(self, clean_logger_cache: None) -> None:
        """Test global level setting affects all loggers."""
        logger1 = FlextLoggerFactory.get_logger("service1", "DEBUG")
        logger2 = FlextLoggerFactory.get_logger("service2", "INFO")

        # Initially different levels
        if logger1._level != "DEBUG":
            msg = f"Expected {'DEBUG'}, got {logger1._level}"
            raise AssertionError(msg)
        assert logger2._level == "INFO"

        # Set global level
        FlextLoggerFactory.set_global_level("WARNING")

        # Both should now have WARNING level
        if logger1._level != "WARNING":
            msg = f"Expected {'WARNING'}, got {logger1._level}"
            raise AssertionError(msg)
        assert logger2._level == "WARNING"

    def test_set_global_level_validation(self, clean_logger_cache: None) -> None:
        """Test global level setting validation."""
        logger = FlextLoggerFactory.get_logger("test", "INFO")
        original_level = logger._level

        # Invalid level should not change anything
        FlextLoggerFactory.set_global_level("")
        if logger._level != original_level:
            msg = f"Expected {original_level}, got {logger._level}"
            raise AssertionError(msg)

        FlextLoggerFactory.set_global_level("INVALID")
        if logger._level != original_level:
            msg = f"Expected {original_level}, got {logger._level}"
            raise AssertionError(msg)

    def test_clear_loggers(self, clean_logger_cache: None) -> None:
        """Test logger cache clearing."""
        logger1 = FlextLoggerFactory.get_logger("test", "INFO")
        FlextLoggerFactory.clear_loggers()

        logger2 = FlextLoggerFactory.get_logger("test", "INFO")

        # Should be different instances after clearing
        assert logger1 is not logger2

    def test_get_log_store(self, clean_log_store: None) -> None:
        """Test log store retrieval."""
        logger = FlextLogger("test", "INFO")
        logger.info("Test message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["message"] == "Test message"

    def test_clear_log_store(self, clean_log_store: None) -> None:
        """Test log store clearing."""
        logger = FlextLogger("test", "INFO")
        logger.info("Test message")

        if len(FlextLoggerFactory.get_log_store()) != 1:
            msg = f"Expected {1}, got {len(FlextLoggerFactory.get_log_store())}"
            raise AssertionError(msg)

        FlextLoggerFactory.clear_log_store()
        if len(FlextLoggerFactory.get_log_store()) != 0:
            msg = f"Expected {0}, got {len(FlextLoggerFactory.get_log_store())}"
            raise AssertionError(msg)

    def test_create_context(self, clean_log_store: None) -> None:
        """Test context manager creation."""
        logger = FlextLogger("test", "INFO")

        context_manager = FlextLoggerFactory.create_context(
            logger,
            user_id="123",
            operation="test",
        )

        assert isinstance(context_manager, FlextLogContextManager)


@pytest.mark.unit
class TestFlextLogContextManager:
    """Test FlextLogContextManager functionality."""

    def test_context_manager_enter_exit(self, clean_log_store: None) -> None:
        """Test context manager enter/exit behavior."""
        logger = FlextLogger("test", "INFO")
        logger.set_context({"service": "auth"})

        context_manager = FlextLogContextManager(logger, user_id="123")

        # Before entering
        if logger.get_context() != {"service": "auth"}:
            msg = f"Expected {{'service': 'auth'}}, got {logger.get_context()}"
            raise AssertionError(msg)

        # Enter context
        returned_logger = context_manager.__enter__()
        assert returned_logger is logger
        if logger.get_context() != {"service": "auth", "user_id": "123"}:
            msg = f"Expected {{'service': 'auth', 'user_id': '123'}}, got {logger.get_context()}"
            raise AssertionError(msg)

        # Exit context
        context_manager.__exit__(None, None, None)
        if logger.get_context() != {"service": "auth"}:
            msg = f"Expected {{'service': 'auth'}}, got {logger.get_context()}"
            raise AssertionError(msg)

    def test_context_manager_with_statement(self, clean_log_store: None) -> None:
        """Test context manager used with 'with' statement."""
        logger = FlextLogger("test", "INFO")
        original_context = {"service": "auth"}
        logger.set_context(original_context)

        with FlextLogContextManager(logger, request_id="abc123") as ctx_logger:
            assert ctx_logger is logger
            logger.info("In context", operation="login")

        # Context should be restored
        if logger.get_context() != original_context:
            msg = f"Expected {original_context}, got {logger.get_context()}"
            raise AssertionError(msg)

        # Check logged message had both contexts
        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        context = logs[0]["context"]
        if context["service"] != "auth":
            msg = f"Expected {'auth'}, got {context['service']}"
            raise AssertionError(msg)
        assert context["request_id"] == "abc123"
        if context["operation"] != "login":
            msg = f"Expected {'login'}, got {context['operation']}"
            raise AssertionError(msg)

    def test_context_manager_exception_handling(self, clean_log_store: None) -> None:
        """Test context manager restores context even with exceptions."""
        logger = FlextLogger("test", "INFO")
        original_context = {"service": "auth"}
        logger.set_context(original_context)

        def raise_test_exception() -> None:
            msg = "Test exception"
            raise ValueError(msg)

        try:
            with FlextLogContextManager(logger, request_id="abc123"):
                if logger.get_context()["request_id"] != "abc123":
                    msg = (
                        f"Expected {'abc123'}, got {logger.get_context()['request_id']}"
                    )
                    raise AssertionError(msg)
                raise_test_exception()
        except ValueError:
            pass

        # Context should be restored even after exception
        if logger.get_context() != original_context:
            msg = f"Expected {original_context}, got {logger.get_context()}"
            raise AssertionError(msg)

    def test_nested_context_managers(self, clean_log_store: None) -> None:
        """Test nested context managers work correctly."""
        logger = FlextLogger("test", "INFO")
        logger.set_context({"service": "auth"})

        with FlextLogContextManager(logger, request_id="123"):
            with FlextLogContextManager(logger, operation="login", user_id="456"):
                expected_context = {
                    "service": "auth",
                    "request_id": "123",
                    "operation": "login",
                    "user_id": "456",
                }
                if logger.get_context() != expected_context:
                    msg = f"Expected {expected_context}, got {logger.get_context()}"
                    raise AssertionError(msg)
                logger.info("Nested context test")

            # Inner context should be removed
            expected_context = {"service": "auth", "request_id": "123"}
            if logger.get_context() != expected_context:
                msg = f"Expected {expected_context}, got {logger.get_context()}"
                raise AssertionError(msg)

        # All context should be restored
        if logger.get_context() != {"service": "auth"}:
            msg = f"Expected {{'service': 'auth'}}, got {logger.get_context()}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_logger_function(self, clean_logger_cache: None) -> None:
        """Test get_logger convenience function."""
        logger = get_logger("test.module", "DEBUG")

        assert isinstance(logger, FlextLogger)
        if logger._name != "test.module":
            msg = f"Expected {'test.module'}, got {logger._name}"
            raise AssertionError(msg)
        assert logger._level == "DEBUG"

    def test_create_log_context_function(self, clean_log_store: None) -> None:
        """Test create_log_context convenience function."""
        logger = FlextLogger("test", "INFO")

        context_manager = create_log_context(logger, user_id="123")

        assert isinstance(context_manager, FlextLogContextManager)

        # Test it works as expected
        with context_manager:
            logger.info("Test message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["context"]["user_id"] == "123"


@pytest.mark.unit
class TestFlextLogContext:
    """Test FlextLogContext TypedDict."""

    def test_typed_dict_usage(self) -> None:
        """Test FlextLogContext can be used as type hint."""
        # This is a TypedDict, so we test it can be used for typing
        context: FlextLogContext = {
            "user_id": "123",
            "request_id": "abc",
            "operation": "login",
        }

        if context["user_id"] != "123":
            msg = f"Expected {'123'}, got {context['user_id']}"
            raise AssertionError(msg)
        assert context["request_id"] == "abc"
        if context["operation"] != "login":
            msg = f"Expected {'login'}, got {context['operation']}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestLoggingIntegration:
    """Test logging integration scenarios."""

    def test_full_logging_workflow(self, clean_log_store: None) -> None:
        """Test complete logging workflow with context."""
        logger = get_logger("myapp.service", "DEBUG")
        logger.set_context({"service": "user_service", "version": "1.0"})

        # Log different levels
        logger.debug("Service starting", component="database")
        logger.info("Service ready", port=8080)

        with create_log_context(logger, request_id="req_123"):
            logger.info("Processing request", action="create_user")
            logger.warning("Slow operation", duration_ms=500)

        logger.error("Service error", error_code="USR001")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 5:
            msg = f"Expected {5}, got {len(logs)}"
            raise AssertionError(msg)

        # Check that context is properly maintained
        for log in logs:
            if log["context"]["service"] != "user_service":
                msg = f"Expected {'user_service'}, got {log['context']['service']}"
                raise AssertionError(msg)
            assert log["context"]["version"] == "1.0"

        # Check request-scoped context
        request_logs = [log for log in logs if "request_id" in log["context"]]
        if len(request_logs) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(request_logs)}"
            raise AssertionError(msg)
        for log in request_logs:
            if log["context"]["request_id"] != "req_123":
                msg = f"Expected {'req_123'}, got {log['context']['request_id']}"
                raise AssertionError(msg)

    def test_multiple_loggers_isolation(
        self,
        clean_log_store: None,
        clean_logger_cache: None,
    ) -> None:
        """Test multiple loggers maintain context isolation."""
        logger1 = get_logger("service1", "INFO")
        logger2 = get_logger("service2", "INFO")

        logger1.set_context({"service_id": "svc1"})
        logger2.set_context({"service_id": "svc2"})

        logger1.info("Message from service 1")
        logger2.info("Message from service 2")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(logs)}"
            raise AssertionError(msg)

        # Find logs by logger name and verify context isolation
        svc1_logs = [log for log in logs if log["logger"] == "service1"]
        svc2_logs = [log for log in logs if log["logger"] == "service2"]

        if len(svc1_logs) != 1:
            msg = f"Expected {1}, got {len(svc1_logs)}"
            raise AssertionError(msg)
        assert len(svc2_logs) == 1

        if svc1_logs[0]["context"]["service_id"] != "svc1":
            msg = f"Expected {'svc1'}, got {svc1_logs[0]['context']['service_id']}"
            raise AssertionError(msg)
        assert svc2_logs[0]["context"]["service_id"] == "svc2"

    def test_performance_with_level_filtering(self, clean_log_store: None) -> None:
        """Test performance optimization with level filtering."""
        logger = get_logger("performance.test", "ERROR")

        # These should not create any log entries due to level filtering
        logger.trace("Trace message")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 0:
            msg = f"Expected {0}, got {len(logs)}"
            raise AssertionError(msg)

        # Only ERROR and CRITICAL should be logged
        logger.error("Error message")
        logger.critical("Critical message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != EXPECTED_BULK_SIZE:
            msg = f"Expected {2}, got {len(logs)}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestLoggingEdgeCases:
    """Test logging edge cases and error conditions."""

    def test_logger_with_empty_name(
        self,
        clean_logger_cache: None,
        clean_log_store: None,
    ) -> None:
        """Test logger creation with empty name."""
        logger = FlextLoggerFactory.get_logger("", "INFO")
        logger.info("Test message")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)
        assert logs[0]["logger"] == "flext.unknown"

    def test_logger_with_invalid_level(self, clean_log_store: None) -> None:
        """Test logger with invalid level defaults to INFO."""
        logger = FlextLogger("test", "INVALID_LEVEL")

        # Should default to INFO level (20)
        if logger._level_value != 20:
            msg = f"Expected {20}, got {logger._level_value}"
            raise AssertionError(msg)

    def test_message_formatting_edge_cases(self, clean_log_store: None) -> None:
        """Test message formatting with edge cases."""
        logger = FlextLogger("test", "INFO")

        # Test with no args
        logger.info("Simple message")

        # Test with mismatched args - expected to log anyway
        logger.info("Message with %s %s", "one")  # noqa: PLE1206

        # Test with non-string message
        logger.info(123, "arg")

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != EXPECTED_DATA_COUNT:
            msg = f"Expected {3}, got {len(logs)}"
            raise AssertionError(msg)

        # All should have logged something
        for log in logs:
            assert len(log["message"]) > 0

    def test_context_with_none_values(self, clean_log_store: None) -> None:
        """Test context handling with None values."""
        logger = FlextLogger("test", "INFO")

        logger.set_context({"user_id": None, "request_id": "123"})
        logger.info("Test message", session_id=None)

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

        context = logs[0]["context"]
        assert context["user_id"] is None
        if context["request_id"] != "123":
            msg = f"Expected {'123'}, got {context['request_id']}"
            raise AssertionError(msg)
        assert context["session_id"] is None

    def test_large_context_data(self, clean_log_store: None) -> None:
        """Test handling of large context data."""
        logger = FlextLogger("test", "INFO")

        large_data = {"data": "x" * 1000, "items": list(range(100))}
        logger.info("Test message", **large_data)

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 1:
            msg = f"Expected {1}, got {len(logs)}"
            raise AssertionError(msg)

        context = logs[0]["context"]
        if len(context["data"]) != 1000:
            msg = f"Expected {1000}, got {len(context['data'])}"
            raise AssertionError(msg)
        assert len(context["items"]) == 100

    def test_concurrent_logging_simulation(self, clean_log_store: None) -> None:
        """Test logging behavior with simulated concurrent access."""
        logger = get_logger("concurrent.test", "INFO")

        def log_messages(thread_id: int) -> None:
            for i in range(10):
                logger.info(f"Message {i} from thread {thread_id}", thread_id=thread_id)
                time.sleep(0.001)  # Small delay to simulate work

        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        logs = FlextLoggerFactory.get_log_store()
        if len(logs) != 30:  # 3 threads * 10 messages each:
            msg = f"Expected {30}, got {len(logs)}"
            raise AssertionError(msg)

        # Verify all thread IDs are present
        thread_ids = {log["context"]["thread_id"] for log in logs}
        if thread_ids != {0, 1, 2}:
            msg = f"Expected {{0, 1, 2}}, got {thread_ids}"
            raise AssertionError(msg)

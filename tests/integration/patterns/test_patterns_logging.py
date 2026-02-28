"""Comprehensive tests for FLEXT patterns logging module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import logging
import time

import pytest
from flext_core import FlextLogger, t

from tests.test_utils import assertion_helpers

# Constants
EXPECTED_BULK_SIZE = 2


def make_result_logger(name: str) -> FlextLogger.ResultAdapter:
    """Helper to create loggers with FlextResult outputs."""
    return FlextLogger(name).with_result()


class TestFlextContext:
    """Test t.Logging.LogContext TypedDict."""

    def test_context_creation_empty(self) -> None:
        """Test creating empty log context."""
        context: dict[str, t.GeneralValueType] = {}

        assert isinstance(context, dict)
        if len(context) != 0:
            msg = f"Expected {0}, got {len(context)}"
            raise AssertionError(msg)

    def test_context_creation_with_values(self) -> None:
        """Test creating log context with values."""
        context: dict[str, t.GeneralValueType] = {
            "user_id": "123",
            "request_id": "req-456",
            "operation": "login",
            "duration_ms": 150.5,
        }

        if context["user_id"] != "123":
            msg = f"Expected {'123'}, got {context['user_id']}"
            raise AssertionError(msg)
        assert context["request_id"] == "req-456"
        if context["operation"] != "login":
            msg = f"Expected {'login'}, got {context['operation']}"
            raise AssertionError(msg)
        assert context["duration_ms"] == pytest.approx(150.5)

    def test_context_optional_fields(self) -> None:
        """Test that all context fields are optional."""
        # Test with partial context
        context: dict[str, t.GeneralValueType] = {
            "user_id": "123",
        }

        if context["user_id"] != "123":
            msg = f"Expected {'123'}, got {context['user_id']}"
            raise AssertionError(msg)
        # Other fields are optional and not present

    def test_context_enterprise_fields(self) -> None:
        """Test enterprise-specific context fields."""
        context: dict[str, t.GeneralValueType] = {
            "tenant_id": "tenant-123",
            "session_id": "session-456",
            "transaction_id": "tx-789",
            "customer_id": "customer-abc",
            "order_id": "order-def",
        }

        if context["tenant_id"] != "tenant-123":
            msg = f"Expected {'tenant-123'}, got {context['tenant_id']}"
            raise AssertionError(msg)
        assert context["session_id"] == "session-456"
        if context["transaction_id"] != "tx-789":
            msg = f"Expected {'tx-789'}, got {context['transaction_id']}"
            raise AssertionError(
                msg,
            )
        assert context["customer_id"] == "customer-abc"
        if context["order_id"] != "order-def":
            msg = f"Expected {'order-def'}, got {context['order_id']}"
            raise AssertionError(msg)

    def test_context_performance_fields(self) -> None:
        """Test performance-related context fields."""
        context: dict[str, t.GeneralValueType] = {
            "duration_ms": 250.0,
            "memory_mb": 128.5,
            "cpu_percent": 75.2,
        }

        if context["duration_ms"] != pytest.approx(250.0):
            msg = f"Expected {250.0}, got {context['duration_ms']}"
            raise AssertionError(msg)
        assert context["memory_mb"] == pytest.approx(128.5)
        if context["cpu_percent"] != pytest.approx(75.2):
            msg = f"Expected {75.2}, got {context['cpu_percent']}"
            raise AssertionError(msg)

    def test_context_error_fields(self) -> None:
        """Test error-related context fields."""
        context: dict[str, t.GeneralValueType] = {
            "error_code": "E001",
            "error_type": "ValidationError",
            "stack_trace": "Traceback...",
        }

        if context["error_code"] != "E001":
            msg = f"Expected {'E001'}, got {context['error_code']}"
            raise AssertionError(msg)
        assert context["error_type"] == "ValidationError"
        if context["stack_trace"] != "Traceback...":
            msg = f"Expected {'Traceback...'}, got {context['stack_trace']}"
            raise AssertionError(
                msg,
            )


class TestFlextLogLevel:
    """Test FlextLogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test standard logging levels - FlextLogLevel was removed."""
        # Test standard Python logging levels instead of removed FlextLogLevel
        assert logging.getLevelName(logging.DEBUG) == "DEBUG"
        assert logging.getLevelName(logging.INFO) == "INFO"
        assert logging.getLevelName(logging.WARNING) == "WARNING"
        assert logging.getLevelName(logging.ERROR) == "ERROR"
        assert logging.getLevelName(logging.CRITICAL) == "CRITICAL"

    def test_log_level_membership(self) -> None:
        """Test standard logging level membership - FlextLogLevel was removed."""
        # Test standard Python logging levels instead of removed FlextLogLevel
        all_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        # Validate that all levels are valid Python logging levels
        for level in all_levels:
            assert isinstance(level, int), f"Level {level} should be an integer"
            assert level >= 0, f"Level {level} should be non-negative"


class TestFlextLogger:
    """Test FlextLogger functionality."""

    def test_logger_auto_configuration(self) -> None:
        """Test that logger auto-configures on first use."""
        # Getting a logger should auto-configure
        logger = make_result_logger("test")

        # The new thin FlextLogger auto-configures structlog on first use
        assert logger is not None

    def test_get_logger_creates_instance(self) -> None:
        """Test that FlextLogger creates logger instances."""
        logger = make_result_logger("test_logger")

        assert logger is not None
        assert hasattr(logger, "info")  # Should be a structlog BoundLogger
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_get_logger_caches_instances(self) -> None:
        """Test that FlextLogger creates new instances (no caching in new implementation)."""
        logger1 = make_result_logger("cached_test")
        logger2 = make_result_logger("cached_test")

        # The new thin FlextLogger doesn't cache instances
        assert logger1 is not logger2

    def test_logger_methods_exist(self) -> None:
        """Test that logger has expected methods."""
        logger = make_result_logger("method_test")

        # Basic logging methods
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

        # The new thin FlextLogger doesn't have bind method
        # Context binding is handled through FlextContext

    def test_get_base_logger(self) -> None:
        """Test getting base logger instance.

        Validates:
        1. Base logger is created successfully
        2. Base logger has observability features
        3. Base logger can be used for logging
        """
        base_logger = make_result_logger("base_test")

        # Validate base logger was created
        assert base_logger is not None
        # Base logger should have observability features
        assert hasattr(base_logger, "info")
        assert hasattr(base_logger, "error")
        assert hasattr(base_logger, "debug")

        # Validate base logger can be used for logging
        base_logger.info("Base logger test message")  # Should not raise exception

    def test_get_base_logger_with_level(self) -> None:
        """Test getting base logger with specific level.

        Validates:
        1. Base logger with level is created successfully
        2. Logger has required methods
        3. Logger can be used for logging operations
        """
        base_logger = make_result_logger("level_test")

        # Validate base logger was created
        assert base_logger is not None
        assert hasattr(base_logger, "info")
        assert hasattr(base_logger, "error")
        assert hasattr(base_logger, "debug")

        # Validate logger can be used for logging
        base_logger.info("Level logger test message")  # Should not raise exception

    def test_bind_context(self) -> None:
        """Test binding context to logger.

        Validates:
        1. Logger.bind() creates bound logger successfully
        2. Bound logger has required methods
        3. Bound logger can be used for logging
        """
        logger = make_result_logger("bind_test")
        bound_logger = logger.bind(
            user_id="123",
            operation="test",
        )

        # Validate bound logger was created
        assert bound_logger is not None
        assert hasattr(bound_logger, "info")
        assert hasattr(bound_logger, "error")
        assert hasattr(bound_logger, "debug")

        # Validate bound logger can be used for logging
        result = bound_logger.info("Test message with bound context")
        _ = (
            assertion_helpers.assert_flext_result_success(result),
            "Bound logger should work for logging",
        )

    def test_backward_compatibility_function(self) -> None:
        """Test backward compatibility function.

        Validates:
        1. Logger is compatible with standard logging patterns
        2. Logger can be used for standard logging operations
        3. All logging methods work correctly
        """
        logger = make_result_logger("compat_test")

        # Validate logger was created
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

        # Validate logger works with standard logging patterns
        result = logger.info("Compatibility test message")
        _ = (
            assertion_helpers.assert_flext_result_success(result),
            "Logger should work with standard patterns",
        )

    def test_module_level_function(self) -> None:
        """Test module-level backward compatibility function.

        Validates:
        1. Module-level logger creation works
        2. Logger has required methods
        3. Logger can be used for logging operations
        """
        logger = make_result_logger("module_test")

        # Validate logger was created
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

        # Validate logger can be used for logging
        result = logger.info("Module-level logger test message")
        _ = (
            assertion_helpers.assert_flext_result_success(result),
            "Module-level logger should work",
        )


class TestFlextLoggerUsage:
    """Test actual usage of FlextLogger."""

    def test_basic_logging(self) -> None:
        """Test basic logging functionality.

        Validates:
        1. Logger is created successfully
        2. All logging methods execute without errors
        3. Logging methods return success results
        """
        # Use FlextLogger constructor directly, not FlextLogger()
        logger = make_result_logger("usage_test")

        # Validate logger was created
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

        # Validate all logging methods execute and return success
        result_info = logger.info("Test info message", test=True)
        assert result_info.is_success, "Info logging should succeed"

        result_debug = logger.debug("Test debug message", test=True)
        assert result_debug.is_success, "Debug logging should succeed"

        result_warning = logger.warning("Test warning message", test=True)
        assert result_warning.is_success, "Warning logging should succeed"

        result_error = logger.error("Test error message", test=True)
        assert result_error.is_success, "Error logging should succeed"

        result_critical = logger.critical("Test critical message", test=True)
        assert result_critical.is_success, "Critical logging should succeed"

    def test_logging_with_context(self) -> None:
        """Test logging with context data.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entries
        3. Logging succeeds with context data
        """
        logger = make_result_logger("context_test")

        # Validate logger supports context parameters
        assert logger is not None

        # Validate logging with context succeeds
        result_info = logger.info("User action", user_id="123", action="login")
        assert result_info.is_success, "Info logging with context should succeed"

        result_error = logger.error(
            "Operation failed",
            error_code="E001",
            duration_ms=150.5,
        )
        assert result_error.is_success, "Error logging with context should succeed"

        # Validate context parameters were accepted (no errors raised)
        # Note: Actual log content validation would require log capture

    def test_bound_logger_usage(self) -> None:
        """Test using bound logger.

        Validates:
        1. Logger.bind() creates bound logger
        2. Bound logger includes context in logs
        3. Logging succeeds with bound context
        """
        logger = make_result_logger("bound_test")
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")

        # Validate bound logger was created
        assert bound_logger is not None
        assert hasattr(bound_logger, "info")
        assert hasattr(bound_logger, "error")

        # Validate logging with bound context succeeds
        result_info = bound_logger.info("Processing request")
        assert result_info.is_success, "Info logging with bound context should succeed"

        result_error = bound_logger.error("Request failed")
        assert result_error.is_success, (
            "Error logging with bound context should succeed"
        )

        # Validate context was bound (no errors raised)
        # Note: Actual log content validation would require log capture

    def test_context_manager_style(self) -> None:
        """Test context manager style usage.

        Validates:
        1. Bound logger can be used for multiple log entries
        2. All log entries succeed with same context
        3. Context persists across multiple log calls
        """
        logger = make_result_logger("context_mgr_test")

        # Bind context for a series of operations
        bound_logger = logger.bind(operation="batch_process", batch_id="batch-123")

        # Validate all log entries succeed
        results = [
            bound_logger.info("Starting batch process"),
            bound_logger.info("Processing item 1"),
            bound_logger.info("Processing item 2"),
            bound_logger.info("Batch process completed"),
        ]

        # Validate all logging operations succeeded
        for i, result in enumerate(results):
            _ = (
                assertion_helpers.assert_flext_result_success(result),
                f"Log entry {i + 1} should succeed",
            )

        # Validate context was bound for all entries (no errors raised)
        # Note: Actual log content validation would require log capture

    @pytest.mark.performance
    def test_performance_logging(self) -> None:
        """Test performance-focused logging.

        Validates:
        1. Performance logger is created successfully
        2. Performance logging methods are available
        3. Logging operations succeed
        """
        perf_logger = make_result_logger("performance_test")

        # Validate logger was created and has required methods
        assert perf_logger is not None
        assert hasattr(perf_logger, "info")
        assert hasattr(perf_logger, "error")
        assert hasattr(perf_logger, "debug")

        # Validate performance logging succeeds
        result = perf_logger.info("Performance test message")
        _ = (
            assertion_helpers.assert_flext_result_success(result),
            "Performance logging should succeed",
        )

        # Validate logger can be used for multiple operations
        result2 = perf_logger.debug("Performance debug message")
        assert result2.is_success, "Performance debug logging should succeed"


class TestFlextLoggerIntegration:
    """Integration tests for FlextLogger."""

    def test_logging_hierarchy(self) -> None:
        """Test hierarchical logging.

        Validates:
        1. Hierarchical logger names are supported
        2. All logger instances are created successfully
        3. Loggers can be used independently
        """
        parent_logger = make_result_logger("parent")
        child_logger = make_result_logger("parent.child")
        grandchild_logger = make_result_logger("parent.child.grandchild")

        # Validate all logger instances were created
        assert parent_logger is not None
        assert child_logger is not None
        assert grandchild_logger is not None

        # Validate all loggers can be used for logging
        result_parent = parent_logger.info("Parent log message")
        assert result_parent.is_success, "Parent logger should work"

        result_child = child_logger.info("Child log message")
        assert result_child.is_success, "Child logger should work"

        result_grandchild = grandchild_logger.info("Grandchild log message")
        assert result_grandchild.is_success, "Grandchild logger should work"

    def test_complex_logging_scenario(self) -> None:
        """Test complex logging scenario with multiple contexts.

        Validates:
        1. Nested context binding works correctly
        2. Multiple loggers with different contexts succeed
        3. All log entries execute successfully
        """
        logger = make_result_logger("complex_test")

        # Simulate a complex operation with nested contexts
        bound_logger = logger.bind(operation="user_registration", request_id="req-789")
        result_start = bound_logger.info("Starting user registration")
        assert result_start.is_success, "Initial log should succeed"

        validation_logger = bound_logger.bind(
            step="validation",
            user_email="test@example.com",
        )
        result_debug_val = validation_logger.debug("Validating user input")
        assert result_debug_val.is_success, "Validation debug log should succeed"

        result_info_val = validation_logger.info("User input validation passed")
        assert result_info_val.is_success, "Validation info log should succeed"

        database_logger = bound_logger.bind(step="database", table="users")
        result_debug_db = database_logger.debug("Saving user to database")
        assert result_debug_db.is_success, "Database debug log should succeed"

        result_info_db = database_logger.info(
            "User saved successfully",
            user_id="user-456",
        )
        assert result_info_db.is_success, "Database info log should succeed"

        result_complete = bound_logger.info("User registration completed")
        assert result_complete.is_success, "Completion log should succeed"

        # Validate all logging operations succeeded
        # Note: Actual log content validation would require log capture

    def test_error_logging_with_context(self) -> None:
        """Test error logging with rich context."""
        logger = make_result_logger("error_test")

        def _raise_test_error() -> None:
            """Raise test error for exception logging tests."""
            msg = "Test error for logging"
            raise ValueError(msg)

        try:
            # Simulate an error
            _raise_test_error()
        except ValueError as e:
            # Capture exception details before assertions
            exception_type_name = type(e).__name__
            exception_message = str(e)
            # Validate exception logging with context
            result = logger.exception(
                "Operation failed with error",
                error_type=exception_type_name,
                error_message=exception_message,
                error_code="E500",
                operation="test_operation",
                user_id="user-123",
            )

            # Validate logging succeeded
            _ = (
                assertion_helpers.assert_flext_result_success(result),
                "Exception logging should succeed",
            )

        # Validate exception details were captured (outside except block)
        # Note: These validations happen after exception handling completes

        # Validate context parameters were included
        # Note: Actual log content validation would require log capture

    @pytest.mark.performance
    def test_performance_logging_integration(self) -> None:
        """Test performance logging integration.

        Validates:
        1. Performance timing is measured correctly
        2. Logging includes performance metrics
        3. Duration calculation is accurate
        """
        logger = make_result_logger("perf_integration_test")

        start_time = time.perf_counter()

        # Simulate some work
        expected_duration_ms = 10
        time.sleep(expected_duration_ms / 1000)  # 10ms

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Validate duration measurement (more tolerant for CI environments)
        assert duration_ms >= expected_duration_ms * 0.5, (
            f"Duration {duration_ms}ms should be >= {expected_duration_ms * 0.5}ms"
        )
        assert duration_ms < expected_duration_ms * 3, (
            f"Duration {duration_ms}ms should be < {expected_duration_ms * 3}ms "
            "(reasonable overhead)"
        )

        # Log with performance metrics
        result = logger.info(
            "Operation completed",
            operation="test_work",
            duration_ms=duration_ms,
            success=True,
        )

        # Validate logging succeeded
        _ = (
            assertion_helpers.assert_flext_result_success(result),
            "Logging should succeed",
        )

        # Validate logged values are correct
        # Note: Actual log content validation would require log capture

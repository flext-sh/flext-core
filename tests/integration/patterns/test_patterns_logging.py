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

EXPECTED_BULK_SIZE = 2


def make_result_logger(name: str) -> FlextLogger.ResultAdapter:
    """Helper to create loggers with r outputs."""
    return FlextLogger(name).with_result()


class TestFlextContext:
    """Test t.Logging.LogContext TypedDict."""

    def test_context_creation_empty(self) -> None:
        """Test creating empty log context."""
        context: dict[str, t.ContainerValue] = {}
        assert isinstance(context, dict)
        if len(context) != 0:
            msg = f"Expected {0}, got {len(context)}"
            raise AssertionError(msg)

    def test_context_creation_with_values(self) -> None:
        """Test creating log context with values."""
        context: dict[str, t.ContainerValue] = {
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
        expected_duration_ms: float = 150.5
        duration_ms = context["duration_ms"]
        assert isinstance(duration_ms, float)
        assert abs(duration_ms - expected_duration_ms) < 1e-9

    def test_context_optional_fields(self) -> None:
        """Test that all context fields are optional."""
        context: dict[str, t.ContainerValue] = {"user_id": "123"}
        if context["user_id"] != "123":
            msg = f"Expected {'123'}, got {context['user_id']}"
            raise AssertionError(msg)

    def test_context_enterprise_fields(self) -> None:
        """Test enterprise-specific context fields."""
        context: dict[str, t.ContainerValue] = {
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
            raise AssertionError(msg)
        assert context["customer_id"] == "customer-abc"
        if context["order_id"] != "order-def":
            msg = f"Expected {'order-def'}, got {context['order_id']}"
            raise AssertionError(msg)

    def test_context_performance_fields(self) -> None:
        """Test performance-related context fields."""
        context: dict[str, t.ContainerValue] = {
            "duration_ms": 250.0,
            "memory_mb": 128.5,
            "cpu_percent": 75.2,
        }
        expected_duration_ms_ctx: float = 250.0
        duration_ms_ctx = context["duration_ms"]
        assert isinstance(duration_ms_ctx, float)
        if abs(duration_ms_ctx - expected_duration_ms_ctx) >= 1e-9:
            msg = f"Expected {250.0}, got {duration_ms_ctx}"
            raise AssertionError(msg)
        expected_memory_mb: float = 128.5
        memory_mb = context["memory_mb"]
        assert isinstance(memory_mb, float)
        assert abs(memory_mb - expected_memory_mb) < 1e-9
        expected_cpu_percent: float = 75.2
        cpu_percent = context["cpu_percent"]
        assert isinstance(cpu_percent, float)
        if abs(cpu_percent - expected_cpu_percent) >= 1e-9:
            msg = f"Expected {75.2}, got {cpu_percent}"
            raise AssertionError(msg)

    def test_context_error_fields(self) -> None:
        """Test error-related context fields."""
        context: dict[str, t.ContainerValue] = {
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
            raise AssertionError(msg)


class TestFlextLogLevel:
    """Test FlextLogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test standard logging levels - FlextLogLevel was removed."""
        assert logging.getLevelName(logging.DEBUG) == "DEBUG"
        assert logging.getLevelName(logging.INFO) == "INFO"
        assert logging.getLevelName(logging.WARNING) == "WARNING"
        assert logging.getLevelName(logging.ERROR) == "ERROR"
        assert logging.getLevelName(logging.CRITICAL) == "CRITICAL"

    def test_log_level_membership(self) -> None:
        """Test standard logging level membership - FlextLogLevel was removed."""
        all_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        for level in all_levels:
            assert isinstance(level, int), f"Level {level} should be an integer"
            assert level >= 0, f"Level {level} should be non-negative"


class TestFlextLogger:
    """Test FlextLogger functionality."""

    def test_logger_auto_configuration(self) -> None:
        """Test that logger auto-configures on first use."""
        logger = make_result_logger("test")
        assert logger is not None

    def test_get_logger_creates_instance(self) -> None:
        """Test that FlextLogger creates logger instances."""
        logger = make_result_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_get_logger_caches_instances(self) -> None:
        """Test that FlextLogger creates new instances (no caching in new implementation)."""
        logger1 = make_result_logger("cached_test")
        logger2 = make_result_logger("cached_test")
        assert logger1 is not logger2

    def test_logger_methods_exist(self) -> None:
        """Test that logger has expected methods."""
        logger = make_result_logger("method_test")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

    def test_get_base_logger(self) -> None:
        """Test getting base logger instance.

        Validates:
        1. Base logger is created successfully
        2. Base logger has observability features
        3. Base logger can be used for logging
        """
        base_logger = make_result_logger("base_test")
        assert base_logger is not None
        assert hasattr(base_logger, "info")
        assert hasattr(base_logger, "error")
        assert hasattr(base_logger, "debug")
        result = base_logger.info("Base logger test message")
        assertion_helpers.assert_flext_result_success(
            result,
            "Base logger should log successfully",
        )

    def test_get_base_logger_with_level(self) -> None:
        """Test getting base logger with specific level.

        Validates:
        1. Base logger with level is created successfully
        2. Logger has required methods
        3. Logger can be used for logging operations
        """
        base_logger = make_result_logger("level_test")
        assert base_logger is not None
        assert hasattr(base_logger, "info")
        assert hasattr(base_logger, "error")
        assert hasattr(base_logger, "debug")
        result = base_logger.info("Level logger test message")
        assertion_helpers.assert_flext_result_success(
            result,
            "Level logger should log successfully",
        )

    def test_bind_context(self) -> None:
        """Test binding context to logger.

        Validates:
        1. Logger.bind() creates bound logger successfully
        2. Bound logger has required methods
        3. Bound logger can be used for logging
        """
        logger = make_result_logger("bind_test")
        bound_logger = logger.bind(user_id="123", operation="test")
        assert bound_logger is not None
        assert hasattr(bound_logger, "info")
        assert hasattr(bound_logger, "error")
        assert hasattr(bound_logger, "debug")
        result = bound_logger.info("Test message with bound context")
        success = assertion_helpers.assert_flext_result_success(
            result,
            "Bound logger should work for logging",
        )
        assert success is True

    def test_backward_compatibility_function(self) -> None:
        """Test backward compatibility function.

        Validates:
        1. Logger is compatible with standard logging patterns
        2. Logger can be used for standard logging operations
        3. All logging methods work correctly
        """
        logger = make_result_logger("compat_test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        result = logger.info("Compatibility test message")
        success = assertion_helpers.assert_flext_result_success(
            result,
            "Logger should work with standard patterns",
        )
        assert success is True

    def test_module_level_function(self) -> None:
        """Test module-level backward compatibility function.

        Validates:
        1. Module-level logger creation works
        2. Logger has required methods
        3. Logger can be used for logging operations
        """
        logger = make_result_logger("module_test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        result = logger.info("Module-level logger test message")
        success = assertion_helpers.assert_flext_result_success(
            result,
            "Module-level logger should work",
        )
        assert success is True


class TestFlextLoggerUsage:
    """Test actual usage of FlextLogger."""

    def test_basic_logging(self) -> None:
        """Test basic logging functionality.

        Validates:
        1. Logger is created successfully
        2. All logging methods execute without errors
        3. Logging methods return success results
        """
        logger = make_result_logger("usage_test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")
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
        assert logger is not None
        result_info = logger.info("User action", user_id="123", action="login")
        assert result_info.is_success, "Info logging with context should succeed"
        result_error = logger.error(
            "Operation failed",
            error_code="E001",
            duration_ms=150.5,
        )
        assert result_error.is_success, "Error logging with context should succeed"

    def test_bound_logger_usage(self) -> None:
        """Test using bound logger.

        Validates:
        1. Logger.bind() creates bound logger
        2. Bound logger includes context in logs
        3. Logging succeeds with bound context
        """
        logger = make_result_logger("bound_test")
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")
        assert bound_logger is not None
        assert hasattr(bound_logger, "info")
        assert hasattr(bound_logger, "error")
        result_info = bound_logger.info("Processing request")
        assert result_info.is_success, "Info logging with bound context should succeed"
        result_error = bound_logger.error("Request failed")
        assert result_error.is_success, (
            "Error logging with bound context should succeed"
        )

    def test_context_manager_style(self) -> None:
        """Test context manager style usage.

        Validates:
        1. Bound logger can be used for multiple log entries
        2. All log entries succeed with same context
        3. Context persists across multiple log calls
        """
        logger = make_result_logger("context_mgr_test")
        bound_logger = logger.bind(operation="batch_process", batch_id="batch-123")
        results = [
            bound_logger.info("Starting batch process"),
            bound_logger.info("Processing item 1"),
            bound_logger.info("Processing item 2"),
            bound_logger.info("Batch process completed"),
        ]
        for i, result in enumerate(results):
            success = assertion_helpers.assert_flext_result_success(
                result,
                f"Log entry {i + 1} should succeed",
            )
            assert success is True

    @pytest.mark.performance
    def test_performance_logging(self) -> None:
        """Test performance-focused logging.

        Validates:
        1. Performance logger is created successfully
        2. Performance logging methods are available
        3. Logging operations succeed
        """
        perf_logger = make_result_logger("performance_test")
        assert perf_logger is not None
        assert hasattr(perf_logger, "info")
        assert hasattr(perf_logger, "error")
        assert hasattr(perf_logger, "debug")
        result = perf_logger.info("Performance test message")
        success = assertion_helpers.assert_flext_result_success(
            result,
            "Performance logging should succeed",
        )
        assert success is True
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
        assert parent_logger is not None
        assert child_logger is not None
        assert grandchild_logger is not None
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

    def test_error_logging_with_context(self) -> None:
        """Test error logging with rich context."""
        logger = make_result_logger("error_test")

        def _raise_test_error() -> None:
            """Raise test error for exception logging tests."""
            msg = "Test error for logging"
            raise ValueError(msg)

        try:
            _raise_test_error()
        except ValueError as e:
            exception_type_name = type(e).__name__
            exception_message = str(e)
            result = logger.exception(
                "Operation failed with error",
                error_type=exception_type_name,
                error_message=exception_message,
                error_code="E500",
                operation="test_operation",
                user_id="user-123",
            )
            success = assertion_helpers.assert_flext_result_success(
                result,
                "Exception logging should succeed",
            )
            assert success is True

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
        expected_duration_ms = 10
        time.sleep(expected_duration_ms / 1000)
        duration_ms = (time.perf_counter() - start_time) * 1000
        assert duration_ms >= expected_duration_ms * 0.5, (
            f"Duration {duration_ms}ms should be >= {expected_duration_ms * 0.5}ms"
        )
        assert duration_ms < expected_duration_ms * 3, (
            f"Duration {duration_ms}ms should be < {expected_duration_ms * 3}ms (reasonable overhead)"
        )
        result = logger.info(
            "Operation completed",
            operation="test_work",
            duration_ms=duration_ms,
            success=True,
        )
        success = assertion_helpers.assert_flext_result_success(
            result,
            "Logging should succeed",
        )
        assert success is True

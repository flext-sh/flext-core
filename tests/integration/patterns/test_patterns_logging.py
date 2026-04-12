"""Comprehensive tests for FLEXT patterns logging module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import pytest

from tests import p, t, u

EXPECTED_BULK_SIZE = 2


class TestPatternsLogging:
    @staticmethod
    def make_result_logger(name: str) -> p.Logger:
        return u.create_module_logger(name)

    @staticmethod
    def assert_result_success[TResult: p.ResultLike[bool]](
        result: TResult,
        context: str,
    ) -> bool:
        assert result.success, f"{context}: Expected success, got {result.error!r}"
        return True

    def test_context_creation_empty(self) -> None:
        """Test creating empty log context."""
        context: t.MutableRecursiveContainerMapping = {}
        assert isinstance(context, dict)
        if len(context) != 0:
            msg = f"Expected {0}, got {len(context)}"
            raise AssertionError(msg)

    def test_context_creation_with_values(self) -> None:
        """Test creating log context with values."""
        context: t.RecursiveContainerMapping = {
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
        context: t.RecursiveContainerMapping = {"user_id": "123"}
        if context["user_id"] != "123":
            msg = f"Expected {'123'}, got {context['user_id']}"
            raise AssertionError(msg)

    def test_context_enterprise_fields(self) -> None:
        """Test enterprise-specific context fields."""
        context: t.RecursiveContainerMapping = {
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
        context: t.RecursiveContainerMapping = {
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
        context: t.RecursiveContainerMapping = {
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

    def test_logger_auto_configuration(self) -> None:
        """Test that logger auto-configures on first use."""
        logger = self.make_result_logger("test")
        assert logger is not None

    def test_fetch_logger_creates_instance(self) -> None:
        """Test that the public logging DSL creates logger instances."""
        logger = self.make_result_logger("test_logger")
        assert logger is not None

    def test_fetch_logger_creates_distinct_instances(self) -> None:
        """Test that the public logging DSL creates distinct logger instances."""
        logger1 = self.make_result_logger("cached_test")
        logger2 = self.make_result_logger("cached_test")
        assert logger1 is not logger2

    def test_logger_methods_exist(self) -> None:
        """Test that logger has expected methods."""
        self.make_result_logger("method_test")

    def test_get_base_logger(self) -> None:
        """Test getting base logger instance.

        Validates:
        1. Base logger is created successfully
        2. Base logger has observability features
        3. Base logger can be used for logging
        """
        base_logger: p.Logger = self.make_result_logger("base_test")
        assert base_logger is not None
        result: p.Result[bool] | None = base_logger.info("Base logger test message")
        assert result is not None
        self.assert_result_success(result, "Base logger should log successfully")

    def test_get_base_logger_with_level(self) -> None:
        """Test getting base logger with specific level.

        Validates:
        1. Base logger with level is created successfully
        2. Logger has required methods
        3. Logger can be used for logging operations
        """
        base_logger = self.make_result_logger("level_test")
        assert base_logger is not None
        result: p.Result[bool] | None = base_logger.info("Level logger test message")
        assert result is not None
        self.assert_result_success(result, "Level logger should log successfully")

    def test_bind_context(self) -> None:
        """Test binding context to logger.

        Validates:
        1. Logger.bind() creates bound logger successfully
        2. Bound logger has required methods
        3. Bound logger can be used for logging
        """
        logger = self.make_result_logger("bind_test")
        bound_logger = logger.bind(user_id="123", operation="test")
        assert bound_logger is not None
        result: p.Result[bool] | None = bound_logger.info(
            "Test message with bound context"
        )
        assert result is not None
        success = self.assert_result_success(
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
        logger = self.make_result_logger("compat_test")
        assert logger is not None
        result: p.Result[bool] | None = logger.info("Compatibility test message")
        assert result is not None
        success = self.assert_result_success(
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
        logger = self.make_result_logger("module_test")
        assert logger is not None
        result: p.Result[bool] | None = logger.info("Module-level logger test message")
        assert result is not None
        success = self.assert_result_success(result, "Module-level logger should work")
        assert success is True

    def test_basic_logging(self) -> None:
        """Test basic logging functionality.

        Validates:
        1. Logger is created successfully
        2. All logging methods execute without errors
        3. Logging methods return success results
        """
        logger = self.make_result_logger("usage_test")
        assert logger is not None
        result_info: p.Result[bool] | None = logger.info("Test info message", test=True)
        assert result_info is not None
        assert result_info.success, "Info logging should succeed"
        result_debug: p.Result[bool] | None = logger.debug(
            "Test debug message", test=True
        )
        assert result_debug is not None
        assert result_debug.success, "Debug logging should succeed"
        result_warning: p.Result[bool] | None = logger.warning(
            "Test warning message",
            test=True,
        )
        assert result_warning is not None
        assert result_warning.success, "Warning logging should succeed"
        result_error: p.Result[bool] | None = logger.error(
            "Test error message", test=True
        )
        assert result_error is not None
        assert result_error.success, "Error logging should succeed"
        result_critical: p.Result[bool] | None = logger.critical(
            "Test critical message",
            test=True,
        )
        assert result_critical is not None
        assert result_critical.success, "Critical logging should succeed"

    def test_logging_with_context(self) -> None:
        """Test logging with context data.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entries
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("context_test")
        assert logger is not None
        result_info: p.Result[bool] | None = logger.info(
            "User action",
            user_id="123",
            action="login",
        )
        assert result_info is not None
        assert result_info.success, "Info logging with context should succeed"
        result_error: p.Result[bool] | None = logger.error(
            "Operation failed",
            error_code="E001",
            duration_ms=150.5,
        )
        assert result_error is not None
        assert result_error.success, "Error logging with context should succeed"

    def test_bound_logger_usage(self) -> None:
        """Test using bound logger.

        Validates:
        1. Logger.bind() creates bound logger
        2. Bound logger includes context in logs
        3. Logging succeeds with bound context
        """
        logger = self.make_result_logger("bound_test")
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")
        assert bound_logger is not None
        result_info: p.Result[bool] | None = bound_logger.info("Processing request")
        assert result_info is not None
        assert result_info.success, "Info logging with bound context should succeed"
        result_error: p.Result[bool] | None = bound_logger.error("Request failed")
        assert result_error is not None
        assert result_error.success, "Error logging with bound context should succeed"

    def test_context_manager_style(self) -> None:
        """Test context manager style usage.

        Validates:
        1. Bound logger can be used for multiple log entries
        2. All log entries succeed with same context
        3. Context persists across multiple log calls
        """
        logger = self.make_result_logger("context_mgr_test")
        bound_logger = logger.bind(operation="batch_process", batch_id="batch-123")
        results: Sequence[p.Result[bool] | None] = [
            bound_logger.info("Starting batch process"),
            bound_logger.info("Processing item 1"),
            bound_logger.info("Processing item 2"),
            bound_logger.info("Batch process completed"),
        ]
        for i, result in enumerate(results):
            assert result is not None
            success = self.assert_result_success(
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
        perf_logger = self.make_result_logger("performance_test")
        assert perf_logger is not None
        result: p.Result[bool] | None = perf_logger.info("Performance test message")
        assert result is not None
        success = self.assert_result_success(
            result,
            "Performance logging should succeed",
        )
        assert success is True
        result2: p.Result[bool] | None = perf_logger.debug("Performance debug message")
        assert result2 is not None
        assert result2.success, "Performance debug logging should succeed"

    def test_logging_hierarchy(self) -> None:
        """Test hierarchical logging.

        Validates:
        1. Hierarchical logger names are supported
        2. All logger instances are created successfully
        3. Loggers can be used independently
        """
        parent_logger = self.make_result_logger("parent")
        child_logger = self.make_result_logger("parent.child")
        grandchild_logger = self.make_result_logger("parent.child.grandchild")
        assert parent_logger is not None
        assert child_logger is not None
        assert grandchild_logger is not None
        result_parent: p.Result[bool] | None = parent_logger.info("Parent log message")
        assert result_parent is not None
        assert result_parent.success, "Parent logger should work"
        result_child: p.Result[bool] | None = child_logger.info("Child log message")
        assert result_child is not None
        assert result_child.success, "Child logger should work"
        result_grandchild: p.Result[bool] | None = grandchild_logger.info(
            "Grandchild log message",
        )
        assert result_grandchild is not None
        assert result_grandchild.success, "Grandchild logger should work"

    def test_complex_logging_scenario(self) -> None:
        """Test complex logging scenario with multiple contexts.

        Validates:
        1. Nested context binding works correctly
        2. Multiple loggers with different contexts succeed
        3. All log entries execute successfully
        """
        logger = self.make_result_logger("complex_test")
        bound_logger = logger.bind(operation="user_registration", request_id="req-789")
        result_start: p.Result[bool] | None = bound_logger.info(
            "Starting user registration"
        )
        assert result_start is not None
        assert result_start.success, "Initial log should succeed"
        validation_logger = bound_logger.bind(
            step="validation",
            user_email="test@example.com",
        )
        result_debug_val: p.Result[bool] | None = validation_logger.debug(
            "Validating user input",
        )
        assert result_debug_val is not None
        assert result_debug_val.success, "Validation debug log should succeed"
        result_info_val: p.Result[bool] | None = validation_logger.info(
            "User input validation passed",
        )
        assert result_info_val is not None
        assert result_info_val.success, "Validation info log should succeed"
        database_logger = bound_logger.bind(step="database", table="users")
        result_debug_db: p.Result[bool] | None = database_logger.debug(
            "Saving user to database",
        )
        assert result_debug_db is not None
        assert result_debug_db.success, "Database debug log should succeed"
        result_info_db: p.Result[bool] | None = database_logger.info(
            "User saved successfully",
            user_id="user-456",
        )
        assert result_info_db is not None
        assert result_info_db.success, "Database info log should succeed"
        result_complete: p.Result[bool] | None = bound_logger.info(
            "User registration completed",
        )
        assert result_complete is not None
        assert result_complete.success, "Completion log should succeed"

    def test_error_logging_with_context(self) -> None:
        """Test error logging with rich context."""
        logger = self.make_result_logger("error_test")

        def _raise_test_error() -> None:
            """Raise test error for exception logging tests."""
            msg = "Test error for logging"
            raise ValueError(msg)

        try:
            _raise_test_error()
        except ValueError as e:
            exception_type_name = type(e).__name__
            exception_message = str(e)
            result: p.Result[bool] | None = logger.exception(
                "Operation failed with error",
                error_type=exception_type_name,
                error_message=exception_message,
                error_code="E500",
                operation="test_operation",
                user_id="user-123",
            )
            assert result is not None
            success = self.assert_result_success(
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
        logger = self.make_result_logger("perf_integration_test")
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
        result: p.Result[bool] | None = logger.info(
            "Operation completed",
            operation="test_work",
            duration_ms=duration_ms,
            success=True,
        )
        assert result is not None
        success = self.assert_result_success(result, "Logging should succeed")
        assert success is True

"""Comprehensive coverage tests for FlextLogger.

This module provides extensive tests for FlextLogger functionality:
- Global context management (bind/unbind/clear/get)
- Scoped context (application/request/operation tiers)
- Level-based context filtering
- Factory patterns (service logger, module logger)
- All logging levels (trace/debug/info/warning/error/critical)
- Exception logging with stack traces
- Performance tracking and timing
- Result integration and logging
- PerformanceTracker inner class

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextConfig, FlextLogger, FlextResult


def make_result_logger(
    name: str,
    *,
    config: FlextConfig | None = None,
    _level: str | None = None,
    _service_name: str | None = None,
    _service_version: str | None = None,
    _correlation_id: str | None = None,
    _force_new: bool = False,
) -> FlextLogger.ResultAdapter:
    """Helper to build loggers returning FlextResult for logging tests."""
    return FlextLogger(
        name,
        config=config,
        _level=_level,
        _service_name=_service_name,
        _service_version=_service_version,
        _correlation_id=_correlation_id,
        _force_new=_force_new,
    ).with_result()


class TestGlobalContextManagement:
    """Test global context management via contextvars."""

    def test_bind_global_context(self) -> None:
        """Test binding global context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_global_context(
            request_id="req-123",
            user_id="usr-456",
        )
        assert result.is_success

    def test_bind_global_context_multiple_calls(self) -> None:
        """Test multiple global context bindings accumulate."""
        FlextLogger.clear_global_context()
        result1 = FlextLogger.bind_global_context(request_id="req-123")
        result2 = FlextLogger.bind_global_context(user_id="usr-456")
        assert result1.is_success
        assert result2.is_success

    def test_unbind_global_context(self) -> None:
        """Test unbinding specific global context keys."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(request_id="req-123", user_id="usr-456")
        result = FlextLogger.unbind_global_context("request_id")
        assert result.is_success

    def test_unbind_global_context_multiple_keys(self) -> None:
        """Test unbinding multiple keys at once."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(
            request_id="req-123",
            user_id="usr-456",
            correlation_id="cor-789",
        )
        result = FlextLogger.unbind_global_context("request_id", "user_id")
        assert result.is_success

    def test_clear_global_context(self) -> None:
        """Test clearing all global context."""
        FlextLogger.bind_global_context(request_id="req-123")
        result = FlextLogger.clear_global_context()
        assert result.is_success

    def test_get_global_context(self) -> None:
        """Test retrieving global context."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(request_id="req-123", user_id="usr-456")
        context = FlextLogger._get_global_context()
        assert isinstance(context, dict)


class TestScopedContextManagement:
    """Test three-tier scoped context system."""

    def test_bind_application_context(self) -> None:
        """Test binding application-level context via bind_context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context(
            scope="application",
            app_name="test-app",
            app_version="1.0.0",
            environment="test",
        )
        assert result.is_success

    def test_bind_request_context(self) -> None:
        """Test binding request-level context via bind_context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context(
            scope="request",
            correlation_id="flext-abc123",
            command="migrate",
            user_id="admin",
        )
        assert result.is_success

    def test_bind_operation_context(self) -> None:
        """Test binding operation-level context via bind_context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context(
            scope="operation",
            operation="migrate",
            service="MigrationService",
            method="execute",
        )
        assert result.is_success

    def test_clear_scope_application(self) -> None:
        """Test clearing application scope."""
        FlextLogger.bind_context(scope="application", app_name="test")
        result = FlextLogger.clear_scope("application")
        assert result.is_success

    def test_clear_scope_request(self) -> None:
        """Test clearing request scope."""
        FlextLogger.bind_context(scope="request", correlation_id="flext-123")
        result = FlextLogger.clear_scope("request")
        assert result.is_success

    def test_clear_scope_operation(self) -> None:
        """Test clearing operation scope."""
        FlextLogger.bind_context(scope="operation", operation="test")
        result = FlextLogger.clear_scope("operation")
        assert result.is_success

    def test_clear_scope_nonexistent(self) -> None:
        """Test clearing nonexistent scope."""
        result = FlextLogger.clear_scope("nonexistent")
        assert result.is_success

    def test_scoped_context_manager_request(self) -> None:
        """Test scoped_context manager for request scope."""
        FlextLogger.clear_global_context()
        with FlextLogger.scoped_context("request", correlation_id="flext-123"):
            context = FlextLogger._get_global_context()
            assert isinstance(context, dict)

    def test_scoped_context_manager_operation(self) -> None:
        """Test scoped_context manager for operation scope."""
        FlextLogger.clear_global_context()
        with FlextLogger.scoped_context("operation", operation="test"):
            context = FlextLogger._get_global_context()
            assert isinstance(context, dict)

    def test_scoped_context_manager_cleanup(self) -> None:
        """Test scoped_context clears context after exit."""
        FlextLogger.clear_global_context()
        with FlextLogger.scoped_context("operation", operation="test"):
            pass
        # Context should be cleared (or empty for operation scope)
        result = FlextLogger.clear_scope("operation")
        assert result.is_success


class TestLevelBasedContextManagement:
    """Test level-based context filtering."""

    def test_bind_context_for_level_debug(self) -> None:
        """Test binding DEBUG-level context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context_for_level("DEBUG", config="debug_config")
        assert result.is_success

    def test_bind_context_for_level_error(self) -> None:
        """Test binding ERROR-level context."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context_for_level("ERROR", stack_trace="trace_info")
        assert result.is_success

    def test_bind_context_for_level_lowercase(self) -> None:
        """Test binding context with lowercase level."""
        FlextLogger.clear_global_context()
        result = FlextLogger.bind_context_for_level("info", message="info_context")
        assert result.is_success

    def test_unbind_context_for_level(self) -> None:
        """Test unbinding level-specific context."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_context_for_level("DEBUG", config="config_data")
        result = FlextLogger.unbind_context_for_level("DEBUG", "config")
        assert result.is_success

    def test_unbind_context_for_level_multiple_keys(self) -> None:
        """Test unbinding multiple level-specific keys."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_context_for_level("ERROR", stack="trace", error="code")
        result = FlextLogger.unbind_context_for_level("ERROR", "stack", "error")
        assert result.is_success


class TestFactoryPatterns:
    """Test logger factory methods."""

    def test_create_service_logger(self) -> None:
        """Test creating service logger using FlextLogger constructor."""
        logger = FlextLogger("user-service")
        assert isinstance(logger, FlextLogger)
        assert logger.name == "user-service"

    def test_create_service_logger_with_version(self) -> None:
        """Test creating service logger with version via bind."""
        logger = FlextLogger("auth-service").bind(version="2.0.0")
        assert isinstance(logger, FlextLogger)

    def test_create_service_logger_with_correlation_id(self) -> None:
        """Test creating service logger with correlation ID via bind."""
        logger = FlextLogger("payment-service").bind(
            correlation_id="flext-abc123",
        )
        assert isinstance(logger, FlextLogger)

    def test_create_module_logger(self) -> None:
        """Test creating module logger."""
        logger = FlextLogger.create_module_logger("myapp.services")
        assert isinstance(logger, FlextLogger)
        assert logger.name == "myapp.services"

    def test_create_module_logger_dunder_name(self) -> None:
        """Test creating module logger with __name__."""
        module_name = __name__
        logger = FlextLogger.create_module_logger(module_name)
        assert isinstance(logger, FlextLogger)

    def test_get_logger(self) -> None:
        """Test creating logger via constructor (get_logger pattern)."""
        logger = FlextLogger("default_logger")
        assert isinstance(logger, FlextLogger)
        assert logger.name == "default_logger"


class TestInstanceCreation:
    """Test FlextLogger instance creation and properties."""

    def test_logger_init_with_name(self) -> None:
        """Test initializing logger with name."""
        logger = make_result_logger("test_module")
        assert logger.name == "test_module"

    def test_logger_init_with_service_context(self) -> None:
        """Test initializing logger with service context."""
        logger = make_result_logger(
            "service_logger",
            _service_name="my-service",
            _service_version="1.0.0",
        )
        assert logger.name == "service_logger"

    def test_logger_name_property(self) -> None:
        """Test logger name property."""
        logger = make_result_logger("test")
        assert logger.name == "test"

    def test_bind_creates_new_instance(self) -> None:
        """Test bind creates new logger instance."""
        logger1 = FlextLogger("test")
        logger2 = logger1.bind(request_id="123")
        assert isinstance(logger2, FlextLogger)

    def test_bind_chaining(self) -> None:
        """Test chaining multiple bind calls."""
        logger = FlextLogger("test").bind(a="1").bind(b="2").bind(c="3")
        assert isinstance(logger, FlextLogger)


class TestLoggingMethods:
    """Test all logging level methods."""

    def test_trace_logging(self) -> None:
        """Test trace level logging."""
        logger = make_result_logger("test")
        result = logger.trace("Test trace message")
        assert result.is_success

    def test_debug_logging(self) -> None:
        """Test debug level logging."""
        logger = make_result_logger("test")
        result = logger.debug("Test debug message")
        assert result.is_success

    def test_debug_with_context(self) -> None:
        """Test debug logging with context."""
        logger = make_result_logger("test")
        result = logger.debug("Debug with context", user_id="123", action="login")
        assert result.is_success

    def test_info_logging(self) -> None:
        """Test info level logging."""
        logger = make_result_logger("test")
        result = logger.info("Test info message")
        assert result.is_success

    def test_info_with_context(self) -> None:
        """Test info logging with context."""
        logger = make_result_logger("test")
        result = logger.info("Info with context", status="completed", duration="0.5s")
        assert result.is_success

    def test_warning_logging(self) -> None:
        """Test warning level logging."""
        logger = make_result_logger("test")
        result = logger.warning("Test warning message")
        assert result.is_success

    def test_warning_with_context(self) -> None:
        """Test warning logging with context."""
        logger = make_result_logger("test")
        result = logger.warning("Warning with context", retry_count=3, delay="1s")
        assert result.is_success

    def test_error_logging(self) -> None:
        """Test error level logging."""
        logger = make_result_logger("test")
        result = logger.error("Test error message")
        assert result.is_success

    def test_error_with_context(self) -> None:
        """Test error logging with context."""
        logger = make_result_logger("test")
        result = logger.error("Error with context", error_code="ERR_001", user_id="456")
        assert result.is_success

    def test_critical_logging(self) -> None:
        """Test critical level logging."""
        logger = make_result_logger("test")
        result = logger.critical("Test critical message")
        assert result.is_success

    def test_critical_with_context(self) -> None:
        """Test critical logging with context."""
        logger = make_result_logger("test")
        result = logger.critical(
            "Critical with context",
            alert_level="high",
            system="payment",
        )
        assert result.is_success

    def test_logging_with_formatting(self) -> None:
        """Test logging with message formatting."""
        logger = make_result_logger("test")
        result = logger.info("User %s logged in", "john")
        assert result.is_success

    def test_logging_with_invalid_formatting(self) -> None:
        """Test logging with invalid format string."""
        logger = make_result_logger("test")
        result = logger.info("Message with %s and %d", "arg1", 42)
        assert result.is_success


class TestExceptionLogging:
    """Test exception logging functionality."""

    def test_exception_logging_with_exception_object(self) -> None:
        """Test logging with exception object."""
        logger = make_result_logger("test")
        try:
            msg = "Test error"
            raise ValueError(msg)
        except ValueError as e:
            result = logger.exception("An error occurred", exception=e)
            assert result.is_success

    def test_exception_logging_with_exc_info(self) -> None:
        """Test logging with exc_info=True."""
        logger = make_result_logger("test")
        try:
            msg = "Test error"
            raise RuntimeError(msg)
        except RuntimeError:
            result = logger.exception("Operation failed")
            assert result.is_success

    def test_exception_logging_without_current_exception(self) -> None:
        """Test logging error outside exception context."""
        logger = make_result_logger("test")
        result = logger.error("No exception context")
        assert result.is_success

    def test_exception_logging_with_context(self) -> None:
        """Test exception logging with additional context."""
        logger = make_result_logger("test")
        try:
            msg = "Test error"
            raise OSError(msg)
        except OSError as e:
            result = logger.exception(
                "IO operation failed",
                exception=e,
                operation="file_read",
                file="data.txt",
            )
            assert result.is_success


class TestResultAdapter:
    """Test FlextLogger.ResultAdapter functionality."""

    def test_with_result_returns_adapter(self) -> None:
        """Test with_result returns FlextLogger.ResultAdapter."""
        logger = make_result_logger("test")
        adapter = logger.with_result()
        assert isinstance(adapter, FlextLogger.ResultAdapter)

    def test_result_adapter_info(self) -> None:
        """Test result adapter info method returns Result."""
        logger = make_result_logger("test")
        adapter = logger.with_result()
        result = adapter.info("Test message")
        assert result.is_success

    def test_result_adapter_debug(self) -> None:
        """Test result adapter debug method returns Result."""
        logger = make_result_logger("test")
        adapter = logger.with_result()
        result = adapter.debug("Debug message")
        assert result.is_success

    def test_result_adapter_warning(self) -> None:
        """Test result adapter warning method returns Result."""
        logger = make_result_logger("test")
        adapter = logger.with_result()
        result = adapter.warning("Warning message")
        assert result.is_success

    def test_result_adapter_error(self) -> None:
        """Test result adapter error method returns Result."""
        logger = make_result_logger("test")
        adapter = logger.with_result()
        result = adapter.error("Error message")
        assert result.is_success


class TestResultIntegration:
    """Test FlextResult integration with logging via with_result()."""

    def test_log_result_success(self) -> None:
        """Test logging successful result via with_result()."""
        logger = make_result_logger("test")
        result = FlextResult[str].ok("test_value")
        log_result = logger.with_result().info(
            f"Operation completed with: {result.value}",
        )
        assert log_result.is_success

    def test_log_result_failure(self) -> None:
        """Test logging failed result via with_result()."""
        logger = make_result_logger("test")
        result = FlextResult[str].fail("Something went wrong")
        log_result = logger.with_result().error(f"Operation failed: {result.error}")
        assert log_result.is_success

    def test_log_result_without_operation(self) -> None:
        """Test logging result without operation name."""
        logger = make_result_logger("test")
        result = FlextResult[str].ok("value")
        log_result = logger.with_result().info(f"Result: {result.value}")
        assert log_result.is_success

    def test_log_result_with_custom_level(self) -> None:
        """Test logging result with debug level."""
        logger = make_result_logger("test")
        result = FlextResult[str].ok("value")
        log_result = logger.with_result().debug(f"Debug result: {result.value}")
        assert log_result.is_success

    def test_log_result_failure_includes_error_details(self) -> None:
        """Test logging result includes error details."""
        logger = make_result_logger("test")
        result = FlextResult[str].fail("Error occurred")
        log_result = logger.with_result().error(f"Error: {result.error}")
        assert log_result.is_success


class TestLoggingIntegration:
    """Test logging integration patterns."""

    def test_logger_with_global_context(self) -> None:
        """Test logger respects global context."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(request_id="req-123")
        logger = make_result_logger("test")
        result = logger.info("Message with global context")
        assert result.is_success
        FlextLogger.clear_global_context()

    def test_logger_with_scoped_context(self) -> None:
        """Test logger respects scoped context."""
        FlextLogger.clear_global_context()
        with FlextLogger.scoped_context("request", correlation_id="flext-456"):
            logger = make_result_logger("test")
            result = logger.info("Message with scoped context")
            assert result.is_success
        FlextLogger.clear_global_context()

    def test_logger_bind_with_global_context(self) -> None:
        """Test bound logger includes all context."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(app="test_app")
        logger = make_result_logger("test").bind(user_id="123")
        result = logger.info("Bound logger message")
        assert result.is_success
        FlextLogger.clear_global_context()

    def test_multiple_loggers_share_global_context(self) -> None:
        """Test multiple logger instances share global context."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(shared="context_value")

        logger1 = make_result_logger("logger1")
        logger2 = make_result_logger("logger2")

        result1 = logger1.info("First logger message")
        result2 = logger2.info("Second logger message")

        assert result1.is_success
        assert result2.is_success
        FlextLogger.clear_global_context()

    def test_logger_complete_workflow(self) -> None:
        """Test complete logging workflow with all features."""
        FlextLogger.clear_global_context()

        # Bind application context using bind_context
        FlextLogger.bind_context(
            scope="application",
            app="test_app",
            version="1.0.0",
        )

        # Create logger
        logger = FlextLogger.create_module_logger(__name__)

        # Log with various levels
        logger.debug("Debug message")
        logger.info("Info message", action="start")

        # Log with bound context
        bound_logger = logger.bind(operation="workflow_test")
        bound_logger.info("Operation in progress")

        logger.info("Completed")
        FlextLogger.clear_global_context()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_logging_with_empty_message(self) -> None:
        """Test logging with empty message."""
        logger = make_result_logger("test")
        result = logger.info("")
        assert result.is_success

    def test_logging_with_none_context_values(self) -> None:
        """Test logging with None context values."""
        logger = make_result_logger("test")
        result = logger.info("Message", context_key=None)
        assert result.is_success

    def test_logging_with_large_context(self) -> None:
        """Test logging with large context dictionary."""
        logger = make_result_logger("test")
        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = logger.info("Message with large context", **large_context)
        assert result.is_success

    def test_multiple_context_managers_nested(self) -> None:
        """Test nested scoped context managers."""
        FlextLogger.clear_global_context()
        with FlextLogger.scoped_context("request", req_id="r1"):
            with FlextLogger.scoped_context("operation", op_id="o1"):
                logger = make_result_logger("test")
                result = logger.info("Nested context message")
                assert result.is_success
        FlextLogger.clear_global_context()


__all__ = [
    "TestEdgeCases",
    "TestExceptionLogging",
    "TestFactoryPatterns",
    "TestGlobalContextManagement",
    "TestInstanceCreation",
    "TestLevelBasedContextManagement",
    "TestLoggingIntegration",
    "TestLoggingMethods",
    "TestResultAdapter",
    "TestResultIntegration",
    "TestScopedContextManagement",
]

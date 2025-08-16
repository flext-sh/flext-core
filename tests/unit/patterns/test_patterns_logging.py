"""Comprehensive tests for FLEXT patterns logging module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from flext_core import (
    FlextLogContext,
    FlextLogger,
    FlextLogLevel,
    flext_get_logger,
)

# Constants
EXPECTED_BULK_SIZE = 2


class TestFlextLogContext:
    """Test FlextLogContext TypedDict."""

    def test_context_creation_empty(self) -> None:
        """Test creating empty log context."""
        context: FlextLogContext = {}

        assert isinstance(context, dict)
        if len(context) != 0:
            raise AssertionError(f"Expected {0}, got {len(context)}")

    def test_context_creation_with_values(self) -> None:
        """Test creating log context with values."""
        context: FlextLogContext = {
            "user_id": "123",
            "request_id": "req-456",
            "operation": "login",
            "duration_ms": 150.5,
        }

        if context["user_id"] != "123":
            raise AssertionError(f"Expected {'123'}, got {context['user_id']}")
        assert context["request_id"] == "req-456"
        if context["operation"] != "login":
            raise AssertionError(f"Expected {'login'}, got {context['operation']}")
        assert context["duration_ms"] == 150.5

    def test_context_optional_fields(self) -> None:
        """Test that all context fields are optional."""
        # Test with partial context
        context: FlextLogContext = {
            "user_id": "123",
        }

        if context["user_id"] != "123":
            raise AssertionError(f"Expected {'123'}, got {context['user_id']}")
        # Other fields are optional and not present

    def test_context_enterprise_fields(self) -> None:
        """Test enterprise-specific context fields."""
        context: FlextLogContext = {
            "tenant_id": "tenant-123",
            "session_id": "session-456",
            "transaction_id": "tx-789",
            "customer_id": "customer-abc",
            "order_id": "order-def",
        }

        if context["tenant_id"] != "tenant-123":
            raise AssertionError(f"Expected {'tenant-123'}, got {context['tenant_id']}")
        assert context["session_id"] == "session-456"
        if context["transaction_id"] != "tx-789":
            raise AssertionError(
                f"Expected {'tx-789'}, got {context['transaction_id']}",
            )
        assert context["customer_id"] == "customer-abc"
        if context["order_id"] != "order-def":
            raise AssertionError(f"Expected {'order-def'}, got {context['order_id']}")

    def test_context_performance_fields(self) -> None:
        """Test performance-related context fields."""
        context: FlextLogContext = {
            "duration_ms": 250.0,
            "memory_mb": 128.5,
            "cpu_percent": 75.2,
        }

        if context["duration_ms"] != 250.0:
            raise AssertionError(f"Expected {250.0}, got {context['duration_ms']}")
        assert context["memory_mb"] == 128.5
        if context["cpu_percent"] != 75.2:
            raise AssertionError(f"Expected {75.2}, got {context['cpu_percent']}")

    def test_context_error_fields(self) -> None:
        """Test error-related context fields."""
        context: FlextLogContext = {
            "error_code": "E001",
            "error_type": "ValidationError",
            "stack_trace": "Traceback...",
        }

        if context["error_code"] != "E001":
            raise AssertionError(f"Expected {'E001'}, got {context['error_code']}")
        assert context["error_type"] == "ValidationError"
        if context["stack_trace"] != "Traceback...":
            raise AssertionError(
                f"Expected {'Traceback...'}, got {context['stack_trace']}",
            )


class TestFlextLogLevel:
    """Test FlextLogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test that log levels have correct string values."""
        if FlextLogLevel.TRACE != "TRACE":
            raise AssertionError(f"Expected {'TRACE'}, got {FlextLogLevel.TRACE}")
        assert FlextLogLevel.DEBUG == "DEBUG"
        if FlextLogLevel.INFO != "INFO":
            raise AssertionError(f"Expected {'INFO'}, got {FlextLogLevel.INFO}")
        assert FlextLogLevel.WARNING == "WARNING"
        if FlextLogLevel.ERROR != "ERROR":
            raise AssertionError(f"Expected {'ERROR'}, got {FlextLogLevel.ERROR}")
        assert FlextLogLevel.CRITICAL == "CRITICAL"

    def test_log_level_membership(self) -> None:
        """Test log level membership."""
        all_levels = [
            FlextLogLevel.TRACE,
            FlextLogLevel.DEBUG,
            FlextLogLevel.INFO,
            FlextLogLevel.WARNING,
            FlextLogLevel.ERROR,
            FlextLogLevel.CRITICAL,
        ]

        for level in all_levels:
            if level not in FlextLogLevel.__dict__.values():
                raise AssertionError(
                    f"Expected {level} in {FlextLogLevel.__dict__.values()}",
                )


class TestFlextLogger:
    """Test FlextLogger functionality."""

    def test_logger_auto_configuration(self) -> None:
        """Test that logger auto-configures on first use."""
        # Clear any existing configuration
        FlextLogger._configured = False

        # Getting a logger should auto-configure
        logger = FlextLogger.get_logger("test")

        if not (FlextLogger._configured):
            raise AssertionError(f"Expected True, got {FlextLogger._configured}")
        assert logger is not None

    def test_get_logger_creates_instance(self) -> None:
        """Test that get_logger creates logger instances."""
        logger = FlextLogger.get_logger("test_logger")

        assert logger is not None
        assert hasattr(logger, "info")  # Should be a structlog BoundLogger
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_get_logger_caches_instances(self) -> None:
        """Test that get_logger caches logger instances."""
        logger1 = FlextLogger.get_logger("cached_test")
        logger2 = FlextLogger.get_logger("cached_test")

        assert logger1 is logger2

    def test_logger_methods_exist(self) -> None:
        """Test that logger has expected methods."""
        logger = FlextLogger.get_logger("method_test")

        # Basic logging methods
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

        # Binding methods
        assert hasattr(logger, "bind")

    def test_configure_with_defaults(self) -> None:
        """Test configuring logger with default settings."""
        # Reset configuration
        FlextLogger._configured = False

        FlextLogger.configure()

        if not (FlextLogger._configured):
            raise AssertionError(f"Expected True, got {FlextLogger._configured}")

    def test_configure_with_custom_settings(self) -> None:
        """Test configuring logger with custom settings."""
        # Reset configuration
        FlextLogger._configured = False

        FlextLogger.configure(
            log_level=FlextLogLevel.DEBUG,
            json_output=True,
            add_timestamp=True,
            add_caller=False,
        )

        if not (FlextLogger._configured):
            raise AssertionError(f"Expected True, got {FlextLogger._configured}")

    def test_configure_idempotent(self) -> None:
        """Test that configure is idempotent."""
        # Configure once
        FlextLogger.configure()
        if not (FlextLogger._configured):
            raise AssertionError(f"Expected True, got {FlextLogger._configured}")

        # Configure again should not raise error
        FlextLogger.configure()
        if not (FlextLogger._configured):
            raise AssertionError(f"Expected True, got {FlextLogger._configured}")

    def test_get_base_logger(self) -> None:
        """Test getting base logger instance."""
        base_logger = FlextLogger.get_base_logger("base_test")

        assert base_logger is not None
        # Base logger should have observability features
        assert hasattr(base_logger, "info")

    def test_get_base_logger_with_level(self) -> None:
        """Test getting base logger with specific level."""
        base_logger = FlextLogger.get_base_logger("level_test", _level="DEBUG")

        assert base_logger is not None

    def test_bind_context(self) -> None:
        """Test binding context to logger."""
        bound_logger = FlextLogger.bind_context(
            user_id="123",
            operation="test",
        )

        assert bound_logger is not None
        assert hasattr(bound_logger, "info")

    def test_clear_context(self) -> None:
        """Test clearing context variables."""
        # Create a logger instance first
        logger = FlextLogger("test_clear_context")
        # This should not raise an error
        logger.clear_context()

    def test_with_performance_tracking(self) -> None:
        """Test getting logger with performance tracking."""
        perf_logger = FlextLogger.with_performance_tracking("perf_test")

        assert perf_logger is not None
        assert hasattr(perf_logger, "info")

    def test_backward_compatibility_function(self) -> None:
        """Test backward compatibility function."""
        logger = FlextLogger.flext_get_logger("compat_test")

        assert logger is not None
        assert hasattr(logger, "info")

    def test_module_level_function(self) -> None:
        """Test module-level backward compatibility function."""
        logger = flext_get_logger("module_test")

        assert logger is not None
        assert hasattr(logger, "info")


class TestFlextLoggerUsage:
    """Test actual usage of FlextLogger."""

    def test_basic_logging(self) -> None:
        """Test basic logging functionality."""
        # Use FlextLogger constructor directly, not get_logger()
        logger = FlextLogger("usage_test", "DEBUG")

        # These should not raise errors
        logger.info("Test info message", test=True)
        logger.debug("Test debug message", test=True)
        logger.warning("Test warning message", test=True)
        logger.error("Test error message", test=True)
        logger.critical("Test critical message", test=True)

    def test_logging_with_context(self) -> None:
        """Test logging with context data."""
        logger = FlextLogger("context_test", "DEBUG")

        # These should not raise errors
        logger.info("User action", user_id="123", action="login")
        logger.error("Operation failed", error_code="E001", duration_ms=150.5)

    def test_bound_logger_usage(self) -> None:
        """Test using bound logger."""
        logger = FlextLogger("bound_test", "DEBUG")
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")

        # Context should be automatically included in these logs
        bound_logger.info("Processing request")
        bound_logger.error("Request failed")

    def test_context_manager_style(self) -> None:
        """Test context manager style usage."""
        logger = FlextLogger("context_mgr_test", "DEBUG")

        # Bind context for a series of operations
        bound_logger = logger.bind(operation="batch_process", batch_id="batch-123")
        bound_logger.info("Starting batch process")
        bound_logger.info("Processing item 1")
        bound_logger.info("Processing item 2")
        bound_logger.info("Batch process completed")

    @pytest.mark.performance
    def test_performance_logging(self) -> None:
        """Test performance-focused logging."""
        perf_logger = FlextLogger.with_performance_tracking("performance_test")

        # Performance loggers should support similar operations
        # Type check to ensure we have a logger-like object
        assert hasattr(perf_logger, "info")
        perf_logger.info("Performance test message")

    def test_logger_factory_called(self) -> None:
        """Test that structlog.get_logger is called appropriately."""
        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Clear cache to force new logger creation
            FlextLogger._loggers.clear()

            logger = FlextLogger.get_logger("factory_test")

            mock_get_logger.assert_called_once_with("factory_test")
        if logger != mock_logger:
            raise AssertionError(f"Expected {mock_logger}, got {logger}")


class TestFlextLoggerIntegration:
    """Integration tests for FlextLogger."""

    def test_multiple_loggers(self) -> None:
        """Test creating and using multiple loggers."""
        logger1 = FlextLogger.get_logger("service1")
        logger2 = FlextLogger.get_logger("service2")
        logger3 = FlextLogger.get_logger("service3")

        # Should be different instances
        assert logger1 is not logger2
        assert logger2 is not logger3

        # But same name should return same instance
        logger1_again = FlextLogger.get_logger("service1")
        assert logger1 is logger1_again

    def test_logging_hierarchy(self) -> None:
        """Test hierarchical logging."""
        parent_logger = FlextLogger.get_logger("parent")
        child_logger = FlextLogger.get_logger("parent.child")
        grandchild_logger = FlextLogger.get_logger("parent.child.grandchild")

        # All should be valid logger instances
        assert parent_logger is not None
        assert child_logger is not None
        assert grandchild_logger is not None

    def test_complex_logging_scenario(self) -> None:
        """Test complex logging scenario with multiple contexts."""
        logger = FlextLogger("complex_test", "DEBUG")

        # Simulate a complex operation with nested contexts
        bound_logger = logger.bind(operation="user_registration", request_id="req-789")
        bound_logger.info("Starting user registration")

        validation_logger = bound_logger.bind(
            step="validation",
            user_email="test@example.com",
        )
        validation_logger.debug("Validating user input")
        validation_logger.info("User input validation passed")

        database_logger = bound_logger.bind(step="database", table="users")
        database_logger.debug("Saving user to database")
        database_logger.info("User saved successfully", user_id="user-456")

        bound_logger.info("User registration completed")

    def test_error_logging_with_context(self) -> None:
        """Test error logging with rich context."""
        logger = FlextLogger("error_test", "DEBUG")

        def _raise_test_error() -> None:
            """Raise test error for exception logging tests."""
            msg = "Test error for logging"
            raise ValueError(msg)

        try:
            # Simulate an error
            _raise_test_error()
        except ValueError as e:
            logger.exception(
                "Operation failed with error",
                error_type=type(e).__name__,
                error_message=str(e),
                error_code="E500",
                operation="test_operation",
                user_id="user-123",
            )

    @pytest.mark.performance
    def test_performance_logging_integration(self) -> None:
        """Test performance logging integration."""
        logger = FlextLogger("perf_integration_test", "DEBUG")

        start_time = time.time()

        # Simulate some work
        time.sleep(0.01)  # 10ms

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "Operation completed",
            operation="test_work",
            duration_ms=duration_ms,
            success=True,
        )

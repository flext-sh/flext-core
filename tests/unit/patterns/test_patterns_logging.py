"""Comprehensive tests for FLEXT patterns logging module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from flext_core.loggings import (
    FlextLogContext,
    FlextLogger,
    FlextLogLevel,
)


class TestFlextLogContext:
    """Test FlextLogContext TypedDict."""

    def test_context_creation_empty(self) -> None:
        """Test creating empty log context."""
        context: FlextLogContext = {}

        assert isinstance(context, dict)
        assert len(context) == 0

    def test_context_creation_with_values(self) -> None:
        """Test creating log context with values."""
        context: FlextLogContext = {
            "user_id": "123",
            "request_id": "req-456",
            "operation": "login",
            "duration_ms": 150.5,
        }

        assert context["user_id"] == "123"
        assert context["request_id"] == "req-456"
        assert context["operation"] == "login"
        assert context["duration_ms"] == 150.5

    def test_context_optional_fields(self) -> None:
        """Test that all context fields are optional."""
        # Test with partial context
        context: FlextLogContext = {
            "user_id": "123",
        }

        assert context["user_id"] == "123"
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

        assert context["tenant_id"] == "tenant-123"
        assert context["session_id"] == "session-456"
        assert context["transaction_id"] == "tx-789"
        assert context["customer_id"] == "customer-abc"
        assert context["order_id"] == "order-def"

    def test_context_performance_fields(self) -> None:
        """Test performance-related context fields."""
        context: FlextLogContext = {
            "duration_ms": 250.0,
            "memory_mb": 128.5,
            "cpu_percent": 75.2,
        }

        assert context["duration_ms"] == 250.0
        assert context["memory_mb"] == 128.5
        assert context["cpu_percent"] == 75.2

    def test_context_error_fields(self) -> None:
        """Test error-related context fields."""
        context: FlextLogContext = {
            "error_code": "E001",
            "error_type": "ValidationError",
            "stack_trace": "Traceback...",
        }

        assert context["error_code"] == "E001"
        assert context["error_type"] == "ValidationError"
        assert context["stack_trace"] == "Traceback..."


class TestFlextLogLevel:
    """Test FlextLogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test that log levels have correct string values."""
        assert FlextLogLevel.TRACE == "TRACE"
        assert FlextLogLevel.DEBUG == "DEBUG"
        assert FlextLogLevel.INFO == "INFO"
        assert FlextLogLevel.WARNING == "WARNING"
        assert FlextLogLevel.ERROR == "ERROR"
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
            assert level in FlextLogLevel.__dict__.values()


class TestFlextLogger:
    """Test FlextLogger functionality."""

    def test_logger_auto_configuration(self) -> None:
        """Test that logger auto-configures on first use."""
        # Clear any existing configuration
        FlextLogger._configured = False

        # Getting a logger should auto-configure
        logger = FlextLogger.get_logger("test")

        assert FlextLogger._configured is True
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

        assert FlextLogger._configured is True

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

        assert FlextLogger._configured is True

    def test_configure_idempotent(self) -> None:
        """Test that configure is idempotent."""
        # Configure once
        FlextLogger.configure()
        assert FlextLogger._configured is True

        # Configure again should not raise error
        FlextLogger.configure()
        assert FlextLogger._configured is True

    def test_get_base_logger(self) -> None:
        """Test getting base logger instance."""
        base_logger = FlextLogger.get_base_logger("base_test")

        assert base_logger is not None
        # Base logger should have observability features
        assert hasattr(base_logger, "info")

    def test_get_base_logger_with_level(self) -> None:
        """Test getting base logger with specific level."""
        base_logger = FlextLogger.get_base_logger("level_test", "DEBUG")

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
        # This should not raise an error
        FlextLogger.clear_context()

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
        from flext_core.loggings import flext_get_logger

        logger = flext_get_logger("module_test")

        assert logger is not None
        assert hasattr(logger, "info")


class TestFlextLoggerUsage:
    """Test actual usage of FlextLogger."""

    def test_basic_logging(self) -> None:
        """Test basic logging functionality."""
        logger = FlextLogger.get_logger("usage_test")

        # These should not raise errors
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.critical("Test critical message")

    def test_logging_with_context(self) -> None:
        """Test logging with context data."""
        logger = FlextLogger.get_logger("context_test")

        # These should not raise errors
        logger.info("User action", user_id="123", action="login")
        logger.error("Operation failed", error_code="E001", duration_ms=150.5)

    def test_bound_logger_usage(self) -> None:
        """Test using bound logger."""
        logger = FlextLogger.get_logger("bound_test")
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")

        # Context should be automatically included in these logs
        bound_logger.info("Processing request")
        bound_logger.error("Request failed")

    def test_context_manager_style(self) -> None:
        """Test context manager style usage."""
        logger = FlextLogger.get_logger("context_mgr_test")

        # Bind context for a series of operations
        with logger.bind(operation="batch_process", batch_id="batch-123"):
            logger.info("Starting batch process")
            logger.info("Processing item 1")
            logger.info("Processing item 2")
            logger.info("Batch process completed")

    def test_performance_logging(self) -> None:
        """Test performance-focused logging."""
        perf_logger = FlextLogger.with_performance_tracking("performance_test")

        # Performance loggers should support similar operations
        perf_logger.info("Performance test message")

    @patch("structlog.get_logger")
    def test_logger_factory_called(self, mock_get_logger: MagicMock) -> None:
        """Test that structlog.get_logger is called appropriately."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Clear cache to force new logger creation
        FlextLogger._loggers.clear()

        logger = FlextLogger.get_logger("factory_test")

        mock_get_logger.assert_called_once_with("factory_test")
        assert logger == mock_logger


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
        logger = FlextLogger.get_logger("complex_test")

        # Simulate a complex operation with nested contexts
        with logger.bind(operation="user_registration", request_id="req-789"):
            logger.info("Starting user registration")

            with logger.bind(step="validation", user_email="test@example.com"):
                logger.debug("Validating user input")
                logger.info("User input validation passed")

            with logger.bind(step="database", table="users"):
                logger.debug("Saving user to database")
                logger.info("User saved successfully", user_id="user-456")

            logger.info("User registration completed")

    def test_error_logging_with_context(self) -> None:
        """Test error logging with rich context."""
        logger = FlextLogger.get_logger("error_test")

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

    def test_performance_logging_integration(self) -> None:
        """Test performance logging integration."""
        import time

        logger = FlextLogger.get_logger("perf_integration_test")

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

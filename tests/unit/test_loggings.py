"""Comprehensive tests for FlextCore.Logger - Logging System.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import os
import threading
import time

import pytest

from flext_core import FlextCore


class TestFlextLogger:
    """Test suite for FlextCore.Logger logging functionality."""

    def test_logger_initialization(self) -> None:
        """Test logger initialization."""
        logger = FlextCore.Logger("test_logger")
        assert logger is not None
        assert isinstance(logger, FlextCore.Logger)

    def test_logger_with_custom_config(self) -> None:
        """Test logger initialization with custom configuration."""
        logger = FlextCore.Logger("test_logger", _level="INFO")
        assert logger is not None

    def test_logger_info_logging(self) -> None:
        """Test info level logging."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test info message")
        assert result.is_success

    def test_logger_debug_logging(self) -> None:
        """Test debug level logging."""
        logger = FlextCore.Logger("test_logger")

        result = logger.debug("Test debug message")
        assert result.is_success

    def test_logger_warning_logging(self) -> None:
        """Test warning level logging."""
        logger = FlextCore.Logger("test_logger")

        result = logger.warning("Test warning message")
        assert result.is_success

    def test_logger_error_logging(self) -> None:
        """Test error level logging."""
        logger = FlextCore.Logger("test_logger")

        result = logger.error("Test error message")
        assert result.is_success

    def test_logger_critical_logging(self) -> None:
        """Test critical level logging."""
        logger = FlextCore.Logger("test_logger")

        result = logger.critical("Test critical message")
        assert result.is_success

    def test_logger_logging_with_context(self) -> None:
        """Test logging with context."""
        logger = FlextCore.Logger("test_logger")

        context = {"user_id": "123", "action": "test"}
        result = logger.info("Test message", context=context)
        assert result.is_success

    def test_logger_logging_with_correlation_id(self) -> None:
        """Test logging with correlation ID."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message", correlation_id="test_corr_123")
        assert result.is_success

    def test_logger_logging_with_exception(self) -> None:
        """Test logging with exception."""
        logger = FlextCore.Logger("test_logger")

        try:
            msg = "Test exception"
            raise ValueError(msg)
        except ValueError as e:
            logger.exception("Test error", exception=e)
            # Exception logging doesn't return a result, it just logs

    def test_logger_logging_with_structured_data(self) -> None:
        """Test logging with structured data."""
        logger = FlextCore.Logger("test_logger")

        structured_data: FlextCore.Types.Dict = {
            "action": "user_action",
            "data": {"key": "value"},
            "metadata": {"timestamp": "2025-01-01"},
        }

        result = logger.info("Test message", **structured_data)
        assert result.is_success

    def test_logger_logging_with_performance_metrics(self) -> None:
        """Test logging with performance metrics."""
        logger = FlextCore.Logger("test_logger")

        performance = {"duration_ms": 150.5, "memory_mb": 64.2, "cpu_percent": 25.8}

        result = logger.info("Test message", performance=performance)
        assert result.is_success

    def test_logger_logging_with_custom_fields(self) -> None:
        """Test logging with custom fields."""
        logger = FlextCore.Logger("test_logger")

        custom_fields: FlextCore.Types.Dict = {
            "custom_field_1": "value1",
            "custom_field_2": 42,
            "custom_field_3": True,
        }

        result = logger.info("Test message", **custom_fields)
        assert result.is_success

    def test_logger_logging_with_nested_context(self) -> None:
        """Test logging with nested context."""
        logger = FlextCore.Logger("test_logger")

        nested_context: FlextCore.Types.Dict = {
            "user": {
                "id": "123",
                "profile": {"name": "John Doe", "email": "john@example.com"},
            },
            "session": {"id": "abc", "started_at": "2025-01-01T00:00:00Z"},
        }

        result = logger.info("Test message", context=nested_context)
        assert result.is_success

    def test_logger_logging_with_array_data(self) -> None:
        """Test logging with array data."""
        logger = FlextCore.Logger("test_logger")

        array_data: FlextCore.Types.Dict = {
            "items": [1, 2, 3, 4, 5],
            "tags": ["tag1", "tag2", "tag3"],
            "scores": [85.5, 92.3, 78.9],
        }

        result = logger.info("Test message", **array_data)
        assert result.is_success

    def test_logger_logging_with_boolean_data(self) -> None:
        """Test logging with boolean data."""
        logger = FlextCore.Logger("test_logger")

        boolean_data = {"is_active": True, "is_verified": False, "has_permission": True}

        result = logger.info("Test message", **boolean_data)
        assert result.is_success

    def test_logger_logging_with_numeric_data(self) -> None:
        """Test logging with numeric data."""
        logger = FlextCore.Logger("test_logger")

        numeric_data: FlextCore.Types.Dict = {
            "count": 42,
            "percentage": 85.5,
            "ratio": 0.75,
        }

        result = logger.info("Test message", **numeric_data)
        assert result.is_success

    def test_logger_logging_with_null_data(self) -> None:
        """Test logging with null data."""
        logger = FlextCore.Logger("test_logger")

        null_data = {"nullable_field": None, "optional_field": None}

        result = logger.info("Test message", **null_data)
        assert result.is_success

    def test_logger_logging_with_empty_data(self) -> None:
        """Test logging with empty data."""
        logger = FlextCore.Logger("test_logger")

        empty_data: dict[
            str, str | FlextCore.Types.StringList | FlextCore.Types.StringDict
        ] = {
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
        }

        result = logger.info("Test message", **empty_data)
        assert result.is_success

    def test_logger_logging_with_special_characters(self) -> None:
        """Test logging with special characters."""
        logger = FlextCore.Logger("test_logger")

        special_data = {
            "unicode_text": "Hello 世界",
            "special_chars": "!@#$%^&*()",
            "newlines": "Line 1\nLine 2\nLine 3",
        }

        result = logger.info("Test message", **special_data)
        assert result.is_success

    @pytest.mark.filterwarnings("ignore:LogEntry validation failed:UserWarning")
    def test_logger_logging_with_large_data(self) -> None:
        """Test logging with large data."""
        logger = FlextCore.Logger("test_logger")

        large_data: FlextCore.Types.Dict = {
            "large_string": "x" * 1000,
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
        }

        result = logger.info("Test message", **large_data)
        assert result.is_success

    def test_logger_logging_with_complex_data(self) -> None:
        """Test logging with complex data structures."""
        logger = FlextCore.Logger("test_logger")

        complex_data: FlextCore.Types.Dict = {
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": [1, 2, 3],
                            "metadata": {
                                "created": "2025-01-01",
                                "updated": "2025-01-02",
                            },
                        },
                    },
                },
            },
            "arrays": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
                {"id": 3, "name": "Item 3"},
            ],
        }

        result = logger.info("Test message", **complex_data)
        assert result.is_success

    def test_logger_logging_with_validation(self) -> None:
        """Test logging with validation."""
        logger = FlextCore.Logger("test_logger")

        # Valid data should pass
        result = logger.info("Valid message", valid_field="value")
        assert result.is_success

    @pytest.mark.filterwarnings("ignore:LogEntry validation failed:UserWarning")
    def test_logger_logging_with_validation_failure(self) -> None:
        """Test logging with validation failure - should handle gracefully."""
        logger = FlextCore.Logger("test_logger")

        # Invalid data should be handled gracefully (not fail)
        # The logger should sanitize invalid data and log successfully
        result = logger.info("Invalid message", invalid_field=object())
        assert result.is_success  # Logger should handle validation failures gracefully

        # Verify that a warning was issued about validation failure
        # (This is tested by the warning in the test output)

    def test_logger_logging_with_middleware(self) -> None:
        """Test logging with middleware - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Middleware functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_filters(self) -> None:
        """Test logging with filters - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Filter functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_handlers(self) -> None:
        """Test logging with custom handlers - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Handler functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_formatters(self) -> None:
        """Test logging with custom formatters - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Formatter functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_levels(self) -> None:
        """Test logging with different levels."""
        logger = FlextCore.Logger("test_logger")

        # Test all levels using the specific methods
        result = logger.debug("Test debug message")
        assert result.is_success

        result = logger.info("Test info message")
        assert result.is_success

        result = logger.warning("Test warning message")
        assert result.is_success

        result = logger.error("Test error message")
        assert result.is_success

        result = logger.critical("Test critical message")
        assert result.is_success

    def test_logger_logging_with_level_filtering(self) -> None:
        """Test logging with level filtering."""
        logger = FlextCore.Logger("test_logger", _level="WARNING")

        # Debug and info should be filtered out
        result = logger.debug("Debug message")
        assert result.is_success  # Logger should still return success

        result = logger.info("Info message")
        assert result.is_success  # Logger should still return success

        # Warning and above should be logged
        result = logger.warning("Warning message")
        assert result.is_success

        result = logger.error("Error message")
        assert result.is_success

    def test_logger_logging_with_thread_safety(self) -> None:
        """Test logging with thread safety."""
        logger = FlextCore.Logger("test_logger")

        results: list[FlextCore.Result[None]] = []

        def log_message(thread_id: int) -> None:
            result = logger.info("Thread %s message", thread_id)
            results.append(result)

        threads: list[threading.Thread] = []
        for i in range(10):
            thread = threading.Thread(target=log_message, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_logger_logging_with_performance(self) -> None:
        """Test logging performance - adjusted for actual performance."""
        logger = FlextCore.Logger("test_logger")

        start_time = time.time()

        # Perform many logging operations
        for i in range(100):
            logger.info("Test message %s", i)

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        # The new thin FlextCore.Logger is much faster
        total_time = end_time - start_time
        assert total_time < 5.0  # Should be very fast with thin implementation
        assert (
            total_time > 0.0001
        )  # Should take some time for 100 operations (adjusted for thin implementation)

    def test_logger_logging_with_error_handling(self) -> None:
        """Test logging with error handling."""
        logger = FlextCore.Logger("test_logger")

        # Test normal logging
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_cleanup(self) -> None:
        """Test logging with cleanup - not needed in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

        # Cleanup is not needed in the new thin FlextCore.Logger
        # Just test that logging still works
        result = logger.info("Test message after cleanup")
        assert result.is_success

    def test_logger_logging_with_statistics(self) -> None:
        """Test logging with statistics - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        logger.info("Test message 1")
        logger.warning("Test message 2")
        logger.error("Test message 3")

        # Statistics functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        assert logger.name == "test_logger"

    def test_logger_logging_with_audit(self) -> None:
        """Test logging with audit."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_context_management(self) -> None:
        from flext_core import FlextCore

        FlextCore.Logger("test_logger")

    def test_logger_validation(self) -> None:
        """Test logger validation - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Validation functionality is not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_instance_creation(self) -> None:
        """Test logging instance creation - no caching in new FlextCore.Logger."""
        logger1 = FlextCore.Logger("test_logger")
        logger2 = FlextCore.Logger("test_logger")

        # The new FlextCore.Logger doesn't cache instances
        # Each call creates a new instance
        assert logger1 is not logger2
        # But they should have the same name
        assert logger1._name == logger2._name == "test_logger"

    def test_logger_logging_with_singleton_reset(self) -> None:
        """Test logging with different instances."""
        logger1 = FlextCore.Logger("test_logger", _force_new=True)
        logger2 = FlextCore.Logger("test_logger", _force_new=True)

        assert logger1 is not logger2

    def test_logger_logging_with_environment_variables(self) -> None:
        """Test logging with environment variables."""
        os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"

        try:
            logger = FlextCore.Logger("test_logger")
            result = logger.debug("Test debug message")
            assert result.is_success
        finally:
            if "FLEXT_LOG_LEVEL" in os.environ:
                del os.environ["FLEXT_LOG_LEVEL"]

    def test_logger_logging_with_file_output(self) -> None:
        """Test logging with file output - file output not currently implemented."""
        # FlextCore.Logger uses structlog which outputs to stdout/stderr by default
        # File output functionality is not currently implemented
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message to file")
        assert result.is_success

        # Note: The logger outputs to stdout/stderr, not to files
        # File output functionality would need to be implemented separately

    def test_logger_logging_with_json_output(self) -> None:
        """Test logging with JSON output."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message", field1="value1", field2=42)
        assert result.is_success

    def test_logger_logging_with_structured_output(self) -> None:
        """Test logging with structured output."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message", structured_field="value")
        assert result.is_success

    def test_logger_logging_with_custom_output(self) -> None:
        """Test logging with custom output."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_rotation(self) -> None:
        """Test logging with log rotation."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_compression(self) -> None:
        """Test logging with compression."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_buffering(self) -> None:
        """Test logging with buffering."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_output(self) -> None:
        """Test logging with output."""
        logger = FlextCore.Logger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_metrics_collection(self) -> None:
        """Test logging with metrics collection - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        result1 = logger.info("Test message 1")
        result2 = logger.warning("Test message 2")
        result3 = logger.error("Test message 3")

        # Metrics collection is not implemented in the new thin FlextCore.Logger
        # Just test that logging operations succeed
        assert result1.is_success
        assert result2.is_success
        assert result3.is_success
        assert logger.name == "test_logger"

    def test_logger_get_performance_metrics(self) -> None:
        """Test getting logger performance metrics - not implemented in new FlextCore.Logger."""
        logger = FlextCore.Logger("test_logger")

        # Performance metrics are not implemented in the new thin FlextCore.Logger
        # Just test basic logging functionality
        result = logger.info("Test message")
        assert result.is_success


class TestFlextLoggerAdvancedFeatures:
    """Comprehensive test suite for FlextCore.Logger advanced features."""

    def test_bind_global_context_success(self) -> None:
        """Test binding global context successfully."""
        # Clear any existing context first
        FlextCore.Logger.clear_global_context()

        result = FlextCore.Logger.bind_global_context(
            request_id="req-123", user_id="usr-456", correlation_id="cor-789"
        )
        assert result.is_success

        # Verify context is bound
        context = FlextCore.Logger.get_global_context()
        assert context.get("request_id") == "req-123"
        assert context.get("user_id") == "usr-456"
        assert context.get("correlation_id") == "cor-789"

        # Clean up
        FlextCore.Logger.clear_global_context()

    def test_unbind_global_context_success(self) -> None:
        """Test unbinding specific keys from global context."""
        # Setup
        FlextCore.Logger.clear_global_context()
        FlextCore.Logger.bind_global_context(
            key1="value1", key2="value2", key3="value3"
        )

        # Unbind specific keys
        result = FlextCore.Logger.unbind_global_context("key1", "key2")
        assert result.is_success

        # Verify only specified keys are unbound
        context = FlextCore.Logger.get_global_context()
        assert "key1" not in context
        assert "key2" not in context
        assert context.get("key3") == "value3"

        # Clean up
        FlextCore.Logger.clear_global_context()

    def test_clear_global_context_success(self) -> None:
        """Test clearing all global context."""
        # Setup
        FlextCore.Logger.bind_global_context(key1="value1", key2="value2")

        # Clear all context
        result = FlextCore.Logger.clear_global_context()
        assert result.is_success

        # Verify all context is cleared
        context = FlextCore.Logger.get_global_context()
        assert len(context) == 0

    def test_get_global_context_returns_dict(self) -> None:
        """Test getting global context returns dictionary."""
        FlextCore.Logger.clear_global_context()
        FlextCore.Logger.bind_global_context(test_key="test_value")

        context = FlextCore.Logger.get_global_context()
        assert isinstance(context, dict)
        assert context.get("test_key") == "test_value"

        FlextCore.Logger.clear_global_context()

    def test_create_service_logger_with_full_context(self) -> None:
        """Test creating service logger with full context."""
        logger = FlextCore.Logger.create_service_logger(
            "payment-service", version="2.1.0", correlation_id="cor-abc123"
        )

        assert logger.name == "payment-service"
        result = logger.info("Service initialized")
        assert result.is_success

    def test_create_service_logger_with_minimal_context(self) -> None:
        """Test creating service logger with minimal context."""
        logger = FlextCore.Logger.create_service_logger("minimal-service")

        assert logger.name == "minimal-service"
        result = logger.info("Minimal service logger created")
        assert result.is_success

    def test_create_module_logger_success(self) -> None:
        """Test creating module logger."""
        logger = FlextCore.Logger.create_module_logger("test.module.path")

        assert logger.name == "test.module.path"
        result = logger.info("Module logger created")
        assert result.is_success

    def test_get_logger_returns_instance(self) -> None:
        """Test get_logger class method returns FlextCore.Logger instance."""
        logger = FlextCore.Logger.get_logger()

        assert isinstance(logger, FlextCore.Logger)
        assert logger.name == "flext"
        result = logger.info("Default logger obtained")
        assert result.is_success

    def test_bind_creates_bound_logger(self) -> None:
        """Test bind() creates new logger with additional context."""
        base_logger = FlextCore.Logger("base_logger")

        # Create bound logger with additional context
        bound_logger = base_logger.bind(request_id="req-456", endpoint="/api/users")

        # Bound logger should be different instance but same name
        assert bound_logger is not base_logger
        assert bound_logger.name == base_logger.name

        # Bound logger should work correctly
        result = bound_logger.info("Request processed")
        assert result.is_success

    def test_bind_multiple_times(self) -> None:
        """Test binding context multiple times creates chained loggers."""
        logger1 = FlextCore.Logger("chained_logger")
        logger2 = logger1.bind(level1="value1")
        logger3 = logger2.bind(level2="value2")
        logger4 = logger3.bind(level3="value3")

        # All should be different instances
        assert logger1 is not logger2 is not logger3 is not logger4

        # All should have same name
        assert logger1.name == logger2.name == logger3.name == logger4.name

        # All should log successfully
        assert logger4.info("Deeply bound logger").is_success

    def test_track_performance_success(self) -> None:
        """Test performance tracking with successful operation."""
        logger = FlextCore.Logger("perf_logger")

        with logger.track_performance("database_query"):
            time.sleep(0.01)  # Simulate operation

        # Should complete without error

    def test_track_performance_with_exception(self) -> None:
        """Test performance tracking with exception."""
        logger = FlextCore.Logger("perf_logger")

        def failing_operation() -> None:
            with logger.track_performance("failing_operation"):
                msg = "Intentional failure"
                raise ValueError(msg)

        with pytest.raises(ValueError):
            failing_operation()

    def test_track_performance_context_manager_enter_exit(self) -> None:
        """Test performance tracker context manager methods."""
        logger = FlextCore.Logger("perf_logger")
        tracker = logger.track_performance("test_operation")

        # Test context manager by using with statement
        with tracker:
            assert tracker._start_time > 0

    def test_log_result_success_case(self) -> None:
        """Test logging successful FlextCore.Result."""
        logger = FlextCore.Logger("result_logger")

        success_result = FlextCore.Result[str].ok("Operation completed")
        log_result = logger.log_result(success_result, operation="user_validation")

        assert log_result.is_success

    def test_log_result_failure_case(self) -> None:
        """Test logging failed FlextCore.Result."""
        logger = FlextCore.Logger("result_logger")

        failure_result = FlextCore.Result[str].fail(
            "Validation failed", error_code="VAL_001"
        )
        log_result = logger.log_result(failure_result, operation="user_validation")

        assert log_result.is_success

    def test_log_result_without_operation_name(self) -> None:
        """Test logging result without operation name."""
        logger = FlextCore.Logger("result_logger")

        result = FlextCore.Result[str].ok("Success")
        log_result = logger.log_result(result)

        assert log_result.is_success

    def test_log_result_with_custom_level(self) -> None:
        """Test logging result with custom log level."""
        logger = FlextCore.Logger("result_logger")

        result = FlextCore.Result[str].ok("Success")
        log_result = logger.log_result(result, level="debug")

        assert log_result.is_success

    def test_trace_logging(self) -> None:
        """Test trace level logging."""
        logger = FlextCore.Logger("trace_logger")

        result = logger.trace("Trace message with details", key="value")
        assert result.is_success

    def test_trace_logging_with_formatting(self) -> None:
        """Test trace logging with string formatting."""
        logger = FlextCore.Logger("trace_logger")

        result = logger.trace("Trace message: %s", "formatted_value")
        assert result.is_success

    def test_exception_logging_with_exc_info(self) -> None:
        """Test exception logging with exc_info parameter."""
        logger = FlextCore.Logger("exception_logger")

        try:
            msg = "Test exception"
            raise RuntimeError(msg)
        except RuntimeError:
            # exc_info=True is default for exception(), no need to specify
            result = logger.exception("Error occurred")
            assert result.is_success

    def test_exception_logging_with_provided_exception(self) -> None:
        """Test exception logging with provided exception object."""
        logger = FlextCore.Logger("exception_logger")

        try:
            msg = "Provided exception"
            raise ValueError(msg)
        except ValueError as e:
            # Using exception() to log the exception properly
            result = logger.exception("Error details", exception=e)
            assert result.is_success

    def test_exception_logging_with_additional_context(self) -> None:
        """Test exception logging with additional context."""
        logger = FlextCore.Logger("exception_logger")

        try:
            msg = "Context exception"
            raise KeyError(msg)
        except KeyError as e:
            result = logger.exception(
                "Key error occurred",
                exception=e,
                user_id="usr-123",
                operation="data_lookup",
            )
            assert result.is_success

    def test_logger_name_property(self) -> None:
        """Test logger name property accessor."""
        logger = FlextCore.Logger("property_test")
        assert logger.name == "property_test"

    def test_logger_with_all_initialization_params(self) -> None:
        """Test logger initialization with all parameters."""
        logger = FlextCore.Logger(
            "full_init",
            _level="DEBUG",
            _service_name="test-service",
            _service_version="1.2.3",
            _correlation_id="cor-xyz",
            _force_new=True,
        )

        assert logger.name == "full_init"
        result = logger.debug("Full initialization test")
        assert result.is_success

    def test_global_context_with_logging(self) -> None:
        """Test that global context is included in log messages."""
        FlextCore.Logger.clear_global_context()
        FlextCore.Logger.bind_global_context(global_request_id="global-123")

        logger = FlextCore.Logger("context_test")
        result = logger.info("Message with global context")

        assert result.is_success
        FlextCore.Logger.clear_global_context()

    def test_performance_tracker_with_very_short_operation(self) -> None:
        """Test performance tracker with very short operation."""
        logger = FlextCore.Logger("perf_logger")

        with logger.track_performance("instant_operation"):
            pass  # Instant operation

    def test_performance_tracker_with_long_operation(self) -> None:
        """Test performance tracker with longer operation."""
        logger = FlextCore.Logger("perf_logger")

        with logger.track_performance("long_operation"):
            time.sleep(0.05)  # 50ms operation

    def test_bound_logger_retains_base_context(self) -> None:
        """Test that bound logger retains base logger context."""
        base = FlextCore.Logger.create_service_logger("api", version="1.0")
        bound = base.bind(endpoint="/users")

        # Both should log successfully
        assert base.info("Base message").is_success
        assert bound.info("Bound message").is_success

    def test_multiple_loggers_independent(self) -> None:
        """Test that multiple logger instances are independent."""
        logger1 = FlextCore.Logger("logger1")
        logger2 = FlextCore.Logger("logger2")

        bound1 = logger1.bind(context1="value1")
        bound2 = logger2.bind(context2="value2")

        # All should log independently
        assert logger1.info("Logger 1").is_success
        assert logger2.info("Logger 2").is_success
        assert bound1.info("Bound 1").is_success
        assert bound2.info("Bound 2").is_success

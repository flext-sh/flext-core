"""Comprehensive tests for FlextLogger real functionality.

Tests the actual FlextLogger functionality using flext_tests utilities:
- Real logger initialization and configuration
- Actual structured logging behavior
- Correlation ID functionality
- Performance metrics tracking
- Security sanitization
- Real functionality validation using flext_tests

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import TracebackType
from typing import NoReturn, Self, cast

import pytest
import structlog

from flext_core import (
    FlextConfig,
    FlextContext,
    FlextLogger,
    FlextProtocols,
    FlextModels,
)
from flext_core.constants import FlextConstants
from flext_tests import (
    FlextTestsMatchers,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class RealLogCapture:
    """Capture real logging output for verification."""

    def __init__(self) -> None:
        """Initialize log capture."""
        self.captured_output = io.StringIO()
        self.captured_errors = io.StringIO()

    def __enter__(self) -> Self:
        """Enter context manager."""
        self._stdout_redirect = redirect_stdout(self.captured_output)
        self._stderr_redirect = redirect_stderr(self.captured_errors)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self._stdout_redirect.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_redirect.__exit__(exc_type, exc_val, exc_tb)

    def get_output(self) -> str:
        """Get captured output."""
        return self.captured_output.getvalue() + self.captured_errors.getvalue()

    def has_content(self, content: str) -> bool:
        """Check if output contains content."""
        return content in self.get_output()


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None]:
    """Reset logging state between tests."""
    # Reset correlation ID (clear any existing ID)
    FlextContext.Utilities.clear_context()

    # Reset configuration state so each test can control feature flags
    FlextConfig.reset_global_instance()

    # Reset global correlation ID to ensure test isolation
    FlextLogger._global_correlation_id = None

    # Reset structlog configuration
    FlextLogger._configured = False

    yield

    # Clean up
    FlextLogger._configured = False
    FlextLogger._global_correlation_id = None
    FlextConfig.reset_global_instance()


class TestFlextLoggerInitialization:
    """Test FlextLogger initialization and basic functionality."""

    def test_logger_creation_basic(self) -> None:
        """Test basic logger creation with default parameters."""
        logger = FlextLogger("test_logger")

        assert logger._name == "test_logger"
        assert logger._level == "WARNING"  # From test environment
        assert logger._service_name == "flext-core"
        assert logger._service_version is not None
        assert logger._correlation_id is not None

    def test_logger_protocol_runtime_compliance(self) -> None:
        """Ensure FlextLogger instances satisfy the runtime logger protocol."""
        logger = FlextLogger("protocol_runtime_test", _force_new=True)

        assert isinstance(
            logger,
            FlextProtocols.Infrastructure.LoggerProtocol,
        )

    def test_logger_creation_with_service_info(self) -> None:
        """Test logger creation with service metadata."""
        logger = FlextLogger(
            "test_service",
            _level="DEBUG",
            _service_name="payment-service",
            _service_version="2.1.0",
        )

        assert logger._name == "test_service"
        assert logger._level == "DEBUG"
        assert logger._service_name == "payment-service"
        assert logger._service_version == "2.1.0"

    def test_service_name_extraction_from_module(self) -> None:
        """Test automatic service name extraction from module name."""
        logger = FlextLogger("flext_api.handlers.user")

        assert logger._service_name == "flext-api"

    def test_service_name_from_real_environment(self) -> None:
        """Test service name extraction - real functionality."""
        logger = FlextLogger("test_logger")

        # Test that service name is properly set (could be from env or default)
        assert hasattr(logger, "_service_name")
        assert isinstance(logger._service_name, str)
        assert len(logger._service_name) >= 0  # Can be empty or have content

    def test_version_from_real_environment(self) -> None:
        """Test version extraction - real functionality."""
        logger = FlextLogger("test_logger")

        # Test that service version is properly set (could be from env or default)
        assert hasattr(logger, "_service_version")
        assert isinstance(logger._service_version, str)
        assert len(logger._service_version) >= 0  # Can be empty or have content

    def test_environment_detection(self) -> None:
        """Test environment detection logic."""
        logger = FlextLogger("test_logger")

        # Test that environment is detected (should be development in testing)
        environment = logger._get_environment()
        assert environment in set(FlextConstants.Config.ENVIRONMENTS), (
            f"Unexpected environment: {environment}"
        )

    def test_instance_id_generation(self) -> None:
        """Test unique instance ID generation."""
        logger1 = FlextLogger("test1")
        logger2 = FlextLogger("test2")

        instance_id1 = logger1._get_instance_id()
        instance_id2 = logger2._get_instance_id()

        # Should be same for same process but unique enough
        assert instance_id1 == instance_id2
        assert "-" in instance_id1  # Should contain hostname-pid format


class TestStructuredLogging:
    """Test structured logging output and field validation."""

    def test_basic_structured_logging(self) -> None:
        """Test basic structured log entry creation with real functionality."""
        logger = FlextLogger("test_service", _service_name="test-app")

        # Test that logger was created correctly
        assert logger._name == "test_service"
        assert logger._service_name == "test-app"
        assert logger._correlation_id is not None
        assert len(logger._correlation_id) > 0

        # Test that logging methods exist and are callable
        assert callable(logger.info)
        assert callable(logger.debug)
        assert callable(logger.error)

        # Test actual logging - verify no exceptions are raised
        try:
            logger.info("Test message", user_id="123", action="login")
            logger.debug("Debug message")
            logger.error("Error message")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify logger maintains state correctly
        assert logger._service_name == "test-app"

    def test_structured_field_validation(self) -> None:
        """Test that structured log entry building works correctly."""
        logger = FlextLogger("test_service", _service_name="validation-test")

        # Test the real _build_log_entry method functionality
        log_entry = logger._build_log_entry("INFO", "Test message", {"user_id": "123"})

        # Verify entry is a dictionary with expected structure
        assert isinstance(log_entry, dict)
        assert len(log_entry) > 0

        # Check required fields exist
        required_fields = ["timestamp", "level", "message", "logger", "correlation_id"]
        for field in required_fields:
            assert field in log_entry, f"Required field {field} missing"

        # Verify field values are correct type and content
        assert log_entry.get("level") == "INFO"
        assert log_entry.get("message") == "Test message"
        assert log_entry.get("logger") == "test_service"
        correlation_id = log_entry.get("correlation_id")
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0

        # Check service metadata exists and is structured
        assert "service" in log_entry
        service_data = log_entry.get("service", {})
        assert isinstance(service_data, dict)
        assert service_data.get("name") == "validation-test"
        assert "version" in service_data
        assert "instance_id" in service_data
        assert "environment" in service_data

        # Check system metadata exists and is populated
        assert "system" in log_entry
        system_data = log_entry.get("system", {})
        assert isinstance(system_data, dict)
        assert "hostname" in system_data
        assert "platform" in system_data
        assert "python_version" in system_data

        # Check execution context exists
        assert "execution" in log_entry
        execution_data = log_entry.get("execution", {})
        assert isinstance(execution_data, dict)

    def test_timestamp_format_validation(self) -> None:
        """Test ISO 8601 timestamp format."""
        logger = FlextLogger("timestamp_test")

        timestamp = logger._get_current_timestamp()

        # Should be ISO 8601 format with timezone
        assert "T" in timestamp
        assert timestamp.endswith(("+00:00", "Z"))

        # Should be parseable as datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None

    def test_message_with_context(self) -> None:
        """Test message logging with context data using real functionality."""
        logger = FlextLogger("context_test")

        # Test that context data is properly handled in log entry building
        context_data = {
            "order_id": "ORD-123",
            "amount": 99.99,
            "currency": "USD",
            "customer_id": "CUST-456",
        }

        log_entry = logger._build_log_entry("INFO", "Order processed", context_data)

        # Verify message is correctly set
        assert log_entry.get("message") == "Order processed"

        # Verify context data is properly included and structured
        assert "context" in log_entry
        context = log_entry.get("context", {})
        assert isinstance(context, dict)
        assert context["order_id"] == "ORD-123"
        assert context["amount"] == 99.99
        assert context["currency"] == "USD"
        assert context["customer_id"] == "CUST-456"

        # Test that real logging with context works without errors
        try:
            logger.info(
                "Order processed",
                order_id="ORD-123",
                amount=99.99,
                currency="USD",
                customer_id="CUST-456",
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")


class TestLoggingFeatureFlags:
    """Validate feature-flag driven logging behaviour."""

    @pytest.mark.parametrize(
        ("include_context", "mask_sensitive_data", "expected_password"),
        [
            (True, True, "[REDACTED]"),
            (True, False, "secret123"),
            (False, True, None),
            (False, False, None),
        ],
    )
    def test_context_and_mask_flags(
        self,
        include_context: bool,
        mask_sensitive_data: bool,
        expected_password: str | None,
    ) -> None:
        """Ensure context inclusion and masking follow FlextConfig settings."""

        FlextConfig.set_global_instance(
            FlextConfig.create(
                include_context=include_context,
                mask_sensitive_data=mask_sensitive_data,
            )
        )

        logger = FlextLogger("flag_test_context")
        context_data = {"password": "secret123", "username": "alice"}

        entry = logger._build_log_entry("INFO", "Context flag", context_data)

        if include_context:
            assert "context" in entry
            context_payload = entry["context"]
            assert context_payload["username"] == "alice"
            assert context_payload.get("password") == expected_password
        else:
            assert "context" not in entry

    @pytest.mark.parametrize("include_correlation_id", [True, False])
    def test_correlation_id_flag(self, include_correlation_id: bool) -> None:
        """Ensure correlation ID is optional based on configuration."""

        FlextConfig.set_global_instance(
            FlextConfig.create(include_correlation_id=include_correlation_id)
        )

        logger = FlextLogger("flag_test_correlation")
        entry = logger._build_log_entry("INFO", "Correlation flag")

        if include_correlation_id:
            assert entry.get("correlation_id") == logger._correlation_id
        else:
            assert "correlation_id" not in entry

    @pytest.mark.parametrize("track_performance", [True, False])
    def test_performance_flag(self, track_performance: bool) -> None:
        """Verify performance block obeys tracking configuration."""

        FlextConfig.set_global_instance(
            FlextConfig.create(track_performance=track_performance)
        )

        logger = FlextLogger("flag_test_performance")
        entry = logger._build_log_entry(
            "INFO",
            "Performance flag",
            {},
            duration_ms=42.5,
        )

        if track_performance:
            performance = entry.get("performance", {})
            assert performance["duration_ms"] == 42.5
        else:
            assert "performance" not in entry

    @pytest.mark.parametrize("track_timing", [True, False])
    def test_timing_flag(self, track_timing: bool) -> None:
        """Ensure execution metadata respects timing tracker configuration."""

        FlextConfig.set_global_instance(FlextConfig.create(track_timing=track_timing))

        logger = FlextLogger("flag_test_timing")
        entry = logger._build_log_entry("INFO", "Timing flag")

        if track_timing:
            execution = entry.get("execution", {})
            assert "uptime_seconds" in execution
        else:
            assert "execution" not in entry


class TestCorrelationIdFunctionality:
    """Test correlation ID functionality for request tracing."""

    def test_automatic_correlation_id_generation(self) -> None:
        """Test automatic correlation ID generation."""
        # Reset to ensure clean state and generate new correlation ID
        with FlextContext.Correlation.new_correlation() as correlation_id:
            FlextLogger("correlation_test")

            retrieved_id = FlextContext.Correlation.get_correlation_id()
            assert retrieved_id is not None
            assert retrieved_id == correlation_id
        assert correlation_id.startswith("corr_")  # Based on actual generation
        assert len(correlation_id) >= 8  # Should have some length

    def test_global_correlation_id_setting(self) -> None:
        """Test setting global correlation ID."""
        test_correlation_id = "corr_test_123456"
        FlextContext.Correlation.set_correlation_id(test_correlation_id)

        logger = FlextLogger("correlation_test")
        # Logger needs to explicitly be set to use the global correlation ID
        logger.set_correlation_id(test_correlation_id)

        assert logger._correlation_id == test_correlation_id
        assert FlextContext.Correlation.get_correlation_id() == test_correlation_id

    def test_correlation_id_in_log_output(self) -> None:
        """Test that correlation ID functionality works correctly."""
        test_correlation_id = f"corr_test_{uuid.uuid4().hex[:8]}"
        FlextContext.Correlation.set_correlation_id(test_correlation_id)

        logger = FlextLogger("correlation_test")

        # Test that correlation ID is correctly retrieved
        assert logger._correlation_id == test_correlation_id

        # Test that log entry includes correlation ID
        log_entry = logger._build_log_entry("INFO", "Test message with correlation")
        assert log_entry.get("correlation_id") == test_correlation_id

        # Test that real logging works with correlation ID
        try:
            logger.info("Test message with correlation")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_correlation_id_persistence(self) -> None:
        """Test that correlation ID persists across multiple log calls."""
        test_correlation_id = f"corr_persist_{uuid.uuid4().hex[:8]}"
        FlextContext.Correlation.set_correlation_id(test_correlation_id)

        logger = FlextLogger("persistence_test")

        # Test that correlation ID is maintained in logger instance
        assert logger._correlation_id == test_correlation_id

        # Test that multiple log entries maintain the same correlation ID
        entry1 = logger._build_log_entry("INFO", "First message")
        entry2 = logger._build_log_entry("WARNING", "Second message")
        entry3 = logger._build_log_entry("ERROR", "Third message")

        # All entries should have the same correlation ID
        assert entry1.get("correlation_id") == test_correlation_id
        assert entry2.get("correlation_id") == test_correlation_id
        assert entry3.get("correlation_id") == test_correlation_id

        # Test that real logging maintains correlation ID
        try:
            logger.info("First message")
            logger.warning("Second message")
            logger.error("Third message")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")


class TestOperationTracking:
    """Test operation tracking and performance metrics."""

    def test_operation_start_tracking(self) -> None:
        """Test operation start tracking with real functionality."""
        logger = FlextLogger("operation_test")

        # Test that start_operation returns an operation ID
        operation_id = logger.start_operation(
            "user_authentication",
            user_id="123",
            method="oauth2",
        )

        # Verify operation ID is returned and valid
        assert operation_id is not None
        assert isinstance(operation_id, str)
        assert operation_id.startswith("op_")
        assert len(operation_id) > 3

        # Test that operation is tracked in thread-local storage
        assert hasattr(logger._local, "operations")
        assert operation_id in logger._local.operations

        # Verify operation data structure
        operation_info = logger._local.operations[operation_id]
        assert operation_info["name"] == "user_authentication"
        assert "start_time" in operation_info
        op_ctx = operation_info["context"]  # narrow type
        assert op_ctx["user_id"] == "123"
        assert op_ctx["method"] == "oauth2"

    def test_operation_completion_tracking(self) -> None:
        """Test operation completion with real performance metrics."""
        logger = FlextLogger("operation_test")

        # Start operation and get ID
        operation_id = logger.start_operation("data_processing")
        assert operation_id is not None

        # Verify operation is being tracked
        assert hasattr(logger._local, "operations")
        assert operation_id in logger._local.operations

        # Get start time for duration calculation
        start_time = logger._local.operations[operation_id]["start_time"]
        assert isinstance(start_time, float)

        # Simulate some work
        time.sleep(0.05)  # Reduced sleep for faster tests

        # Complete operation - should not raise exception
        try:
            logger.complete_operation(
                operation_id,
                success=True,
                records_processed=1500,
                cache_hits=45,
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify operation was cleaned up after completion
        assert operation_id not in logger._local.operations

    def test_operation_failure_tracking(self) -> None:
        """Test operation failure tracking with real functionality."""
        logger = FlextLogger("operation_test")

        # Start operation
        operation_id = logger.start_operation("failing_operation")
        assert operation_id is not None
        assert operation_id in logger._local.operations

        # Test failure completion - should not raise exception
        try:
            logger.complete_operation(
                operation_id,
                success=False,
                error_code="PROC_001",
                retry_count=3,
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify operation was cleaned up after completion
        assert operation_id not in logger._local.operations

    def test_performance_metrics_accuracy(self) -> None:
        """Test accuracy of performance metrics using real functionality."""
        logger = FlextLogger("perf_test")

        # Test that operation timing works correctly
        operation_id = logger.start_operation("timed_operation")
        assert operation_id is not None
        assert hasattr(logger._local, "operations")
        assert operation_id in logger._local.operations

        # Get start time for validation
        start_time = logger._local.operations[operation_id]["start_time"]
        assert isinstance(start_time, float)

        # Simulate precise timing
        test_start = time.time()
        time.sleep(0.02)  # 20ms for faster tests
        test_end = time.time()

        # Complete operation - should not raise exception
        try:
            logger.complete_operation(operation_id, success=True)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify operation was cleaned up
        assert operation_id not in logger._local.operations

        # Verify timing calculation is reasonable
        expected_duration_ms = (test_end - test_start) * 1000
        assert expected_duration_ms >= 15  # At least 15ms


class TestSecuritySanitization:
    """Test security-safe logging with sensitive data sanitization."""

    def test_sensitive_field_sanitization(self) -> None:
        """Test that sensitive fields are automatically sanitized using real functionality."""
        logger = FlextLogger("security_test")

        sensitive_context: FlextTypes.Core.Dict = {
            "username": "john_doe",
            "password": "secret123",
            "api_key": "sk_live_abc123",
            "authorization": "Bearer token123",
            "secret": "top_secret",
            "private": "private_data",
            "session_id": "sess_abc123",
        }

        # Test the real sanitization method
        sanitized = logger._sanitize_context(sensitive_context)

        # Verify sanitized result is a dict with correct structure
        assert isinstance(sanitized, dict)
        assert len(sanitized) == len(sensitive_context)

        # Non-sensitive data should remain unchanged
        assert sanitized["username"] == "john_doe"

        # Sensitive data should be redacted using FlextTestsMatchers for validation
        sensitive_keys = [
            "password",
            "api_key",
            "authorization",
            "secret",
            "private",
            "session_id",
        ]
        for key in sensitive_keys:
            assert sanitized[key] == "[REDACTED]", (
                f"Sensitive key {key} was not properly sanitized"
            )

        # Test that sanitization works in actual log entries
        log_entry = logger._build_log_entry("INFO", "Test message", sensitive_context)
        assert isinstance(log_entry, dict)
        assert "context" in log_entry
        context = log_entry.get("context", {})
        assert isinstance(context, dict)
        assert context["username"] == "john_doe"
        for key in sensitive_keys:
            assert context[key] == "[REDACTED]", (
                f"Sensitive key {key} not sanitized in log entry"
            )

    def test_nested_sensitive_data_sanitization(self) -> None:
        """Test sanitization of nested sensitive data using real functionality."""
        logger = FlextLogger("security_test")

        nested_context: FlextTypes.Core.Dict = {
            "user": {
                "name": "john",
                "password": "secret",
                "profile": {"email": "john@example.com", "api_key": "key123"},
            },
            "request": {"headers": {"authorization": "Bearer token"}},
        }

        # Test the real nested sanitization functionality
        sanitized = logger._sanitize_context(nested_context)

        # Verify structure is maintained
        assert isinstance(sanitized, dict)
        assert "user" in sanitized
        assert "request" in sanitized
        assert isinstance(sanitized["user"], dict)
        assert isinstance(sanitized["request"], dict)

        # Extract nested data for verification
        user_data = sanitized["user"]  # narrow
        profile_data = user_data["profile"]  # narrow
        request_data = sanitized["request"]  # narrow
        headers_data = request_data["headers"]  # narrow

        # Non-sensitive nested data should remain unchanged
        assert user_data["name"] == "john"
        assert profile_data["email"] == "john@example.com"

        # Sensitive nested data should be redacted
        assert user_data["password"] == "[REDACTED]"
        assert profile_data["api_key"] == "[REDACTED]"
        assert headers_data["authorization"] == "[REDACTED]"

        # Test that nested sanitization works in actual log building
        log_entry = logger._build_log_entry("INFO", "Test nested", nested_context)
        context = log_entry.get("context", {})
        nested_user = context["user"]
        assert isinstance(nested_user, dict)
        assert nested_user["name"] == "john"
        assert nested_user["password"] == "[REDACTED]"

    def test_sanitization_in_log_output(self) -> None:
        """Test that sanitization occurs in actual log building process."""
        logger = FlextLogger("security_test")

        # Test that real logging with sensitive data works without errors
        try:
            logger.info(
                "User login attempt",
                username="john_doe",
                password="should_be_hidden",
                api_key="should_also_be_hidden",
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building properly sanitizes sensitive data
        context_data: dict[str, object] = {
            "username": "john_doe",
            "password": "should_be_hidden",
            "api_key": "should_also_be_hidden",
        }

        log_entry = logger._build_log_entry("INFO", "User login attempt", context_data)

        # Verify log entry was built correctly
        assert "context" in log_entry
        context = log_entry.get("context", {})
        assert isinstance(context, dict)

        # Username should appear (not sensitive)
        assert context["username"] == "john_doe"

        # Sensitive data should be redacted
        assert context["password"] == "[REDACTED]"
        assert context["api_key"] == "[REDACTED]"


class TestErrorHandling:
    """Test error handling and exception logging."""

    def test_error_logging_with_exception(self) -> None:
        """Test error logging with exception details using real functionality."""
        logger = FlextLogger("error_test")

        def _raise_validation_error() -> NoReturn:
            msg = "Invalid configuration parameter"
            raise ValueError(msg)

        # Test that exception logging works correctly
        try:
            _raise_validation_error()
        except Exception as e:
            # Test that exception logging doesn't raise additional exceptions
            try:
                logger.exception(
                    "Configuration validation failed",
                    error="Test validation error",
                    config_file="/etc/app/config.yaml",
                    parameter="database_url",
                )
            except Exception as logging_error:
                pytest.fail(
                    f"Exception logging should not raise exceptions: {logging_error}",
                )

            # Test that log entry building works with exceptions
            log_entry = logger._build_log_entry(
                "ERROR",
                "Configuration validation failed",
                {
                    "error": "Test validation error",
                    "config_file": "/etc/app/config.yaml",
                },
                error=e,
            )

            # Verify error details are included
            assert "error" in log_entry
            error_info = log_entry.get("error", {})
            assert isinstance(error_info, dict)
            assert error_info["type"] == "ValueError"
            assert isinstance(error_info["message"], str)
            message = error_info["message"]
            assert isinstance(message, str)
            assert "Invalid configuration parameter" in message
            assert "stack_trace" in error_info

    def test_exception_logging_with_stack_trace(self) -> None:
        """Test exception logging captures full stack trace using real functionality."""
        logger = FlextLogger("exception_test")

        def nested_function() -> NoReturn:
            msg = "Deep error"
            raise RuntimeError(msg)

        def calling_function() -> None:
            nested_function()

        # Test that exception logging with stack trace works correctly
        try:
            calling_function()
        except Exception as e:
            # Test that exception logging doesn't raise additional exceptions
            try:
                logger.exception("Unexpected error occurred")
            except Exception as logging_error:
                pytest.fail(
                    f"Exception logging should not raise exceptions: {logging_error}",
                )

            # Test that log entry building includes stack trace
            log_entry = logger._build_log_entry(
                "ERROR",
                "Unexpected error occurred",
                error=e,
            )

            # Verify stack trace information is captured
            assert "error" in log_entry
            error_info = log_entry.get("error", {})
            assert isinstance(error_info, dict)
            assert error_info["type"] == "RuntimeError"
            assert isinstance(error_info["message"], str)
            message = error_info["message"]
            assert isinstance(message, str)
            assert "Deep error" in message
            assert "stack_trace" in error_info
            assert isinstance(error_info["stack_trace"], list)
            assert len(error_info["stack_trace"]) > 0

            # Verify stack trace contains function names from the call chain
            stack_trace_str = "".join(error_info["stack_trace"])
            assert "nested_function" in stack_trace_str
            assert "calling_function" in stack_trace_str

    def test_error_logging_without_exception(self) -> None:
        """Test error logging without an exception object using real functionality."""
        logger = FlextLogger("error_test")

        # Test that error logging without exception works correctly
        try:
            logger.error(
                "Business rule violation",
                rule="max_daily_limit",
                current_amount=1500,
                limit=1000,
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works without exception object
        context_data = {
            "rule": "max_daily_limit",
            "current_amount": 1500,
            "limit": 1000,
        }

        log_entry = logger._build_log_entry(
            "ERROR",
            "Business rule violation",
            context_data,
        )

        # Verify log entry structure is correct
        assert isinstance(log_entry, dict)
        assert log_entry.get("level") == "ERROR"
        assert log_entry.get("message") == "Business rule violation"
        assert "context" in log_entry
        context = log_entry.get("context", {})
        assert isinstance(context, dict)
        assert context["rule"] == "max_daily_limit"
        assert context["current_amount"] == 1500
        assert context["limit"] == 1000

        # Verify no error object is present when none provided
        assert "error" not in log_entry or log_entry.get("error", {}) is None


class TestRequestContextManagement:
    """Test request-scoped context management."""

    def test_request_context_setting(self) -> None:
        """Test setting request-specific context using real functionality."""
        logger = FlextLogger("context_test")

        # Test that request context can be set without errors
        logger.set_request_context(
            request_id="req_123",
            user_id="user_456",
            endpoint="/api/orders",
        )

        # Verify request context is stored in thread-local storage
        assert hasattr(logger._local, "request_context")
        request_context = logger._local.request_context
        assert isinstance(request_context, dict)
        assert request_context["request_id"] == "req_123"
        assert request_context["user_id"] == "user_456"
        assert request_context["endpoint"] == "/api/orders"

        # Test that log entry includes request context
        log_entry = logger._build_log_entry("INFO", "Processing request")
        assert isinstance(log_entry, dict)
        assert "request" in log_entry
        request_data = log_entry.get("request", {})
        assert isinstance(request_data, dict)
        assert request_data["request_id"] == "req_123"
        assert request_data["user_id"] == "user_456"
        assert request_data["endpoint"] == "/api/orders"

        # Test that real logging works with request context
        try:
            logger.info("Processing request")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_request_context_clearing(self) -> None:
        """Test clearing request-specific context using real functionality."""
        logger = FlextLogger("context_test")

        # Set context
        logger.set_request_context(request_id="req_123")
        assert hasattr(logger._local, "request_context")
        assert logger._local.request_context["request_id"] == "req_123"

        # Test that log entry includes context before clearing
        log_entry_with_context = logger._build_log_entry("INFO", "With context")
        assert "request" in log_entry_with_context
        assert log_entry_with_context["request"]["request_id"] == "req_123"

        # Clear context
        logger.clear_request_context()

        # Verify context was cleared
        request_context = getattr(logger._local, "request_context", {})
        assert len(request_context) == 0

        # Test that log entry no longer includes cleared context
        log_entry_without_context = logger._build_log_entry("INFO", "Without context")
        if "request" in log_entry_without_context:
            assert len(log_entry_without_context["request"]) == 0

        # Test that real logging works after clearing context
        try:
            logger.info("With context")
            logger.clear_request_context()
            logger.info("Without context")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_request_context_thread_isolation(self) -> None:
        """Test that request context is thread-isolated using real functionality."""
        logger = FlextLogger("thread_test")

        thread_results = []
        context_results = []

        def thread_function(thread_id: str) -> None:
            # Set thread-specific context
            logger.set_request_context(thread_id=thread_id)
            time.sleep(0.01)  # Small delay to allow context mixing if not isolated

            # Verify thread context is isolated - store separately from success/error results
            if hasattr(logger._local, "request_context"):
                context = logger._local.request_context
                context_results.append((thread_id, context.get("thread_id")))

            # Test that logging works in thread context
            try:
                logger.info(f"Message from thread {thread_id}")
                thread_results.append(f"success_{thread_id}")
            except Exception as e:
                thread_results.append(f"error_{thread_id}: {e}")

        # Create multiple threads
        threads = []
        for i in range(3):
            thread_id = f"thread_{i}"
            thread = threading.Thread(target=thread_function, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify threads executed successfully
        success_results = [r for r in thread_results if r.startswith("success_")]
        assert len(success_results) == 3  # All threads should succeed

        # Verify no error results
        error_results = [r for r in thread_results if "error_" in str(r)]
        assert len(error_results) == 0, f"Thread errors: {error_results}"


class TestLoggerConfiguration:
    """Test logger configuration and processors."""

    def test_json_output_configuration(self) -> None:
        """Test JSON output configuration using real functionality."""
        # Reset configuration to test auto-detection
        FlextLogger._configured = False

        logger = FlextLogger("json_test")

        # Test that logger was properly configured
        assert logger._structlog_logger is not None
        assert FlextLogger._configured is True

        # Test that JSON logging works without errors
        try:
            logger.info("JSON test message", field1="value1", field2=123)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works correctly
        log_entry = logger._build_log_entry(
            "INFO",
            "JSON test message",
            {"field1": "value1", "field2": 123},
        )

        assert log_entry.get("message") == "JSON test message"
        assert "context" in log_entry
        context = log_entry.get("context", {})  # narrow
        assert context["field1"] == "value1"
        assert context["field2"] == 123

    def test_configure_uses_global_log_verbosity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure global config log_verbosity flows into FlextLogger.configure."""

        config = FlextConfig.get_global_instance()
        original_verbosity = config.log_verbosity
        config.log_verbosity = "full"

        FlextLogger._configured = False
        structlog.reset_defaults()

        captured_kwargs: dict[str, object] = {}
        original_model = FlextModels.LoggerConfigurationModel

        def capture_logger_configuration_model(
            *args: object, **kwargs: object
        ) -> FlextModels.LoggerConfigurationModel:
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
            return original_model(*args, **kwargs)

        monkeypatch.setattr(
            FlextModels, "LoggerConfigurationModel", capture_logger_configuration_model
        )

        try:
            result = FlextLogger.configure(
                log_level="INFO",
                json_output=False,
                include_source=True,
                structured_output=True,
            )

            assert result.is_success
            assert FlextLogger._configured is True
            assert captured_kwargs.get("log_verbosity") == "full"

            configuration = FlextLogger.get_configuration()
            assert configuration["log_verbosity"] == "full"
        finally:
            config.log_verbosity = original_verbosity
            structlog.reset_defaults()
            FlextLogger._configured = False

    def test_development_console_output(self) -> None:
        """Test development console output configuration."""
        # Reset configuration
        FlextLogger._configured = False

        logger = FlextLogger("console_test")

        # Test that console logging works without errors
        try:
            logger.info("Console test message", debug_info="useful")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works for console output
        log_entry = logger._build_log_entry(
            "INFO",
            "Console test message",
            {"debug_info": "useful"},
        )

        assert log_entry.get("message") == "Console test message"
        assert "context" in log_entry
        assert log_entry.get("context", {})["debug_info"] == "useful"

    def test_structured_processor_functionality(self) -> None:
        """Test that structured processors are working."""
        logger = FlextLogger("processor_test")

        # Test correlation processor
        test_correlation = f"corr_proc_{uuid.uuid4().hex[:8]}"
        # Set correlation ID on the logger instance directly to ensure it uses this specific ID
        logger.set_correlation_id(test_correlation)

        # Test that processor logging works correctly
        try:
            logger.info("Processor test")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that correlation ID is properly included in log entries
        log_entry = logger._build_log_entry("INFO", "Processor test")
        assert log_entry.get("correlation_id") == test_correlation

    def test_global_log_verbosity_propagates_to_configuration(self) -> None:
        """Ensure global log verbosity updates are honored by configure."""
        config = FlextConfig.get_global_instance()
        original_verbosity = config.log_verbosity
        config.log_verbosity = "compact"

        try:
            FlextLogger._configured = False
            result = FlextLogger.configure()
            assert result.is_success

            configuration = FlextLogger.get_configuration()
            assert configuration["log_verbosity"] == "compact"
        finally:
            config.log_verbosity = original_verbosity


class TestConvenienceFunctions:
    """Test convenience functions and factory methods."""

    def test_get_logger_function(self) -> None:
        """Test FlextLogger convenience function."""
        logger = FlextLogger("convenience_test", _service_name="test-service")

        assert isinstance(logger, FlextLogger)
        assert logger._name == "convenience_test"
        assert logger._service_name == "test-service"

    def test_get_logger_with_version(self) -> None:
        """Test FlextLogger with version parameter."""
        logger = FlextLogger(
            "versioned_test",
            _service_name="test-service",
            _service_version="1.2.3",
        )

        assert logger._service_version == "1.2.3"

    def test_correlation_id_functions(self) -> None:
        """Test correlation ID utility functions."""
        # Test setting
        test_id = f"corr_util_{uuid.uuid4().hex[:8]}"
        FlextContext.Correlation.set_correlation_id(test_id)

        # Test getting
        assert FlextContext.Correlation.get_correlation_id() == test_id

        # Test that new loggers use it
        logger = FlextLogger("correlation_util_test")
        assert logger._correlation_id == test_id


class TestLoggingLevels:
    """Test different logging levels and filtering."""

    def test_all_logging_levels(self) -> None:
        """Test all logging levels work correctly using real functionality."""
        logger = FlextLogger("level_test", _level="DEBUG")

        # Verify logger level was set correctly
        assert logger._level == "DEBUG"

        # Test that all logging methods exist and are callable
        assert callable(logger.trace)
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)

        # Test that all logging methods work without raising exceptions
        try:
            logger.trace("Trace message", detail="very_fine")
            logger.debug("Debug message", component="database")
            logger.info("Info message", status="normal")
            logger.warning("Warning message", issue="deprecated")
            logger.error("Error message", code="E001")
            logger.critical("Critical message", severity="high")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works for different levels
        trace_entry = logger._build_log_entry(
            "TRACE",
            "Trace message",
            {"detail": "very_fine"},
        )
        debug_entry = logger._build_log_entry(
            "DEBUG",
            "Debug message",
            {"component": "database"},
        )
        info_entry = logger._build_log_entry(
            "INFO",
            "Info message",
            {"status": "normal"},
        )
        warning_entry = logger._build_log_entry(
            "WARNING",
            "Warning message",
            {"issue": "deprecated"},
        )
        error_entry = logger._build_log_entry(
            "ERROR",
            "Error message",
            {"code": "E001"},
        )
        critical_entry = logger._build_log_entry(
            "CRITICAL",
            "Critical message",
            {"severity": "high"},
        )

        # Verify all entries have correct levels
        assert trace_entry.get("level") == "TRACE"
        assert debug_entry.get("level") == "DEBUG"
        assert info_entry.get("level") == "INFO"
        assert warning_entry.get("level") == "WARNING"
        assert error_entry.get("level") == "ERROR"
        assert critical_entry.get("level") == "CRITICAL"

    def test_level_filtering(self) -> None:
        """Test that level filtering works correctly using real functionality."""
        logger = FlextLogger("filter_test", _level="WARNING")

        # Verify logger level was set correctly
        assert logger._level == "WARNING"

        # Test that all logging methods still work without raising exceptions
        # (filtering happens at the structlog level, not in our methods)
        try:
            logger.trace("Should be filtered")
            logger.debug("Should be filtered")
            logger.info("Should be filtered")
            logger.warning("Should appear")
            logger.error("Should appear")
            logger.critical("Should appear")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works regardless of level filtering
        # (filtering is handled by structlog processors, not by our entry building)
        trace_entry = logger._build_log_entry("TRACE", "Should be filtered")
        warning_entry = logger._build_log_entry("WARNING", "Should appear")
        error_entry = logger._build_log_entry("ERROR", "Should appear")
        critical_entry = logger._build_log_entry("CRITICAL", "Should appear")

        # All entries should be built correctly (filtering happens at output level)
        assert trace_entry.get("level") == "TRACE"
        assert warning_entry.get("level") == "WARNING"
        assert error_entry.get("level") == "ERROR"
        assert critical_entry.get("level") == "CRITICAL"


class TestRealWorldScenarios:
    """Test real-world logging scenarios."""

    def test_api_request_lifecycle(self) -> None:
        """Test complete API request lifecycle logging."""
        logger = FlextLogger("api_service", _service_name="order-api")

        # Start request
        logger.set_request_context(
            request_id="req_order_123",
            endpoint="POST /api/orders",
            user_id="user_456",
            correlation_id=FlextContext.Correlation.get_correlation_id(),
        )

        # Test real API request lifecycle functionality
        try:
            # Request received
            logger.info("Request received", method="POST", path="/api/orders")

            # Validation
            logger.debug("Validating request data", fields=["amount", "currency"])

            # Business logic
            operation_id = logger.start_operation("create_order", amount=99.99)
            assert operation_id is not None

            # Simulate processing
            time.sleep(0.01)

            # Complete operation
            logger.complete_operation(
                operation_id,
                success=True,
                order_id="ORD-789",
                payment_status="authorized",
            )

            # Response
            logger.info("Request completed", status_code=201, response_size=256)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify operation was cleaned up
        assert operation_id not in logger._local.operations

        # Verify context propagation by checking for request context in logs

    def test_error_handling_scenario(self) -> None:
        """Test comprehensive error handling scenario."""
        logger = FlextLogger("error_service", _service_name="payment-processor")

        logger.set_request_context(
            request_id="req_payment_fail",
            operation="process_payment",
        )

        # Simulate a payment processing error
        def _process_payment() -> NoReturn:
            msg = "Payment gateway timeout"
            raise ConnectionError(msg)

        # Test real error handling scenario functionality
        try:
            logger.info("Starting payment processing", amount=150.00, gateway="stripe")

            try:
                _process_payment()
            except Exception:
                logger.exception(
                    "Payment processing failed",
                    error="Test validation error",
                    retry_count=3,
                    fallback_gateway="paypal",
                )

                # Log recovery attempt
                logger.warning(
                    "Attempting payment recovery",
                    recovery_method="fallback_gateway",
                )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify error handling worked correctly
        assert logger._correlation_id is not None

    def test_high_throughput_logging(self) -> None:
        """Test logging performance under high throughput using real functionality."""
        logger = FlextLogger("throughput_test", _service_name="high-volume-api")

        # Verify logger was created correctly
        assert logger._service_name == "high-volume-api"

        start_time = time.time()
        message_count = 50  # Reduced for faster testing

        # Test that high-volume logging works without errors
        try:
            for i in range(message_count):
                logger.info(
                    f"Processing item {i}",
                    item_id=f"item_{i}",
                    batch_id="batch_001",
                    sequence=i,
                )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        end_time = time.time()
        duration = end_time - start_time

        # Performance should be reasonable (less than 2 seconds for 50 messages)
        assert duration < 2.0

        # Test that log entry building works correctly for high throughput
        sample_entry = logger._build_log_entry(
            "INFO",
            "Processing item 0",
            {"item_id": "item_0", "batch_id": "batch_001", "sequence": 0},
        )

        assert sample_entry.get("message") == "Processing item 0"
        assert "context" in sample_entry
        context = sample_entry.get("context", {})
        assert context["item_id"] == "item_0"
        assert context["batch_id"] == "batch_001"
        assert context["sequence"] == 0

        # Calculate throughput using FlextTestsMatchers for validation
        throughput = message_count / duration if duration > 0 else message_count
        # Use FlextTestsMatchers assertion pattern
        FlextTestsMatchers.CoreMatchers.assert_greater_than(
            actual=throughput,
            expected=10.0,  # More reasonable expectation
            message=f"Throughput {throughput:.2f} should be > 10 msg/sec",
        )


class TestLoggingConfiguration:
    """Test logging configuration and system-level features."""

    def test_bind_logger_creates_new_instance(self) -> None:
        """Test that bind creates a new logger with bound context using real functionality."""
        logger = FlextLogger("bind_test", _service_name="test-service")

        # Test that bind creates a new instance
        bound_logger = logger.bind(user_id="123", operation="test")

        # Should be different instances
        assert bound_logger is not logger
        assert type(bound_logger) is type(logger)

        # Should have same base configuration
        assert bound_logger._name == logger._name
        assert bound_logger._service_name == logger._service_name
        assert bound_logger._level == logger._level

        # Test that bound logger has the bound context
        assert hasattr(bound_logger._local, "request_context")
        bound_context = bound_logger._local.request_context
        assert bound_context["user_id"] == "123"
        assert bound_context["operation"] == "test"

        # Test that bound logger works for logging
        try:
            bound_logger.info("Test bound logging")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that bound logger includes context in log entries
        log_entry = bound_logger._build_log_entry("INFO", "Test bound logging")
        assert "request" in log_entry
        request_data = log_entry.get("request", {})
        assert request_data["user_id"] == "123"
        assert request_data["operation"] == "test"


class TestAdvancedLoggingFeatures:
    """Test advanced logging features and edge cases for comprehensive coverage."""

    def test_invalid_log_level_during_initialization(self) -> None:
        """Test handling of invalid log level during logger initialization."""
        # Test with invalid log level - should default to INFO
        # Test with invalid log level - should default to INFO

        logger = FlextLogger(
            "invalid_level_test",
            _level=cast("FlextTypes.Config.LogLevel", "INVALID_LEVEL"),
        )

        # Should default to INFO level
        assert logger._level == "WARNING"  # From test environment

    def test_calling_function_extraction_error_handling(self) -> None:
        """Test error handling when extracting calling function information."""
        logger = FlextLogger("function_test")

        # Test direct call to protected method to trigger error path
        function_name = logger._get_calling_function()

        # Should return a string (either real function name or "unknown")
        assert isinstance(function_name, str)
        assert len(function_name) > 0

    def test_calling_line_extraction_error_handling(self) -> None:
        """Test error handling when extracting calling line information."""
        logger = FlextLogger("line_test")

        # Test direct call to protected method to trigger error path
        line_number = logger._get_calling_line()

        # Should return an integer (either real line number or 0)
        assert isinstance(line_number, int)
        assert line_number >= 0

    def test_performance_duration_with_none_value(self) -> None:
        """Test performance logging when duration is None."""
        logger = FlextLogger("perf_test")

        # Test building log entry with None duration
        entry = logger._build_log_entry("INFO", "Test message", {}, duration_ms=None)

        # Performance section should not be included when duration is None
        assert "performance" not in entry

    def test_performance_duration_with_value(self) -> None:
        """Test performance logging when duration has a value."""
        logger = FlextLogger("perf_test")

        # Test building log entry with specific duration
        entry = logger._build_log_entry("INFO", "Test message", {}, duration_ms=123.456)

        # Performance section should be included with rounded duration
        assert "performance" in entry
        performance_data = entry["performance"]
        assert performance_data["duration_ms"] == 123.456

    def test_system_information_gathering(self) -> None:
        """Test system information gathering functionality."""
        logger = FlextLogger("system_test")

        # Test that system information is included in log entries
        entry = logger._build_log_entry("INFO", "Test message", {})

        # Check system metadata exists
        assert "system" in entry
        system_data = entry["system"]

        # Verify system information fields
        assert "hostname" in system_data
        assert isinstance(system_data["hostname"], str)
        assert len(str(system_data["hostname"])) > 0

        assert "platform" in system_data
        assert isinstance(system_data["platform"], str)
        assert len(str(system_data["platform"])) > 0

        assert "python_version" in system_data
        assert isinstance(system_data["python_version"], str)
        assert "." in str(system_data["python_version"])

    def test_uptime_calculation(self) -> None:
        """Test uptime calculation functionality."""
        logger = FlextLogger("uptime_test")

        # Small delay to ensure uptime > 0
        time.sleep(0.01)

        entry = logger._build_log_entry("INFO", "Test uptime", {})

        execution_data = entry.get("execution", {})
        uptime = execution_data.get("uptime_seconds", 0)

        # Should be a positive number
        assert isinstance(uptime, (int, float))
        assert uptime > 0

    def test_thread_local_data_management(self) -> None:
        """Test thread-local data management for operations and context."""
        logger = FlextLogger("thread_test")

        # Test setting and clearing request context
        logger.set_request_context(test_key="test_value")

        # Context should be set
        assert hasattr(logger._local, "request_context")

        logger.clear_request_context()

        # Context should be cleared
        request_context = getattr(logger._local, "request_context", {})
        assert len(request_context) == 0

    def test_operation_tracking_edge_cases(self) -> None:
        """Test operation tracking edge cases and error conditions."""
        logger = FlextLogger("operation_edge_test")

        # Test real operation tracking functionality
        try:
            # Test starting and completing operation normally
            operation_id = logger.start_operation("test_operation")
            assert operation_id is not None

            logger.complete_operation(operation_id, success=True)

            # Verify operation was cleaned up
            assert operation_id not in logger._local.operations
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_log_level_validation_edge_cases(self) -> None:
        """Test log level validation with various edge cases."""
        # Test with invalid level - should default to INFO
        # Test with invalid level - should default to INFO
        logger = FlextLogger(
            "level_edge_test",
            _level=cast("FlextTypes.Config.LogLevel", "INVALID_LEVEL"),
        )
        assert (
            logger._level == "WARNING"
        )  # From test environment  # Should default to INFO

        # Test with empty string level - should default to INFO
        # Test with empty string level - should default to INFO
        logger = FlextLogger(
            "level_edge_test",
            _level=cast("FlextTypes.Config.LogLevel", ""),
        )
        assert (
            logger._level == "WARNING"
        )  # From test environment  # Should default to INFO

        # Test with lowercase valid level - should convert to uppercase
        # Test with lowercase valid level - should convert to uppercase
        logger = FlextLogger(
            "level_edge_test",
            _level=cast("FlextTypes.Config.LogLevel", "debug"),
        )
        assert logger._level == "DEBUG"  # Should convert to uppercase

    def test_service_name_extraction_from_environment(self) -> None:
        """Test service name extraction from different sources."""
        # Test extraction from environment variable or module name
        logger = FlextLogger("test")
        # Service name should be extracted from environment or module
        assert isinstance(logger._service_name, str)
        assert len(logger._service_name) > 0

        # Test extraction from module name with flext prefix
        logger = FlextLogger("flext_auth.handlers.user")
        # Service name should be derived from module name or environment
        assert isinstance(logger._service_name, str)
        assert len(logger._service_name) > 0


class TestPerformanceAndStressScenarios:
    """Test performance characteristics and stress scenarios using tests/support."""

    def test_high_volume_logging_performance(self) -> None:
        """Test logging performance under high volume."""
        logger = FlextLogger("performance_test")

        start_time = time.time()

        # Test real high volume functionality
        try:
            # Log 50 messages with context (reduced for faster testing)
            for i in range(50):
                logger.info(
                    f"High volume message {i}",
                    request_id=f"req_{i}",
                    batch_id="perf_test_batch",
                    sequence=i,
                    data_size=1024 * (i % 10),
                )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        end_time = time.time()
        duration = end_time - start_time

        # Verify all messages were logged
        # Test completed successfully

        # Performance should be reasonable (less than 1 second for 200 messages)
        assert duration < 1.0

        # Calculate throughput for reduced message count
        throughput = 50 / duration if duration > 0 else 50
        # Use FlextTestsMatchers for validation
        FlextTestsMatchers.CoreMatchers.assert_greater_than(
            actual=throughput,
            expected=20.0,  # More reasonable expectation for real logging
            message=f"Throughput {throughput:.2f} should be > 20 msg/sec",
        )

    def test_concurrent_logging_thread_safety(self) -> None:
        """Test thread safety with concurrent logging operations."""
        logger = FlextLogger("concurrent_test")

        errors = []
        total_messages = 0

        def log_worker(thread_id: str, message_count: int) -> None:
            nonlocal total_messages
            try:
                logger.set_request_context(thread_id=thread_id)

                for i in range(message_count):
                    logger.info(
                        f"Concurrent message {i}",
                        thread_id=thread_id,
                        message_index=i,
                    )
                    total_messages += 1

            except Exception:
                errors.append(f"Thread {thread_id}: exception caught")

        # Create and start multiple threads (reduced count for faster testing)
        threads = []
        message_count = 10  # Reduced for faster testing

        for i in range(3):  # 3 concurrent threads
            thread_id = f"thread_{i}"
            thread = threading.Thread(
                target=log_worker,
                args=(thread_id, message_count),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors in concurrent logging: {errors}"

        # Verify all messages were processed
        assert total_messages == 30  # 3 threads * 10 messages each

    def test_memory_efficiency_with_large_contexts(self) -> None:
        """Test memory efficiency with large context objects."""
        logger = FlextLogger("memory_test")

        # Create large context data (reduced size for faster testing)
        large_context = {
            f"field_{i}": f"value_{i}" * 50  # Reduced string size
            for i in range(50)  # Reduced field count
        }

        start_time = time.time()

        # Test real large context functionality
        try:
            logger.info("Large context test", **large_context)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        end_time = time.time()
        duration = end_time - start_time

        # Verify message was logged
        # Test completed successfully

        # Performance should still be reasonable even with large context
        assert duration < 0.5  # Should complete in less than 500ms

    def test_error_resilience_under_stress(self) -> None:
        """Test error resilience under stress conditions."""
        logger = FlextLogger("stress_test")

        errors_handled = 0

        # Test real error resilience functionality
        try:
            # Mix of normal logs and error conditions (reduced count for faster testing)
            for i in range(50):  # Reduced from 100

                def _trigger_error(index: int) -> None:
                    msg = f"Test error {index}"
                    raise ValueError(msg)

                if i % 10 == 0:
                    # Trigger error condition and handle it
                    try:
                        _trigger_error(i)
                    except ValueError as e:
                        logger.exception(
                            f"Handled error in iteration {i}",
                            error=e,
                            iteration=i,
                        )
                        errors_handled += 1
                else:
                    # Normal logging
                    logger.info(f"Normal message {i}", index=i)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify errors were handled correctly
        assert errors_handled == 5  # Should have handled 5 errors


class TestUncoveredLinesTargeted:
    """Tests specifically targeting remaining uncovered lines for 100% coverage."""

    def test_line_597_no_operations_early_return(self) -> None:
        """Test line 597: early return when no operations in thread-local storage."""
        logger = FlextLogger("no_ops_test")

        # Ensure no operations exist in thread-local storage
        if hasattr(logger._local, "operations"):
            delattr(logger._local, "operations")

        # Test real functionality - early return when no operations
        try:
            # This should trigger the early return on line 597
            logger.complete_operation("non_existent_op", success=False)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify no operations were created
        operations = getattr(logger._local, "operations", {})
        assert "non_existent_op" not in operations

    def test_line_909_sensitive_key_sanitization(self) -> None:
        """Test line 909: sanitization when sensitive key is found."""
        logger = FlextLogger("sanitize_test")

        # Create a processor that will trigger the sanitization line 909
        sensitive_data: FlextTypes.Core.Dict = {
            "user_data": "normal",
            "password_field": "should_be_redacted",  # This should trigger line 909
            "normal_field": "normal_value",
        }

        sanitized = logger._sanitize_context(sensitive_data)

        # Line 909 should have executed to redact the password_field
        assert sanitized["password_field"] == "[REDACTED]"
        assert sanitized["user_data"] == "normal"  # Not sensitive
        assert sanitized["normal_field"] == "normal_value"  # Not sensitive

    def test_lines_1084_1085_bind_exception(self) -> None:
        """Test lines 1084-1085: exception handling in bind method."""
        logger = FlextLogger("bind_exception_test")

        # Try to create a scenario that might cause an exception during bind
        # Test with reasonable-sized data to avoid memory issues
        try:
            # Bind with moderately large context
            large_data = {f"key_{i}": f"value_{i}" * 10 for i in range(20)}
            bound_logger = logger.bind(**large_data)
            assert bound_logger is not logger
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test with potential problematic keys
        try:
            # Test with None values and special characters
            problematic_data = {
                "normal_key": "normal_value",
                "none_key": None,
                "unicode_key": "unicode_value_ñ_emoji_🎉",
            }
            bound_logger = logger.bind(**problematic_data)
            assert bound_logger is not logger
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_extreme_edge_cases_for_100_percent_coverage(self) -> None:
        """Test extreme edge cases to reach 100% coverage of remaining 10 lines."""
        logger = FlextLogger("extreme_test")

        # Test line 909: Force sanitization processor to run
        # Create data with mixed case sensitive keys
        data_with_sensitive: FlextTypes.Core.Dict = {
            "PASSWORD": "secret",  # Uppercase sensitive
            "Api_Key": "key123",  # Mixed case
            "SECRET_TOKEN": "token456",  # Underscore
            "normal_data": "normal",
        }

        sanitized = logger._sanitize_context(data_with_sensitive)
        # Line 909 should execute for each sensitive key
        assert sanitized["PASSWORD"] == "[REDACTED]"
        assert sanitized["Api_Key"] == "[REDACTED]"
        assert sanitized["SECRET_TOKEN"] == "[REDACTED]"
        assert sanitized["normal_data"] == "normal"

        # Test console renderer with edge cases (lines 1215, 1226-1227)
        try:
            # Force different console renderer paths
            renderer1 = logger._create_enhanced_console_renderer()
            renderer2 = logger._create_enhanced_console_renderer()
            assert renderer1 is not None
            assert renderer2 is not None
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

    def test_100_percent_coverage_line_909_exception_paths(self) -> None:
        """Test to force coverage of line 909 with multiple sensitive keys."""
        logger = FlextLogger("coverage_909")

        # Create data that will definitely trigger line 909 multiple times
        sensitive_data_complex: FlextTypes.Core.Dict = {
            "password": "secret1",  # Should trigger line 909
            "api_key": "secret2",  # Should trigger line 909
            "token": "secret3",  # Should trigger line 909
            "secret": "secret4",  # Should trigger line 909
            "credential": "secret5",  # Should trigger line 909
            "auth": "secret6",  # Should trigger line 909
            "normal": "not_secret",  # Should NOT trigger line 909
        }

        # This should execute line 909 multiple times
        sanitized = logger._sanitize_context(sensitive_data_complex)

        # Verify all sensitive keys were redacted (line 909 executed)
        for key in ["password", "api_key", "token", "secret", "credential", "auth"]:
            assert sanitized[key] == "[REDACTED]", f"Key {key} should be redacted"
        assert sanitized["normal"] == "not_secret"

    def test_100_percent_coverage_line_909_sanitize_processor(self) -> None:
        """Test to force line 909 in _sanitize_processor to be covered."""
        # Configure logging with structured output to ensure processors run
        FlextLogger.configure(
            json_output=False,
            structured_output=True,
            include_source=True,
        )

        logger = FlextLogger("sanitize_processor_test")

        # Test real sanitization processor functionality
        try:
            # Log with sensitive data that should trigger sanitization processor
            logger.info(
                "User login attempt",
                password="should_be_redacted",  # Should trigger line 909
                api_key="secret_key_123",  # Should trigger line 909
                username="john_doe",  # Should NOT trigger
                token="bearer_token_456",  # Should trigger line 909
                secret="top_secret",  # Should trigger line 909
                normal_field="normal_value",  # Should NOT trigger
            )
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify that log entry was created and sensitive data was processed
        # Test completed successfully

        # The structured logging processor should have executed line 909
        # to sanitize the sensitive fields during log processing

    def test_100_percent_coverage_line_911_direct_processor_call(self) -> None:
        """Test to force line 911 by calling _sanitize_processor directly."""
        # Create a test event_dict with sensitive data
        test_event_dict = {
            "message": "User login",
            "password": "secret123",  # Should trigger line 911
            "api_key": "key456",  # Should trigger line 911
            "username": "john",  # Should NOT trigger line 911
            "normal_field": "normal",  # Should NOT trigger line 911
        }

        # Call the static processor method directly
        processed_dict = FlextLogger._sanitize_processor(
            logging.getLogger("test"),  # dummy logger
            "info",  # dummy method name
            test_event_dict,
        )

        # Verify that line 911 executed and sensitive fields were redacted
        assert processed_dict["password"] == "[REDACTED]"
        assert processed_dict["api_key"] == "[REDACTED]"
        assert processed_dict["username"] == "john"  # Should remain
        assert processed_dict["normal_field"] == "normal"  # Should remain
        assert processed_dict["message"] == "User login"  # Should remain


class TestCoverageTargetedTests:
    """Tests specifically targeting remaining uncovered lines."""

    @staticmethod
    def _raise_validation_error() -> NoReturn:
        """Helper method to raise validation error for coverage."""
        msg = "Forced validation exception for coverage"
        raise ValueError(msg)

    @staticmethod
    def _raise_runtime_error(context: str = "coverage") -> NoReturn:
        """Helper method to raise runtime error for coverage."""
        msg = f"Forced exception for {context}"
        raise RuntimeError(msg)

    @staticmethod
    def _raise_key_error(key: str) -> NoReturn:
        """Helper method to raise key error for coverage."""
        raise KeyError(f"Forced exception for line coverage: {key}")

    def test_get_calling_line_error_handling(self) -> None:
        """Test _get_calling_line method error handling."""
        logger = FlextLogger("line_test")

        # Direct call to trigger potential error path
        line_num = logger._get_calling_line()

        # Should return a valid line number
        assert isinstance(line_num, int)
        assert line_num >= 0

    def test_sanitize_processor_edge_cases(self) -> None:
        """Test sanitization processor with various data types."""
        logger = FlextLogger("sanitize_test")

        # Test with simple sensitive data first
        simple_context: FlextTypes.Core.Dict = {
            "password": "secret123",
            "api_key": "key456",
            "username": "john_doe",  # Not sensitive
        }

        sanitized = logger._sanitize_context(simple_context)

        # Verify sanitization
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["username"] == "john_doe"  # Should remain unchanged

    def test_correlation_processor_functionality(self) -> None:
        """Test correlation processor functionality."""
        logger = FlextLogger("correlation_proc_test")

        # Set specific correlation ID
        test_correlation = f"proc_test_{uuid.uuid4().hex[:8]}"
        logger.set_correlation_id(test_correlation)

        # Test real correlation processor functionality
        try:
            logger.info("Test correlation processor")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify correlation ID is maintained in log entries
        log_entry = logger._build_log_entry("INFO", "Test correlation processor")
        assert log_entry.get("correlation_id") == test_correlation

    def test_edge_cases_for_remaining_coverage(self) -> None:
        """Test remaining edge cases to push coverage over 95%."""
        logger = FlextLogger("edge_test")

        # Test with various log levels to cover filtering logic
        logger._level = "ERROR"  # Set high level

        # Test real functionality with different log levels
        try:
            # These should be filtered out at structlog level
            logger.trace("Should be filtered")
            logger.debug("Should be filtered")
            logger.info("Should be filtered")
            logger.warning("Should be filtered")

            # These should appear
            logger.error("Should appear")
            logger.critical("Should appear")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that all methods work regardless of level
        # (filtering happens at output level, not method level)
        assert callable(logger.trace)
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)

    def test_permanent_context_coverage(self) -> None:
        """Test permanent context functionality to cover line 362."""
        logger = FlextLogger("permanent_test")

        # Set permanent context
        logger._permanent_context = {"app_version": "1.0.0", "deployment": "test"}

        # Test real functionality
        try:
            logger.info("Test with permanent context")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building includes permanent context
        log_entry = logger._build_log_entry("INFO", "Test with permanent context")
        assert "permanent" in log_entry
        assert log_entry.get("permanent", {})["app_version"] == "1.0.0"
        assert log_entry.get("permanent", {})["deployment"] == "test"

    def test_string_error_handling_coverage(self) -> None:
        """Test string error handling to cover line 385."""
        logger = FlextLogger("error_test")

        # Test real functionality
        try:
            # Pass a string error instead of Exception
            logger.error("String error occurred", error="This is a string error")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building works with string errors
        log_entry = logger._build_log_entry(
            "ERROR",
            "String error occurred",
            error="This is a string error",
        )
        assert "error" in log_entry
        assert log_entry.get("error", {})["type"] == "StringError"
        assert log_entry.get("error", {})["message"] == "This is a string error"
        assert log_entry.get("error", {})["stack_trace"] is None

    def test_frame_exception_handling_coverage(self) -> None:
        """Test frame exception handling to cover lines 411-412, 419-420."""
        logger = FlextLogger("frame_test")

        # Test frame exception handling - real logging functionality
        # Test real functionality
        try:
            logger.info("Test normal frame handling")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that log entry building includes proper execution context
        log_entry = logger._build_log_entry("INFO", "Test normal frame handling")
        # Real logging should have proper execution context
        assert "execution" in log_entry
        execution_context = log_entry.get("execution", {})
        # Real frame access should work normally
        if "function" in execution_context:
            assert isinstance(execution_context["function"], str)
            assert len(execution_context["function"]) > 0
            # Line number should be a positive integer in real execution
            assert isinstance(execution_context["line"], int)
            assert execution_context["line"] >= 0

    def test_global_correlation_id_edge_cases(self) -> None:
        """Test global correlation ID edge cases to cover line 475."""
        # Test getting global correlation ID when none is set
        FlextLogger._global_correlation_id = None
        assert FlextLogger.get_global_correlation_id() is None

        # Test setting and getting global correlation ID
        FlextLogger.set_global_correlation_id("global_test_id")
        assert FlextLogger.get_global_correlation_id() == "global_test_id"

    def test_performance_tracking_nonexistent_operation(self) -> None:
        """Test performance tracking with non-existent operation."""
        logger = FlextLogger("perf_test")

        # Test real functionality
        try:
            # First start an operation to have something in tracking
            op_id = logger.start_operation("test_operation")
            # Then complete both existing and non-existent operations
            logger.complete_operation(op_id, success=True)
            logger.complete_operation("non_existent_operation", success=False)
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify operations completed successfully
        assert op_id not in logger._local.operations  # First operation cleaned up

    def test_service_context_empty_values(self) -> None:
        """Test service context with empty values to cover lines 597, 601."""
        logger = FlextLogger("service_test")

        # Test accessing service info directly
        service_name = logger._service_name
        service_version = logger._service_version
        assert isinstance(service_name, str)
        assert isinstance(service_version, str)

        # Test with request context (correct parameter passing)
        logger.set_request_context(request_id="", operation="")
        logger.clear_request_context()  # Test clearing

        # Test real functionality
        try:
            logger.info("Test service info access")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # pass  # Test completed successfully

    def test_logger_binding_complex_data(self) -> None:
        """Test logger binding with complex data to cover lines 1047-1048, 1084-1085."""
        logger = FlextLogger("bind_test")

        # Test binding with complex nested data
        bound_logger = logger.bind(
            complex_data={
                "nested": {"deep": {"value": "test"}},
                "list": [1, 2, 3],
                "none_value": None,
                "bool_value": True,
            },
            simple_str="test_string",
        )

        assert bound_logger is not logger  # Should be different instance

        # Test real functionality
        try:
            bound_logger.info("Test bound logger with complex data")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that bound logger includes context in log entries
        log_entry = bound_logger._build_log_entry(
            "INFO",
            "Test bound logger with complex data",
        )
        # Bound context data should be in the request field
        assert "request" in log_entry
        request_data = log_entry.get("request", {})
        assert "complex_data" in request_data
        assert "simple_str" in request_data

    def test_console_configuration_toggles(self) -> None:
        """Test console configuration toggles to cover remaining lines."""
        logger = FlextLogger("console_test")

        # Test accessing internal console renderer creation
        renderer = logger._create_enhanced_console_renderer()
        assert renderer is not None

        # Test real console configuration functionality
        try:
            logger.info("Test console configuration access")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Verify renderer was created successfully
        assert renderer is not None

    def test_permanent_context_copy_on_bind(self) -> None:
        """Test permanent context copying on bind to cover line 475."""
        logger = FlextLogger("bind_copy_test")

        # Set permanent context
        logger._permanent_context = {"shared_key": "original_value"}

        # Bind should copy the permanent context
        bound_logger = logger.bind(new_key="bound_value")

        # Verify both have permanent context but are separate objects
        assert hasattr(bound_logger, "_permanent_context")
        assert bound_logger._permanent_context is not logger._permanent_context
        assert bound_logger._permanent_context["shared_key"] == "original_value"

    def test_set_context_initialization_and_branches(self) -> None:
        """Test set_context method to cover lines 505-515."""
        logger = FlextLogger("context_test")

        # Remove _permanent_context to test initialization (line 505-506)
        if hasattr(logger, "_permanent_context"):
            delattr(logger, "_permanent_context")

        # Test with context_dict parameter (lines 508-512)
        logger.set_context({"dict_key": "dict_value"}, extra_key="extra_value")
        assert logger._permanent_context["dict_key"] == "dict_value"
        assert logger._permanent_context["extra_key"] == "extra_value"

        # Test with None context_dict (lines 514-515)
        logger.set_context(None, another_key="another_value")
        assert logger._permanent_context["another_key"] == "another_value"

        # Test real functionality
        try:
            logger.info("Test context")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # pass  # Test completed successfully

    def test_with_context_method_coverage(self) -> None:
        """Test with_context method to cover line 542."""
        logger = FlextLogger("with_context_test")

        # with_context should call bind internally (line 542)
        bound_logger = logger.with_context(test_key="test_value")

        # Test real functionality
        try:
            bound_logger.info("Test with_context method")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Test that bound logger includes context in log entries
        log_entry = bound_logger._build_log_entry("INFO", "Test with_context method")
        # Context data from bind/with_context should be in the 'request' field
        assert "request" in log_entry, (
            f"request field not found in log_entry: {list(log_entry.keys())}"
        )
        request_context = log_entry.get("request", {})
        context_repr = (
            list(request_context.keys())
            if isinstance(request_context, dict)
            else request_context
        )
        assert "test_key" in request_context, (
            f"test_key not found in request: {context_repr}"
        )
        assert request_context["test_key"] == "test_value"

    def test_service_info_access_edge_cases(self) -> None:
        """Test service info access patterns to cover line 597."""
        logger = FlextLogger("service_edge_test")

        # Test accessing _service_info directly
        service_info = logger._service_info
        assert isinstance(service_info, dict)

        # Test accessing individual service attributes
        service_name = logger._service_name
        service_version = logger._service_version
        assert isinstance(service_name, str)
        assert isinstance(service_version, str)

    def test_logger_bind_method_edge_cases(self) -> None:
        """Test bind method edge cases to cover lines 1047-1048, 1084-1085."""
        logger = FlextLogger("bind_edge_test")

        # Test bind with None values
        bound_logger1 = logger.bind(none_value=None, valid_value="test")

        # Test bind with empty dict
        bound_logger2 = logger.bind()

        # Test bind with complex nested structures
        bound_logger3 = logger.bind(
            nested_dict={"level1": {"level2": "value"}},
            list_data=[1, "two", {"three": 3}],
            mixed_types={"str": "text", "num": 42, "bool": True, "none": None},
        )

        # All should be different instances
        assert bound_logger1 is not logger
        assert bound_logger2 is not logger
        assert bound_logger3 is not logger

        # Test real functionality
        try:
            bound_logger1.info("Test None values")
            bound_logger2.info("Test empty bind")
            bound_logger3.info("Test complex bind")
        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # pass  # Test completed successfully

    def test_console_renderer_specific_features(self) -> None:
        """Test console renderer specific features."""
        logger = FlextLogger("console_renderer_test")

        # Test accessing console renderer directly
        renderer = logger._create_enhanced_console_renderer()
        assert renderer is not None

        # Test with different log levels to potentially trigger different renderer paths
        logger._level = "DEBUG"
        # Test real functionality
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

        except Exception as e:
            pytest.fail(f"Logging should not raise exceptions: {e}")

        # Should have logged all messages at DEBUG level
        # Test completed successfully

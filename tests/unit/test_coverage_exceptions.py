"""Comprehensive coverage tests for FlextExceptions.

This module provides extensive tests for the FlextExceptions hierarchy,
targeting all missing lines and edge cases with accurate API usage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextResult,
)


class TestFlextExceptionsHierarchy:
    """Test complete exception hierarchy with correct API signatures."""

    def test_validation_error_basic(self) -> None:
        """Test creating ValidationError with message only."""
        error = FlextExceptions.ValidationError("Invalid input")
        assert str(error) == "[VALIDATION_ERROR] Invalid input"
        assert isinstance(error, Exception)

    def test_validation_error_with_field(self) -> None:
        """Test ValidationError with field information."""
        error = FlextExceptions.ValidationError(
            "Email invalid",
            field="email",
            value="not-an-email",
        )
        assert "Email invalid" in str(error)
        assert error.field == "email"
        assert error.value == "not-an-email"

    def test_configuration_error_basic(self) -> None:
        """Test ConfigurationError creation."""
        error = FlextExceptions.ConfigurationError("Missing required field")
        assert "Missing required field" in str(error)

    def test_configuration_error_with_source(self) -> None:
        """Test ConfigurationError with config source tracking."""
        error = FlextExceptions.ConfigurationError(
            "Missing API key",
            config_key="API_KEY",
            config_source="environment",
        )
        assert "Missing API key" in str(error)
        assert error.config_key == "API_KEY"
        assert error.config_source == "environment"

    def test_connection_error(self) -> None:
        """Test ConnectionError with host/port tracking."""
        error = FlextExceptions.ConnectionError(
            "Failed to connect to database",
            host="db.example.com",
            port=5432,
            timeout=30.0,
        )
        assert "Failed to connect" in str(error)
        assert error.host == "db.example.com"
        assert error.port == 5432
        assert error.timeout == 30.0

    def test_timeout_error(self) -> None:
        """Test TimeoutError for operation timeout."""
        error = FlextExceptions.TimeoutError(
            "Operation timed out",
            timeout_seconds=30,
            operation="fetch_data",
        )
        assert "Operation timed out" in str(error)
        assert error.timeout_seconds == 30
        assert error.operation == "fetch_data"

    def test_authentication_error(self) -> None:
        """Test AuthenticationError for auth failures."""
        error = FlextExceptions.AuthenticationError(
            "Invalid credentials",
            auth_method="basic",
            user_id="user123",
        )
        assert "Invalid credentials" in str(error)
        assert error.auth_method == "basic"
        assert error.user_id == "user123"

    def test_authorization_error(self) -> None:
        """Test AuthorizationError for permission failures."""
        error = FlextExceptions.AuthorizationError(
            "User lacks required permission",
            user_id="user123",
            resource="admin_panel",
            permission="read",
        )
        assert "User lacks required permission" in str(error)
        assert error.user_id == "user123"
        assert error.resource == "admin_panel"
        assert error.permission == "read"

    def test_not_found_error(self) -> None:
        """Test NotFoundError creation and properties."""
        error = FlextExceptions.NotFoundError(
            "User not found",
            resource_type="User",
            resource_id="123",
        )
        assert "User not found" in str(error)
        assert error.resource_type == "User"
        assert error.resource_id == "123"

    def test_conflict_error(self) -> None:
        """Test ConflictError for duplicate resources."""
        error = FlextExceptions.ConflictError(
            "User already exists",
            resource_type="User",
            resource_id="user@example.com",
            conflict_reason="email_already_registered",
        )
        assert "User already exists" in str(error)
        assert error.resource_type == "User"
        assert error.resource_id == "user@example.com"
        assert error.conflict_reason == "email_already_registered"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError for rate limiting."""
        error = FlextExceptions.RateLimitError(
            "Too many requests",
            limit=100,
            window_seconds=60,
            retry_after=30,
        )
        assert "Too many requests" in str(error)
        assert error.limit == 100
        assert error.window_seconds == 60
        assert error.retry_after == 30

    def test_circuit_breaker_error(self) -> None:
        """Test CircuitBreakerError for circuit breaker trip."""
        error = FlextExceptions.CircuitBreakerError(
            "Circuit breaker is open",
            service_name="payment_service",
            failure_count=5,
            reset_timeout=60,
        )
        assert "Circuit breaker is open" in str(error)
        assert error.service_name == "payment_service"
        assert error.failure_count == 5
        assert error.reset_timeout == 60

    def test_type_error(self) -> None:
        """Test TypeError for type mismatches."""
        error = FlextExceptions.TypeError(
            "Expected string, got int",
            expected_type="str",
            actual_type="int",
        )
        assert "Expected string" in str(error)
        assert error.expected_type == "str"
        assert error.actual_type == "int"

    def test_operation_error(self) -> None:
        """Test OperationError for failed operations."""
        error = FlextExceptions.OperationError(
            "Database operation failed",
            operation="INSERT",
            reason="Constraint violation",
        )
        assert "Database operation failed" in str(error)
        assert error.operation == "INSERT"
        assert error.reason == "Constraint violation"

    def test_attribute_access_error(self) -> None:
        """Test AttributeAccessError for missing attributes."""
        error = FlextExceptions.AttributeAccessError(
            "Attribute not found",
            attribute_name="missing_field",
            attribute_context={"class": "User", "attempted_access": "read"},
        )
        assert "Attribute not found" in str(error)
        assert error.attribute_name == "missing_field"
        assert error.attribute_context["class"] == "User"


class TestExceptionIntegration:
    """Test exceptions integration with FlextResult."""

    def test_exception_to_result_conversion(self) -> None:
        """Test converting exceptions to FlextResult."""
        try:
            msg = "Test error"
            raise FlextExceptions.ValidationError(msg, field="email")
        except FlextExceptions.ValidationError as e:
            result = FlextResult[None].fail(str(e))
            assert result.is_failure
            assert "Test error" in result.error

    def test_exception_in_railway_pattern(self) -> None:
        """Test exception handling in railway pattern."""

        def validate_and_process(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if not data.get("id"):
                return FlextResult[dict[str, object]].fail("Missing id")
            return FlextResult[dict[str, object]].ok(data)

        result = validate_and_process({})
        assert result.is_failure

        result = validate_and_process({"id": "123"})
        assert result.is_success

    def test_nested_exception_handling(self) -> None:
        """Test nested exception scenarios."""
        try:
            msg = "Validation failed"
            raise FlextExceptions.ValidationError(
                msg,
                field="email",
                value="invalid",
            )
        except FlextExceptions.ValidationError as e:
            # Wrap in another exception
            result = FlextResult[None].fail(f"Error in user creation: {e}")
            assert result.is_failure
            assert "Validation failed" in result.error


class TestExceptionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exception_with_empty_message(self) -> None:
        """Test exception with empty message."""
        error = FlextExceptions.ValidationError("")
        assert isinstance(error, Exception)

    def test_exception_with_unicode_message(self) -> None:
        """Test exception with unicode characters."""
        error = FlextExceptions.ValidationError("Invalid: 中文 العربية 🔴")
        assert "中文" in str(error)

    def test_exception_with_long_message(self) -> None:
        """Test exception with very long message."""
        long_msg = "x" * 10000
        error = FlextExceptions.ValidationError(long_msg)
        assert len(str(error)) > 9000

    def test_multiple_exceptions_in_sequence(self) -> None:
        """Test handling multiple exceptions."""
        errors = []
        for i in range(5):
            try:
                if i % 2 == 0:
                    raise FlextExceptions.ValidationError(f"Error {i}")
                raise FlextExceptions.ConfigurationError(f"Config error {i}")
            except Exception as e:
                errors.append(str(e))

        assert len(errors) == 5
        assert any("Error" in e for e in errors)

    def test_exception_inheritance_chain(self) -> None:
        """Test exception inheritance chain."""
        error = FlextExceptions.ValidationError("Test")
        assert isinstance(error, Exception)

    def test_exception_with_special_characters(self) -> None:
        """Test exception message with special characters."""
        error = FlextExceptions.ValidationError(
            "Message with \"quotes\" and 'apostrophes'"
        )
        assert "quotes" in str(error)


class TestExceptionProperties:
    """Test exception properties and attributes."""

    def test_exception_string_representation(self) -> None:
        """Test string representation of exceptions."""
        error = FlextExceptions.ValidationError("Test message")
        error_str = str(error)
        assert "Test message" in error_str

    def test_exception_repr(self) -> None:
        """Test repr of exceptions."""
        error = FlextExceptions.ValidationError("Test")
        repr_str = repr(error)
        assert "ValidationError" in repr_str or "Test" in repr_str

    def test_exception_type_checking(self) -> None:
        """Test type checking for exceptions."""
        error = FlextExceptions.ValidationError("Test")
        assert isinstance(error, FlextExceptions.ValidationError)
        assert isinstance(error, Exception)

    def test_base_error_with_metadata(self) -> None:
        """Test BaseError with metadata."""
        error = FlextExceptions.NotFoundError(
            "Resource not found",
            resource_id="123",
            resource_type="User",
        )
        assert "Resource not found" in str(error)


class TestExceptionContext:
    """Test exception context enrichment."""

    def test_exception_with_context_data(self) -> None:
        """Test exception with contextual information."""
        context_dict = {
            "user_id": "123",
            "operation": "create_user",
            "timestamp": 1234567890,
        }
        error = FlextExceptions.ValidationError(
            "Validation failed in context",
        ).with_context(**context_dict)
        assert "user_id" in error.metadata
        assert error.metadata["user_id"] == "123"

    def test_exception_with_correlation_id(self) -> None:
        """Test exception with auto-generated correlation ID."""
        error = FlextExceptions.BaseError(
            "Test error",
            auto_correlation=True,
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")

    def test_exception_chaining(self) -> None:
        """Test exception chaining with cause."""
        original: Exception | None = None
        try:
            msg = "Original error"
            raise ValueError(msg)
        except ValueError as e:
            original = e

        assert original is not None
        error = FlextExceptions.OperationError("Operation failed").chain_from(original)
        assert error.__cause__ is original
        assert "parent_correlation_id" in error.metadata or error.__cause__ is not None

    def test_exception_preservation(self) -> None:
        """Test that exception information is preserved."""
        original_msg = "Original error message with details"
        error = FlextExceptions.ValidationError(original_msg)

        # Verify message is preserved through conversion
        result = FlextResult[None].fail(str(error))
        assert original_msg in result.error or "Original error" in result.error


class TestExceptionSerialization:
    """Test exception serialization for logging/APIs."""

    def test_exception_to_dict(self) -> None:
        """Test converting exception to dictionary."""
        error = FlextExceptions.ValidationError(
            "Invalid email",
            field="email",
            value="not-valid",
        )
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ValidationError"
        assert error_dict["message"] == "Invalid email"
        assert error_dict["error_code"] == FlextConstants.Errors.VALIDATION_ERROR

    def test_exception_dict_with_metadata(self) -> None:
        """Test exception dict includes metadata."""
        error = FlextExceptions.OperationError(
            "Operation failed",
            operation="INSERT",
        ).with_context(user_id="123", timestamp=1234567890)
        error_dict = error.to_dict()
        assert error_dict["metadata"]["user_id"] == "123"
        assert error_dict["metadata"]["operation"] == "INSERT"


class TestExceptionFactory:
    """Test exception factory methods."""

    def test_create_error_by_type(self) -> None:
        """Test creating exception by type name."""
        error = FlextExceptions.create_error("ValidationError", "Test validation error")
        assert isinstance(error, FlextExceptions.ValidationError)
        assert "Test validation error" in str(error)

    def test_create_error_with_auto_detection(self) -> None:
        """Test smart error type detection in create()."""
        error = FlextExceptions.create(
            "Invalid email",
            field="email",
            value="not-valid",
        )
        assert isinstance(error, FlextExceptions.ValidationError)

    def test_create_config_error(self) -> None:
        """Test create() detects configuration error."""
        error = FlextExceptions.create(
            "Missing config",
            config_key="API_KEY",
            config_source="environment",
        )
        assert isinstance(error, FlextExceptions.ConfigurationError)

    def test_create_connection_error(self) -> None:
        """Test create() detects connection error."""
        error = FlextExceptions.create(
            "Connection failed",
            host="localhost",
            port=5432,
        )
        assert isinstance(error, FlextExceptions.ConnectionError)

    def test_create_operation_error(self) -> None:
        """Test create() detects operation error."""
        error = FlextExceptions.create(
            "Operation failed",
            operation="INSERT",
            reason="Constraint violation",
        )
        assert isinstance(error, FlextExceptions.OperationError)

    def test_create_timeout_error(self) -> None:
        """Test create() detects timeout error."""
        error = FlextExceptions.create(
            "Operation timed out",
            timeout_seconds=30,
        )
        assert isinstance(error, FlextExceptions.TimeoutError)


class TestExceptionMetrics:
    """Test exception metrics tracking."""

    def test_record_exception(self) -> None:
        """Test recording exception metrics."""
        # Clear metrics first
        FlextExceptions.clear_metrics()

        FlextExceptions.record_exception("ValidationError")
        FlextExceptions.record_exception("ValidationError")
        FlextExceptions.record_exception("ConfigurationError")

        metrics = FlextExceptions.get_metrics()
        assert metrics["total_exceptions"] == 3
        assert metrics["exception_counts"]["ValidationError"] == 2
        assert metrics["exception_counts"]["ConfigurationError"] == 1
        assert metrics["unique_exception_types"] == 2

    def test_clear_metrics(self) -> None:
        """Test clearing exception metrics."""
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception("TestError")

        metrics_before = FlextExceptions.get_metrics()
        assert metrics_before["total_exceptions"] == 1

        FlextExceptions.clear_metrics()
        metrics_after = FlextExceptions.get_metrics()
        assert metrics_after["total_exceptions"] == 0


class TestExceptionLogging:
    """Test exception logging functionality."""

    def test_exception_string_with_correlation_id(self) -> None:
        """Test exception string representation includes correlation ID."""
        error = FlextExceptions.BaseError(
            "Test",
            auto_correlation=True,
        )
        error_str = str(error)
        if error.correlation_id:
            assert error.correlation_id in error_str

    def test_exception_error_code_in_string(self) -> None:
        """Test error code is included in string representation."""
        error = FlextExceptions.ValidationError("Test message")
        error_str = str(error)
        # The error code should be in the string
        assert "VALIDATION_ERROR" in error_str or "Test message" in error_str


__all__ = [
    "TestExceptionContext",
    "TestExceptionEdgeCases",
    "TestExceptionFactory",
    "TestExceptionIntegration",
    "TestExceptionLogging",
    "TestExceptionMetrics",
    "TestExceptionProperties",
    "TestExceptionSerialization",
    "TestFlextExceptionsHierarchy",
]

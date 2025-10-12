"""Comprehensive tests for FlextCore.Exceptions - Exception Type Definitions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

import pytest

from flext_core import FlextCore


class TestFlextExceptions:
    """Test suite for FlextCore.Exceptions exception types."""

    def test_base_error_initialization(self) -> None:
        """Test BaseError initialization with basic message."""
        error = FlextCore.Exceptions.BaseError("Test error")
        assert error.message == "Test error"
        assert error.error_code == "UNKNOWN_ERROR"  # Default error code
        assert error.correlation_id is None
        assert isinstance(error.metadata, dict)
        assert isinstance(error.timestamp, float)

    def test_base_error_with_error_code(self) -> None:
        """Test BaseError with error code."""
        error = FlextCore.Exceptions.BaseError("Test error", error_code="TEST_001")
        assert error.error_code == "TEST_001"
        assert str(error) == "[TEST_001] Test error"

    def test_base_error_with_correlation_id(self) -> None:
        """Test BaseError with correlation ID."""
        error = FlextCore.Exceptions.BaseError("Test error", correlation_id="corr-123")
        assert error.correlation_id == "corr-123"

    def test_base_error_with_metadata(self) -> None:
        """Test BaseError with metadata."""
        metadata: FlextCore.Types.Dict = {"field": "email", "value": "invalid"}
        error = FlextCore.Exceptions.BaseError("Test error", metadata=metadata)
        assert error.metadata["field"] == "email"
        assert error.metadata["value"] == "invalid"

    def test_base_error_with_extra_kwargs(self) -> None:
        """Test BaseError with extra keyword arguments."""
        error = FlextCore.Exceptions.BaseError(
            "Test error", field="email", value="invalid"
        )
        assert error.metadata["field"] == "email"
        assert error.metadata["value"] == "invalid"

    def test_base_error_to_dict(self) -> None:
        """Test BaseError to_dict conversion."""
        error = FlextCore.Exceptions.BaseError(
            "Test error",
            error_code="TEST_001",
            correlation_id="corr-123",
            field="email",
        )
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "BaseError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["correlation_id"] == "corr-123"
        assert "timestamp" in error_dict
        metadata = error_dict["metadata"]
        if isinstance(metadata, dict):
            assert metadata["field"] == "email"

    def test_base_error_str_without_code(self) -> None:
        """Test BaseError string representation with default error code."""
        error = FlextCore.Exceptions.BaseError("Test error")
        assert str(error) == "[UNKNOWN_ERROR] Test error"  # Default error code is shown

    def test_base_error_str_with_code(self) -> None:
        """Test BaseError string representation with error code."""
        error = FlextCore.Exceptions.BaseError("Test error", error_code="TEST_001")
        assert str(error) == "[TEST_001] Test error"

    def test_validation_error(self) -> None:
        """Test ValidationError exception type."""
        error = FlextCore.Exceptions.ValidationError(
            "Invalid email", field="email", error_code="VAL_EMAIL"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Invalid email"
        assert error.error_code == "VAL_EMAIL"
        assert error.field == "email"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError exception type."""
        error = FlextCore.Exceptions.ConfigurationError(
            "Missing config", config_key="database.host"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Missing config"
        assert error.config_key == "database.host"
        assert error.error_code is not None

    def test_connection_error(self) -> None:
        """Test ConnectionError exception type."""
        error = FlextCore.Exceptions.ConnectionError(
            "Connection failed", host="localhost", port=5432
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Connection failed"
        assert error.host == "localhost"
        assert error.port == 5432

    def test_timeout_error(self) -> None:
        """Test TimeoutError exception type."""
        error = FlextCore.Exceptions.TimeoutError(
            "Operation timeout", timeout_seconds=30.0, operation="database_query"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Operation timeout"
        assert error.timeout_seconds == 30.0
        assert error.operation == "database_query"

    def test_authentication_error(self) -> None:
        """Test AuthenticationError exception type."""
        error = FlextCore.Exceptions.AuthenticationError(
            "Invalid credentials", user_id="testuser", auth_method="password"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Invalid credentials"
        assert error.user_id == "testuser"
        assert error.auth_method == "password"

    def test_authorization_error(self) -> None:
        """Test AuthorizationError exception type."""
        error = FlextCore.Exceptions.AuthorizationError(
            "Access denied",
            user_id="user123",
            resource="REDACTED_LDAP_BIND_PASSWORD_panel",
            permission="read",
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Access denied"
        assert error.user_id == "user123"
        assert error.resource == "REDACTED_LDAP_BIND_PASSWORD_panel"
        assert error.permission == "read"

    def test_not_found_error(self) -> None:
        """Test NotFoundError exception type."""
        error = FlextCore.Exceptions.NotFoundError(
            "Resource not found", resource_type="User", resource_id="123"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Resource not found"
        assert error.resource_type == "User"
        assert error.resource_id == "123"

    def test_conflict_error(self) -> None:
        """Test ConflictError exception type."""
        error = FlextCore.Exceptions.ConflictError(
            "Resource conflict",
            resource_id="user_123",
            conflict_reason="duplicate_email",
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Resource conflict"
        assert error.resource_id == "user_123"
        assert error.conflict_reason == "duplicate_email"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError exception type."""
        error = FlextCore.Exceptions.RateLimitError(
            "Rate limit exceeded", limit=100, window_seconds=60
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Rate limit exceeded"
        assert error.limit == 100
        assert error.window_seconds == 60

    def test_circuit_breaker_error(self) -> None:
        """Test CircuitBreakerError exception type."""
        error = FlextCore.Exceptions.CircuitBreakerError(
            "Circuit breaker open", service_name="payment_service", failure_count=5
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Circuit breaker open"
        assert error.service_name == "payment_service"
        assert error.failure_count == 5

    def test_type_error(self) -> None:
        """Test TypeError exception type."""
        error = FlextCore.Exceptions.TypeError(
            "Invalid type", expected_type="str", actual_type="int"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Invalid type"
        assert error.expected_type == "str"
        assert error.actual_type == "int"

    def test_operation_error(self) -> None:
        """Test OperationError exception type."""
        error = FlextCore.Exceptions.OperationError(
            "Operation failed", operation="backup", reason="disk_full"
        )
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert error.message == "Operation failed"
        assert error.operation == "backup"
        assert error.reason == "disk_full"

    def test_create_error_validation(self) -> None:
        """Test create_error factory for ValidationError."""
        error = FlextCore.Exceptions.create_error("ValidationError", "Test error")
        assert isinstance(error, FlextCore.Exceptions.ValidationError)
        assert error.message == "Test error"

    def test_create_error_configuration(self) -> None:
        """Test create_error factory for ConfigurationError."""
        error = FlextCore.Exceptions.create_error("ConfigurationError", "Config error")
        assert isinstance(error, FlextCore.Exceptions.ConfigurationError)
        assert error.message == "Config error"

    def test_create_error_connection(self) -> None:
        """Test create_error factory for ConnectionError."""
        error = FlextCore.Exceptions.create_error("ConnectionError", "Conn error")
        assert isinstance(error, FlextCore.Exceptions.ConnectionError)
        assert error.message == "Conn error"

    def test_create_error_timeout(self) -> None:
        """Test create_error factory for TimeoutError."""
        error = FlextCore.Exceptions.create_error("TimeoutError", "Timeout error")
        assert isinstance(error, FlextCore.Exceptions.TimeoutError)
        assert error.message == "Timeout error"

    def test_create_error_authentication(self) -> None:
        """Test create_error factory for AuthenticationError."""
        error = FlextCore.Exceptions.create_error("AuthenticationError", "Auth error")
        assert isinstance(error, FlextCore.Exceptions.AuthenticationError)
        assert error.message == "Auth error"

    def test_create_error_authorization(self) -> None:
        """Test create_error factory for AuthorizationError."""
        error = FlextCore.Exceptions.create_error("AuthorizationError", "Authz error")
        assert isinstance(error, FlextCore.Exceptions.AuthorizationError)
        assert error.message == "Authz error"

    def test_create_error_not_found(self) -> None:
        """Test create_error factory for NotFoundError."""
        error = FlextCore.Exceptions.create_error("NotFoundError", "Not found error")
        assert isinstance(error, FlextCore.Exceptions.NotFoundError)
        assert error.message == "Not found error"

    def test_create_error_conflict(self) -> None:
        """Test create_error factory for ConflictError."""
        error = FlextCore.Exceptions.create_error("ConflictError", "Conflict error")
        assert isinstance(error, FlextCore.Exceptions.ConflictError)
        assert error.message == "Conflict error"

    def test_create_error_rate_limit(self) -> None:
        """Test create_error factory for RateLimitError."""
        error = FlextCore.Exceptions.create_error("RateLimitError", "Rate limit error")
        assert isinstance(error, FlextCore.Exceptions.RateLimitError)
        assert error.message == "Rate limit error"

    def test_create_error_circuit_breaker(self) -> None:
        """Test create_error factory for CircuitBreakerError."""
        error = FlextCore.Exceptions.create_error(
            "CircuitBreakerError", "Circuit breaker error"
        )
        assert isinstance(error, FlextCore.Exceptions.CircuitBreakerError)
        assert error.message == "Circuit breaker error"

    def test_create_error_type(self) -> None:
        """Test create_error factory for TypeError."""
        error = FlextCore.Exceptions.create_error("TypeError", "Type error")
        assert isinstance(error, FlextCore.Exceptions.TypeError)
        assert error.message == "Type error"

    def test_create_error_operation(self) -> None:
        """Test create_error factory for OperationError."""
        error = FlextCore.Exceptions.create_error("OperationError", "Operation error")
        assert isinstance(error, FlextCore.Exceptions.OperationError)
        assert error.message == "Operation error"

    def test_create_error_invalid_type(self) -> None:
        """Test create_error with invalid error type."""
        with pytest.raises(ValueError, match="Unknown error type"):
            FlextCore.Exceptions.create_error("InvalidError", "Test error")

    def test_exception_raising(self) -> None:
        """Test that exceptions can be raised and caught."""
        error_msg = "Test error"
        with pytest.raises(FlextCore.Exceptions.ValidationError) as exc_info:
            raise FlextCore.Exceptions.ValidationError(error_msg, error_code="TEST_001")
        assert exc_info.value.message == "Test error"
        assert exc_info.value.error_code == "TEST_001"

    def test_exception_inheritance(self) -> None:
        """Test exception inheritance from BaseError."""
        error = FlextCore.Exceptions.ValidationError("Test error")
        assert isinstance(error, FlextCore.Exceptions.BaseError)
        assert isinstance(error, Exception)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = FlextCore.Exceptions.BaseError("Test error")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata: FlextCore.Types.Dict = {"existing": "value"}
        error = FlextCore.Exceptions.BaseError(
            "Test error", metadata=metadata, new_field="new_value"
        )
        assert error.metadata["existing"] == "value"
        assert error.metadata["new_field"] == "new_value"


class TestFlextExceptionsComprehensive:
    """Comprehensive test suite for all FlextCore.Exceptions classes."""

    def test_all_exception_instantiation(self) -> None:
        """Test instantiation of all exception types."""
        from flext_core import FlextCore

        # Test BaseError
        exc = FlextCore.Exceptions.BaseError("Base error message")
        assert str(exc) == "[UNKNOWN_ERROR] Base error message"
        assert exc.error_code == "UNKNOWN_ERROR"

        # Test ConfigurationError
        exc = FlextCore.Exceptions.ConfigurationError("Config error")
        assert "Config error" in str(exc)

        # Test ValidationError
        exc = FlextCore.Exceptions.ValidationError("Validation failed")
        assert "Validation failed" in str(exc)

        # Test NotFoundError
        exc = FlextCore.Exceptions.NotFoundError("Resource not found")
        assert "Resource not found" in str(exc)

        # Test ConflictError
        exc = FlextCore.Exceptions.ConflictError("Resource conflict")
        assert "Resource conflict" in str(exc)

        # Test AuthenticationError
        exc = FlextCore.Exceptions.AuthenticationError("Auth failed")
        assert "Auth failed" in str(exc)

        # Test AuthorizationError
        exc = FlextCore.Exceptions.AuthorizationError("Unauthorized")
        assert "Unauthorized" in str(exc)

        # Test TimeoutError
        exc = FlextCore.Exceptions.TimeoutError("Operation timed out")
        assert "Operation timed out" in str(exc)

        # Test ConnectionError
        exc = FlextCore.Exceptions.ConnectionError("Connection failed")
        assert "Connection failed" in str(exc)

        # Test RateLimitError
        exc = FlextCore.Exceptions.RateLimitError("Rate limit exceeded")
        assert "Rate limit exceeded" in str(exc)

        # Test CircuitBreakerError
        exc = FlextCore.Exceptions.CircuitBreakerError("Circuit open")
        assert "Circuit open" in str(exc)

        # Test AttributeAccessError
        exc = FlextCore.Exceptions.AttributeAccessError("Attribute error")
        assert "Attribute error" in str(exc)

        # Test OperationError
        exc = FlextCore.Exceptions.OperationError("Operation failed")
        assert "Operation failed" in str(exc)

        # Test TypeError
        exc = FlextCore.Exceptions.TypeError("Type mismatch")
        assert "Type mismatch" in str(exc)

    def test_exception_with_error_codes(self) -> None:
        """Test exceptions with custom error codes."""
        from flext_core import FlextCore

        validation_exc = FlextCore.Exceptions.ValidationError(
            "Validation error", error_code="VALIDATION_FAILED"
        )
        assert validation_exc.error_code == "VALIDATION_FAILED"

        not_found_exc = FlextCore.Exceptions.NotFoundError(
            "Not found", error_code="RESOURCE_NOT_FOUND"
        )
        assert not_found_exc.error_code == "RESOURCE_NOT_FOUND"

        timeout_exc = FlextCore.Exceptions.TimeoutError(
            "Timeout", error_code="TIMEOUT_EXCEEDED"
        )
        assert timeout_exc.error_code == "TIMEOUT_EXCEEDED"

    def test_exception_with_metadata(self) -> None:
        """Test exceptions with additional metadata."""
        from flext_core import FlextCore

        metadata: FlextCore.Types.Dict = {"resource_id": "123", "user_id": "456"}
        not_found_exc = FlextCore.Exceptions.NotFoundError(
            "Resource not found", metadata=metadata
        )
        assert not_found_exc.metadata == metadata

        validation_exc = FlextCore.Exceptions.ValidationError(
            "Invalid", metadata={"field": "email"}
        )
        assert validation_exc.metadata["field"] == "email"

    def test_exception_inheritance(self) -> None:
        """Test exception class inheritance hierarchy."""
        from flext_core import FlextCore

        # All exceptions should inherit from BaseError
        assert issubclass(
            FlextCore.Exceptions.ValidationError, FlextCore.Exceptions.BaseError
        )
        assert issubclass(
            FlextCore.Exceptions.NotFoundError, FlextCore.Exceptions.BaseError
        )
        assert issubclass(
            FlextCore.Exceptions.AuthenticationError, FlextCore.Exceptions.BaseError
        )
        assert issubclass(
            FlextCore.Exceptions.TimeoutError, FlextCore.Exceptions.BaseError
        )

        # And from Python Exception
        assert issubclass(FlextCore.Exceptions.BaseError, Exception)

    def test_exception_raising_and_catching(self) -> None:
        """Test raising and catching exceptions."""
        from flext_core import FlextCore

        # Test ValidationError
        error_msg = "Invalid input"
        with pytest.raises(FlextCore.Exceptions.ValidationError, match=error_msg):
            raise FlextCore.Exceptions.ValidationError(error_msg)

        # Test NotFoundError
        error_msg = "User not found"
        with pytest.raises(FlextCore.Exceptions.NotFoundError, match=error_msg):
            raise FlextCore.Exceptions.NotFoundError(error_msg)

        # Test AuthenticationError
        error_msg = "Invalid credentials"
        with pytest.raises(FlextCore.Exceptions.AuthenticationError, match=error_msg):
            raise FlextCore.Exceptions.AuthenticationError(error_msg)

    def test_exception_context_propagation(self) -> None:
        """Test exception context propagation."""
        from flext_core import FlextCore

        error_msg = "Config error"
        operation_error_msg = "Operation failed"

        # Test that OperationError properly chains from ConfigurationError
        with pytest.raises(FlextCore.Exceptions.OperationError) as exc_info:
            raise FlextCore.Exceptions.OperationError(
                operation_error_msg
            ) from FlextCore.Exceptions.ConfigurationError(error_msg)

        assert exc_info.value.__cause__ is not None
        assert isinstance(
            exc_info.value.__cause__, FlextCore.Exceptions.ConfigurationError
        )

    def test_exception_with_various_data_types(self) -> None:
        """Test exceptions with various metadata data types."""
        from flext_core import FlextCore

        # String metadata
        exc = FlextCore.Exceptions.BaseError("Error", metadata={"msg": "test"})
        assert exc.metadata["msg"] == "test"

        # Numeric metadata
        exc = FlextCore.Exceptions.BaseError("Error", metadata={"count": 42})
        assert exc.metadata["count"] == 42

        # List metadata
        exc = FlextCore.Exceptions.BaseError("Error", metadata={"items": [1, 2, 3]})
        assert exc.metadata["items"] == [1, 2, 3]

        # Nested dict metadata
        exc = FlextCore.Exceptions.BaseError(
            "Error", metadata={"nested": {"key": "val"}}
        )
        nested_dict = exc.metadata["nested"]
        assert isinstance(nested_dict, dict) and nested_dict["key"] == "val"

    def test_all_exception_repr(self) -> None:
        """Test repr() for all exception types."""
        from flext_core import FlextCore

        exceptions_to_test: list[FlextCore.Exceptions.BaseError] = [
            FlextCore.Exceptions.BaseError("base"),
            FlextCore.Exceptions.ValidationError("validation"),
            FlextCore.Exceptions.NotFoundError("not_found"),
            FlextCore.Exceptions.ConflictError("conflict"),
            FlextCore.Exceptions.AuthenticationError("auth"),
            FlextCore.Exceptions.AuthorizationError("authz"),
            FlextCore.Exceptions.TimeoutError("timeout"),
            FlextCore.Exceptions.ConnectionError("connection"),
            FlextCore.Exceptions.RateLimitError("rate_limit"),
            FlextCore.Exceptions.CircuitBreakerError("circuit"),
            FlextCore.Exceptions.ConfigurationError("config"),
            FlextCore.Exceptions.AttributeAccessError("attribute"),
            FlextCore.Exceptions.OperationError("operation"),
            FlextCore.Exceptions.TypeError("type"),
        ]

        for exc in exceptions_to_test:
            repr_str = repr(exc)
            assert repr_str is not None
            assert len(repr_str) > 0

    def test_exception_equality(self) -> None:
        """Test exception equality comparison."""
        from flext_core import FlextCore

        exc1 = FlextCore.Exceptions.ValidationError("Error")
        exc2 = FlextCore.Exceptions.ValidationError("Error")

        # Exceptions are equal by type and message
        assert type(exc1) is type(exc2)
        assert str(exc1) == str(exc2)

    def test_exception_serialization(self) -> None:
        """Test exception serialization to dict."""
        from flext_core import FlextCore

        exc = FlextCore.Exceptions.ValidationError(
            "Validation failed",
            error_code="INVALID_INPUT",
            metadata={"field": "email", "value": "invalid"},
        )

        # Test if exception has serialization method
        if hasattr(exc, "to_dict"):
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error_code" in result or "message" in result

    def test_all_exception_types_creation(self) -> None:
        """Test creating all exception types with full parameters."""
        from flext_core import FlextCore

        test_cases: list[tuple[type[FlextCore.Exceptions.BaseError], str]] = [
            (FlextCore.Exceptions.BaseError, "Base"),
            (FlextCore.Exceptions.ValidationError, "Validation"),
            (FlextCore.Exceptions.NotFoundError, "NotFound"),
            (FlextCore.Exceptions.ConflictError, "Conflict"),
            (FlextCore.Exceptions.AuthenticationError, "Authentication"),
            (FlextCore.Exceptions.AuthorizationError, "Authorization"),
            (FlextCore.Exceptions.TimeoutError, "Timeout"),
            (FlextCore.Exceptions.ConnectionError, "Connection"),
            (FlextCore.Exceptions.RateLimitError, "RateLimit"),
            (FlextCore.Exceptions.CircuitBreakerError, "CircuitBreaker"),
            (FlextCore.Exceptions.ConfigurationError, "Configuration"),
            (FlextCore.Exceptions.AttributeAccessError, "AttributeAccess"),
            (FlextCore.Exceptions.OperationError, "Operation"),
            (FlextCore.Exceptions.TypeError, "Type"),
        ]

        for exc_class, name in test_cases:
            # Test with just message
            exc: FlextCore.Exceptions.BaseError = exc_class(f"{name} error")
            assert f"{name} error" in str(exc)

            # Test with error_code
            exc = exc_class(f"{name} error", error_code=f"{name.upper()}_ERROR")
            assert exc.error_code == f"{name.upper()}_ERROR"

            # Test with metadata
            exc = exc_class(f"{name} error", metadata={"test": "data"})
            # Metadata should contain at least the passed data
            assert "test" in exc.metadata
            assert exc.metadata["test"] == "data"

            # Test with all parameters
            exc = exc_class(
                f"{name} error",
                error_code=f"{name.upper()}_CODE",
                metadata={"key": "value"},
            )
            assert f"{name} error" in str(exc)
            assert exc.error_code == f"{name.upper()}_CODE"
            assert exc.metadata["key"] == "value"

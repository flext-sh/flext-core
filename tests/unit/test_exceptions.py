"""Comprehensive tests for FlextExceptions - Exception Type Definitions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

import pytest

from flext_core import FlextExceptions


class TestFlextExceptions:
    """Test suite for FlextExceptions exception types."""

    def test_base_error_initialization(self) -> None:
        """Test BaseError initialization with basic message."""
        error = FlextExceptions.BaseError("Test error")
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.correlation_id is None
        assert isinstance(error.metadata, dict)
        assert isinstance(error.timestamp, float)

    def test_base_error_with_error_code(self) -> None:
        """Test BaseError with error code."""
        error = FlextExceptions.BaseError("Test error", error_code="TEST_001")
        assert error.error_code == "TEST_001"
        assert str(error) == "[TEST_001] Test error"

    def test_base_error_with_correlation_id(self) -> None:
        """Test BaseError with correlation ID."""
        error = FlextExceptions.BaseError("Test error", correlation_id="corr-123")
        assert error.correlation_id == "corr-123"

    def test_base_error_with_metadata(self) -> None:
        """Test BaseError with metadata."""
        metadata = {"field": "email", "value": "invalid"}
        error = FlextExceptions.BaseError("Test error", metadata=metadata)
        assert error.metadata["field"] == "email"
        assert error.metadata["value"] == "invalid"

    def test_base_error_with_extra_kwargs(self) -> None:
        """Test BaseError with extra keyword arguments."""
        error = FlextExceptions.BaseError("Test error", field="email", value="invalid")
        assert error.metadata["field"] == "email"
        assert error.metadata["value"] == "invalid"

    def test_base_error_to_dict(self) -> None:
        """Test BaseError to_dict conversion."""
        error = FlextExceptions.BaseError(
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
        assert error_dict["metadata"]["field"] == "email"

    def test_base_error_str_without_code(self) -> None:
        """Test BaseError string representation without error code."""
        error = FlextExceptions.BaseError("Test error")
        assert str(error) == "Test error"

    def test_base_error_str_with_code(self) -> None:
        """Test BaseError string representation with error code."""
        error = FlextExceptions.BaseError("Test error", error_code="TEST_001")
        assert str(error) == "[TEST_001] Test error"

    def test_validation_error(self) -> None:
        """Test ValidationError exception type."""
        error = FlextExceptions.ValidationError(
            "Invalid email", field="email", error_code="VAL_EMAIL"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Invalid email"
        assert error.error_code == "VAL_EMAIL"
        assert error.field == "email"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError exception type."""
        error = FlextExceptions.ConfigurationError(
            "Missing config", config_key="database.host"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Missing config"
        assert error.config_key == "database.host"
        assert error.error_code is not None

    def test_connection_error(self) -> None:
        """Test ConnectionError exception type."""
        error = FlextExceptions.ConnectionError(
            "Connection failed", host="localhost", port=5432
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Connection failed"
        assert error.host == "localhost"
        assert error.port == 5432

    def test_timeout_error(self) -> None:
        """Test TimeoutError exception type."""
        error = FlextExceptions.TimeoutError(
            "Operation timeout", timeout_seconds=30.0, operation="database_query"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Operation timeout"
        assert error.timeout_seconds == 30.0
        assert error.operation == "database_query"

    def test_authentication_error(self) -> None:
        """Test AuthenticationError exception type."""
        error = FlextExceptions.AuthenticationError(
            "Invalid credentials", user_id="testuser", auth_method="password"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Invalid credentials"
        assert error.user_id == "testuser"
        assert error.auth_method == "password"

    def test_authorization_error(self) -> None:
        """Test AuthorizationError exception type."""
        error = FlextExceptions.AuthorizationError(
            "Access denied",
            user_id="user123",
            resource="REDACTED_LDAP_BIND_PASSWORD_panel",
            permission="read",
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Access denied"
        assert error.user_id == "user123"
        assert error.resource == "REDACTED_LDAP_BIND_PASSWORD_panel"
        assert error.permission == "read"

    def test_not_found_error(self) -> None:
        """Test NotFoundError exception type."""
        error = FlextExceptions.NotFoundError(
            "Resource not found", resource_type="User", resource_id="123"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Resource not found"
        assert error.resource_type == "User"
        assert error.resource_id == "123"

    def test_conflict_error(self) -> None:
        """Test ConflictError exception type."""
        error = FlextExceptions.ConflictError(
            "Resource conflict",
            resource_id="user_123",
            conflict_reason="duplicate_email",
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Resource conflict"
        assert error.resource_id == "user_123"
        assert error.conflict_reason == "duplicate_email"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError exception type."""
        error = FlextExceptions.RateLimitError(
            "Rate limit exceeded", limit=100, window_seconds=60.0
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Rate limit exceeded"
        assert error.limit == 100
        assert error.window_seconds == 60.0

    def test_circuit_breaker_error(self) -> None:
        """Test CircuitBreakerError exception type."""
        error = FlextExceptions.CircuitBreakerError(
            "Circuit breaker open", service_name="payment_service", failure_count=5
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Circuit breaker open"
        assert error.service_name == "payment_service"
        assert error.failure_count == 5

    def test_type_error(self) -> None:
        """Test TypeError exception type."""
        error = FlextExceptions.TypeError(
            "Invalid type", expected_type="str", actual_type="int"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Invalid type"
        assert error.expected_type == "str"
        assert error.actual_type == "int"

    def test_operation_error(self) -> None:
        """Test OperationError exception type."""
        error = FlextExceptions.OperationError(
            "Operation failed", operation="backup", reason="disk_full"
        )
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Operation failed"
        assert error.operation == "backup"
        assert error.reason == "disk_full"

    def test_create_error_validation(self) -> None:
        """Test create_error factory for ValidationError."""
        error = FlextExceptions.create_error("ValidationError", "Test error")
        assert isinstance(error, FlextExceptions.ValidationError)
        assert error.message == "Test error"

    def test_create_error_configuration(self) -> None:
        """Test create_error factory for ConfigurationError."""
        error = FlextExceptions.create_error("ConfigurationError", "Config error")
        assert isinstance(error, FlextExceptions.ConfigurationError)
        assert error.message == "Config error"

    def test_create_error_connection(self) -> None:
        """Test create_error factory for ConnectionError."""
        error = FlextExceptions.create_error("ConnectionError", "Conn error")
        assert isinstance(error, FlextExceptions.ConnectionError)
        assert error.message == "Conn error"

    def test_create_error_timeout(self) -> None:
        """Test create_error factory for TimeoutError."""
        error = FlextExceptions.create_error("TimeoutError", "Timeout error")
        assert isinstance(error, FlextExceptions.TimeoutError)
        assert error.message == "Timeout error"

    def test_create_error_authentication(self) -> None:
        """Test create_error factory for AuthenticationError."""
        error = FlextExceptions.create_error("AuthenticationError", "Auth error")
        assert isinstance(error, FlextExceptions.AuthenticationError)
        assert error.message == "Auth error"

    def test_create_error_authorization(self) -> None:
        """Test create_error factory for AuthorizationError."""
        error = FlextExceptions.create_error("AuthorizationError", "Authz error")
        assert isinstance(error, FlextExceptions.AuthorizationError)
        assert error.message == "Authz error"

    def test_create_error_not_found(self) -> None:
        """Test create_error factory for NotFoundError."""
        error = FlextExceptions.create_error("NotFoundError", "Not found error")
        assert isinstance(error, FlextExceptions.NotFoundError)
        assert error.message == "Not found error"

    def test_create_error_conflict(self) -> None:
        """Test create_error factory for ConflictError."""
        error = FlextExceptions.create_error("ConflictError", "Conflict error")
        assert isinstance(error, FlextExceptions.ConflictError)
        assert error.message == "Conflict error"

    def test_create_error_rate_limit(self) -> None:
        """Test create_error factory for RateLimitError."""
        error = FlextExceptions.create_error("RateLimitError", "Rate limit error")
        assert isinstance(error, FlextExceptions.RateLimitError)
        assert error.message == "Rate limit error"

    def test_create_error_circuit_breaker(self) -> None:
        """Test create_error factory for CircuitBreakerError."""
        error = FlextExceptions.create_error(
            "CircuitBreakerError", "Circuit breaker error"
        )
        assert isinstance(error, FlextExceptions.CircuitBreakerError)
        assert error.message == "Circuit breaker error"

    def test_create_error_type(self) -> None:
        """Test create_error factory for TypeError."""
        error = FlextExceptions.create_error("TypeError", "Type error")
        assert isinstance(error, FlextExceptions.TypeError)
        assert error.message == "Type error"

    def test_create_error_operation(self) -> None:
        """Test create_error factory for OperationError."""
        error = FlextExceptions.create_error("OperationError", "Operation error")
        assert isinstance(error, FlextExceptions.OperationError)
        assert error.message == "Operation error"

    def test_create_error_invalid_type(self) -> None:
        """Test create_error with invalid error type."""
        with pytest.raises(ValueError, match="Unknown error type"):
            FlextExceptions.create_error("InvalidError", "Test error")

    def test_exception_raising(self) -> None:
        """Test that exceptions can be raised and caught."""
        error_msg = "Test error"
        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            raise FlextExceptions.ValidationError(error_msg, error_code="TEST_001")
        assert exc_info.value.message == "Test error"
        assert exc_info.value.error_code == "TEST_001"

    def test_exception_inheritance(self) -> None:
        """Test exception inheritance from BaseError."""
        error = FlextExceptions.ValidationError("Test error")
        assert isinstance(error, FlextExceptions.BaseError)
        assert isinstance(error, Exception)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = FlextExceptions.BaseError("Test error")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata = {"existing": "value"}
        error = FlextExceptions.BaseError(
            "Test error", metadata=metadata, new_field="new_value"
        )
        assert error.metadata["existing"] == "value"
        assert error.metadata["new_field"] == "new_value"

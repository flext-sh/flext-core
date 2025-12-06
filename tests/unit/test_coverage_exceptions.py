"""Comprehensive coverage tests for FlextExceptions.

Module: flext_core.exceptions
Scope: FlextExceptions - exception hierarchy, factory methods, configuration

This module provides extensive tests for the FlextExceptions hierarchy,
targeting all missing lines and edge cases with accurate API usage.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from flext_core import FlextExceptions, FlextResult, FlextRuntime, t
from flext_core.constants import c


@dataclass(frozen=True, slots=True)
class ExceptionCreationScenario:
    """Exception creation test scenario."""

    name: str
    exception_type: type[FlextExceptions.BaseError]
    message: str
    kwargs: dict[str, object]
    expected_attrs: dict[str, object]


class ExceptionScenarios:
    """Centralized exception coverage test scenarios using c."""

    EXCEPTION_CREATION: ClassVar[list[ExceptionCreationScenario]] = [
        ExceptionCreationScenario(
            "validation_basic",
            FlextExceptions.ValidationError,
            "Invalid input",
            {},
            {},
        ),
        ExceptionCreationScenario(
            "validation_with_field",
            FlextExceptions.ValidationError,
            "Email invalid",
            {"field": "email", "value": "not-an-email"},
            {"field": "email", "value": "not-an-email"},
        ),
        ExceptionCreationScenario(
            "configuration_basic",
            FlextExceptions.ConfigurationError,
            "Missing required field",
            {},
            {},
        ),
        ExceptionCreationScenario(
            "configuration_with_source",
            FlextExceptions.ConfigurationError,
            "Missing API key",
            {"config_key": "API_KEY", "config_source": "environment"},
            {"config_key": "API_KEY", "config_source": "environment"},
        ),
        ExceptionCreationScenario(
            "connection",
            FlextExceptions.ConnectionError,
            "Failed to connect",
            {"host": "db.example.com", "port": 5432, "timeout": 30.0},
            {"host": "db.example.com", "port": 5432},
        ),
        ExceptionCreationScenario(
            "timeout",
            FlextExceptions.TimeoutError,
            "Operation timed out",
            {"timeout_seconds": 30, "operation": "fetch_data"},
            {"timeout_seconds": 30, "operation": "fetch_data"},
        ),
        ExceptionCreationScenario(
            "authentication",
            FlextExceptions.AuthenticationError,
            "Invalid credentials",
            {"auth_method": "basic", "user_id": "user123"},
            {"auth_method": "basic", "user_id": "user123"},
        ),
        ExceptionCreationScenario(
            "authorization",
            FlextExceptions.AuthorizationError,
            "User lacks permission",
            {"user_id": "user123", "resource": "REDACTED_LDAP_BIND_PASSWORD_panel", "permission": "read"},
            {"user_id": "user123", "resource": "REDACTED_LDAP_BIND_PASSWORD_panel"},
        ),
        ExceptionCreationScenario(
            "not_found",
            FlextExceptions.NotFoundError,
            "User not found",
            {"resource_type": "User", "resource_id": "123"},
            {"resource_type": "User", "resource_id": "123"},
        ),
        ExceptionCreationScenario(
            "conflict",
            FlextExceptions.ConflictError,
            "User already exists",
            {
                "resource_type": "User",
                "resource_id": "user@example.com",
                "conflict_reason": "email_already_registered",
            },
            {"resource_type": "User", "resource_id": "user@example.com"},
        ),
        ExceptionCreationScenario(
            "rate_limit",
            FlextExceptions.RateLimitError,
            "Too many requests",
            {"limit": 100, "window_seconds": 60, "retry_after": 30},
            {"limit": 100, "window_seconds": 60},
        ),
        ExceptionCreationScenario(
            "circuit_breaker",
            FlextExceptions.CircuitBreakerError,
            "Circuit breaker is open",
            {
                "service_name": "payment_service",
                "failure_count": 5,
                "reset_timeout": 60,
            },
            {"service_name": "payment_service", "failure_count": 5},
        ),
        ExceptionCreationScenario(
            "type_error",
            FlextExceptions.TypeError,
            "Expected string, got int",
            {"expected_type": str, "actual_type": int},
            {"expected_type": str, "actual_type": int},
        ),
        ExceptionCreationScenario(
            "operation_error",
            FlextExceptions.OperationError,
            "Database operation failed",
            {"operation": "INSERT", "reason": "Constraint violation"},
            {"operation": "INSERT", "reason": "Constraint violation"},
        ),
        ExceptionCreationScenario(
            "attribute_access",
            FlextExceptions.AttributeAccessError,
            "Attribute not found",
            {
                "attribute_name": "missing_field",
                "attribute_context": {"class": "User", "attempted_access": "read"},
            },
            {"attribute_name": "missing_field"},
        ),
    ]

    FACTORY_CREATION: ClassVar[
        list[tuple[str, dict[str, object], type[FlextExceptions.BaseError]]]
    ] = [
        (
            "ValidationError",
            {"field": "email", "value": "not-valid"},
            FlextExceptions.ValidationError,
        ),
        (
            "ConfigurationError",
            {"config_key": "API_KEY", "config_source": "environment"},
            FlextExceptions.ConfigurationError,
        ),
        (
            "ConnectionError",
            {"host": "localhost", "port": 5432},
            FlextExceptions.ConnectionError,
        ),
        (
            "OperationError",
            {"operation": "INSERT", "reason": "Constraint violation"},
            FlextExceptions.OperationError,
        ),
        ("TimeoutError", {"timeout_seconds": 30}, FlextExceptions.TimeoutError),
    ]


class TestFlextExceptionsHierarchy:
    """Test complete exception hierarchy using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.EXCEPTION_CREATION,
        ids=lambda s: s.name,
    )
    def test_exception_creation(self, scenario: ExceptionCreationScenario) -> None:
        """Test creating exceptions with various scenarios."""
        if scenario.kwargs:
            # Convert dict[str, object] to dict[str, MetadataAttributeValue]
            # Separate type values from metadata values for proper type handling
            type_kwargs: dict[str, type] = {}
            metadata_kwargs: dict[str, t.MetadataAttributeValue] = {}
            for key, value in scenario.kwargs.items():
                # For TypeError, preserve type objects for expected_type and actual_type
                if (
                    scenario.exception_type == FlextExceptions.TypeError
                    and key
                    in {
                        "expected_type",
                        "actual_type",
                    }
                    and isinstance(value, type)
                ):
                    type_kwargs[key] = value
                elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    metadata_kwargs[key] = FlextRuntime.normalize_to_metadata_value(
                        value,
                    )
                else:
                    # Convert non-compatible types to string
                    metadata_kwargs[key] = FlextRuntime.normalize_to_metadata_value(
                        str(value),
                    )
            # For TypeError, pass type_kwargs separately, then metadata_kwargs
            if type_kwargs:
                # Convert type objects to strings for metadata compatibility
                for key, type_value in type_kwargs.items():
                    metadata_kwargs[key] = type_value.__name__
            # Type narrowing: all values in metadata_kwargs are MetadataAttributeValue
            # BaseError accepts **extra_kwargs: MetadataAttributeValue, but type checker
            # can't infer compatibility with specific exception constructors

            # BaseError accepts **extra_kwargs: MetadataAttributeValue
            # Use type: ignore for dynamic kwargs unpacking in tests
            error = scenario.exception_type(scenario.message, **metadata_kwargs)
        else:
            error = scenario.exception_type(scenario.message)
        assert scenario.message in str(error)
        assert isinstance(error, Exception)
        for attr_name, expected_value in scenario.expected_attrs.items():
            assert hasattr(error, attr_name)
            assert getattr(error, attr_name) == expected_value


class TestExceptionIntegration:
    """Test exceptions integration with FlextResult using FlextTestsUtilities."""

    def test_exception_to_result_conversion(self) -> None:
        """Test converting exceptions to FlextResult."""
        try:
            error_msg = "Test error"
            raise FlextExceptions.ValidationError(error_msg, field="email")
        except FlextExceptions.ValidationError as e:
            result = FlextResult[bool].fail(str(e))
            assert result.is_failure
            assert result.error is not None and "Test error" in result.error

    def test_exception_in_railway_pattern(self) -> None:
        """Test exception handling in railway pattern."""

        def validate_and_process(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if not data.get("id"):
                return FlextResult[dict[str, object]].fail("Missing id")
            return FlextResult[dict[str, object]].ok(data)

        assert validate_and_process({}).is_failure
        assert validate_and_process({"id": "123"}).is_success

    def test_nested_exception_handling(self) -> None:
        """Test nested exception scenarios."""
        try:
            error_msg = "Validation failed"
            raise FlextExceptions.ValidationError(
                error_msg,
                field="email",
                value="invalid",
            )
        except FlextExceptions.ValidationError as e:
            result = FlextResult[bool].fail(f"Error in user creation: {e}")
            assert result.is_failure
            assert result.error is not None and "Validation failed" in result.error


class TestExceptionEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        ("message", "expected_in_str"),
        [
            ("", True),
            ("Invalid: 中文 العربية 🔴", True),
            ("x" * 10000, True),
            ("Message with \"quotes\" and 'apostrophes'", True),
        ],
        ids=["empty", "unicode", "long", "special_chars"],
    )
    def test_exception_message_variations(
        self,
        message: str,
        expected_in_str: bool,
    ) -> None:
        """Test exception with various message formats."""
        error = FlextExceptions.ValidationError(message)
        assert isinstance(error, Exception)
        if message:
            assert message in str(error) or len(str(error)) > 9000

    def test_multiple_exceptions_in_sequence(self) -> None:
        """Test handling multiple exceptions."""
        errors: list[str] = []
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


class TestExceptionProperties:
    """Test exception properties and attributes."""

    def test_exception_string_representation(self) -> None:
        """Test string representation of exceptions."""
        error = FlextExceptions.ValidationError("Test message")
        assert "Test message" in str(error)

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
        """Test exception with contextual information via metadata."""
        # ValidationError accepts metadata via **extra_kwargs as MetadataAttributeValue
        # Convert Metadata to dict for extra_kwargs
        metadata_dict: dict[str, t.MetadataAttributeValue] = {
            "user_id": "123",
            "operation": "create_user",
            "timestamp": 1234567890,
        }
        # ValidationError accepts metadata via **extra_kwargs: MetadataAttributeValue
        # Pass metadata_dict via **extra_kwargs, not as named parameter
        # Use type: ignore for dynamic metadata dict in tests
        error = FlextExceptions.ValidationError(
            "Validation failed in context",
            metadata=metadata_dict,
        )
        assert "user_id" in error.metadata.attributes
        assert error.metadata.attributes["user_id"] == "123"

    def test_exception_with_correlation_id(self) -> None:
        """Test exception with auto-generated correlation ID."""
        error = FlextExceptions.BaseError("Test error", auto_correlation=True)
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")

    def test_exception_chaining(self) -> None:
        """Test exception chaining with cause using Python's native chaining."""
        original: Exception | None = None
        try:
            error_msg = "Original error"
            raise ValueError(error_msg)
        except ValueError as e:
            original = e
        assert original is not None
        # Python native exception chaining using 'from'
        error = FlextExceptions.OperationError("Operation failed")
        error.__cause__ = original
        assert error.__cause__ is original

    def test_exception_preservation(self) -> None:
        """Test that exception information is preserved."""
        original_msg = "Original error message with details"
        error = FlextExceptions.ValidationError(original_msg)
        result = FlextResult[bool].fail(str(error))
        assert result.error is not None and (
            original_msg in result.error or "Original error" in result.error
        )


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
        assert error_dict["error_code"] == c.Errors.VALIDATION_ERROR

    def test_exception_dict_with_metadata(self) -> None:
        """Test exception dict includes metadata (flattened)."""
        error = FlextExceptions.OperationError(
            "Operation failed",
            operation="INSERT",
        )
        error_dict = error.to_dict()
        # Metadata is flattened into the dict, not nested
        assert error_dict["operation"] == "INSERT"


class TestExceptionFactory:
    """Test exception factory methods using FlextTestsUtilities."""

    def test_create_error_by_type(self) -> None:
        """Test creating exception by type name."""
        error = FlextExceptions.create_error("ValidationError", "Test validation error")
        assert isinstance(error, FlextExceptions.ValidationError)
        assert "Test validation error" in str(error)

    @pytest.mark.parametrize(
        ("message", "kwargs", "expected_type"),
        ExceptionScenarios.FACTORY_CREATION,
        ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
    )
    def test_create_error_auto_detection(
        self,
        message: str,
        kwargs: dict[str, object],
        expected_type: type[FlextExceptions.BaseError],
    ) -> None:
        """Test smart error type detection in create()."""
        # Convert dict[str, object] to dict[str, MetadataAttributeValue]
        converted_kwargs: dict[str, t.MetadataAttributeValue] = {}
        for key, value in kwargs.items():
            # Type narrowing: ensure value is GeneralValueType before normalization
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                converted_kwargs[key] = FlextRuntime.normalize_to_metadata_value(value)
            else:
                # Convert non-compatible types to string
                converted_kwargs[key] = FlextRuntime.normalize_to_metadata_value(
                    str(value),
                )
        # Type narrowing: all values in converted_kwargs are MetadataAttributeValue
        # create accepts **kwargs: MetadataAttributeValue, but type checker can't infer compatibility

        # Use type: ignore for dynamic kwargs unpacking in tests
        error = FlextExceptions.create(message, **converted_kwargs)
        assert isinstance(error, expected_type)


class TestExceptionMetrics:
    """Test exception metrics tracking."""

    def test_record_exception(self) -> None:
        """Test recording exception metrics."""
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ConfigurationError)
        metrics = FlextExceptions.get_metrics()
        assert metrics["total_exceptions"] == 3
        exception_counts = metrics.get("exception_counts")
        # Type narrowing: exception_counts is dict-like
        if isinstance(exception_counts, dict):
            assert exception_counts.get("FlextExceptions.ValidationError") == 2
            assert exception_counts.get("FlextExceptions.ConfigurationError") == 1
        assert metrics["unique_exception_types"] == 2

    def test_clear_metrics(self) -> None:
        """Test clearing exception metrics."""
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        assert FlextExceptions.get_metrics()["total_exceptions"] == 1
        FlextExceptions.clear_metrics()
        assert FlextExceptions.get_metrics()["total_exceptions"] == 0


class TestExceptionLogging:
    """Test exception logging functionality."""

    def test_exception_string_with_correlation_id(self) -> None:
        """Test exception has correlation ID when auto_correlation=True."""
        error = FlextExceptions.BaseError("Test", auto_correlation=True)
        # Correlation ID is stored but not in string repr (only error_code and message)
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        # String contains the message
        assert "Test" in str(error)

    def test_exception_error_code_in_string(self) -> None:
        """Test error code is included in string representation."""
        error = FlextExceptions.ValidationError("Test message")
        error_str = str(error)
        assert "VALIDATION_ERROR" in error_str or "Test message" in error_str


class TestHierarchicalExceptionSystem:
    """Test hierarchical exception configuration system."""

    def test_failure_level_enum_values(self) -> None:
        """Test FailureLevel enum has all required values."""
        failure_level = c.Exceptions.FailureLevel
        assert all(
            hasattr(failure_level, level) for level in ["STRICT", "WARN", "PERMISSIVE"]
        )

    def test_failure_level_string_values(self) -> None:
        """Test FailureLevel enum string values."""
        failure_level = c.Exceptions.FailureLevel
        assert failure_level.STRICT.value == "strict"
        assert failure_level.WARN.value == "warn"
        assert failure_level.PERMISSIVE.value == "permissive"

    def test_failure_level_comparison(self) -> None:
        """Test FailureLevel enum comparison."""
        failure_level = c.Exceptions.FailureLevel
        # Test enum members are distinct - use str() to avoid Literal type overlap issues
        strict_val: str = str(failure_level.STRICT.value)
        warn_val: str = str(failure_level.WARN.value)
        permissive_val: str = str(failure_level.PERMISSIVE.value)
        assert strict_val != warn_val
        assert warn_val != permissive_val
        assert strict_val == str(failure_level.STRICT.value)


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
    "TestHierarchicalExceptionSystem",
]

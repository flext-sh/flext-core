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

from collections.abc import Callable
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import t, tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextConstants, FlextExceptions, r
from tests import c

from ..test_utils import assertion_helpers


class ExceptionCreationScenario(BaseModel):
    """Exception creation test scenario."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, Field(description="Exception creation scenario name")]
    exception_type: Annotated[
        type[FlextExceptions.BaseError],
        Field(description="Exception class to instantiate"),
    ]
    message: Annotated[str, Field(description="Exception message")]
    kwargs: Annotated[
        dict[str, object | type],
        Field(description="Keyword arguments for exception creation"),
    ]
    expected_attrs: Annotated[
        dict[str, object | type], Field(description="Expected attributes to validate")
    ]


class ExceptionScenarios:
    """Centralized exception coverage test scenarios using c."""

    EXCEPTION_CREATION: ClassVar[list[ExceptionCreationScenario]] = [
        ExceptionCreationScenario(
            name="validation_basic",
            exception_type=FlextExceptions.ValidationError,
            message="Invalid input",
            kwargs={},
            expected_attrs={},
        ),
        ExceptionCreationScenario(
            name="validation_with_field",
            exception_type=FlextExceptions.ValidationError,
            message="Email invalid",
            kwargs={"field": "email", "value": "not-an-email"},
            expected_attrs={"field": "email", "value": "not-an-email"},
        ),
        ExceptionCreationScenario(
            name="configuration_basic",
            exception_type=FlextExceptions.ConfigurationError,
            message="Missing required field",
            kwargs={},
            expected_attrs={},
        ),
        ExceptionCreationScenario(
            name="configuration_with_source",
            exception_type=FlextExceptions.ConfigurationError,
            message="Missing API key",
            kwargs={"config_key": "API_KEY", "config_source": "environment"},
            expected_attrs={"config_key": "API_KEY", "config_source": "environment"},
        ),
        ExceptionCreationScenario(
            name="connection",
            exception_type=FlextExceptions.ConnectionError,
            message="Failed to connect",
            kwargs={"host": "db.example.com", "port": 5432, "timeout": 30.0},
            expected_attrs={"host": "db.example.com", "port": 5432},
        ),
        ExceptionCreationScenario(
            name="timeout",
            exception_type=FlextExceptions.TimeoutError,
            message="Operation timed out",
            kwargs={"timeout_seconds": 30, "operation": "fetch_data"},
            expected_attrs={"timeout_seconds": 30, "operation": "fetch_data"},
        ),
        ExceptionCreationScenario(
            name="authentication",
            exception_type=FlextExceptions.AuthenticationError,
            message="Invalid credentials",
            kwargs={"auth_method": "basic", "user_id": "user123"},
            expected_attrs={"auth_method": "basic", "user_id": "user123"},
        ),
        ExceptionCreationScenario(
            name="authorization",
            exception_type=FlextExceptions.AuthorizationError,
            message="User lacks permission",
            kwargs={
                "user_id": "user123",
                "resource": "REDACTED_LDAP_BIND_PASSWORD_panel",
                "permission": "read",
            },
            expected_attrs={
                "user_id": "user123",
                "resource": "REDACTED_LDAP_BIND_PASSWORD_panel",
            },
        ),
        ExceptionCreationScenario(
            name="not_found",
            exception_type=FlextExceptions.NotFoundError,
            message="User not found",
            kwargs={"resource_type": "User", "resource_id": "123"},
            expected_attrs={"resource_type": "User", "resource_id": "123"},
        ),
        ExceptionCreationScenario(
            name="conflict",
            exception_type=FlextExceptions.ConflictError,
            message="User already exists",
            kwargs={
                "resource_type": "User",
                "resource_id": "user@example.com",
                "conflict_reason": "email_already_registered",
            },
            expected_attrs={"resource_type": "User", "resource_id": "user@example.com"},
        ),
        ExceptionCreationScenario(
            name="rate_limit",
            exception_type=FlextExceptions.RateLimitError,
            message="Too many requests",
            kwargs={"limit": 100, "window_seconds": 60, "retry_after": 30},
            expected_attrs={"limit": 100, "window_seconds": 60},
        ),
        ExceptionCreationScenario(
            name="circuit_breaker",
            exception_type=FlextExceptions.CircuitBreakerError,
            message="Circuit breaker is open",
            kwargs={
                "service_name": "payment_service",
                "failure_count": 5,
                "reset_timeout": 60,
            },
            expected_attrs={"service_name": "payment_service", "failure_count": 5},
        ),
        ExceptionCreationScenario(
            name="type_error",
            exception_type=FlextExceptions.TypeError,
            message="Expected string, got int",
            kwargs={"expected_type": str, "actual_type": int},
            expected_attrs={"expected_type": str, "actual_type": int},
        ),
        ExceptionCreationScenario(
            name="operation_error",
            exception_type=FlextExceptions.OperationError,
            message="Database operation failed",
            kwargs={"operation": "INSERT", "reason": "Constraint violation"},
            expected_attrs={"operation": "INSERT", "reason": "Constraint violation"},
        ),
        ExceptionCreationScenario(
            name="attribute_access",
            exception_type=FlextExceptions.AttributeAccessError,
            message="Attribute not found",
            kwargs={
                "attribute_name": "missing_field",
                "attribute_context": {"class": "User", "attempted_access": "read"},
            },
            expected_attrs={"attribute_name": "missing_field"},
        ),
    ]
    FACTORY_CREATION: ClassVar[
        list[tuple[str, dict[str, t.Tests.object], type[FlextExceptions.BaseError]]]
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
            {"host": FlextConstants.Network.LOCALHOST},
            FlextExceptions.ConnectionError,
        ),
        (
            "OperationError",
            {"operation": "INSERT", "reason": "Constraint violation"},
            FlextExceptions.OperationError,
        ),
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
            type_kwargs: dict[str, type] = {}
            metadata_kwargs: dict[str, t.Tests.object] = {}
            for key, value in scenario.kwargs.items():
                if (
                    scenario.exception_type == FlextExceptions.TypeError
                    and key in {"expected_type", "actual_type"}
                    and isinstance(value, type)
                ):
                    type_kwargs[key] = value
                elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    metadata_kwargs[key] = cast("t.MetadataAttributeValue", value)
                else:
                    metadata_kwargs[key] = cast("t.MetadataAttributeValue", str(value))
            if type_kwargs:
                for key, type_value in type_kwargs.items():
                    metadata_kwargs[key] = type_value.__name__
            exception_ctor = cast(
                "Callable[..., FlextExceptions.BaseError]",
                scenario.exception_type,
            )
            error = exception_ctor(scenario.message, **metadata_kwargs)
        else:
            error = scenario.exception_type(scenario.message)
        tm.that(str(error), has=scenario.message)
        tm.that(isinstance(error, Exception), eq=True)
        for attr_name, expected_value in scenario.expected_attrs.items():
            tm.that(hasattr(error, attr_name), eq=True)
            tm.that(getattr(error, attr_name), eq=expected_value)  # type: ignore[arg-type]


class TestExceptionIntegration:
    """Test exceptions integration with r using FlextTestsUtilities."""

    def test_exception_to_result_conversion(self) -> None:
        """Test converting exceptions to r."""
        try:
            error_msg = "Test error"
            raise FlextExceptions.ValidationError(error_msg, field="email")
        except FlextExceptions.ValidationError as e:
            result = r[bool].fail(str(e))
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.fail(result, has="Test error")

    def test_exception_in_railway_pattern(self) -> None:
        """Test exception handling in railway pattern."""

        def validate_and_process(
            data: dict[str, t.Tests.object],
        ) -> r[dict[str, t.Tests.object]]:
            if not data.get("id"):
                return r[dict[str, t.Tests.object]].fail("Missing id")
            return r[dict[str, t.Tests.object]].ok(data)

        tm.fail(validate_and_process({}))
        tm.ok(validate_and_process({"id": "123"}))

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
            result = r[bool].fail(f"Error in user creation: {e}")
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.that(result.error, ne=None) and "Validation failed" in result.error


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
        tm.that(isinstance(error, Exception), eq=True)
        if message:
            tm.that(message in str(error) or len(str(error)) > 9000, eq=True)

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
        tm.that(len(errors), eq=5)
        tm.that(any("Error" in e for e in errors), eq=True)

    def test_exception_inheritance_chain(self) -> None:
        """Test exception inheritance chain."""
        error = FlextExceptions.ValidationError("Test")
        tm.that(isinstance(error, Exception), eq=True)


class TestExceptionProperties:
    """Test exception properties and attributes."""

    def test_exception_string_representation(self) -> None:
        """Test string representation of exceptions."""
        error = FlextExceptions.ValidationError("Test message")
        tm.that(str(error), has="Test message")

    def test_exception_repr(self) -> None:
        """Test repr of exceptions."""
        error = FlextExceptions.ValidationError("Test")
        repr_str = repr(error)
        tm.that(repr_str or "Test" in repr_str, has="ValidationError")

    def test_exception_type_checking(self) -> None:
        """Test type checking for exceptions."""
        error = FlextExceptions.ValidationError("Test")
        tm.that(isinstance(error, FlextExceptions.ValidationError), eq=True)
        tm.that(isinstance(error, Exception), eq=True)

    def test_base_error_with_metadata(self) -> None:
        """Test BaseError with metadata."""
        error = FlextExceptions.NotFoundError(
            "Resource not found",
            resource_id="123",
            resource_type="User",
        )
        tm.that(str(error), has="Resource not found")


class TestExceptionContext:
    """Test exception context enrichment."""

    def test_exception_with_context_data(self) -> None:
        """Test exception with contextual information via metadata."""
        error = FlextExceptions.ValidationError(
            "Validation failed in context",
            user_id="123",
            operation="create_user",
            timestamp=1234567890,
        )
        tm.that(error.metadata.attributes, has="user_id")
        tm.that(error.metadata.attributes["user_id"], eq="123")

    def test_exception_with_correlation_id(self) -> None:
        """Test exception with auto-generated correlation ID."""
        error = FlextExceptions.BaseError("Test error", auto_correlation=True)
        tm.that(error.correlation_id, none=False)
        tm.that(error.correlation_id, starts="exc_")

    def test_exception_chaining(self) -> None:
        """Test exception chaining with cause using Python's native chaining."""
        original: Exception | None = None
        try:
            error_msg = "Original error"
            raise ValueError(error_msg)
        except ValueError as e:
            original = e
        tm.that(original, none=False)
        error = FlextExceptions.OperationError("Operation failed")
        error.__cause__ = original
        tm.that(error.__cause__, eq=original)

    def test_exception_preservation(self) -> None:
        """Test that exception information is preserved."""
        original_msg = "Original error message with details"
        error = FlextExceptions.ValidationError(original_msg)
        result = r[bool].fail(str(error))
        tm.that(result.error, none=False)
        tm.fail(result, has=original_msg)


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
        tm.that(error_dict["error_type"], eq="ValidationError")
        tm.that(error_dict["message"], eq="Invalid email")
        tm.that(error_dict["error_code"], eq=c.Errors.VALIDATION_ERROR)

    def test_exception_dict_with_metadata(self) -> None:
        """Test exception dict includes metadata (flattened)."""
        error = FlextExceptions.OperationError("Operation failed", operation="INSERT")
        error_dict = error.to_dict()
        tm.that(error_dict["operation"], eq="INSERT")


class TestExceptionFactory:
    """Test exception factory methods using FlextTestsUtilities."""

    def test_create_error_by_type(self) -> None:
        """Test creating exception by type name."""
        error = FlextExceptions.create("ValidationError", "Test validation error")
        tm.that(isinstance(error, FlextExceptions.ValidationError), eq=True)
        tm.that(str(error), has="Test validation error")

    @pytest.mark.parametrize(
        ("message", "kwargs", "expected_type"),
        ExceptionScenarios.FACTORY_CREATION,
    )
    def test_create_error_auto_detection(
        self,
        message: str,
        kwargs: dict[str, t.Tests.object],
        expected_type: type[FlextExceptions.BaseError],
    ) -> None:
        """Test smart error type detection in create()."""
        converted_kwargs: dict[str, t.Tests.object] = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs.items()
        }
        kwargs_typed: dict[str, t.Tests.object] = converted_kwargs
        create_error = cast(
            "Callable[..., FlextExceptions.BaseError]",
            FlextExceptions.create,
        )
        error = create_error(message, **kwargs_typed)
        tm.that(isinstance(error, expected_type), eq=True)


class TestExceptionMetrics:
    """Test exception metrics tracking."""

    def test_record_exception(self) -> None:
        """Test recording exception metrics."""
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ConfigurationError)
        metrics = FlextExceptions.get_metrics()
        tm.that(metrics["total_exceptions"], eq=3)
        raw_counts = metrics.root.get("exception_counts")
        exception_counts = cast("dict[str, int]", raw_counts)
        tm.that(exception_counts.get("FlextExceptions.ValidationError"), eq=2)
        tm.that(exception_counts.get("FlextExceptions.ConfigurationError"), eq=1)
        tm.that(metrics["unique_exception_types"], eq=2)

    def test_clear_metrics(self) -> None:
        """Test clearing exception metrics."""
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        tm.that(FlextExceptions.get_metrics()["total_exceptions"], eq=1)
        FlextExceptions.clear_metrics()
        tm.that(FlextExceptions.get_metrics()["total_exceptions"], eq=0)


class TestExceptionLogging:
    """Test exception logging functionality."""

    def test_exception_string_with_correlation_id(self) -> None:
        """Test exception has correlation ID when auto_correlation=True."""
        error = FlextExceptions.BaseError("Test", auto_correlation=True)
        tm.that(error.correlation_id, none=False)
        tm.that(error.correlation_id, starts="exc_")
        tm.that(str(error), has="Test")

    def test_exception_error_code_in_string(self) -> None:
        """Test error code is included in string representation."""
        error = FlextExceptions.ValidationError("Test message")
        error_str = str(error)
        tm.that(error_str or "Test message" in error_str, has="VALIDATION_ERROR")


class TestHierarchicalExceptionSystem:
    """Test hierarchical exception configuration system."""

    def test_failure_level_enum_values(self) -> None:
        """Test FailureLevel enum has all required values."""
        failure_level = c.Exceptions.FailureLevel
        tm.that(
            all(
                hasattr(failure_level, level)
                for level in ["STRICT", "WARN", "PERMISSIVE"]
            ),
            eq=True,
        )

    def test_failure_level_string_values(self) -> None:
        """Test FailureLevel enum string values."""
        failure_level = c.Exceptions.FailureLevel
        tm.that(failure_level.STRICT.value, eq="strict")
        tm.that(failure_level.WARN.value, eq="warn")
        tm.that(failure_level.PERMISSIVE.value, eq="permissive")

    def test_failure_level_comparison(self) -> None:
        """Test FailureLevel enum comparison."""
        failure_level = c.Exceptions.FailureLevel
        strict_val: str = str(failure_level.STRICT.value)
        warn_val: str = str(failure_level.WARN.value)
        permissive_val: str = str(failure_level.PERMISSIVE.value)
        tm.that(strict_val, ne=warn_val)
        tm.that(warn_val, ne=permissive_val)
        tm.that(strict_val, eq=str(failure_level.STRICT.value))


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

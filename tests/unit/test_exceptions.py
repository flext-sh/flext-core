"""Tests for FlextExceptions - Exception Type Definitions and Implementations.

Module: flext_core.exceptions
Scope: FlextExceptions - all exception types and factory methods

Tests FlextExceptions functionality including:
- BaseError initialization and configuration
- Exception type hierarchy (ValidationError, ConfigurationError, etc.)
- Exception with metadata, correlation IDs, error codes
- Exception factory methods (create_error, create)
- Exception serialization and string representation
- Exception chaining and context propagation

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import FlextExceptions
from flext_core._models.metadata import Metadata


class ExceptionScenarioType(StrEnum):
    """Exception test scenario types for organization."""

    BASE_ERROR = "base_error"
    WITH_CODE = "with_code"
    WITH_CORRELATION = "with_correlation"
    WITH_METADATA = "with_metadata"
    WITH_EXTRA_KWARGS = "with_extra_kwargs"
    TO_DICT = "to_dict"
    STRING_REPRESENTATION = "string_representation"
    SPECIFIC_TYPE = "specific_type"
    FACTORY_METHOD = "factory_method"
    FACTORY_INVALID = "factory_invalid"
    EXCEPTION_RAISING = "exception_raising"
    EXCEPTION_CHAINING = "exception_chaining"


class ExceptionTypeScenarioType(StrEnum):
    """Exception type testing scenarios."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    TYPE_ERROR = "type_error"
    OPERATION = "operation"


@dataclass(frozen=True, slots=True)
class ExceptionScenario:
    """Exception test scenario definition."""

    name: str
    scenario_type: ExceptionScenarioType
    exception_type: type[FlextExceptions.BaseError] | None = None
    should_raise: bool = False
    error_factory_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTypeScenario:
    """Exception type instantiation test scenario."""

    name: str
    scenario_type: ExceptionTypeScenarioType
    exception_class: type[FlextExceptions.BaseError]


class ExceptionTestHelpers:
    """Generalized helpers for exception testing."""

    @staticmethod
    def create_metadata_object(attributes: dict[str, object] | None = None) -> Metadata:
        """Create Metadata object from attributes dict."""
        return Metadata(attributes=attributes or {})

    @staticmethod
    def get_exception_class_by_type(error_type: str) -> type[FlextExceptions.BaseError]:
        """Get exception class by type name."""
        return getattr(FlextExceptions, error_type)


class ExceptionScenarios:
    """Centralized exception test scenarios using FlextConstants."""

    BASE_SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            "base_error_init",
            ExceptionScenarioType.BASE_ERROR,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_code",
            ExceptionScenarioType.WITH_CODE,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_correlation",
            ExceptionScenarioType.WITH_CORRELATION,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_metadata",
            ExceptionScenarioType.WITH_METADATA,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_kwargs",
            ExceptionScenarioType.WITH_EXTRA_KWARGS,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_to_dict",
            ExceptionScenarioType.TO_DICT,
            FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            "base_error_str_repr",
            ExceptionScenarioType.STRING_REPRESENTATION,
            FlextExceptions.BaseError,
        ),
    ]

    SPECIFIC_TYPE_SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            "validation_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.ValidationError,
        ),
        ExceptionScenario(
            "configuration_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.ConfigurationError,
        ),
        ExceptionScenario(
            "connection_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.ConnectionError,
        ),
        ExceptionScenario(
            "timeout_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.TimeoutError,
        ),
        ExceptionScenario(
            "authentication_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.AuthenticationError,
        ),
        ExceptionScenario(
            "authorization_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.AuthorizationError,
        ),
        ExceptionScenario(
            "not_found_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.NotFoundError,
        ),
        ExceptionScenario(
            "conflict_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.ConflictError,
        ),
        ExceptionScenario(
            "rate_limit_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.RateLimitError,
        ),
        ExceptionScenario(
            "circuit_breaker_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.CircuitBreakerError,
        ),
        ExceptionScenario(
            "type_error", ExceptionScenarioType.SPECIFIC_TYPE, FlextExceptions.TypeError,
        ),
        ExceptionScenario(
            "operation_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            FlextExceptions.OperationError,
        ),
    ]

    FACTORY_SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            "create_error_validation",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "ValidationError",
        ),
        ExceptionScenario(
            "create_error_configuration",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "ConfigurationError",
        ),
        ExceptionScenario(
            "create_error_connection",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "ConnectionError",
        ),
        ExceptionScenario(
            "create_error_timeout",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "TimeoutError",
        ),
        ExceptionScenario(
            "create_error_authentication",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "AuthenticationError",
        ),
        ExceptionScenario(
            "create_error_authorization",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "AuthorizationError",
        ),
        ExceptionScenario(
            "create_error_not_found",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "NotFoundError",
        ),
        ExceptionScenario(
            "create_error_conflict",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "ConflictError",
        ),
        ExceptionScenario(
            "create_error_rate_limit",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "RateLimitError",
        ),
        ExceptionScenario(
            "create_error_circuit_breaker",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "CircuitBreakerError",
        ),
        ExceptionScenario(
            "create_error_type",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "TypeError",
        ),
        ExceptionScenario(
            "create_error_operation",
            ExceptionScenarioType.FACTORY_METHOD,
            None,
            False,
            "OperationError",
        ),
        ExceptionScenario(
            "create_error_invalid", ExceptionScenarioType.FACTORY_INVALID,
        ),
        ExceptionScenario(
            "exception_raising",
            ExceptionScenarioType.EXCEPTION_RAISING,
            FlextExceptions.ValidationError,
            True,
        ),
        ExceptionScenario(
            "exception_chaining",
            ExceptionScenarioType.EXCEPTION_CHAINING,
            FlextExceptions.OperationError,
            True,
        ),
    ]

    TYPE_SCENARIOS: ClassVar[list[ExceptionTypeScenario]] = [
        ExceptionTypeScenario(
            "instantiate_validation",
            ExceptionTypeScenarioType.VALIDATION,
            FlextExceptions.ValidationError,
        ),
        ExceptionTypeScenario(
            "instantiate_configuration",
            ExceptionTypeScenarioType.CONFIGURATION,
            FlextExceptions.ConfigurationError,
        ),
        ExceptionTypeScenario(
            "instantiate_connection",
            ExceptionTypeScenarioType.CONNECTION,
            FlextExceptions.ConnectionError,
        ),
        ExceptionTypeScenario(
            "instantiate_timeout",
            ExceptionTypeScenarioType.TIMEOUT,
            FlextExceptions.TimeoutError,
        ),
        ExceptionTypeScenario(
            "instantiate_authentication",
            ExceptionTypeScenarioType.AUTHENTICATION,
            FlextExceptions.AuthenticationError,
        ),
        ExceptionTypeScenario(
            "instantiate_authorization",
            ExceptionTypeScenarioType.AUTHORIZATION,
            FlextExceptions.AuthorizationError,
        ),
        ExceptionTypeScenario(
            "instantiate_not_found",
            ExceptionTypeScenarioType.NOT_FOUND,
            FlextExceptions.NotFoundError,
        ),
        ExceptionTypeScenario(
            "instantiate_conflict",
            ExceptionTypeScenarioType.CONFLICT,
            FlextExceptions.ConflictError,
        ),
        ExceptionTypeScenario(
            "instantiate_rate_limit",
            ExceptionTypeScenarioType.RATE_LIMIT,
            FlextExceptions.RateLimitError,
        ),
        ExceptionTypeScenario(
            "instantiate_circuit_breaker",
            ExceptionTypeScenarioType.CIRCUIT_BREAKER,
            FlextExceptions.CircuitBreakerError,
        ),
        ExceptionTypeScenario(
            "instantiate_type_error",
            ExceptionTypeScenarioType.TYPE_ERROR,
            FlextExceptions.TypeError,
        ),
        ExceptionTypeScenario(
            "instantiate_operation",
            ExceptionTypeScenarioType.OPERATION,
            FlextExceptions.OperationError,
        ),
    ]


class TestFlextExceptions:
    """Comprehensive test suite for FlextExceptions using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario", ExceptionScenarios.BASE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_base_exception_scenarios(self, scenario: ExceptionScenario) -> None:
        """Test base exception creation and behavior."""
        if scenario.scenario_type == ExceptionScenarioType.BASE_ERROR:
            error = FlextExceptions.BaseError("Test error")
            assert error.message == "Test error"
            assert error.error_code == "UNKNOWN_ERROR"
            assert error.correlation_id is None
            assert isinstance(error.metadata.attributes, dict)
        elif scenario.scenario_type == ExceptionScenarioType.WITH_CODE:
            error = FlextExceptions.BaseError("Test error", error_code="TEST_001")
            assert error.error_code == "TEST_001"
            assert str(error) == "[TEST_001] Test error"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_CORRELATION:
            error = FlextExceptions.BaseError("Test error", correlation_id="corr-123")
            assert error.correlation_id == "corr-123"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_METADATA:
            metadata = ExceptionTestHelpers.create_metadata_object({
                "field": "email",
                "value": "invalid",
            })
            error = FlextExceptions.BaseError("Test error", metadata=metadata)
            assert error.metadata.attributes["field"] == "email"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_EXTRA_KWARGS:
            error = FlextExceptions.BaseError(
                "Test error", field="email", value="invalid",
            )
            assert error.metadata.attributes["field"] == "email"
        elif scenario.scenario_type == ExceptionScenarioType.TO_DICT:
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
        elif scenario.scenario_type == ExceptionScenarioType.STRING_REPRESENTATION:
            error1 = FlextExceptions.BaseError("Test error")
            assert str(error1) == "[UNKNOWN_ERROR] Test error"
            error2 = FlextExceptions.BaseError("Test error", error_code="TEST_001")
            assert str(error2) == "[TEST_001] Test error"

    @pytest.mark.parametrize(
        "scenario", ExceptionScenarios.SPECIFIC_TYPE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_specific_exception_types(self, scenario: ExceptionScenario) -> None:
        """Test specific exception type instantiation."""
        assert scenario.exception_type is not None
        if scenario.exception_type == FlextExceptions.ValidationError:
            error: FlextExceptions.BaseError = FlextExceptions.ValidationError(
                "Invalid email", field="email", error_code="VAL_EMAIL",
            )
            assert error.message == "Invalid email" and error.error_code == "VAL_EMAIL"
        elif scenario.exception_type == FlextExceptions.ConfigurationError:
            config_error: FlextExceptions.ConfigurationError = (
                FlextExceptions.ConfigurationError(
                    "Missing config", config_key="database.host",
                )
            )
            assert (
                config_error.message == "Missing config"
                and config_error.config_key == "database.host"
            )
        elif scenario.exception_type == FlextExceptions.ConnectionError:
            conn_error: FlextExceptions.ConnectionError = (
                FlextExceptions.ConnectionError(
                    "Connection failed", host="localhost", port=5432,
                )
            )
            assert (
                conn_error.message == "Connection failed"
                and conn_error.host == "localhost"
            )
        elif scenario.exception_type == FlextExceptions.TimeoutError:
            timeout_error: FlextExceptions.TimeoutError = FlextExceptions.TimeoutError(
                "Operation timeout", timeout_seconds=30.0, operation="database_query",
            )
            assert (
                timeout_error.message == "Operation timeout"
                and timeout_error.timeout_seconds == 30.0
            )
        elif scenario.exception_type == FlextExceptions.AuthenticationError:
            auth_error: FlextExceptions.AuthenticationError = (
                FlextExceptions.AuthenticationError(
                    "Invalid credentials", user_id="testuser", auth_method="password",
                )
            )
            assert (
                auth_error.message == "Invalid credentials"
                and auth_error.user_id == "testuser"
            )
        elif scenario.exception_type == FlextExceptions.AuthorizationError:
            authz_error: FlextExceptions.AuthorizationError = (
                FlextExceptions.AuthorizationError(
                    "Access denied",
                    user_id="user123",
                    resource="admin_panel",
                    permission="read",
                )
            )
            assert (
                authz_error.message == "Access denied"
                and authz_error.user_id == "user123"
            )
        elif scenario.exception_type == FlextExceptions.NotFoundError:
            not_found_error: FlextExceptions.NotFoundError = (
                FlextExceptions.NotFoundError(
                    "Resource not found", resource_type="User", resource_id="123",
                )
            )
            assert (
                not_found_error.message == "Resource not found"
                and not_found_error.resource_type == "User"
            )
        elif scenario.exception_type == FlextExceptions.ConflictError:
            conflict_error: FlextExceptions.ConflictError = (
                FlextExceptions.ConflictError(
                    "Resource conflict",
                    resource_id="user_123",
                    conflict_reason="duplicate_email",
                )
            )
            assert (
                conflict_error.message == "Resource conflict"
                and conflict_error.resource_id == "user_123"
            )
        elif scenario.exception_type == FlextExceptions.RateLimitError:
            rate_limit_error: FlextExceptions.RateLimitError = (
                FlextExceptions.RateLimitError(
                    "Rate limit exceeded", limit=100, window_seconds=60,
                )
            )
            assert (
                rate_limit_error.message == "Rate limit exceeded"
                and rate_limit_error.limit == 100
            )
        elif scenario.exception_type == FlextExceptions.CircuitBreakerError:
            circuit_error: FlextExceptions.CircuitBreakerError = (
                FlextExceptions.CircuitBreakerError(
                    "Circuit breaker open",
                    service_name="payment_service",
                    failure_count=5,
                )
            )
            assert (
                circuit_error.message == "Circuit breaker open"
                and circuit_error.service_name == "payment_service"
            )
        elif scenario.exception_type == FlextExceptions.TypeError:
            type_error: FlextExceptions.TypeError = FlextExceptions.TypeError(
                "Invalid type", expected_type=str, actual_type=int,
            )
            assert (
                type_error.message == "Invalid type" and type_error.expected_type is str
            )
        elif scenario.exception_type == FlextExceptions.OperationError:
            op_error: FlextExceptions.OperationError = FlextExceptions.OperationError(
                "Operation failed", operation="backup", reason="disk_full",
            )
            assert (
                op_error.message == "Operation failed"
                and op_error.operation == "backup"
            )

    @pytest.mark.parametrize(
        "scenario", ExceptionScenarios.FACTORY_SCENARIOS, ids=lambda s: s.name,
    )
    def test_factory_methods(self, scenario: ExceptionScenario) -> None:
        """Test exception factory methods."""
        if scenario.scenario_type == ExceptionScenarioType.FACTORY_METHOD:
            assert scenario.error_factory_type is not None
            error = FlextExceptions.create_error(
                scenario.error_factory_type, "Test error",
            )
            assert type(error).__name__ == scenario.error_factory_type
            assert error.message == "Test error"
        elif scenario.scenario_type == ExceptionScenarioType.FACTORY_INVALID:
            with pytest.raises(ValueError, match="Unknown error type"):
                FlextExceptions.create_error("InvalidError", "Test error")
        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_RAISING:
            error_msg = "Test error"
            with pytest.raises(FlextExceptions.ValidationError) as exc_info:
                raise FlextExceptions.ValidationError(error_msg, error_code="TEST_001")
            assert exc_info.value.message == error_msg
            assert exc_info.value.error_code == "TEST_001"
        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_CHAINING:
            operation_error_msg = "Operation failed"
            with pytest.raises(FlextExceptions.OperationError) as exc_op_info:
                raise FlextExceptions.OperationError(
                    operation_error_msg,
                ) from FlextExceptions.ConfigurationError("Config error")
            assert exc_op_info.value.__cause__ is not None
            assert isinstance(
                exc_op_info.value.__cause__, FlextExceptions.ConfigurationError,
            )

    @pytest.mark.parametrize(
        "scenario", ExceptionScenarios.TYPE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_exception_type_scenarios(self, scenario: ExceptionTypeScenario) -> None:
        """Test comprehensive exception type instantiation."""
        exc = scenario.exception_class(f"{scenario.scenario_type} error")
        assert f"{scenario.scenario_type} error" in str(exc)
        exc = scenario.exception_class(
            f"{scenario.scenario_type} error",
            error_code=f"{scenario.scenario_type.upper()}_ERROR",
        )
        assert exc.error_code == f"{scenario.scenario_type.upper()}_ERROR"
        exc = scenario.exception_class(
            f"{scenario.scenario_type} error",
            metadata=ExceptionTestHelpers.create_metadata_object({"test": "data"}),
        )
        assert "test" in exc.metadata.attributes
        assert exc.metadata.attributes["test"] == "data"

    def test_exception_class_hierarchy(self) -> None:
        """Test exception class inheritance hierarchy."""
        assert all(
            issubclass(cls, FlextExceptions.BaseError)
            for cls in [
                FlextExceptions.ValidationError,
                FlextExceptions.NotFoundError,
                FlextExceptions.AuthenticationError,
                FlextExceptions.TimeoutError,
            ]
        )
        assert issubclass(FlextExceptions.BaseError, Exception)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = FlextExceptions.BaseError("Test error")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata = ExceptionTestHelpers.create_metadata_object({"existing": "value"})
        error = FlextExceptions.BaseError(
            "Test error", metadata=metadata, new_field="new_value",
        )
        assert error.metadata.attributes["existing"] == "value"
        assert error.metadata.attributes["new_field"] == "new_value"

    def test_exception_repr(self) -> None:
        """Test repr() for all exception types."""
        exceptions_to_test: list[FlextExceptions.BaseError] = [
            FlextExceptions.BaseError("base"),
            FlextExceptions.ValidationError("validation"),
            FlextExceptions.NotFoundError("not_found"),
            FlextExceptions.ConflictError("conflict"),
            FlextExceptions.AuthenticationError("auth"),
            FlextExceptions.AuthorizationError("authz"),
            FlextExceptions.TimeoutError("timeout"),
            FlextExceptions.ConnectionError("connection"),
            FlextExceptions.RateLimitError("rate_limit"),
            FlextExceptions.CircuitBreakerError("circuit"),
            FlextExceptions.ConfigurationError("config"),
            FlextExceptions.OperationError("operation"),
            FlextExceptions.TypeError("type"),
        ]
        for exc in exceptions_to_test:
            repr_str = repr(exc)
            assert repr_str is not None and len(repr_str) > 0

    def test_exception_serialization(self) -> None:
        """Test exception serialization to dict."""
        exc = FlextExceptions.ValidationError(
            "Validation failed",
            error_code="INVALID_INPUT",
            metadata=ExceptionTestHelpers.create_metadata_object({
                "field": "email",
                "value": "invalid",
            }),
        )
        if hasattr(exc, "to_dict"):
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error_code" in result or "message" in result


__all__ = ["TestFlextExceptions"]

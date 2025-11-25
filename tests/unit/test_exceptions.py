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
- Comprehensive exception instantiation and handling

Uses Python 3.13 patterns (StrEnum, frozen dataclasses with slots),
centralized constants, and parametrization for DRY testing.

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

# =========================================================================
# Exception Scenario Type Enumerations
# =========================================================================


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


# =========================================================================
# Test Case Structures
# =========================================================================


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


# =========================================================================
# Helper Functions
# =========================================================================


def create_metadata_object(attributes: dict[str, object] | None = None) -> Metadata:
    """Helper to create Metadata object from attributes dict.

    Properly creates a Metadata instance as expected by FlextExceptions.
    """
    if attributes is None:
        attributes = {}
    return Metadata(attributes=attributes)


# =========================================================================
# Test Scenario Factories
# =========================================================================


class ExceptionScenarios:
    """Factory for exception test scenarios."""

    SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            name="base_error_init",
            scenario_type=ExceptionScenarioType.BASE_ERROR,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_with_code",
            scenario_type=ExceptionScenarioType.WITH_CODE,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_with_correlation",
            scenario_type=ExceptionScenarioType.WITH_CORRELATION,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_with_metadata",
            scenario_type=ExceptionScenarioType.WITH_METADATA,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_with_kwargs",
            scenario_type=ExceptionScenarioType.WITH_EXTRA_KWARGS,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_to_dict",
            scenario_type=ExceptionScenarioType.TO_DICT,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="base_error_str_repr",
            scenario_type=ExceptionScenarioType.STRING_REPRESENTATION,
            exception_type=FlextExceptions.BaseError,
        ),
        ExceptionScenario(
            name="validation_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.ValidationError,
        ),
        ExceptionScenario(
            name="configuration_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.ConfigurationError,
        ),
        ExceptionScenario(
            name="connection_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.ConnectionError,
        ),
        ExceptionScenario(
            name="timeout_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.TimeoutError,
        ),
        ExceptionScenario(
            name="authentication_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.AuthenticationError,
        ),
        ExceptionScenario(
            name="authorization_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.AuthorizationError,
        ),
        ExceptionScenario(
            name="not_found_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.NotFoundError,
        ),
        ExceptionScenario(
            name="conflict_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.ConflictError,
        ),
        ExceptionScenario(
            name="rate_limit_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.RateLimitError,
        ),
        ExceptionScenario(
            name="circuit_breaker_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.CircuitBreakerError,
        ),
        ExceptionScenario(
            name="type_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.TypeError,
        ),
        ExceptionScenario(
            name="operation_error",
            scenario_type=ExceptionScenarioType.SPECIFIC_TYPE,
            exception_type=FlextExceptions.OperationError,
        ),
        ExceptionScenario(
            name="create_error_validation",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="ValidationError",
        ),
        ExceptionScenario(
            name="create_error_configuration",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="ConfigurationError",
        ),
        ExceptionScenario(
            name="create_error_connection",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="ConnectionError",
        ),
        ExceptionScenario(
            name="create_error_timeout",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="TimeoutError",
        ),
        ExceptionScenario(
            name="create_error_authentication",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="AuthenticationError",
        ),
        ExceptionScenario(
            name="create_error_authorization",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="AuthorizationError",
        ),
        ExceptionScenario(
            name="create_error_not_found",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="NotFoundError",
        ),
        ExceptionScenario(
            name="create_error_conflict",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="ConflictError",
        ),
        ExceptionScenario(
            name="create_error_rate_limit",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="RateLimitError",
        ),
        ExceptionScenario(
            name="create_error_circuit_breaker",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="CircuitBreakerError",
        ),
        ExceptionScenario(
            name="create_error_type",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="TypeError",
        ),
        ExceptionScenario(
            name="create_error_operation",
            scenario_type=ExceptionScenarioType.FACTORY_METHOD,
            error_factory_type="OperationError",
        ),
        ExceptionScenario(
            name="create_error_invalid",
            scenario_type=ExceptionScenarioType.FACTORY_INVALID,
        ),
        ExceptionScenario(
            name="exception_raising",
            scenario_type=ExceptionScenarioType.EXCEPTION_RAISING,
            exception_type=FlextExceptions.ValidationError,
            should_raise=True,
        ),
        ExceptionScenario(
            name="exception_chaining",
            scenario_type=ExceptionScenarioType.EXCEPTION_CHAINING,
            exception_type=FlextExceptions.OperationError,
            should_raise=True,
        ),
    ]


class ExceptionTypeScenarios:
    """Factory for exception type test scenarios."""

    SCENARIOS: ClassVar[list[ExceptionTypeScenario]] = [
        ExceptionTypeScenario(
            name="instantiate_validation",
            scenario_type=ExceptionTypeScenarioType.VALIDATION,
            exception_class=FlextExceptions.ValidationError,
        ),
        ExceptionTypeScenario(
            name="instantiate_configuration",
            scenario_type=ExceptionTypeScenarioType.CONFIGURATION,
            exception_class=FlextExceptions.ConfigurationError,
        ),
        ExceptionTypeScenario(
            name="instantiate_connection",
            scenario_type=ExceptionTypeScenarioType.CONNECTION,
            exception_class=FlextExceptions.ConnectionError,
        ),
        ExceptionTypeScenario(
            name="instantiate_timeout",
            scenario_type=ExceptionTypeScenarioType.TIMEOUT,
            exception_class=FlextExceptions.TimeoutError,
        ),
        ExceptionTypeScenario(
            name="instantiate_authentication",
            scenario_type=ExceptionTypeScenarioType.AUTHENTICATION,
            exception_class=FlextExceptions.AuthenticationError,
        ),
        ExceptionTypeScenario(
            name="instantiate_authorization",
            scenario_type=ExceptionTypeScenarioType.AUTHORIZATION,
            exception_class=FlextExceptions.AuthorizationError,
        ),
        ExceptionTypeScenario(
            name="instantiate_not_found",
            scenario_type=ExceptionTypeScenarioType.NOT_FOUND,
            exception_class=FlextExceptions.NotFoundError,
        ),
        ExceptionTypeScenario(
            name="instantiate_conflict",
            scenario_type=ExceptionTypeScenarioType.CONFLICT,
            exception_class=FlextExceptions.ConflictError,
        ),
        ExceptionTypeScenario(
            name="instantiate_rate_limit",
            scenario_type=ExceptionTypeScenarioType.RATE_LIMIT,
            exception_class=FlextExceptions.RateLimitError,
        ),
        ExceptionTypeScenario(
            name="instantiate_circuit_breaker",
            scenario_type=ExceptionTypeScenarioType.CIRCUIT_BREAKER,
            exception_class=FlextExceptions.CircuitBreakerError,
        ),
        ExceptionTypeScenario(
            name="instantiate_type_error",
            scenario_type=ExceptionTypeScenarioType.TYPE_ERROR,
            exception_class=FlextExceptions.TypeError,
        ),
        ExceptionTypeScenario(
            name="instantiate_operation",
            scenario_type=ExceptionTypeScenarioType.OPERATION,
            exception_class=FlextExceptions.OperationError,
        ),
    ]


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextExceptions:
    """Comprehensive test suite for FlextExceptions exception types."""

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_exception_scenarios(self, scenario: ExceptionScenario) -> None:
        """Test exception creation and behavior across scenarios."""
        if scenario.scenario_type == ExceptionScenarioType.BASE_ERROR:
            error = FlextExceptions.BaseError("Test error")
            assert error.message == "Test error"
            assert error.error_code == "UNKNOWN_ERROR"
            assert error.correlation_id is None
            assert isinstance(error.metadata.attributes, dict)
            assert isinstance(error.timestamp, float)

        elif scenario.scenario_type == ExceptionScenarioType.WITH_CODE:
            error = FlextExceptions.BaseError("Test error", error_code="TEST_001")
            assert error.error_code == "TEST_001"
            assert str(error) == "[TEST_001] Test error"

        elif scenario.scenario_type == ExceptionScenarioType.WITH_CORRELATION:
            error = FlextExceptions.BaseError("Test error", correlation_id="corr-123")
            assert error.correlation_id == "corr-123"

        elif scenario.scenario_type == ExceptionScenarioType.WITH_METADATA:
            metadata = create_metadata_object({"field": "email", "value": "invalid"})
            error = FlextExceptions.BaseError("Test error", metadata=metadata)
            assert error.metadata.attributes["field"] == "email"
            assert error.metadata.attributes["value"] == "invalid"

        elif scenario.scenario_type == ExceptionScenarioType.WITH_EXTRA_KWARGS:
            error = FlextExceptions.BaseError(
                "Test error", field="email", value="invalid"
            )
            assert error.metadata.attributes["field"] == "email"
            assert error.metadata.attributes["value"] == "invalid"

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
            assert error_dict["correlation_id"] == "corr-123"
            assert "timestamp" in error_dict
            if isinstance(error_dict["metadata"], dict):
                assert error_dict["metadata"]["field"] == "email"

        elif scenario.scenario_type == ExceptionScenarioType.STRING_REPRESENTATION:
            error1 = FlextExceptions.BaseError("Test error")
            assert str(error1) == "[UNKNOWN_ERROR] Test error"
            error2 = FlextExceptions.BaseError("Test error", error_code="TEST_001")
            assert str(error2) == "[TEST_001] Test error"

        elif scenario.scenario_type == ExceptionScenarioType.SPECIFIC_TYPE:
            assert scenario.exception_type is not None
            if scenario.exception_type == FlextExceptions.ValidationError:
                error = FlextExceptions.ValidationError(
                    "Invalid email", field="email", error_code="VAL_EMAIL"
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Invalid email"
                assert error.error_code == "VAL_EMAIL"
                assert error.field == "email"
            elif scenario.exception_type == FlextExceptions.ConfigurationError:
                error = FlextExceptions.ConfigurationError(
                    "Missing config", config_key="database.host"
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Missing config"
                assert error.config_key == "database.host"
            elif scenario.exception_type == FlextExceptions.ConnectionError:
                error = FlextExceptions.ConnectionError(
                    "Connection failed", host="localhost", port=5432
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Connection failed"
                assert error.host == "localhost"
                assert error.port == 5432
            elif scenario.exception_type == FlextExceptions.TimeoutError:
                error = FlextExceptions.TimeoutError(
                    "Operation timeout",
                    timeout_seconds=30.0,
                    operation="database_query",
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Operation timeout"
                assert error.timeout_seconds == 30.0
                assert error.operation == "database_query"
            elif scenario.exception_type == FlextExceptions.AuthenticationError:
                error = FlextExceptions.AuthenticationError(
                    "Invalid credentials", user_id="testuser", auth_method="password"
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Invalid credentials"
                assert error.user_id == "testuser"
                assert error.auth_method == "password"
            elif scenario.exception_type == FlextExceptions.AuthorizationError:
                error = FlextExceptions.AuthorizationError(
                    "Access denied",
                    user_id="user123",
                    resource="admin_panel",
                    permission="read",
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Access denied"
                assert error.user_id == "user123"
                assert error.resource == "admin_panel"
                assert error.permission == "read"
            elif scenario.exception_type == FlextExceptions.NotFoundError:
                error = FlextExceptions.NotFoundError(
                    "Resource not found", resource_type="User", resource_id="123"
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Resource not found"
                assert error.resource_type == "User"
                assert error.resource_id == "123"
            elif scenario.exception_type == FlextExceptions.ConflictError:
                error = FlextExceptions.ConflictError(
                    "Resource conflict",
                    resource_id="user_123",
                    conflict_reason="duplicate_email",
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Resource conflict"
                assert error.resource_id == "user_123"
                assert error.conflict_reason == "duplicate_email"
            elif scenario.exception_type == FlextExceptions.RateLimitError:
                error = FlextExceptions.RateLimitError(
                    "Rate limit exceeded", limit=100, window_seconds=60
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Rate limit exceeded"
                assert error.limit == 100
                assert error.window_seconds == 60
            elif scenario.exception_type == FlextExceptions.CircuitBreakerError:
                error = FlextExceptions.CircuitBreakerError(
                    "Circuit breaker open",
                    service_name="payment_service",
                    failure_count=5,
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Circuit breaker open"
                assert error.service_name == "payment_service"
                assert error.failure_count == 5
            elif scenario.exception_type == FlextExceptions.TypeError:
                error = FlextExceptions.TypeError(
                    "Invalid type", expected_type=str, actual_type=int
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Invalid type"
                assert error.expected_type is str
                assert error.actual_type is int
            elif scenario.exception_type == FlextExceptions.OperationError:
                error = FlextExceptions.OperationError(
                    "Operation failed", operation="backup", reason="disk_full"
                )
                assert isinstance(error, FlextExceptions.BaseError)
                assert error.message == "Operation failed"
                assert error.operation == "backup"
                assert error.reason == "disk_full"

        elif scenario.scenario_type == ExceptionScenarioType.FACTORY_METHOD:
            assert scenario.error_factory_type is not None
            error = FlextExceptions.create_error(
                scenario.error_factory_type, "Test error"
            )
            expected_class_name = scenario.error_factory_type
            assert type(error).__name__ == expected_class_name
            assert error.message == "Test error"

        elif scenario.scenario_type == ExceptionScenarioType.FACTORY_INVALID:
            with pytest.raises(ValueError, match="Unknown error type"):
                FlextExceptions.create_error("InvalidError", "Test error")

        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_RAISING:
            error_msg = "Test error"
            with pytest.raises(FlextExceptions.ValidationError) as exc_val_info:
                raise FlextExceptions.ValidationError(error_msg, error_code="TEST_001")
            assert exc_val_info.value.message == "Test error"
            assert exc_val_info.value.error_code == "TEST_001"

        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_CHAINING:
            operation_error_msg = "Operation failed"
            with pytest.raises(FlextExceptions.OperationError) as exc_op_info:
                raise FlextExceptions.OperationError(
                    operation_error_msg,
                ) from FlextExceptions.ConfigurationError("Config error")
            assert exc_op_info.value.__cause__ is not None
            assert isinstance(
                exc_op_info.value.__cause__, FlextExceptions.ConfigurationError
            )

    @pytest.mark.parametrize(
        "scenario",
        ExceptionTypeScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_exception_type_scenarios(self, scenario: ExceptionTypeScenario) -> None:
        """Test comprehensive exception type instantiation."""
        # Test with just message
        exc = scenario.exception_class(f"{scenario.scenario_type} error")
        assert f"{scenario.scenario_type} error" in str(exc)

        # Test with error_code
        exc = scenario.exception_class(
            f"{scenario.scenario_type} error",
            error_code=f"{scenario.scenario_type.upper()}_ERROR",
        )
        assert exc.error_code == f"{scenario.scenario_type.upper()}_ERROR"

        # Test with metadata
        exc = scenario.exception_class(
            f"{scenario.scenario_type} error",
            metadata=create_metadata_object({"test": "data"}),
        )
        assert "test" in exc.metadata.attributes
        assert exc.metadata.attributes["test"] == "data"

    def test_exception_class_hierarchy(self) -> None:
        """Test exception class inheritance hierarchy."""
        assert issubclass(FlextExceptions.ValidationError, FlextExceptions.BaseError)
        assert issubclass(FlextExceptions.NotFoundError, FlextExceptions.BaseError)
        assert issubclass(
            FlextExceptions.AuthenticationError, FlextExceptions.BaseError
        )
        assert issubclass(FlextExceptions.TimeoutError, FlextExceptions.BaseError)
        assert issubclass(FlextExceptions.BaseError, Exception)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = FlextExceptions.BaseError("Test error")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata = create_metadata_object({"existing": "value"})
        error = FlextExceptions.BaseError(
            "Test error", metadata=metadata, new_field="new_value"
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
            assert repr_str is not None
            assert len(repr_str) > 0

    def test_exception_serialization(self) -> None:
        """Test exception serialization to dict."""
        exc = FlextExceptions.ValidationError(
            "Validation failed",
            error_code="INVALID_INPUT",
            metadata=create_metadata_object({"field": "email", "value": "invalid"}),
        )

        if hasattr(exc, "to_dict"):
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error_code" in result or "message" in result


__all__ = ["TestFlextExceptions"]

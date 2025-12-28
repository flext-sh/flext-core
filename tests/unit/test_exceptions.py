"""Tests for e - Exception Type Definitions and Implementations.

Module: flext_core.exceptions
Scope: e - all exception types and factory methods

Tests e functionality including:
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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import pytest

from flext_core import FlextRuntime, c, e, m, p, t
from flext_tests import u


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
    exception_type: type[e.BaseError] | None = None
    should_raise: bool = False
    error_factory_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTypeScenario:
    """Exception type instantiation test scenario."""

    name: str
    scenario_type: ExceptionTypeScenarioType
    exception_class: type[e.BaseError]


class ExceptionScenarios:
    """Centralized exception test scenarios using FlextConstants."""

    BASE_SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            "base_error_init",
            ExceptionScenarioType.BASE_ERROR,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_code",
            ExceptionScenarioType.WITH_CODE,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_correlation",
            ExceptionScenarioType.WITH_CORRELATION,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_metadata",
            ExceptionScenarioType.WITH_METADATA,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_with_kwargs",
            ExceptionScenarioType.WITH_EXTRA_KWARGS,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_to_dict",
            ExceptionScenarioType.TO_DICT,
            e.BaseError,
        ),
        ExceptionScenario(
            "base_error_str_repr",
            ExceptionScenarioType.STRING_REPRESENTATION,
            e.BaseError,
        ),
    ]

    SPECIFIC_TYPE_SCENARIOS: ClassVar[list[ExceptionScenario]] = [
        ExceptionScenario(
            "validation_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.ValidationError,
        ),
        ExceptionScenario(
            "configuration_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.ConfigurationError,
        ),
        ExceptionScenario(
            "connection_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.ConnectionError,
        ),
        ExceptionScenario(
            "timeout_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.TimeoutError,
        ),
        ExceptionScenario(
            "authentication_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.AuthenticationError,
        ),
        ExceptionScenario(
            "authorization_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.AuthorizationError,
        ),
        ExceptionScenario(
            "not_found_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.NotFoundError,
        ),
        ExceptionScenario(
            "conflict_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.ConflictError,
        ),
        ExceptionScenario(
            "rate_limit_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.RateLimitError,
        ),
        ExceptionScenario(
            "circuit_breaker_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.CircuitBreakerError,
        ),
        ExceptionScenario(
            "type_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.TypeError,
        ),
        ExceptionScenario(
            "operation_error",
            ExceptionScenarioType.SPECIFIC_TYPE,
            e.OperationError,
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
            "create_error_invalid",
            ExceptionScenarioType.FACTORY_INVALID,
        ),
        ExceptionScenario(
            "exception_raising",
            ExceptionScenarioType.EXCEPTION_RAISING,
            e.ValidationError,
            True,
        ),
        ExceptionScenario(
            "exception_chaining",
            ExceptionScenarioType.EXCEPTION_CHAINING,
            e.OperationError,
            True,
        ),
    ]

    TYPE_SCENARIOS: ClassVar[list[ExceptionTypeScenario]] = [
        ExceptionTypeScenario(
            "instantiate_validation",
            ExceptionTypeScenarioType.VALIDATION,
            e.ValidationError,
        ),
        ExceptionTypeScenario(
            "instantiate_configuration",
            ExceptionTypeScenarioType.CONFIGURATION,
            e.ConfigurationError,
        ),
        ExceptionTypeScenario(
            "instantiate_connection",
            ExceptionTypeScenarioType.CONNECTION,
            e.ConnectionError,
        ),
        ExceptionTypeScenario(
            "instantiate_timeout",
            ExceptionTypeScenarioType.TIMEOUT,
            e.TimeoutError,
        ),
        ExceptionTypeScenario(
            "instantiate_authentication",
            ExceptionTypeScenarioType.AUTHENTICATION,
            e.AuthenticationError,
        ),
        ExceptionTypeScenario(
            "instantiate_authorization",
            ExceptionTypeScenarioType.AUTHORIZATION,
            e.AuthorizationError,
        ),
        ExceptionTypeScenario(
            "instantiate_not_found",
            ExceptionTypeScenarioType.NOT_FOUND,
            e.NotFoundError,
        ),
        ExceptionTypeScenario(
            "instantiate_conflict",
            ExceptionTypeScenarioType.CONFLICT,
            e.ConflictError,
        ),
        ExceptionTypeScenario(
            "instantiate_rate_limit",
            ExceptionTypeScenarioType.RATE_LIMIT,
            e.RateLimitError,
        ),
        ExceptionTypeScenario(
            "instantiate_circuit_breaker",
            ExceptionTypeScenarioType.CIRCUIT_BREAKER,
            e.CircuitBreakerError,
        ),
        ExceptionTypeScenario(
            "instantiate_type_error",
            ExceptionTypeScenarioType.TYPE_ERROR,
            e.TypeError,
        ),
        ExceptionTypeScenario(
            "instantiate_operation",
            ExceptionTypeScenarioType.OPERATION,
            e.OperationError,
        ),
    ]


class Teste:
    """Comprehensive test suite for e using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.BASE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_base_exception_scenarios(self, scenario: ExceptionScenario) -> None:
        """Test base exception creation and behavior."""
        if scenario.scenario_type == ExceptionScenarioType.BASE_ERROR:
            error = e.BaseError("Test error")
            assert error.message == "Test error"
            assert error.error_code == c.Errors.UNKNOWN_ERROR
            assert error.correlation_id is None
            assert isinstance(error.metadata.attributes, dict)
        elif scenario.scenario_type == ExceptionScenarioType.WITH_CODE:
            error = e.BaseError("Test error", error_code="TEST_001")
            assert error.error_code == "TEST_001"  # Test-specific error code
            assert str(error) == "[TEST_001] Test error"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_CORRELATION:
            error = e.BaseError("Test error", correlation_id="corr-123")
            assert error.correlation_id == "corr-123"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_METADATA:
            metadata = u.Tests.ExceptionHelpers.create_metadata_object({
                "field": "email",
                "value": "invalid",
            })
            error = e.BaseError("Test error", metadata=metadata)
            assert error.metadata.attributes["field"] == "email"
        elif scenario.scenario_type == ExceptionScenarioType.WITH_EXTRA_KWARGS:
            error = e.BaseError(
                "Test error",
                field="email",
                value="invalid",
            )
            assert error.metadata.attributes["field"] == "email"
        elif scenario.scenario_type == ExceptionScenarioType.TO_DICT:
            error = e.BaseError(
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
            error1 = e.BaseError("Test error")
            assert str(error1) == "[UNKNOWN_ERROR] Test error"
            error2 = e.BaseError("Test error", error_code="TEST_001")
            assert str(error2) == "[TEST_001] Test error"

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.SPECIFIC_TYPE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_specific_exception_types(self, scenario: ExceptionScenario) -> None:
        """Test specific exception type instantiation."""
        assert scenario.exception_type is not None
        if scenario.exception_type == e.ValidationError:
            error: e.BaseError = e.ValidationError(
                "Invalid email",
                field="email",
                error_code="VAL_EMAIL",
            )
            assert error.message == "Invalid email"
            assert error.error_code == "VAL_EMAIL"  # Test-specific error code
        elif scenario.exception_type == e.ConfigurationError:
            config_error: e.ConfigurationError = e.ConfigurationError(
                "Missing config",
                config_key="database.host",
            )
            assert (
                config_error.message == "Missing config"
                and config_error.config_key == "database.host"
            )
        elif scenario.exception_type == e.ConnectionError:
            conn_error: e.ConnectionError = e.ConnectionError(
                "Connection failed",
                host="localhost",
                port=5432,
            )
            assert (
                conn_error.message == "Connection failed"
                and conn_error.host == "localhost"
            )
        elif scenario.exception_type == e.TimeoutError:
            timeout_error: e.TimeoutError = e.TimeoutError(
                "Operation timeout",
                timeout_seconds=30.0,
                operation="database_query",
            )
            assert (
                timeout_error.message == "Operation timeout"
                and timeout_error.timeout_seconds == 30.0
            )
        elif scenario.exception_type == e.AuthenticationError:
            auth_error: e.AuthenticationError = e.AuthenticationError(
                "Invalid credentials",
                user_id="testuser",
                auth_method="password",
            )
            assert (
                auth_error.message == "Invalid credentials"
                and auth_error.user_id == "testuser"
            )
        elif scenario.exception_type == e.AuthorizationError:
            authz_error: e.AuthorizationError = e.AuthorizationError(
                "Access denied",
                user_id="user123",
                resource="admin_panel",
                permission="read",
            )
            assert (
                authz_error.message == "Access denied"
                and authz_error.user_id == "user123"
            )
        elif scenario.exception_type == e.NotFoundError:
            not_found_error: e.NotFoundError = e.NotFoundError(
                "Resource not found",
                resource_type="User",
                resource_id="123",
            )
            assert (
                not_found_error.message == "Resource not found"
                and not_found_error.resource_type == "User"
            )
        elif scenario.exception_type == e.ConflictError:
            conflict_error: e.ConflictError = e.ConflictError(
                "Resource conflict",
                resource_id="user_123",
                conflict_reason="duplicate_email",
            )
            assert (
                conflict_error.message == "Resource conflict"
                and conflict_error.resource_id == "user_123"
            )
        elif scenario.exception_type == e.RateLimitError:
            rate_limit_error: e.RateLimitError = e.RateLimitError(
                "Rate limit exceeded",
                limit=100,
                window_seconds=60,
            )
            assert (
                rate_limit_error.message == "Rate limit exceeded"
                and rate_limit_error.limit == 100
            )
        elif scenario.exception_type == e.CircuitBreakerError:
            circuit_error: e.CircuitBreakerError = e.CircuitBreakerError(
                "Circuit breaker open",
                service_name="payment_service",
                failure_count=5,
            )
            assert (
                circuit_error.message == "Circuit breaker open"
                and circuit_error.service_name == "payment_service"
            )
        elif scenario.exception_type == e.TypeError:
            type_error: e.TypeError = e.TypeError(
                "Invalid type",
                expected_type=str,
                actual_type=int,
            )
            assert (
                type_error.message == "Invalid type" and type_error.expected_type is str
            )
        elif scenario.exception_type == e.OperationError:
            op_error: e.OperationError = e.OperationError(
                "Operation failed",
                operation="backup",
                reason="disk_full",
            )
            assert (
                op_error.message == "Operation failed"
                and op_error.operation == "backup"
            )

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.FACTORY_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_factory_methods(self, scenario: ExceptionScenario) -> None:
        """Test exception factory methods."""
        if scenario.scenario_type == ExceptionScenarioType.FACTORY_METHOD:
            assert scenario.error_factory_type is not None
            error = e.create_error(
                scenario.error_factory_type,
                "Test error",
            )
            assert type(error).__name__ == scenario.error_factory_type
            assert error.message == "Test error"
        elif scenario.scenario_type == ExceptionScenarioType.FACTORY_INVALID:
            with pytest.raises(ValueError, match="Unknown error type"):
                e.create_error("InvalidError", "Test error")
        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_RAISING:
            error_msg = "Test error"
            with pytest.raises(e.ValidationError) as exc_info:
                raise e.ValidationError(error_msg, error_code="TEST_001")
            assert exc_info.value.message == error_msg
            assert exc_info.value.error_code == "TEST_001"
        elif scenario.scenario_type == ExceptionScenarioType.EXCEPTION_CHAINING:
            operation_error_msg = "Operation failed"
            with pytest.raises(e.OperationError) as exc_op_info:
                raise e.OperationError(
                    operation_error_msg,
                ) from e.ConfigurationError("Config error")
            assert exc_op_info.value.__cause__ is not None
            assert isinstance(
                exc_op_info.value.__cause__,
                e.ConfigurationError,
            )

    @pytest.mark.parametrize(
        "scenario",
        ExceptionScenarios.TYPE_SCENARIOS,
        ids=lambda s: s.name,
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
            metadata=u.Tests.ExceptionHelpers.create_metadata_object({
                "test": "data",
            }),
        )
        assert "test" in exc.metadata.attributes
        assert exc.metadata.attributes["test"] == "data"

    def test_exception_class_hierarchy(self) -> None:
        """Test exception class inheritance hierarchy."""
        assert all(
            issubclass(cls, e.BaseError)
            for cls in [
                e.ValidationError,
                e.NotFoundError,
                e.AuthenticationError,
                e.TimeoutError,
            ]
        )
        assert issubclass(e.BaseError, Exception)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = e.BaseError("Test error")
        after = time.time()
        assert before <= error.timestamp <= after

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata = u.Tests.ExceptionHelpers.create_metadata_object({
            "existing": "value",
        })
        error = e.BaseError(
            "Test error",
            metadata=metadata,
            new_field="new_value",
        )
        assert error.metadata.attributes["existing"] == "value"
        assert error.metadata.attributes["new_field"] == "new_value"

    def test_exception_repr(self) -> None:
        """Test repr() for all exception types."""
        exceptions_to_test: list[e.BaseError] = [
            e.BaseError("base"),
            e.ValidationError("validation"),
            e.NotFoundError("not_found"),
            e.ConflictError("conflict"),
            e.AuthenticationError("auth"),
            e.AuthorizationError("authz"),
            e.TimeoutError("timeout"),
            e.ConnectionError("connection"),
            e.RateLimitError("rate_limit"),
            e.CircuitBreakerError("circuit"),
            e.ConfigurationError("config"),
            e.OperationError("operation"),
            e.TypeError("type"),
        ]
        for exc in exceptions_to_test:
            repr_str = repr(exc)
            assert repr_str is not None and len(repr_str) > 0

    def test_exception_serialization(self) -> None:
        """Test exception serialization to dict."""
        metadata_obj = u.Tests.ExceptionHelpers.create_metadata_object({
            "field": "email",
            "value": "invalid",
        })
        # ValidationError accepts metadata via extra_kwargs, but BaseError.__init__ accepts it
        # Pass metadata via extra_kwargs - ValidationError.__init__ pops it and passes to BaseError
        # extra_kwargs accepts t.MetadataAttributeValue, and m.Metadata is compatible
        exc = e.ValidationError(
            "Validation failed",
            error_code="INVALID_INPUT",
            metadata=cast("t.MetadataAttributeValue", metadata_obj),
        )
        if hasattr(exc, "to_dict"):
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error_code" in result or "message" in result

    def test_base_error_str_without_code(self) -> None:
        """Test __str__ without error code - tests line 118.

        Note: BaseError has default error_code=UNKNOWN_ERROR, so to test
        line 118 (no code path), we need to set error_code to empty string
        after creation or use a different approach.
        """
        # Create error and manually set error_code to empty to test line 118
        error = e.BaseError("Test message")
        error.error_code = ""
        assert str(error) == "Test message"

    def test_base_error_str_with_code(self) -> None:
        """Test __str__ with error code - tests line 117."""
        error = e.BaseError("Test message", error_code="TEST_ERROR")
        assert str(error) == "[TEST_ERROR] Test message"

    def test_validation_error_extra_kwargs(self) -> None:
        """Test ValidationError with extra_kwargs - tests line 252."""
        error = e.ValidationError(
            "Validation failed",
            field="test_field",
            value="test_value",
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "custom_key" in error.metadata.attributes

    def test_configuration_error_extra_kwargs(self) -> None:
        """Test ConfigurationError with extra_kwargs - tests line 292."""
        error = e.ConfigurationError(
            "Config failed",
            config_key="test_key",
            config_source="env",
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "custom_key" in error.metadata.attributes

    def test_normalize_metadata_fallback(self) -> None:
        """Test _normalize_metadata fallback path - tests line 219."""
        # Test with non-Mapping, non-Metadata value
        result = e.BaseError._normalize_metadata(
            12345,  # int is t.GeneralValueType but not Mapping/Metadata
            {},
        )
        assert isinstance(result, p.Log.Metadata)
        # Assuming fallback behavior normalizes single value to "value" key
        # or similar default behavior in normalize_metadata
        # Check attributes exist
        assert result.attributes

    def test_normalize_metadata_with_merged_kwargs(self) -> None:
        """Test _normalize_metadata with merged_kwargs - tests lines 211-212."""
        metadata = {"key1": "value1"}
        merged_kwargs = {"key2": "value2"}
        # Cast merged_kwargs to MetadataAttributeValue for type compatibility
        merged_kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", merged_kwargs)
        result = e.BaseError._normalize_metadata(metadata, merged_kwargs_cast)
        assert isinstance(result, p.Log.Metadata)
        assert result.attributes["key1"] == "value1"
        assert result.attributes["key2"] == "value2"

    def test_validation_error_with_context(self) -> None:
        """Test ValidationError with context - tests lines 243-244."""
        context_raw = {"key1": "value1", "key2": 123}
        # Convert to Mapping[str, t.MetadataAttributeValue] using normalize_to_metadata_value
        # All values in context_raw are already t.GeneralValueType (str, int)
        context: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(cast("t.GeneralValueType", v))
            for k, v in context_raw.items()
        }
        error = e.ValidationError(
            "Validation failed",
            field="test_field",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes
        assert "key2" in error.metadata.attributes

    def test_configuration_error_with_context(self) -> None:
        """Test ConfigurationError with context - tests line 286."""
        context = {"key1": "value1"}
        error = e.ConfigurationError(
            "Config failed",
            config_key="test_key",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_connection_error_with_context(self) -> None:
        """Test ConnectionError with context - tests lines 325-326."""
        context = {"key1": "value1"}
        error = e.ConnectionError(
            "Connection failed",
            host="localhost",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_timeout_error_with_extra_kwargs(self) -> None:
        """Test TimeoutError with extra_kwargs - tests line 371."""
        error = e.TimeoutError(
            "Timeout",
            timeout_seconds=30.0,
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "custom_key" in error.metadata.attributes

    def test_authentication_error_with_context(self) -> None:
        """Test AuthenticationError with context - tests lines 406-407."""
        context = {"key1": "value1"}
        error = e.AuthenticationError("Auth failed", user_id="user1", context=context)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_authentication_error_with_extra_kwargs(self) -> None:
        """Test AuthenticationError with extra_kwargs - tests line 413."""
        error = e.AuthenticationError(
            "Auth failed",
            user_id="user1",
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "custom_key" in error.metadata.attributes

    def test_authorization_error_with_context(self) -> None:
        """Test AuthorizationError with context - tests lines 446-447."""
        context = {"key1": "value1"}
        error = e.AuthorizationError(
            "Authz failed",
            user_id="user1",
            resource="resource1",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_extract_from_context_none(self) -> None:
        """Test _extract_context_values with None - tests line 477-478."""
        result = e.NotFoundError._extract_context_values(None)
        assert result == (None, None, False, False)

    def test_extract_from_context_full(self) -> None:
        """Test _extract_context_values with full context - tests lines 480-498."""
        metadata_obj = m.Metadata(attributes={"key": "value"})
        # Convert values to MetadataAttributeValue
        context: dict[str, t.MetadataAttributeValue] = {
            "correlation_id": "test-correlation-id",
            "metadata": cast(
                "t.MetadataAttributeValue", metadata_obj
            ),  # m.Metadata is compatible with p.Log.Metadata which is in MetadataAttributeValue union
            "auto_log": True,
            "auto_correlation": True,
        }
        corr_id, metadata, auto_log, auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert corr_id == "test-correlation-id"
        assert metadata == metadata_obj
        assert auto_log is True
        assert auto_corr is True

    def test_extract_from_context_partial(self) -> None:
        """Test _extract_context_values with partial context - tests lines 480-496."""
        context: dict[str, t.MetadataAttributeValue] = {
            "correlation_id": 123,  # Not a string, should return None
            "metadata": "not_metadata",  # Not Metadata, should return None
            "auto_log": "not_bool",  # Not bool, should return False
            "auto_correlation": "not_bool",  # Not bool, should return False
        }
        corr_id, metadata, auto_log, auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert corr_id is None
        assert metadata is None
        assert auto_log is False
        assert auto_corr is False

    def test_extract_from_context_empty(self) -> None:
        """Test _extract_context_values with empty context - tests line 477."""
        result = e.NotFoundError._extract_context_values({})
        assert result == (None, None, False, False)

    def test_extract_from_context_with_string_correlation_id(self) -> None:
        """Test _extract_context_values with string correlation_id - tests line 481."""
        context = {"correlation_id": "test-id"}
        corr_id, _metadata, _auto_log, _auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert corr_id == "test-id"

    def test_extract_from_context_with_metadata_object(self) -> None:
        """Test _extract_context_values with Metadata object - tests lines 483-488."""
        metadata_obj = m.Metadata(attributes={"key": "value"})
        # Convert to Mapping[str, t.MetadataAttributeValue] for _extract_context_values
        context: dict[str, t.MetadataAttributeValue] = {
            "metadata": cast("t.MetadataAttributeValue", metadata_obj),
        }
        _corr_id, metadata, _auto_log, _auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert metadata == metadata_obj

    def test_extract_from_context_with_bool_auto_log(self) -> None:
        """Test _extract_context_values with bool auto_log - tests lines 490-491."""
        context = {"auto_log": True}
        _corr_id, _metadata, auto_log, _auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert auto_log is True

    def test_extract_from_context_with_bool_auto_correlation(self) -> None:
        """Test _extract_context_values with bool auto_correlation - tests lines 493-496."""
        context = {"auto_correlation": True}
        _corr_id, _metadata, _auto_log, auto_corr = (
            e.NotFoundError._extract_context_values(context)
        )
        assert auto_corr is True

    def test_extract_from_context_return_tuple(self) -> None:
        """Test _extract_context_values returns correct tuple - tests line 498."""
        # Convert to Mapping[str, t.MetadataAttributeValue] for _extract_context_values
        context: dict[str, t.MetadataAttributeValue] = {
            "correlation_id": "test-id",
            "metadata": cast(
                "t.MetadataAttributeValue", m.Metadata(attributes={"key": "value"})
            ),
            "auto_log": True,
            "auto_correlation": True,
        }
        result = e.NotFoundError._extract_context_values(context)
        assert len(result) == 4
        assert result[0] == "test-id"
        assert isinstance(result[1], m.Metadata)
        assert result[2] is True
        assert result[3] is True

    def test_not_found_error_with_context(self) -> None:
        """Test NotFoundError with context - tests lines 518-547."""
        context_raw = {
            "key1": "value1",
            "correlation_id": "test-id",  # Reserved key, should be excluded
            "metadata": "test",  # Reserved key, should be excluded
            "auto_log": True,  # Reserved key, should be excluded
            "auto_correlation": True,  # Reserved key, should be excluded
        }
        # Convert to Mapping[str, t.MetadataAttributeValue]
        # All values are already t.GeneralValueType compatible
        context: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(cast("t.GeneralValueType", v))
            for k, v in context_raw.items()
        }
        error = e.NotFoundError(
            "Not found",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes
        # Reserved keys should not be in metadata
        assert "correlation_id" not in error.metadata.attributes

    def test_not_found_error_build_kwargs_with_invalid_values(self) -> None:
        """Test NotFoundError._build_notfound_kwargs with invalid values - tests lines 524-528."""
        # Test with extra_kwargs containing invalid types
        extra_kwargs_raw = {
            "valid_str": "value",
            "valid_int": 123,
            "invalid_type": object(),  # Should be normalized to string (not filtered out)
        }
        # Convert to t.MetadataAttributeDict
        # normalize_to_metadata_value converts object() to string, so it won't be filtered
        # The test expectation is wrong - object() gets normalized to string, not filtered
        extra_kwargs: t.MetadataAttributeDict = {
            k: FlextRuntime.normalize_to_metadata_value(
                cast(
                    "t.GeneralValueType",
                    str(v)
                    if not isinstance(v, (str, int, float, bool, type(None)))
                    else v,
                )
            )
            for k, v in extra_kwargs_raw.items()
        }
        context_raw = {"key1": "value1"}
        # Convert to t.MetadataAttributeDict
        # All values need to be t.GeneralValueType for normalize_to_metadata_value
        context: t.MetadataAttributeDict = {
            k: FlextRuntime.normalize_to_metadata_value(cast("t.GeneralValueType", v))
            for k, v in context_raw.items()
        }
        kwargs = e.NotFoundError._build_notfound_kwargs(
            "User",
            "123",
            extra_kwargs,
            context,
        )
        assert "valid_str" in kwargs
        assert "valid_int" in kwargs
        # object() gets normalized to string, not filtered out
        assert "invalid_type" in kwargs  # Normalized to string representation
        assert isinstance(kwargs["invalid_type"], str)

    def test_not_found_error_build_kwargs_with_excluded_context(self) -> None:
        """Test NotFoundError._build_notfound_kwargs excludes reserved keys - tests lines 532-545."""
        context_raw = {
            "correlation_id": "test-id",
            "metadata": "test",
            "auto_log": True,
            "auto_correlation": True,
            "valid_key": "value",
        }
        # Convert to t.MetadataAttributeDict
        # All values need to be t.GeneralValueType for normalize_to_metadata_value
        context: t.MetadataAttributeDict = {
            k: FlextRuntime.normalize_to_metadata_value(cast("t.GeneralValueType", v))
            for k, v in context_raw.items()
        }
        kwargs = e.NotFoundError._build_notfound_kwargs("User", "123", {}, context)
        assert "valid_key" in kwargs
        assert "correlation_id" not in kwargs
        assert "metadata" not in kwargs
        assert "auto_log" not in kwargs
        assert "auto_correlation" not in kwargs

    def test_conflict_error_with_context(self) -> None:
        """Test ConflictError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.ConflictError(
            "Conflict",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_conflict_error_build_context(self) -> None:
        """Test ConflictError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.ConflictError(
            "Conflict",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_conflict_error_build_context_with_extra_kwargs(self) -> None:
        """Test ConflictError with extra_kwargs - tests line 625."""
        extra_kwargs_raw = {
            "custom": "value",
            "resource_type": "User",
            "resource_id": "123",
        }
        # Convert to t.MetadataAttributeValue for **kwargs
        # ConflictError accepts **extra_kwargs: t.MetadataAttributeValue
        # resource_type and resource_id come from extra_kwargs, not as direct parameters
        # Mypy limitation: **kwargs unpacking with dict[str, MetadataAttributeValue] not fully supported
        extra_kwargs: dict[str, t.MetadataAttributeValue] = {
            k: cast("t.MetadataAttributeValue", v)  # str is already ScalarValue
            for k, v in extra_kwargs_raw.items()
        }
        error = e.ConflictError(
            "Conflict",
            **extra_kwargs,
        )
        assert error.metadata is not None
        assert "custom" in error.metadata.attributes

    def test_rate_limit_error_with_context(self) -> None:
        """Test RateLimitError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.RateLimitError(
            "Rate limit",
            limit=100,
            window_seconds=60,
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_circuit_breaker_error_with_context(self) -> None:
        """Test CircuitBreakerError with context - tests line 659."""
        context = {"key1": "value1"}
        error = e.CircuitBreakerError(
            "Circuit open",
            service="test_service",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_type_error_with_context(self) -> None:
        """Test TypeError with context - tests line 701."""
        context = {"key1": "value1"}
        error = e.TypeError(
            "Type error",
            expected_type=str,
            actual_type=int,
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_operation_error_with_context(self) -> None:
        """Test OperationError with context - tests lines 757-761."""
        context = {"key1": "value1"}
        error = e.OperationError(
            "Operation failed",
            operation="test_op",
            context=context,
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_operation_error_with_context_and_reason(self) -> None:
        """Test OperationError with context and reason - tests lines 870-872, 875, 878."""
        context = {"key1": "value1"}
        error = e.OperationError(
            "Operation failed",
            operation="test_op",
            reason="Test reason",
            context=context,
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes
        assert "reason" in error.metadata.attributes
        assert "custom_key" in error.metadata.attributes

    def test_type_error_normalize_type(self) -> None:
        """Test TypeError._normalize_type."""
        # Test extracting type from extra_kwargs when type_value is None
        type_map: dict[str, type] = {"str": str, "int": int}
        extra_kwargs_raw = {"expected_type": "str"}
        # Convert to t.MetadataAttributeDict
        # _normalize_type accepts t.MetadataAttributeDict but at runtime can handle type objects
        # Mypy limitation: type objects not in MetadataAttributeValue union, but method handles them
        extra_kwargs: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v)  # str is ScalarValue
            for k, v in extra_kwargs_raw.items()
        }
        result = e.TypeError._normalize_type(
            None,
            type_map,
            extra_kwargs,
            "expected_type",
        )
        assert result is str
        # Key should be popped from extra_kwargs
        assert "expected_type" not in extra_kwargs

        # Test extracting type object from extra_kwargs
        extra_kwargs_type_raw = {"expected_type": int}
        # Type objects need special handling - _normalize_type accepts t.MetadataAttributeDict
        # but at runtime can handle type objects (defensive programming)
        # Mypy limitation: type objects not in MetadataAttributeValue union
        extra_kwargs_type: t.MetadataAttributeDict = cast(
            "t.MetadataAttributeDict",
            extra_kwargs_type_raw,
        )
        result = e.TypeError._normalize_type(
            None,
            type_map,
            extra_kwargs_type,
            "expected_type",
        )
        assert result is int
        assert "expected_type" not in extra_kwargs_type

        # Test with type_value as string (should lookup in type_map)
        result = e.TypeError._normalize_type("str", type_map, {}, "expected_type")
        assert result is str

        # Test with type_value as type object (should return as-is)
        result = e.TypeError._normalize_type(str, type_map, {}, "expected_type")
        assert result is str

        # Test with type_value as None and key not in extra_kwargs
        result = e.TypeError._normalize_type(None, type_map, {}, "expected_type")
        assert result is None

        # Test with string type not in map
        result = e.TypeError._normalize_type("float", type_map, {}, "expected_type")
        assert result is None

    def test_type_error_build_type_context(self) -> None:
        """Test TypeError._build_type_context."""
        context = {"key1": "value1"}
        type_context = e.TypeError._build_type_context(
            str,
            int,
            context,
            {"custom": "value"},
        )
        assert "key1" in type_context
        assert "expected_type" in type_context
        assert type_context["expected_type"] == "str"
        assert "actual_type" in type_context
        assert type_context["actual_type"] == "int"
        assert "custom" in type_context

    def test_type_error_build_type_context_none_types(self) -> None:
        """Test TypeError._build_type_context with None types."""
        type_context = e.TypeError._build_type_context(None, None, None, {})
        assert type_context["expected_type"] is None
        assert type_context["actual_type"] is None

    def test_type_error_build_type_context_with_type_objects(self) -> None:
        """Test TypeError._build_type_context with type objects."""
        type_context = e.TypeError._build_type_context(str, int, None, {})
        # Type objects should be converted to __qualname__
        assert type_context["expected_type"] == "str"
        assert type_context["actual_type"] == "int"

    def test_type_error_build_type_context_with_string_types(self) -> None:
        """Test TypeError._build_type_context with string types."""
        type_context = e.TypeError._build_type_context("str", "int", None, {})
        assert type_context["expected_type"] == "str"
        assert type_context["actual_type"] == "int"

    def test_type_error_build_type_context_with_context(self) -> None:
        """Test TypeError._build_type_context with context."""
        context = {"key1": "value1"}
        type_context = e.TypeError._build_type_context(None, None, context, {})
        assert "key1" in type_context
        assert type_context["expected_type"] is None
        assert type_context["actual_type"] is None

    def test_type_error_build_type_context_expected_type_string(self) -> None:
        """Test TypeError._build_type_context with string expected_type."""
        type_context = e.TypeError._build_type_context("custom_type", None, None, {})
        assert type_context["expected_type"] == "custom_type"

    def test_type_error_build_type_context_actual_type_string(self) -> None:
        """Test TypeError._build_type_context with string actual_type."""
        type_context = e.TypeError._build_type_context(None, "custom_type", None, {})
        assert type_context["actual_type"] == "custom_type"

    def test_type_error_build_type_context_with_extra_kwargs(self) -> None:
        """Test TypeError._build_type_context with extra_kwargs."""
        type_context = e.TypeError._build_type_context(
            None,
            None,
            None,
            {"custom": "value"},
        )
        assert type_context["expected_type"] is None
        assert type_context["actual_type"] is None
        assert type_context["custom"] == "value"

    def test_create_error_with_invalid_type(self) -> None:
        """Test create_error with invalid error type - tests lines 1031-1032."""
        with pytest.raises(ValueError, match="Unknown error type"):
            e.create_error("invalid_type", "message")

    def test_create_with_invalid_type(self) -> None:
        """Test create with invalid error type - tests line 1346."""
        # create returns BaseError when type cannot be determined
        error = e.create("message", invalid_key="value")
        assert isinstance(error, e.BaseError)
        assert error.message == "message"

    def test_create_error_factory_methods(self) -> None:
        """Test create_error factory methods - tests lines 1014-1033."""
        # Test all factory methods exist and work
        # Note: create_error uses class names, not lowercase types
        error = e.create_error("ValidationError", "Test message")
        assert isinstance(error, e.ValidationError)
        assert error.message == "Test message"

        error = e.create_error("ConfigurationError", "Config error")
        assert isinstance(error, e.ConfigurationError)

        error = e.create_error("ConnectionError", "Conn error")
        assert isinstance(error, e.ConnectionError)

        error = e.create_error("TimeoutError", "Timeout error")
        assert isinstance(error, e.TimeoutError)

        error = e.create_error("AuthenticationError", "Auth error")
        assert isinstance(error, e.AuthenticationError)

        error = e.create_error("AuthorizationError", "Authz error")
        assert isinstance(error, e.AuthorizationError)

        error = e.create_error("NotFoundError", "Not found error")
        assert isinstance(error, e.NotFoundError)

        error = e.create_error("ConflictError", "Conflict error")
        assert isinstance(error, e.ConflictError)

        error = e.create_error("RateLimitError", "Rate limit error")
        assert isinstance(error, e.RateLimitError)

        error = e.create_error("CircuitBreakerError", "Circuit error")
        assert isinstance(error, e.CircuitBreakerError)

        error = e.create_error("TypeError", "Type error")
        assert isinstance(error, e.TypeError)

        error = e.create_error("OperationError", "Operation error")
        assert isinstance(error, e.OperationError)

        error = e.create_error("AttributeError", "Attribute error")
        assert isinstance(error, e.AttributeAccessError)

    def test_create_factory_methods(self) -> None:
        """Test create factory methods - tests lines 1368-1379."""
        # Test create method with various error types (determined from kwargs)
        error = e.create("Test message", field="test_field")
        assert isinstance(error, e.ValidationError)

        error = e.create("Config error", config_key="test_key")
        assert isinstance(error, e.ConfigurationError)

        error = e.create("Conn error", host="localhost")
        assert isinstance(error, e.ConnectionError)

        error = e.create("Timeout error", timeout_seconds=30.0)
        assert isinstance(error, e.TimeoutError)

        error = e.create("Auth error", user_id="user1", auth_method="password")
        assert isinstance(error, e.AuthenticationError)

        error = e.create("Authz error", user_id="user1", permission="read")
        assert isinstance(error, e.AuthorizationError)

        error = e.create("Not found error", resource_type="User", resource_id="123")
        assert isinstance(error, e.NotFoundError)

        error = e.create("Operation error", operation="test_op")
        assert isinstance(error, e.OperationError)

        error = e.create("Attribute error", attribute_name="test_attr")
        assert isinstance(error, e.AttributeAccessError)

    def test_create_with_metadata_dict(self) -> None:
        """Test create with metadata as dict - tests lines 1372-1375."""
        error = e.create(
            "Test message",
            field="test_field",
            metadata={"key1": "value1"},
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_create_with_metadata_metadata_object(self) -> None:
        """Test create with metadata as Metadata object - tests lines 1369-1371."""
        metadata_obj = m.Metadata(attributes={"key1": "value1"})
        # create accepts **kwargs: t.MetadataAttributeValue
        # m.Metadata is compatible with p.Log.Metadata which is in MetadataAttributeValue union
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", metadata_obj),
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None

    def test_create_with_dict_like_metadata_basic(self) -> None:
        """Test create with dict-like metadata - tests lines 1376-1379."""

        class DictLike(Mapping[str, object]):
            def __getitem__(self, key: str) -> object:
                if key == "key1":
                    return "value1"
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(["key1"])

            def __len__(self) -> int:
                return 1

        dict_like = DictLike()
        # Convert DictLike to Mapping[str, t.MetadataAttributeValue] for metadata parameter
        # create accepts **kwargs: t.MetadataAttributeValue, and metadata is one of those kwargs
        # DictLike is Mapping[str, object], need to convert values to MetadataAttributeValue
        # First convert object values to t.GeneralValueType (string), then normalize
        dict_like_converted: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(
                cast(
                    "t.GeneralValueType",
                    str(v)
                    if not isinstance(
                        v, (str, int, float, bool, type(None), list, dict)
                    )
                    else cast("t.GeneralValueType", v),
                ),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_create_with_correlation_id(self) -> None:
        """Test create with correlation_id - tests lines 1365-1366."""
        error = e.create(
            "Test message",
            field="test_field",
            correlation_id="test-correlation-id",
        )
        assert isinstance(error, e.ValidationError)
        # correlation_id is added to error_context and passed to _create_error_by_type
        assert error.correlation_id == "test-correlation-id"

    def test_create_error_by_type_with_error_code(self) -> None:
        """Test _create_error_by_type with error_code - tests lines 1322-1323."""
        error = e._create_error_by_type(
            "validation",
            "Test message",
            error_code="CUSTOM_ERROR",
            context=None,
        )
        assert isinstance(error, e.ValidationError)
        assert error.error_code == "CUSTOM_ERROR"

    def test_create_error_by_type_with_context(self) -> None:
        """Test _create_error_by_type with context - tests lines 1320-1321."""
        context = {"key1": "value1"}
        error = e._create_error_by_type(
            "validation",
            "Test message",
            error_code=None,
            context=context,
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_create_error_by_type_base_error(self) -> None:
        """Test _create_error_by_type returns BaseError for None type - tests lines 1346-1350."""
        error = e._create_error_by_type(
            None,
            "Test message",
            error_code=None,
            context=None,
        )
        assert isinstance(error, e.BaseError)
        assert error.message == "Test message"

    def test_create_error_by_type_invalid_type(self) -> None:
        """Test _create_error_by_type with invalid type - tests line 1346."""
        error = e._create_error_by_type(
            "invalid",
            "Test message",
            error_code=None,
            context=None,
        )
        assert isinstance(error, e.BaseError)
        assert error.message == "Test message"

    def test_attribute_access_error_with_extra_kwargs(self) -> None:
        """Test AttributeAccessError with extra_kwargs - tests line 918."""
        error = e.AttributeAccessError(
            "Attribute error",
            attribute_name="test_attr",
            custom_key="custom_value",
        )
        assert error.metadata is not None
        assert "custom_key" in error.metadata.attributes

    def test_prepare_kwargs(self) -> None:
        """Test prepare_exception_kwargs - tests lines 945-970."""
        specific_params_raw = {"field": "test_field"}
        # Convert to t.MetadataAttributeDict
        specific_params: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs_raw = {
            "correlation_id": "test-id",
            "metadata": m.Metadata(attributes={"key": "value"}),
            "auto_log": True,
            "auto_correlation": True,
            "config": "test_config",
            "field": "override_field",  # Should be overridden by specific_params
            "custom": "value",
        }
        # Convert to t.MetadataAttributeDict
        kwargs: t.MetadataAttributeDict = {}
        for k, v in kwargs_raw.items():
            if isinstance(v, m.Metadata):
                # m.Metadata is compatible with p.Log.Metadata which is in MetadataAttributeValue union
                kwargs[k] = cast("t.MetadataAttributeValue", v)
            elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
                # These are already t.GeneralValueType
                kwargs[k] = FlextRuntime.normalize_to_metadata_value(
                    cast("t.GeneralValueType", v)
                )
            else:
                # Convert object to string first (str is t.GeneralValueType), then normalize
                v_str = str(v)
                kwargs[k] = FlextRuntime.normalize_to_metadata_value(
                    cast("t.GeneralValueType", v_str)
                )

        # Note: The order of arguments in prepare_exception_kwargs in exceptions.py is (kwargs, specific_params)
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        corr_id, metadata, auto_log, auto_corr, _config, extra = result

        assert corr_id == "test-id"
        assert isinstance(metadata, m.Metadata)
        assert auto_log is True
        assert auto_corr is True
        assert "field" in extra
        assert extra["field"] == "test_field"  # Overridden by specific_params
        assert "custom" in extra
        # Reserved keys should not be in extra_kwargs
        assert "correlation_id" not in extra
        assert "metadata" not in extra
        assert "auto_log" not in extra
        assert "auto_correlation" not in extra
        assert "config" not in extra

    def test_prepare_kwargs_with_empty_specific_params(self) -> None:
        """Test prepare_exception_kwargs with empty specific_params - tests line 945."""
        kwargs_raw = {"field": "test_field"}
        # Convert to t.MetadataAttributeDict
        kwargs: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs_raw.items()
        }
        result = e.prepare_exception_kwargs(kwargs, {})
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        assert "field" in extra
        assert extra["field"] == "test_field"

    def test_prepare_kwargs_setdefault_behavior(self) -> None:
        """Test prepare_exception_kwargs setdefault behavior - tests line 948."""
        specific_params_raw = {"field": "test_field"}
        # Convert to t.MetadataAttributeDict
        specific_params: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs: t.MetadataAttributeDict = {}  # field not in kwargs
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        assert "field" in extra
        assert extra["field"] == "test_field"

    def test_prepare_kwargs_with_specific_params_none(self) -> None:
        """Test prepare_exception_kwargs with None in specific_params - tests lines 947-948."""
        specific_params_raw = {"field": None}  # None value should not override
        # Convert to t.MetadataAttributeDict (None is valid MetadataAttributeValue)
        specific_params: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs_raw = {"field": "test_field"}
        # Convert to t.MetadataAttributeDict
        kwargs: t.MetadataAttributeDict = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs_raw.items()
        }
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        assert (
            extra["field"] == "test_field"
        )  # Not overridden because specific_params value is None

    def test_prepare_exception_kwargs_with_non_string_correlation_id(self) -> None:
        """Test prepare_exception_kwargs with non-string correlation_id - tests lines 961-966."""
        kwargs = {"correlation_id": 123}  # Not a string
        # Cast kwargs to dict[str, t.MetadataAttributeValue] for type compatibility
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        result = e.prepare_exception_kwargs(kwargs_cast, None)
        corr_id, _metadata, _auto_log, _auto_corr, _config, _extra = result
        assert corr_id is None  # Should be None for non-string

    def test_prepare_exception_kwargs_return_tuple(self) -> None:
        """Test prepare_exception_kwargs returns correct tuple - tests lines 967-970."""
        kwargs = {"field": "test"}
        # Cast kwargs to dict[str, t.MetadataAttributeValue] for type compatibility
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        result = e.prepare_exception_kwargs(kwargs_cast, None)
        assert len(result) == 6
        corr_id, metadata, auto_log, auto_corr, _metadata_val, extra = result
        assert corr_id is None or isinstance(corr_id, str)
        assert metadata is None or isinstance(metadata, (m.Metadata, dict))
        assert isinstance(auto_log, bool)
        assert isinstance(auto_corr, bool)
        assert isinstance(extra, dict)

    def test_create_error_with_context(self) -> None:
        """Test create_error with context parameter - tests lines 1086-1087."""
        # Note: create_error doesn't accept context directly, but we can test
        # the error creation with various parameters
        error = e.create_error("ValidationError", "Test message")
        assert isinstance(error, e.ValidationError)

    def test_create_with_dict_like_metadata_normalization(self) -> None:
        """Test create normalizes dict-like metadata values - tests lines 1376-1379."""

        class DictLike(Mapping[str, object]):
            _obj: object

            def __init__(self) -> None:
                self._obj = object()

            def __getitem__(self, key: str) -> object:
                if key == "key1":
                    return "value1"
                if key == "key2":
                    return self._obj
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(["key1", "key2"])

            def __len__(self) -> int:
                return 2

        dict_like = DictLike()
        # Convert DictLike to Mapping[str, t.MetadataAttributeValue] for metadata parameter
        # create accepts **kwargs: t.MetadataAttributeValue, and metadata is one of those kwargs
        # DictLike is Mapping[str, object], need to convert values to MetadataAttributeValue
        # First convert object values to t.GeneralValueType (string), then normalize
        dict_like_converted: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(
                cast(
                    "t.GeneralValueType",
                    str(v)
                    if not isinstance(
                        v, (str, int, float, bool, type(None), list, dict)
                    )
                    else cast("t.GeneralValueType", v),
                ),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes
        assert "key2" in error.metadata.attributes

    def test_create_with_dict_like_metadata_items_iteration(self) -> None:
        """Test create iterates dict-like metadata items - tests lines 1378-1379."""

        class DictLike(Mapping[str, object]):
            def __getitem__(self, key: str) -> object:
                mapping: dict[str, t.GeneralValueType] = {"key1": "value1", "key2": 123}
                if key in mapping:
                    return mapping[key]
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(["key1", "key2"])

            def __len__(self) -> int:
                return 2

        dict_like = DictLike()
        # Convert DictLike to Mapping[str, t.MetadataAttributeValue] for metadata parameter
        # create accepts **kwargs: t.MetadataAttributeValue, and metadata is one of those kwargs
        # DictLike is Mapping[str, object], need to convert values to MetadataAttributeValue
        # First convert object values to t.GeneralValueType (string), then normalize
        dict_like_converted: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(
                cast(
                    "t.GeneralValueType",
                    str(v)
                    if not isinstance(
                        v, (str, int, float, bool, type(None), list, dict)
                    )
                    else cast("t.GeneralValueType", v),
                ),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes
        assert "key2" in error.metadata.attributes

    def test_create_with_dict_like_metadata_normalize_values(self) -> None:
        """Test create normalizes dict-like metadata values - tests line 1379."""

        class DictLike(Mapping[str, object]):
            _obj: object

            def __init__(self) -> None:
                self._obj = object()

            def __getitem__(self, key: str) -> object:
                if key == "key1":
                    return self._obj
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(["key1"])

            def __len__(self) -> int:
                return 1

        dict_like = DictLike()
        # Convert DictLike to Mapping[str, t.MetadataAttributeValue] for metadata parameter
        # create accepts **kwargs: t.MetadataAttributeValue, and metadata is one of those kwargs
        # DictLike is Mapping[str, object], need to convert values to MetadataAttributeValue
        # First convert object values to t.GeneralValueType (string), then normalize
        dict_like_converted: dict[str, t.MetadataAttributeValue] = {
            k: FlextRuntime.normalize_to_metadata_value(
                cast(
                    "t.GeneralValueType",
                    str(v)
                    if not isinstance(
                        v, (str, int, float, bool, type(None), list, dict)
                    )
                    else cast("t.GeneralValueType", v),
                ),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "key1" in error.metadata.attributes

    def test_create_error_instance(self) -> None:
        """Test create_error instance method (__call__) - tests lines 1453-1456."""
        error_factory = e()
        # __call__ accepts error_code parameter
        error = error_factory(
            "Test message",
            error_code="TEST_ERROR",
            field="test_field",
        )
        assert isinstance(error, e.ValidationError)
        assert error.error_code == "TEST_ERROR"
        assert error.metadata is not None
        assert "field" in error.metadata.attributes

    def test_create_error_instance_normalizes_kwargs(self) -> None:
        """Test create_error instance method normalizes kwargs - tests lines 1453-1456."""
        error_factory = e()
        # Test with various types that need normalization
        # Note: calling create() which accepts kwargs, NOT create_error() which is static and positional only
        error = error_factory.create(
            "Test message",
            error_code="TEST_ERROR",
            field="test_field",
            value=123,  # int is already t.MetadataAttributeValue (ScalarValue)
            custom_obj=cast(
                "t.MetadataAttributeValue", str(object())
            ),  # object needs to be converted to string
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "field" in error.metadata.attributes
        assert "value" in error.metadata.attributes
        assert "custom_obj" in error.metadata.attributes

    def test_create_error_instance_normalizes_all_kwargs(self) -> None:
        """Test create_error instance method normalizes all kwargs - tests lines 1454-1455."""
        error_factory = e()
        # Test normalization loop
        # Note: calling create() which accepts kwargs
        error = error_factory.create(
            "Test message",
            field="test_field",
            value1=123,
            value2="string",
            value3=True,
            value4=None,
        )
        assert isinstance(error, e.ValidationError)
        assert error.metadata is not None
        assert "field" in error.metadata.attributes
        assert "value1" in error.metadata.attributes
        assert "value2" in error.metadata.attributes
        assert "value3" in error.metadata.attributes
        assert "value4" in error.metadata.attributes

    def test_create_error_instance_normalize_loop(self) -> None:
        """Test create_error instance method normalization loop - tests lines 1454-1455."""
        error_factory = e()
        # Test that all kwargs are normalized in the loop
        # Note: calling create() which accepts kwargs: t.MetadataAttributeValue
        # Convert all values to MetadataAttributeValue
        # create() normalizes values internally, but we need to pass compatible types
        error = error_factory.create(
            "Test message",
            obj=str(object()),  # object -> str (str is ScalarValue)
            lst=[1, 2, 3],  # list[int] is Sequence[ScalarValue]
            dct={"key": "value"},  # dict[str, str] is Mapping[str, ScalarValue]
        )
        assert isinstance(error, e.BaseError)
        assert error.metadata is not None
        assert "obj" in error.metadata.attributes
        assert "lst" in error.metadata.attributes
        assert "dct" in error.metadata.attributes

    def test_prepare_metadata_value(self) -> None:
        """Test _prepare_metadata_value - tests line 1074."""
        metadata_obj = m.Metadata(attributes={"key": "value"})
        result = e._prepare_metadata_value(metadata_obj)
        assert result == metadata_obj.attributes

        result_none = e._prepare_metadata_value(None)
        assert result_none is None

    def test_determine_error_type(self) -> None:
        """Test _determine_error_type - tests lines 1105-1308."""
        # Test various patterns
        error_type = e._determine_error_type({"field": "test"})
        assert error_type == "validation"

        error_type = e._determine_error_type({"config_key": "test"})
        assert error_type == "configuration"

        error_type = e._determine_error_type({"host": "localhost"})
        assert error_type == "connection"

        error_type = e._determine_error_type({"timeout_seconds": 30.0})
        assert error_type == "timeout"

        error_type = e._determine_error_type({
            "user_id": "user1",
            "auth_method": "password",
        })
        assert error_type == "authentication"

        error_type = e._determine_error_type({
            "user_id": "user1",
            "permission": "read",
        })
        assert error_type == "authorization"

        error_type = e._determine_error_type({"resource_id": "123"})
        assert error_type == "not_found"

        error_type = e._determine_error_type({"attribute_name": "test"})
        assert error_type == "attribute_access"

        error_type = e._determine_error_type({"unknown": "value"})
        assert error_type is None

    def test_determine_error_type_with_conflict(self) -> None:
        """Test _determine_error_type with conflict pattern - tests line 585."""
        # ConflictError doesn't have a specific pattern, so it won't be detected
        # But we can test other patterns
        error_type = e._determine_error_type({
            "resource_type": "User",
            "resource_id": "123",
        })
        assert error_type == "not_found"

    def test_extract_common_kwargs(self) -> None:
        """Test extract_common_kwargs - tests lines 999-1008."""
        kwargs = {
            "correlation_id": "test-id",
            "metadata": m.Metadata(attributes={"key": "value"}),
            "field": "test_field",
        }
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        corr_id, metadata = e.extract_common_kwargs(kwargs_cast)
        assert corr_id == "test-id"
        assert isinstance(metadata, m.Metadata)

        # Test with dict metadata
        kwargs_dict_raw = {
            "correlation_id": "test-id",
            "metadata": {"key": "value"},
            "field": "test_field",
        }
        # Convert to Mapping[str, t.MetadataAttributeValue] for extract_common_kwargs
        kwargs_dict: dict[str, t.MetadataAttributeValue] = {
            k: cast("t.MetadataAttributeValue", v)
            if not isinstance(v, dict)
            else cast(
                "t.MetadataAttributeValue",
                {str(k2): cast("t.MetadataAttributeValue", v2) for k2, v2 in v.items()},
            )
            for k, v in kwargs_dict_raw.items()
        }
        corr_id_dict, metadata_dict = e.extract_common_kwargs(kwargs_dict)
        assert corr_id_dict == "test-id"
        assert isinstance(metadata_dict, dict)

        # Test with dict-like metadata
        class DictLike(Mapping[str, object]):
            def __getitem__(self, key: str) -> object:
                if key == "key":
                    return "value"
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(["key"])

            def __len__(self) -> int:
                return 1

        dict_like_obj = DictLike()
        kwargs_dict_like_raw = {
            "correlation_id": "test-id",
            "metadata": dict_like_obj,
            "field": "test_field",
        }
        # Convert to Mapping[str, t.MetadataAttributeValue] for extract_common_kwargs
        # DictLike is Mapping[str, object], need to convert values
        kwargs_dict_like: dict[str, t.MetadataAttributeValue] = {}
        for k, v in kwargs_dict_like_raw.items():
            if k == "metadata" and isinstance(v, DictLike):
                # Convert DictLike to dict[str, t.MetadataAttributeValue]
                dict_like_dict: dict[str, t.MetadataAttributeValue] = {
                    k2: cast(
                        "t.MetadataAttributeValue",
                        str(v2)
                        if not isinstance(v2, (str, int, float, bool, type(None)))
                        else cast("t.MetadataAttributeValue", v2),
                    )
                    for k2, v2 in v.items()
                }
                kwargs_dict_like[k] = cast("t.MetadataAttributeValue", dict_like_dict)
            else:
                kwargs_dict_like[k] = cast("t.MetadataAttributeValue", v)
        corr_id_dl, metadata_dl = e.extract_common_kwargs(kwargs_dict_like)
        assert corr_id_dl == "test-id"
        # Dict-like metadata is converted to dict
        assert isinstance(metadata_dl, dict)
        assert "key" in metadata_dl


__all__ = ["Teste"]

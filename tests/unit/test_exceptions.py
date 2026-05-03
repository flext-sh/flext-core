"""Behavioral tests for the public exception contract."""

from __future__ import annotations

import time
from collections.abc import (
    Callable,
)

import pytest

from tests import c, e, m, p, t


class TestsFlextExceptions:
    """Validate the public behavior exposed by flext_core."""

    def test_exception_hierarchy_uses_base_error(self) -> None:
        assert issubclass(e.ValidationError, e.BaseError)
        assert issubclass(e.NotFoundError, e.BaseError)
        assert issubclass(e.AuthenticationError, e.BaseError)
        assert issubclass(e.TimeoutError, e.BaseError)
        assert issubclass(e.BaseError, Exception)

    def test_base_error_sets_timestamp_and_formats_string(self) -> None:
        before = time.time()
        error = e.BaseError("Test message", error_code="TEST_ERROR")
        assert before <= error.timestamp <= time.time()
        assert str(error) == "[TEST_ERROR] Test message"
        error.error_code = ""
        assert str(error) == "Test message"

    def test_base_error_merges_metadata_context_and_extra_kwargs(self) -> None:
        error = e.BaseError(
            "Test error",
            context={"scope": "service"},
            metadata={"existing": "value"},
            new_field="new_value",
        )
        attributes = error.metadata.attributes
        assert attributes["existing"] == "value"
        assert attributes["scope"] == "service"
        assert attributes["new_field"] == "new_value"

    def test_fail_operation_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_operation(
            "register service",
            ValueError("boom"),
        )
        assert result.failure
        assert result.error is not None
        assert "Failed to register service" in result.error
        assert "boom" in result.error
        assert result.error_code == c.ErrorCode.OPERATION_ERROR
        assert result.error_data is not None
        assert result.error_data["operation"] == "register service"
        assert result.error_data["reason"] == "boom"

    def test_fail_not_found_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_not_found(
            "service",
            "command_bus",
        )
        assert result.failure
        assert result.error is not None
        assert "Service 'command_bus' not found" in result.error
        assert result.error_code == c.ErrorCode.NOT_FOUND_ERROR
        assert result.error_data is not None
        assert result.error_data["resource_type"] == "service"
        assert result.error_data["resource_id"] == "command_bus"

    def test_fail_type_mismatch_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_type_mismatch(
            "Dispatcher",
            "str",
        )
        assert result.failure
        assert result.error is not None
        assert "Dispatcher" in result.error
        assert result.error_code == c.ErrorCode.TYPE_ERROR
        assert result.error_data is not None
        assert result.error_data["expected_type"] == "Dispatcher"
        assert result.error_data["actual_type"] == "str"

    def test_fail_type_mismatch_accepts_service_lookup_params(self) -> None:
        result: p.Result[bool] = e.fail_type_mismatch(
            m.ServiceLookupParams(
                service_name="connection",
                expected_type="ldap3.Connection",
                actual_type="str",
            ),
        )

        assert result.failure
        assert result.error is not None
        assert "ldap3.Connection" in result.error
        assert result.error_data is not None
        assert result.error_data["service_name"] == "connection"
        assert result.error_data["expected_type"] == "ldap3.Connection"
        assert result.error_data["actual_type"] == "str"

    def test_fail_validation_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field="name", value=""),
            error="empty",
        )
        assert result.failure
        assert result.error is not None
        assert "validate name" in result.error
        assert result.error_code == c.ErrorCode.VALIDATION_ERROR
        assert result.error_data is not None
        assert result.error_data["field"] == "name"
        assert result.error_data["value"] == ""
        assert result.error_data["cause"] == "empty"

    def test_fail_validation_accepts_validation_error_params(self) -> None:
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field="email", value="bad"),
            error="invalid",
        )

        assert result.failure
        assert result.error is not None
        assert "validate email" in result.error
        assert result.error_data is not None
        assert result.error_data["field"] == "email"
        assert result.error_data["value"] == "bad"
        assert result.error_data["cause"] == "invalid"

    def test_declarative_error_supports_public_auto_correlation(self) -> None:
        error = e.ValidationError(
            "Validation failed",
            field="email",
            auto_correlation=True,
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        assert error.field == "email"

    @pytest.mark.parametrize(
        ("factory", "expected_type"),
        [
            (
                lambda: e.ValidationError("Test message", field="email"),
                e.ValidationError,
            ),
            (
                lambda: e.ConfigurationError("Test message", config_key="timeout"),
                e.ConfigurationError,
            ),
            (
                lambda: e.ConnectionError("Test message", host=c.LOCALHOST, port=8080),
                e.ConnectionError,
            ),
            (
                lambda: e.TimeoutError("Test message", timeout_seconds=30.0),
                e.TimeoutError,
            ),
            (
                lambda: e.AuthenticationError(
                    "Test message",
                    auth_method="password",
                    user_id="u-1",
                ),
                e.AuthenticationError,
            ),
            (
                lambda: e.AuthorizationError(
                    "Test message", user_id="u-1", permission="read"
                ),
                e.AuthorizationError,
            ),
            (
                lambda: e.NotFoundError(
                    "Test message",
                    resource_type="User",
                    resource_id="123",
                ),
                e.NotFoundError,
            ),
            (
                lambda: e.OperationError("Test message", operation="dispatch"),
                e.OperationError,
            ),
            (
                lambda: e.AttributeAccessError("Test message", attribute_name="logger"),
                e.AttributeAccessError,
            ),
        ],
    )
    def test_typed_exception_instances_are_correct_type(
        self,
        factory: Callable[[], e.BaseError],
        expected_type: type[e.BaseError],
    ) -> None:
        error = factory()
        assert isinstance(error, expected_type)
        assert error.message == "Test message"

    def test_base_error_handles_correlation_and_metadata(self) -> None:
        err = e.BaseError(
            "boom",
            correlation_id="corr-001",
            metadata={"scope": "service"},
        )
        assert err.correlation_id == "corr-001"
        assert err.metadata.attributes.get("scope") == "service"

    def test_not_found_error_excludes_internal_context_keys_from_metadata(self) -> None:
        error = e.NotFoundError(
            "Not found",
            resource_type="User",
            resource_id="123",
            context={
                "key1": "value1",
                "correlation_id": "corr-001",
                "metadata": "skip-me",
            },
        )
        attributes = error.metadata.attributes
        assert attributes["key1"] == "value1"
        assert "correlation_id" not in attributes
        assert "metadata" not in attributes

    def test_type_error_normalizes_expected_and_actual_type(self) -> None:
        error = e.TypeError(
            "Type mismatch",
            expected_type="str",
            actual_type=int,
            context={"source": "api"},
        )
        assert error.expected_type is str
        assert error.actual_type is int
        attributes = error.metadata.attributes
        assert attributes["source"] == "api"
        assert attributes["expected_type"] == "str"
        assert attributes["actual_type"] == "int"

    def test_metrics_are_recorded_and_reset_through_public_api(self) -> None:
        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.TimeoutError)
        metrics = e.resolve_metrics_snapshot()
        assert isinstance(metrics, m.ExceptionMetricsSnapshot)
        assert metrics.total_exceptions == 3
        assert metrics.exception_counts[e.ValidationError.__qualname__] == 2
        assert metrics.exception_counts[e.TimeoutError.__qualname__] == 1
        e.clear_metrics()
        cleared_metrics = e.resolve_metrics_snapshot()
        assert cleared_metrics.total_exceptions == 0


type ErrorFactory = Callable[[], e.BaseError]
type FailureFactory = Callable[[], p.Result[bool]]


class TestsFlextCoverageExceptions:
    """Validate public exception behavior without depending on internals."""

    STRUCTURED_ERRORS: t.SequenceOf[
        tuple[str, ErrorFactory, str, str, dict[str, str | int | None]]
    ] = [
        (
            "validation",
            lambda: e.ValidationError("Invalid input", field="email", value="bad"),
            c.ErrorDomain.VALIDATION.value,
            c.ErrorCode.VALIDATION_ERROR,
            {"field": "email", "value": "bad"},
        ),
        (
            "configuration",
            lambda: e.ConfigurationError(
                "Missing key",
                config_key="API_KEY",
                config_source="environment",
            ),
            c.ErrorDomain.INTERNAL.value,
            c.ErrorCode.CONFIGURATION_ERROR,
            {"config_key": "API_KEY", "config_source": "environment"},
        ),
        (
            "connection",
            lambda: e.ConnectionError(
                "Connect failed",
                host="db.internal",
                port=5432,
                timeout=5,
            ),
            c.ErrorDomain.NETWORK.value,
            c.ErrorCode.CONNECTION_ERROR,
            {"host": "db.internal", "port": 5432, "timeout": 5},
        ),
        (
            "timeout",
            lambda: e.TimeoutError(
                "Timed out",
                timeout_seconds=30,
                operation="dispatch",
            ),
            c.ErrorDomain.TIMEOUT.value,
            c.ErrorCode.TIMEOUT_ERROR,
            {"timeout_seconds": 30, "operation": "dispatch"},
        ),
        (
            "authentication",
            lambda: e.AuthenticationError(
                "Auth failed",
                auth_method="token",
                user_id="u-1",
            ),
            c.ErrorDomain.AUTH.value,
            c.ErrorCode.AUTHENTICATION_ERROR,
            {"auth_method": "token", "user_id": "u-1"},
        ),
        (
            "authorization",
            lambda: e.AuthorizationError(
                "Denied",
                user_id="u-1",
                resource="admin.panel",
                permission="write",
            ),
            c.ErrorDomain.AUTH.value,
            c.ErrorCode.AUTHORIZATION_ERROR,
            {"user_id": "u-1", "resource": "admin.panel", "permission": "write"},
        ),
        (
            "not_found",
            lambda: e.NotFoundError(
                "User missing",
                resource_type="User",
                resource_id="123",
            ),
            c.ErrorDomain.NOT_FOUND.value,
            c.ErrorCode.NOT_FOUND_ERROR,
            {"resource_type": "User", "resource_id": "123"},
        ),
        (
            "conflict",
            lambda: e.ConflictError(
                "User exists",
                resource_type="User",
                resource_id="123",
                conflict_reason="duplicate",
            ),
            c.ErrorDomain.VALIDATION.value,
            c.ErrorCode.ALREADY_EXISTS,
            {
                "resource_type": "User",
                "resource_id": "123",
                "conflict_reason": "duplicate",
            },
        ),
        (
            "circuit_breaker",
            lambda: e.CircuitBreakerError(
                "Circuit open",
                service_name="payments",
                failure_count=5,
                reset_timeout=60,
            ),
            c.ErrorDomain.NETWORK.value,
            c.ErrorCode.EXTERNAL_SERVICE_ERROR,
            {"service_name": "payments", "failure_count": 5, "reset_timeout": 60},
        ),
        (
            "type_error",
            lambda: e.TypeError(
                "Wrong type",
                expected_type="str",
                actual_type=int,
            ),
            c.ErrorDomain.VALIDATION.value,
            c.ErrorCode.TYPE_ERROR,
            {"expected_type": "str", "actual_type": "int"},
        ),
    ]

    FAILURES: t.SequenceOf[
        tuple[str, FailureFactory, str, str, dict[str, str | int | None]]
    ] = [
        (
            "config",
            lambda: e.fail_config_error(
                "API_KEY",
                "environment",
                options=m.ExceptionFactoryOptions(error="missing"),
            ),
            "read config key 'API_KEY'",
            c.ErrorCode.CONFIGURATION_ERROR,
            {"config_key": "API_KEY", "config_source": "environment"},
        ),
        (
            "connection",
            lambda: e.fail_connection(
                "db.internal",
                params=m.ConnectionErrorParams(
                    host="db.internal", port=5432, timeout=5
                ),
                options=m.ExceptionFactoryOptions(error="refused"),
            ),
            "connect to db.internal",
            c.ErrorCode.CONNECTION_ERROR,
            {"host": "db.internal", "port": 5432, "timeout": 5},
        ),
        (
            "timeout",
            lambda: e.fail_timeout(30, "dispatch"),
            "dispatch",
            c.ErrorCode.TIMEOUT_ERROR,
            {"timeout_seconds": 30, "operation": "dispatch"},
        ),
        (
            "auth",
            lambda: e.fail_auth(
                "token",
                "u-1",
                options=m.ExceptionFactoryOptions(error="denied"),
            ),
            "authenticate user u-1",
            c.ErrorCode.AUTHENTICATION_ERROR,
            {"auth_method": "token", "user_id": "u-1"},
        ),
        (
            "authz",
            lambda: e.fail_authz("u-1", "admin.panel", "write"),
            "authorize",
            c.ErrorCode.AUTHORIZATION_ERROR,
            {"user_id": "u-1", "resource": "admin.panel", "permission": "write"},
        ),
        (
            "conflict",
            lambda: e.fail_conflict("user", "u-1", "duplicate"),
            "create user",
            c.ErrorCode.ALREADY_EXISTS,
            {
                "resource_type": "user",
                "resource_id": "u-1",
                "conflict_reason": "duplicate",
            },
        ),
        (
            "operation",
            lambda: e.fail_operation("resolve service", "timeout"),
            "resolve service",
            c.ErrorCode.OPERATION_ERROR,
            {"operation": "resolve service", "reason": "timeout"},
        ),
        (
            "not_found",
            lambda: e.fail_not_found("service", "my-service"),
            "my-service",
            c.ErrorCode.NOT_FOUND_ERROR,
            {"resource_type": "service", "resource_id": "my-service"},
        ),
        (
            "type_mismatch",
            lambda: e.fail_type_mismatch("FlextLogger", "str"),
            "FlextLogger",
            c.ErrorCode.TYPE_ERROR,
            {"expected_type": "FlextLogger", "actual_type": "str"},
        ),
        (
            "validation",
            lambda: e.fail_validation(
                m.ValidationErrorParams(field="email", value="bad"),
                error="invalid",
            ),
            "validate email",
            c.ErrorCode.VALIDATION_ERROR,
            {"field": "email", "value": "bad"},
        ),
    ]

    @pytest.mark.parametrize(
        ("name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_expose_public_contract(
        self,
        name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: dict[str, str | int | None],
    ) -> None:
        error = factory()

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == expected_domain
        assert error.error_code == expected_code
        assert error.error_message == error.message
        assert error.matches_error_domain(expected_domain)

        for key, value in expected_payload.items():
            assert error.metadata.attributes[key] == value

    def test_base_error_defaults_to_unknown_domain_protocol_surface(self) -> None:
        error = e.BaseError("Base failure")

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == c.ErrorDomain.UNKNOWN.value
        assert error.error_code == c.ErrorCode.UNKNOWN_ERROR
        assert error.error_message == "Base failure"
        assert error.matches_error_domain(c.ErrorDomain.UNKNOWN.value)

    def test_metadata_attributes_preserve_correlation_id(self) -> None:
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

        assert error.message == "Insert failed"
        assert error.error_code == c.ErrorCode.OPERATION_ERROR
        assert error.error_domain == c.ErrorDomain.INTERNAL.value
        assert error.correlation_id == "corr-123"
        assert error.metadata.attributes["scope"] == "users"
        assert error.metadata.attributes["operation"] == "insert_user"
        assert error.metadata.attributes["reason"] == "constraint"
        assert error.metadata.attributes["attempt"] == 2

    @pytest.mark.parametrize(
        ("name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_factories_return_public_result_contract(
        self,
        name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: dict[str, str | int | None],
    ) -> None:
        result = factory()
        result_data = result.error_data or {}

        assert result.failure
        assert result.error is not None
        assert expected_fragment in result.error
        assert result.error_code == expected_code
        assert result.error_data is not None
        for key, value in expected_data.items():
            assert result_data[key] == value

    def test_metrics_are_exposed_through_public_behavior(self) -> None:
        e.clear_metrics()
        for exception_type in (
            e.ValidationError,
            e.ValidationError,
            e.TimeoutError,
            e.TimeoutError,
            e.TimeoutError,
        ):
            e.record_exception(exception_type)

        metrics = e.resolve_metrics_snapshot()
        assert isinstance(metrics, m.ExceptionMetricsSnapshot)
        assert metrics.total_exceptions == 5
        assert metrics.unique_exception_types == 2
        assert metrics.exception_counts[e.ValidationError.__qualname__] == 2
        assert metrics.exception_counts[e.TimeoutError.__qualname__] == 3
        summary = metrics.exception_counts_summary
        assert "ValidationError:2" in summary
        assert "TimeoutError:3" in summary

        e.clear_metrics()
        cleared = e.resolve_metrics_snapshot()
        assert cleared.total_exceptions == 0
        assert cleared.unique_exception_types == 0
        assert cleared.exception_counts_summary == ""

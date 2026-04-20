"""Additional behavioral coverage for the public exception APIs."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)

import pytest

from tests import c, e, m, p, t

type ErrorFactory = Callable[[], e.BaseError]
type FailureFactory = Callable[[], p.Result[bool]]


class TestCoverageExceptions:
    """Validate public exception behavior without depending on internals."""

    STRUCTURED_ERRORS: Sequence[
        tuple[str, ErrorFactory, str, str, Mapping[str, t.MetadataValue | None]]
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

    FAILURES: Sequence[
        tuple[str, FailureFactory, str, str, Mapping[str, t.MetadataValue | None]]
    ] = [
        (
            "config",
            lambda: e.fail_config_error("API_KEY", "environment", error="missing"),
            "read config key 'API_KEY'",
            c.ErrorCode.CONFIGURATION_ERROR,
            {"config_key": "API_KEY", "config_source": "environment"},
        ),
        (
            "connection",
            lambda: e.fail_connection(
                "db.internal",
                5432,
                timeout=5,
                error="refused",
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
            lambda: e.fail_auth("token", "u-1", error="denied"),
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
    ]

    @pytest.mark.parametrize(
        ("_name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_expose_public_contract(
        self,
        _name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: Mapping[str, t.MetadataValue | None],
    ) -> None:
        error = factory()

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == expected_domain
        assert error.error_code == expected_code
        assert error.error_message == error.message
        assert error.matches_error_domain(expected_domain)

        payload = error.to_dict()
        assert payload["message"] == error.message
        assert payload["error_type"] == type(error).__name__
        assert payload["error_code"] == expected_code
        assert payload["error_domain"] == expected_domain
        for key, value in expected_payload.items():
            assert payload[key] == value

    def test_base_error_defaults_to_unknown_domain_protocol_surface(self) -> None:
        error = e.BaseError("Base failure")

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == c.ErrorDomain.UNKNOWN.value
        assert error.error_code == c.ErrorCode.UNKNOWN_ERROR
        assert error.error_message == "Base failure"
        assert error.matches_error_domain(c.ErrorDomain.UNKNOWN.value)

    def test_to_dict_flattens_metadata_and_preserves_correlation_id(self) -> None:
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

        payload = error.to_dict()
        assert payload["message"] == "Insert failed"
        assert payload["error_code"] == c.ErrorCode.OPERATION_ERROR
        assert payload["error_domain"] == c.ErrorDomain.INTERNAL.value
        assert payload["correlation_id"] == "corr-123"
        assert payload["scope"] == "users"
        assert payload["operation"] == "insert_user"
        assert payload["reason"] == "constraint"
        assert payload["attempt"] == 2

    @pytest.mark.parametrize(
        ("_name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_factories_return_public_result_contract(
        self,
        _name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: Mapping[str, t.MetadataValue | None],
    ) -> None:
        result = factory()

        assert result.failure
        assert result.error is not None
        assert expected_fragment in result.error
        assert result.error_code == expected_code
        assert result.error_data is not None
        for key, value in expected_data.items():
            assert result.error_data[key] == value

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

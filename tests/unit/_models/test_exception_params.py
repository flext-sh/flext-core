"""Tests for exception parameter models via FlextModels facade.

Covers FlextModelsExceptionParams: ParamsModel base, ValidationErrorParams,
ConfigurationErrorParams, ConnectionErrorParams, TimeoutErrorParams,
AuthenticationErrorParams, AuthorizationErrorParams, NotFoundErrorParams,
ConflictErrorParams, RateLimitErrorParams, CircuitBreakerErrorParams,
TypeErrorParams, OperationErrorParams, AttributeAccessErrorParams.
"""

from __future__ import annotations

import math

import pytest
from flext_tests import tm

from pydantic import ValidationError
from tests import m, t


class TestFlextModelsExceptionParams:
    """Tests for flext_core via the m facade."""

    # ── ParamsModel base ──────────────────────────────────────

    def test_params_model_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            m.ParamsModel.model_validate({"unknown_field": "value"})

    def test_params_model_strict_mode(self) -> None:
        """ParamsModel enforces strict=True in settings."""
        tm.that(m.ParamsModel.model_config.get("strict"), eq=True)
        tm.that(m.ParamsModel.model_config.get("extra"), eq="forbid")
        tm.that(m.ParamsModel.model_config.get("validate_assignment"), eq=True)

    # ── ValidationErrorParams ─────────────────────────────────

    def test_validation_error_params_defaults(self) -> None:
        params = m.ValidationErrorParams()
        tm.that(params.field, none=True)
        tm.that(params.value, none=True)

    def test_validation_error_params_with_values(self) -> None:
        params = m.ValidationErrorParams(field="email", value="bad@")
        tm.that(params.field, eq="email")
        tm.that(params.value, eq="bad@")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("username", "john"),
            ("age", 25),
            ("ratio", math.pi),
            ("active", True),
            (None, None),
        ],
        ids=["str-value", "int-value", "float-value", "bool-value", "all-none"],
    )
    def test_validation_error_params_scalar_values(
        self, field: str | None, value: t.Scalar | None
    ) -> None:
        params = m.ValidationErrorParams(field=field, value=value)
        tm.that(params.field, eq=field)
        tm.that(params.value, eq=value)

    def test_validation_error_params_serialization(self) -> None:
        params = m.ValidationErrorParams(field="email", value=42)
        data = params.model_dump()
        tm.that(data["field"], eq="email")
        tm.that(data["value"], eq=42)

    # ── ConfigurationErrorParams ──────────────────────────────

    def test_configuration_error_params_defaults(self) -> None:
        params = m.ConfigurationErrorParams()
        tm.that(params.config_key, none=True)
        tm.that(params.config_source, none=True)

    def test_configuration_error_params_with_values(self) -> None:
        params = m.ConfigurationErrorParams(
            config_key="database_url", config_source=".env"
        )
        tm.that(params.config_key, eq="database_url")
        tm.that(params.config_source, eq=".env")

    def test_configuration_error_params_serialization(self) -> None:
        params = m.ConfigurationErrorParams(
            config_key="api_key",
            config_source="vault",
        )
        data = params.model_dump()
        tm.that(data["config_key"], eq="api_key")
        tm.that(data["config_source"], eq="vault")

    # ── ConnectionErrorParams ─────────────────────────────────

    def test_connection_error_params_defaults(self) -> None:
        params = m.ConnectionErrorParams()
        tm.that(params.host, none=True)
        tm.that(params.port, none=True)
        tm.that(params.timeout, none=True)

    def test_connection_error_params_with_values(self) -> None:
        params = m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5.5)
        tm.that(params.host, eq="db.internal")
        tm.that(params.port, eq=5432)
        tm.that(params.timeout, eq=5.5)

    @pytest.mark.parametrize(
        "timeout_val",
        [5, 5.5, 0, None],
        ids=["int-timeout", "float-timeout", "zero", "none"],
    )
    def test_connection_error_params_timeout_types(
        self, timeout_val: float | None
    ) -> None:
        params = m.ConnectionErrorParams(timeout=timeout_val)
        tm.that(params.timeout, eq=timeout_val)

    def test_connection_error_params_serialization(self) -> None:
        params = m.ConnectionErrorParams(host="localhost", port=8080, timeout=10)
        data = params.model_dump()
        tm.that(data["host"], eq="localhost")
        tm.that(data["port"], eq=8080)
        tm.that(data["timeout"], eq=10)

    # ── TimeoutErrorParams ────────────────────────────────────

    def test_timeout_error_params_defaults(self) -> None:
        params = m.TimeoutErrorParams()
        tm.that(params.timeout_seconds, none=True)
        tm.that(params.operation, none=True)

    def test_timeout_error_params_with_values(self) -> None:
        params = m.TimeoutErrorParams(timeout_seconds=30, operation="dispatch")
        tm.that(params.timeout_seconds, eq=30)
        tm.that(params.operation, eq="dispatch")

    @pytest.mark.parametrize(
        "seconds",
        [30, 30.0, 0.5],
        ids=["int", "float", "fraction"],
    )
    def test_timeout_error_params_numeric_seconds(self, seconds: float) -> None:
        params = m.TimeoutErrorParams(timeout_seconds=seconds)
        tm.that(params.timeout_seconds, eq=seconds)

    # ── AuthenticationErrorParams ─────────────────────────────

    def test_authentication_error_params_defaults(self) -> None:
        params = m.AuthenticationErrorParams()
        tm.that(params.auth_method, none=True)
        tm.that(params.user_id, none=True)

    def test_authentication_error_params_with_values(self) -> None:
        params = m.AuthenticationErrorParams(auth_method="token", user_id="user-123")
        tm.that(params.auth_method, eq="token")
        tm.that(params.user_id, eq="user-123")

    def test_authentication_error_params_serialization(self) -> None:
        params = m.AuthenticationErrorParams(
            auth_method="oauth2", user_id="svc-account"
        )
        data = params.model_dump()
        tm.that(data["auth_method"], eq="oauth2")
        tm.that(data["user_id"], eq="svc-account")

    # ── AuthorizationErrorParams ──────────────────────────────

    def test_authorization_error_params_defaults(self) -> None:
        params = m.AuthorizationErrorParams()
        tm.that(params.user_id, none=True)
        tm.that(params.resource, none=True)
        tm.that(params.permission, none=True)

    def test_authorization_error_params_with_values(self) -> None:
        params = m.AuthorizationErrorParams(
            user_id="user-123", resource="invoice:12345", permission="write"
        )
        tm.that(params.user_id, eq="user-123")
        tm.that(params.resource, eq="invoice:12345")
        tm.that(params.permission, eq="write")

    def test_authorization_error_params_serialization(self) -> None:
        params = m.AuthorizationErrorParams(
            user_id="admin", resource="settings", permission="delete"
        )
        data = params.model_dump()
        tm.that(data["user_id"], eq="admin")
        tm.that(data["resource"], eq="settings")
        tm.that(data["permission"], eq="delete")

    # ── NotFoundErrorParams ───────────────────────────────────

    def test_not_found_error_params_defaults(self) -> None:
        params = m.NotFoundErrorParams()
        tm.that(params.resource_type, none=True)
        tm.that(params.resource_id, none=True)

    def test_not_found_error_params_with_values(self) -> None:
        params = m.NotFoundErrorParams(resource_type="user", resource_id="42")
        tm.that(params.resource_type, eq="user")
        tm.that(params.resource_id, eq="42")

    @pytest.mark.parametrize(
        ("resource_type", "resource_id"),
        [
            ("user", "42"),
            ("order", "ord-1001"),
            ("invoice", "INV-2024-001"),
            (None, None),
        ],
        ids=["user", "order", "invoice", "all-none"],
    )
    def test_not_found_error_params_resource_variants(
        self, resource_type: str | None, resource_id: str | None
    ) -> None:
        params = m.NotFoundErrorParams(
            resource_type=resource_type, resource_id=resource_id
        )
        tm.that(params.resource_type, eq=resource_type)
        tm.that(params.resource_id, eq=resource_id)

    # ── ConflictErrorParams ───────────────────────────────────

    def test_conflict_error_params_defaults(self) -> None:
        params = m.ConflictErrorParams()
        tm.that(params.resource_type, none=True)
        tm.that(params.resource_id, none=True)
        tm.that(params.conflict_reason, none=True)

    def test_conflict_error_params_with_values(self) -> None:
        params = m.ConflictErrorParams(
            resource_type="order",
            resource_id="ord-1001",
            conflict_reason="version_mismatch",
        )
        tm.that(params.resource_type, eq="order")
        tm.that(params.resource_id, eq="ord-1001")
        tm.that(params.conflict_reason, eq="version_mismatch")

    def test_conflict_error_params_serialization(self) -> None:
        params = m.ConflictErrorParams(
            resource_type="invoice",
            resource_id="inv-99",
            conflict_reason="duplicate_entry",
        )
        data = params.model_dump()
        tm.that(data["resource_type"], eq="invoice")
        tm.that(data["resource_id"], eq="inv-99")
        tm.that(data["conflict_reason"], eq="duplicate_entry")

    # ── RateLimitErrorParams ──────────────────────────────────

    def test_rate_limit_error_params_defaults(self) -> None:
        params = m.RateLimitErrorParams()
        tm.that(params.limit, none=True)
        tm.that(params.window_seconds, none=True)
        tm.that(params.retry_after, none=True)

    def test_rate_limit_error_params_with_values(self) -> None:
        params = m.RateLimitErrorParams(limit=100, window_seconds=60, retry_after=1.5)
        tm.that(params.limit, eq=100)
        tm.that(params.window_seconds, eq=60)
        tm.that(params.retry_after, eq=1.5)

    @pytest.mark.parametrize(
        "retry_after",
        [1, 1.5, 0, None],
        ids=["int", "float", "zero", "none"],
    )
    def test_rate_limit_error_params_retry_after_types(
        self, retry_after: float | None
    ) -> None:
        params = m.RateLimitErrorParams(retry_after=retry_after)
        tm.that(params.retry_after, eq=retry_after)

    # ── CircuitBreakerErrorParams ─────────────────────────────

    def test_circuit_breaker_error_params_defaults(self) -> None:
        params = m.CircuitBreakerErrorParams()
        tm.that(params.service_name, none=True)
        tm.that(params.failure_count, none=True)
        tm.that(params.reset_timeout, none=True)

    def test_circuit_breaker_error_params_with_values(self) -> None:
        params = m.CircuitBreakerErrorParams(
            service_name="payments-api", failure_count=5, reset_timeout=30.0
        )
        tm.that(params.service_name, eq="payments-api")
        tm.that(params.failure_count, eq=5)
        tm.that(params.reset_timeout, eq=30.0)

    def test_circuit_breaker_error_params_serialization(self) -> None:
        params = m.CircuitBreakerErrorParams(
            service_name="auth-svc", failure_count=3, reset_timeout=15
        )
        data = params.model_dump()
        tm.that(data["service_name"], eq="auth-svc")
        tm.that(data["failure_count"], eq=3)
        tm.that(data["reset_timeout"], eq=15)

    # ── TypeErrorParams ───────────────────────────────────────

    def test_type_error_params_defaults(self) -> None:
        params = m.TypeErrorParams()
        tm.that(params.expected_type, none=True)
        tm.that(params.actual_type, none=True)

    def test_type_error_params_with_values(self) -> None:
        params = m.TypeErrorParams(expected_type="str", actual_type="int")
        tm.that(params.expected_type, eq="str")
        tm.that(params.actual_type, eq="int")

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("str", "int"),
            ("list", "dict"),
            ("BaseModel", "NoneType"),
        ],
        ids=["str-int", "list-dict", "model-none"],
    )
    def test_type_error_params_type_pairs(self, expected: str, actual: str) -> None:
        params = m.TypeErrorParams(expected_type=expected, actual_type=actual)
        tm.that(params.expected_type, eq=expected)
        tm.that(params.actual_type, eq=actual)

    # ── OperationErrorParams ──────────────────────────────────

    def test_operation_error_params_defaults(self) -> None:
        params = m.OperationErrorParams()
        tm.that(params.operation, none=True)
        tm.that(params.reason, none=True)

    def test_operation_error_params_with_values(self) -> None:
        params = m.OperationErrorParams(
            operation="publish_events", reason="transient_backend_error"
        )
        tm.that(params.operation, eq="publish_events")
        tm.that(params.reason, eq="transient_backend_error")

    def test_operation_error_params_serialization(self) -> None:
        params = m.OperationErrorParams(operation="save_state", reason="disk_full")
        data = params.model_dump()
        tm.that(data["operation"], eq="save_state")
        tm.that(data["reason"], eq="disk_full")

    # ── AttributeAccessErrorParams ────────────────────────────

    def test_attribute_access_error_params_defaults(self) -> None:
        params = m.AttributeAccessErrorParams()
        tm.that(params.attribute_name, none=True)
        tm.that(params.attribute_context, none=True)

    def test_attribute_access_error_params_with_values(self) -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="token", attribute_context={"owner": "session"}
        )
        tm.that(params.attribute_name, eq="token")
        tm.that(params.attribute_context, eq={"owner": "session"})

    def test_attribute_access_error_params_serialization(self) -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="settings", attribute_context="runtime"
        )
        data = params.model_dump()
        tm.that(data["attribute_name"], eq="settings")
        tm.that(data["attribute_context"], eq="runtime")

    # ── Cross-cutting: extra fields forbidden ─────────────────

    @pytest.mark.parametrize(
        "model_cls",
        [
            m.ValidationErrorParams,
            m.ConfigurationErrorParams,
            m.ConnectionErrorParams,
            m.TimeoutErrorParams,
            m.AuthenticationErrorParams,
            m.AuthorizationErrorParams,
            m.NotFoundErrorParams,
            m.ConflictErrorParams,
            m.RateLimitErrorParams,
            m.CircuitBreakerErrorParams,
            m.TypeErrorParams,
            m.OperationErrorParams,
            m.AttributeAccessErrorParams,
        ],
        ids=[
            "validation",
            "configuration",
            "connection",
            "timeout",
            "authentication",
            "authorization",
            "not-found",
            "conflict",
            "rate-limit",
            "circuit-breaker",
            "type-error",
            "operation",
            "attribute-access",
        ],
    )
    def test_all_params_reject_extra_fields(
        self,
        model_cls: type[m.ParamsModel],
    ) -> None:
        with pytest.raises(ValidationError):
            model_cls.model_validate({"bogus_field": "nope"})

    # ── Cross-cutting: all defaults to None ───────────────────

    @pytest.mark.parametrize(
        "model_cls",
        [
            m.ValidationErrorParams,
            m.ConfigurationErrorParams,
            m.ConnectionErrorParams,
            m.TimeoutErrorParams,
            m.AuthenticationErrorParams,
            m.AuthorizationErrorParams,
            m.NotFoundErrorParams,
            m.ConflictErrorParams,
            m.RateLimitErrorParams,
            m.CircuitBreakerErrorParams,
            m.TypeErrorParams,
            m.OperationErrorParams,
            m.AttributeAccessErrorParams,
        ],
        ids=[
            "validation",
            "configuration",
            "connection",
            "timeout",
            "authentication",
            "authorization",
            "not-found",
            "conflict",
            "rate-limit",
            "circuit-breaker",
            "type-error",
            "operation",
            "attribute-access",
        ],
    )
    def test_all_params_instantiate_with_no_args(self, model_cls: type) -> None:
        instance = model_cls()
        data = instance.model_dump()
        for value in data.values():
            tm.that(value, none=True)

    # ── Cross-cutting: validate_assignment ────────────────────

    def test_validate_assignment_enforced(self) -> None:
        """Assigning a wrong type to a strict str field raises."""
        params = m.ValidationErrorParams(field="email")
        with pytest.raises(ValidationError):
            setattr(params, "field", 123)

    # ── Cross-cutting: roundtrip model_validate ───────────────

    def test_roundtrip_connection_error_params(self) -> None:
        original = m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5)
        rebuilt = m.ConnectionErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.host, eq=original.host)
        tm.that(rebuilt.port, eq=original.port)
        tm.that(rebuilt.timeout, eq=original.timeout)

    def test_roundtrip_authorization_error_params(self) -> None:
        original = m.AuthorizationErrorParams(
            user_id="u-1", resource="docs:secret", permission="read"
        )
        rebuilt = m.AuthorizationErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.user_id, eq=original.user_id)
        tm.that(rebuilt.resource, eq=original.resource)
        tm.that(rebuilt.permission, eq=original.permission)

    def test_roundtrip_rate_limit_error_params(self) -> None:
        original = m.RateLimitErrorParams(
            limit=1000, window_seconds=3600, retry_after=2.5
        )
        rebuilt = m.RateLimitErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.limit, eq=original.limit)
        tm.that(rebuilt.window_seconds, eq=original.window_seconds)
        tm.that(rebuilt.retry_after, eq=original.retry_after)

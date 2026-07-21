"""Exception parameter base and connection/authentication tests."""

from __future__ import annotations

import math

import pytest

from flext_tests import tm
from tests import c, m, t


class TestsFlextModelsExceptionParamsCore:
    def test_params_model_forbids_extra_fields(self) -> None:
        with pytest.raises(c.ValidationError):
            m.ParamsModel.model_validate({"unknown_field": "value"})

    def test_strict_mode_rejects_type_coercion(self) -> None:
        """strict=True: a string is not coerced into an int port."""
        with pytest.raises(c.ValidationError):
            m.ConnectionErrorParams(port="5432")

    def test_validate_assignment_rejects_bad_value(self) -> None:
        """validate_assignment=True: reassigning an invalid type raises."""
        params = m.ConnectionErrorParams(port=1)
        with pytest.raises(c.ValidationError):
            setattr(params, "port", "bad")

    def test_validate_assignment_rejects_extra_field(self) -> None:
        """extra=forbid + validate_assignment: unknown attribute assignment raises."""
        params = m.ConnectionErrorParams()
        with pytest.raises(c.ValidationError):
            setattr(params, "unknown_field", "value")

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
        params = m.ConfigurationErrorParams(config_key="api_key", config_source="vault")
        data = params.model_dump()
        tm.that(data["config_key"], eq="api_key")
        tm.that(data["config_source"], eq="vault")

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

    @pytest.mark.parametrize(
        ("host", "port", "expected"),
        [
            ("db.internal", 5432, "db.internal:5432"),
            ("db.internal", None, "db.internal"),
            (None, 5432, "unknown:5432"),
            (None, None, "unknown"),
        ],
        ids=["host-and-port", "host-only", "port-only", "neither"],
    )
    def test_connection_target_formats_host_port(
        self, host: str | None, port: int | None, expected: str
    ) -> None:
        params = m.ConnectionErrorParams(host=host, port=port)
        tm.that(params.connection_target, eq=expected)

    def test_connection_error_params_serialization(self) -> None:
        params = m.ConnectionErrorParams(host="localhost", port=8080, timeout=10)
        data = params.model_dump()
        tm.that(data["host"], eq="localhost")
        tm.that(data["port"], eq=8080)
        tm.that(data["timeout"], eq=10)

    def test_timeout_error_params_defaults(self) -> None:
        params = m.TimeoutErrorParams()
        tm.that(params.timeout_seconds, none=True)
        tm.that(params.operation, none=True)

    def test_timeout_error_params_with_values(self) -> None:
        params = m.TimeoutErrorParams(timeout_seconds=30, operation="dispatch")
        tm.that(params.timeout_seconds, eq=30)
        tm.that(params.operation, eq="dispatch")

    @pytest.mark.parametrize(
        "seconds", [30, 30.0, 0.5], ids=["int", "float", "fraction"]
    )
    def test_timeout_error_params_numeric_seconds(self, seconds: float) -> None:
        params = m.TimeoutErrorParams(timeout_seconds=seconds)
        tm.that(params.timeout_seconds, eq=seconds)

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

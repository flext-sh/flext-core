"""Tests for FlextModelsErrors and FlextModelsDecorators via facade.

Source: flext_core._models/errors.py (117 LOC) + _models/decorators.py (49 LOC)
Tested through facade: m.Error, m.TimeoutConfig

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flext_core import c
from flext_tests import tm
from tests import m, t


class TestFlextModelsErrors:
    """Tests for Error model, from_exception, to_dict, and TimeoutConfig."""

    # -- Error creation and fields --

    def test_error_creation_minimal(self) -> None:
        """Error can be created with code and message only."""
        error = m.Error(code="TEST_001", message="Something failed")
        tm.that(error.code, eq="TEST_001")
        tm.that(error.message, eq="Something failed")
        tm.that(error.domain, eq=c.ErrorDomain.UNKNOWN)
        tm.that(error.source, eq=None)

    def test_error_creation_with_domain(self) -> None:
        """Error accepts explicit domain."""
        error = m.Error(
            domain=c.ErrorDomain.VALIDATION,
            code="INVALID_EMAIL",
            message="Email format is wrong",
        )
        tm.that(error.domain, eq=c.ErrorDomain.VALIDATION)
        tm.that(error.code, eq="INVALID_EMAIL")

    def test_error_creation_with_all_fields(self) -> None:
        """Error can be created with all fields populated."""
        details = t.ConfigMap(root={"field": "email", "value": "bad"})
        exc = ValueError("original")
        error = m.Error(
            domain=c.ErrorDomain.NETWORK,
            code="CONN_REFUSED",
            message="Connection refused",
            details=details,
            source=exc,
        )
        tm.that(error.domain, eq=c.ErrorDomain.NETWORK)
        tm.that(error.code, eq="CONN_REFUSED")
        tm.that(error.message, eq="Connection refused")
        tm.that(error.details["field"], eq="email")
        tm.that(error.source, eq=exc)

    def test_error_default_details_is_empty_configmap(self) -> None:
        """Error defaults details to empty ConfigMap."""
        error = m.Error(code="X", message="y")
        tm.that(len(error.details), eq=0)

    # -- Error domain enum --

    DOMAIN_MEMBERS: tuple[tuple[str, str], ...] = (
        ("VALIDATION", "VALIDATION"),
        ("NETWORK", "NETWORK"),
        ("AUTH", "AUTH"),
        ("NOT_FOUND", "NOT_FOUND"),
        ("TIMEOUT", "TIMEOUT"),
        ("INTERNAL", "INTERNAL"),
        ("UNKNOWN", "UNKNOWN"),
    )

    @pytest.mark.parametrize(("name", "value"), DOMAIN_MEMBERS)
    def test_error_domain_members(self, name: str, value: str) -> None:
        """Each ErrorDomain member has expected name and value."""
        member = c.ErrorDomain[name]
        tm.that(member.value, eq=value)
        tm.that(str(member), eq=value)

    # -- from_exception factory --

    def test_from_exception_basic(self) -> None:
        """from_exception creates Error from exception."""
        exc = ValueError("bad input")
        error = m.Error.from_exception(exc)
        tm.that(error.code, eq="ValueError")
        tm.that(error.message, eq="bad input")
        tm.that(error.domain, eq=c.ErrorDomain.INTERNAL)
        tm.that(error.source, eq=exc)

    def test_from_exception_custom_domain(self) -> None:
        """from_exception accepts custom domain."""
        exc = ConnectionError("timeout")
        error = m.Error.from_exception(exc, domain=c.ErrorDomain.NETWORK)
        tm.that(error.domain, eq=c.ErrorDomain.NETWORK)
        tm.that(error.code, eq="ConnectionError")

    def test_from_exception_custom_code(self) -> None:
        """from_exception accepts custom code overriding exception class name."""
        exc = RuntimeError("oops")
        error = m.Error.from_exception(exc, code="CUSTOM_CODE")
        tm.that(error.code, eq="CUSTOM_CODE")
        tm.that(error.message, eq="oops")

    def test_from_exception_with_domain_and_code(self) -> None:
        """from_exception accepts both domain and code."""
        exc = PermissionError("denied")
        error = m.Error.from_exception(
            exc, domain=c.ErrorDomain.AUTH, code="ACCESS_DENIED"
        )
        tm.that(error.domain, eq=c.ErrorDomain.AUTH)
        tm.that(error.code, eq="ACCESS_DENIED")
        tm.that(error.message, eq="denied")
        tm.that(error.source, eq=exc)

    # -- to_dict --

    def test_to_dict_basic(self) -> None:
        """to_dict returns ConfigMap with domain, code, message."""
        error = m.Error(
            domain=c.ErrorDomain.VALIDATION,
            code="FIELD_REQUIRED",
            message="Name is required",
        )
        result = error.to_dict()
        tm.that(result["domain"], eq="VALIDATION")
        tm.that(result["code"], eq="FIELD_REQUIRED")
        tm.that(result["message"], eq="Name is required")

    def test_to_dict_includes_details(self) -> None:
        """to_dict merges details into output."""
        details = t.ConfigMap(root={"field": "email"})
        error = m.Error(
            domain=c.ErrorDomain.VALIDATION,
            code="INVALID",
            message="Bad email",
            details=details,
        )
        result = error.to_dict()
        tm.that(result["field"], eq="email")
        tm.that(result["domain"], eq="VALIDATION")

    def test_to_dict_returns_configmap(self) -> None:
        """to_dict returns a ConfigMap instance."""
        error = m.Error(code="X", message="y")
        result = error.to_dict()
        tm.that(result, is_=t.ConfigMap)

    # -- __str__ --

    def test_str_representation(self) -> None:
        """__str__ returns 'code: message' format."""
        error = m.Error(code="ERR_001", message="Something broke")
        tm.that(str(error), eq="ERR_001: Something broke")

    def test_str_from_exception(self) -> None:
        """__str__ works on from_exception-created errors."""
        exc = TypeError("wrong type")
        error = m.Error.from_exception(exc)
        tm.that(str(error), eq="TypeError: wrong type")

    # -- TimeoutConfig --

    def test_timeout_config_creation(self) -> None:
        """TimeoutConfig can be created with timeout_seconds."""
        cfg = m.TimeoutConfig(timeout_seconds=5.0)
        tm.that(cfg.timeout_seconds, eq=5.0)
        tm.that(cfg.error_code, eq=None)

    def test_timeout_config_with_error_code(self) -> None:
        """TimeoutConfig accepts optional error_code."""
        cfg = m.TimeoutConfig(timeout_seconds=30.0, error_code="TIMEOUT_001")
        tm.that(cfg.timeout_seconds, eq=30.0)
        tm.that(cfg.error_code, eq="TIMEOUT_001")

    def test_timeout_config_frozen(self) -> None:
        """TimeoutConfig is frozen (immutable)."""
        cfg = m.TimeoutConfig(timeout_seconds=5.0)
        with pytest.raises(ValidationError):
            cfg.timeout_seconds = 10.0  # type: ignore[misc]

    def test_timeout_config_forbids_extra(self) -> None:
        """TimeoutConfig forbids extra fields."""
        with pytest.raises(ValidationError):
            m.TimeoutConfig(timeout_seconds=5.0, unknown_field="x")  # type: ignore[call-arg]

    def test_timeout_config_rejects_zero(self) -> None:
        """TimeoutConfig rejects zero timeout (uses PositiveFloat)."""
        with pytest.raises(ValidationError):
            m.TimeoutConfig(timeout_seconds=0.0)

    def test_timeout_config_rejects_negative(self) -> None:
        """TimeoutConfig rejects negative timeout."""
        with pytest.raises(ValidationError):
            m.TimeoutConfig(timeout_seconds=-1.0)

    def test_timeout_config_serialization(self) -> None:
        """TimeoutConfig serializes to dict."""
        cfg = m.TimeoutConfig(timeout_seconds=10.5, error_code="SLOW")
        data = cfg.model_dump()
        tm.that(data["timeout_seconds"], eq=10.5)
        tm.that(data["error_code"], eq="SLOW")

    def test_timeout_config_json_round_trip(self) -> None:
        """TimeoutConfig survives JSON round-trip."""
        cfg = m.TimeoutConfig(timeout_seconds=7.5, error_code="TO")
        json_str = cfg.model_dump_json()
        restored = m.TimeoutConfig.model_validate_json(json_str)
        tm.that(restored.timeout_seconds, eq=7.5)
        tm.that(restored.error_code, eq="TO")


__all__ = ["TestFlextModelsErrors"]

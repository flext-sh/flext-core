"""Exception parameter resource and rate tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m


class TestsFlextModelsExceptionParamsResources:
    @staticmethod
    def test_authorization_error_params_defaults() -> None:
        params = m.AuthorizationErrorParams()
        tm.that(params.user_id, none=True)
        tm.that(params.resource, none=True)
        tm.that(params.permission, none=True)

    @staticmethod
    def test_authorization_error_params_with_values() -> None:
        params = m.AuthorizationErrorParams(
            user_id="user-123",
            resource="invoice:12345",
            permission="write",
        )
        tm.that(params.user_id, eq="user-123")
        tm.that(params.resource, eq="invoice:12345")
        tm.that(params.permission, eq="write")

    @staticmethod
    def test_authorization_error_params_serialization() -> None:
        params = m.AuthorizationErrorParams(
            user_id="admin",
            resource="settings",
            permission="delete",
        )
        data = params.model_dump()
        tm.that(data["user_id"], eq="admin")
        tm.that(data["resource"], eq="settings")
        tm.that(data["permission"], eq="delete")

    @staticmethod
    def test_authorization_error_params_round_trip() -> None:
        params = m.AuthorizationErrorParams(user_id="u1", permission="read")
        restored = m.AuthorizationErrorParams.model_validate(params.model_dump())
        tm.that(restored.user_id, eq="u1")
        tm.that(restored.permission, eq="read")
        tm.that(restored.resource, none=True)

    @staticmethod
    def test_authorization_error_params_forbids_extra_field() -> None:
        with pytest.raises(c.ValidationError):
            m.AuthorizationErrorParams.model_validate({"role": "admin"})

    @staticmethod
    def test_not_found_error_params_defaults() -> None:
        params = m.NotFoundErrorParams()
        tm.that(params.resource_type, none=True)
        tm.that(params.resource_id, none=True)

    @staticmethod
    def test_not_found_error_params_with_values() -> None:
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
    @staticmethod
    def test_not_found_error_params_resource_variants(
        resource_type: str | None,
        resource_id: str | None,
    ) -> None:
        params = m.NotFoundErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
        )
        tm.that(params.resource_type, eq=resource_type)
        tm.that(params.resource_id, eq=resource_id)

    @staticmethod
    def test_not_found_error_params_forbids_extra_field() -> None:
        with pytest.raises(c.ValidationError):
            m.NotFoundErrorParams.model_validate({"resource_name": "x"})

    @staticmethod
    def test_conflict_error_params_defaults() -> None:
        params = m.ConflictErrorParams()
        tm.that(params.resource_type, none=True)
        tm.that(params.resource_id, none=True)
        tm.that(params.conflict_reason, none=True)

    @staticmethod
    def test_conflict_error_params_with_values() -> None:
        params = m.ConflictErrorParams(
            resource_type="order",
            resource_id="ord-1001",
            conflict_reason="version_mismatch",
        )
        tm.that(params.resource_type, eq="order")
        tm.that(params.resource_id, eq="ord-1001")
        tm.that(params.conflict_reason, eq="version_mismatch")

    @staticmethod
    def test_conflict_error_params_serialization() -> None:
        params = m.ConflictErrorParams(
            resource_type="invoice",
            resource_id="inv-99",
            conflict_reason="duplicate_entry",
        )
        data = params.model_dump()
        tm.that(data["resource_type"], eq="invoice")
        tm.that(data["resource_id"], eq="inv-99")
        tm.that(data["conflict_reason"], eq="duplicate_entry")

    @staticmethod
    def test_rate_limit_error_params_defaults() -> None:
        params = m.RateLimitErrorParams()
        tm.that(params.limit, none=True)
        tm.that(params.window_seconds, none=True)
        tm.that(params.retry_after, none=True)

    @staticmethod
    def test_rate_limit_error_params_with_values() -> None:
        params = m.RateLimitErrorParams(limit=100, window_seconds=60, retry_after=1.5)
        tm.that(params.limit, eq=100)
        tm.that(params.window_seconds, eq=60)
        tm.that(params.retry_after, eq=1.5)

    @pytest.mark.parametrize(
        "retry_after",
        [1, 1.5, 0, None],
        ids=["int", "float", "zero", "none"],
    )
    @staticmethod
    def test_rate_limit_error_params_retry_after_types(
        retry_after: float | None,
    ) -> None:
        params = m.RateLimitErrorParams(retry_after=retry_after)
        tm.that(params.retry_after, eq=retry_after)

    @staticmethod
    def test_rate_limit_error_params_strict_rejects_string_limit() -> None:
        """strict=True: a numeric string is not coerced into an int limit."""
        with pytest.raises(c.ValidationError):
            m.RateLimitErrorParams(limit="100")

    @staticmethod
    def test_rate_limit_error_params_validate_assignment_rejects_bad_value() -> None:
        """validate_assignment=True: reassigning an invalid type raises."""
        params = m.RateLimitErrorParams(limit=10)
        tm.rejects_assignment(params, "limit", "not-an-int", expected=c.ValidationError)

    @staticmethod
    def test_circuit_breaker_error_params_defaults() -> None:
        params = m.CircuitBreakerErrorParams()
        tm.that(params.service_name, none=True)
        tm.that(params.failure_count, none=True)
        tm.that(params.reset_timeout, none=True)

    @staticmethod
    def test_circuit_breaker_error_params_with_values() -> None:
        params = m.CircuitBreakerErrorParams(
            service_name="payments-api",
            failure_count=5,
            reset_timeout=30.0,
        )
        tm.that(params.service_name, eq="payments-api")
        tm.that(params.failure_count, eq=5)
        tm.that(params.reset_timeout, eq=30.0)

    @staticmethod
    def test_circuit_breaker_error_params_serialization() -> None:
        params = m.CircuitBreakerErrorParams(
            service_name="auth-svc",
            failure_count=3,
            reset_timeout=15,
        )
        data = params.model_dump()
        tm.that(data["service_name"], eq="auth-svc")
        tm.that(data["failure_count"], eq=3)
        tm.that(data["reset_timeout"], eq=15)

    @staticmethod
    def test_circuit_breaker_error_params_forbids_extra_field() -> None:
        with pytest.raises(c.ValidationError):
            m.CircuitBreakerErrorParams.model_validate({"threshold": 1})

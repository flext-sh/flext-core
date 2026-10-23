"""Public contract tests for the FlextExceptions MRO facade.

These exercise observable behavior of the `FlextExceptions` facade (exposed as
`e`): the composed typed exception hierarchy, error-domain routing, template
rendering, occurrence metrics, the fail-DSL factories, and the enforcement
violation types — never private attributes or internal collaborators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final, cast

import pytest

from tests import c, e, m, p, tm


class TestsFlextCoreExceptions:
    """Behavioral contract of the FlextExceptions facade."""

    @staticmethod
    @pytest.mark.parametrize(
        "subclass",
        [
            e.ValidationError,
            e.ConfigurationError,
            e.FlextConnectionError,
            e.FlextTimeoutError,
            e.AuthenticationError,
            e.AuthorizationError,
            e.NotFoundError,
            e.ConflictError,
            e.RateLimitError,
            e.CircuitBreakerError,
            e.FlextTypeError,
            e.OperationError,
            e.AttributeAccessError,
        ],
    )
    def test_facade_exposes_typed_errors_as_base_error_subclasses(
        subclass: type[e.BaseError],
    ) -> None:
        # Act / Assert — every typed error is reachable on the facade and is a
        # catchable member of the BaseError hierarchy.
        """Test facade exposes typed errors as base error subclasses."""
        tm.that(subclass, where=e.BaseError.__subclasscheck__)
        tm.that(subclass, where=Exception.__subclasscheck__)

    @staticmethod
    @pytest.mark.parametrize(
        ("error", "expected_domain"),
        [
            (e.ValidationError("bad", field="email"), c.ErrorDomain.VALIDATION),
            (e.AuthenticationError("nope"), c.ErrorDomain.AUTH),
            (e.AuthorizationError("nope"), c.ErrorDomain.AUTH),
            (e.NotFoundError("missing"), c.ErrorDomain.NOT_FOUND),
            (e.FlextConnectionError("down"), c.ErrorDomain.NETWORK),
            (e.FlextTimeoutError("slow"), c.ErrorDomain.TIMEOUT),
        ],
    )
    def test_error_domain_routes_typed_errors(
        error: e.BaseError,
        expected_domain: str,
    ) -> None:
        # Assert — the public routing domain is derived from the error code.
        """Test error domain routes typed errors."""
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.matches_error_domain(expected_domain) is True, truth=True)
        tm.that(error.matches_error_domain("UNRELATED_DOMAIN") is False, truth=True)

    @staticmethod
    def test_error_message_property_exposes_message() -> None:
        # Arrange / Act
        """Test error message property exposes message."""
        error = e.ValidationError("field is required", field="name")
        # Assert — public message accessor and str formatting.
        tm.that(error.error_message, eq="field is required")
        tm.that("field is required" in str(error), truth=True)

    @staticmethod
    def test_typed_error_raises_and_carries_public_fields() -> None:
        # Act / Assert
        """Test typed error raises and carries public fields.

        Raises:
            NotFoundError: If not found.

        """
        message = "not found"
        with pytest.raises(e.BaseError) as excinfo:
            raise e.NotFoundError(
                message, resource_type=c.ContextScope.USER, resource_id="u-1"
            )
        raised = cast("e.NotFoundError", excinfo.value)
        tm.that(raised, is_=e.NotFoundError)
        tm.that(raised.resource_type, eq=c.ContextScope.USER)
        tm.that(raised.resource_id, eq="u-1")

    @staticmethod
    def test_auto_correlation_generates_public_correlation_id() -> None:
        # Act
        """Test auto correlation generates public correlation id."""
        error = e.ValidationError(
            "bad",
            field="email",
            options=m.ExceptionInitOptions(auto_correlation=True),
        )
        # Assert — correlation id is generated and exposed publicly.
        tm.that(error.correlation_id, none=False)
        tm.that(str(error.correlation_id).startswith("exc_"), truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("declared", "expected"),
        [("int", int), (str, str), ("dict", dict)],
    )
    def test_type_error_resolves_type_names_to_types(
        declared: type | str,
        expected: type,
    ) -> None:
        # Act — TypeError accepts either a type or its name and resolves both.
        """Test type error resolves type names to types."""
        error = e.FlextTypeError(
            "type mismatch",
            expected_type=declared,
            actual_type=declared,
        )
        # Assert
        tm.that(error.expected_type is expected, truth=True)
        tm.that(error.actual_type is expected, truth=True)

    @staticmethod
    def test_render_template_substitutes_provided_values() -> None:
        # Act
        """Test render template substitutes provided values."""
        rendered = e.render_template("Hello {name} from {place}", name="A", place="B")
        # Assert
        tm.that(rendered, eq="Hello A from B")

    @staticmethod
    def test_render_template_fails_fast_on_missing_placeholder() -> None:
        # Assert — a missing placeholder is a loud ValueError, never silent.
        """Test render template fails fast on missing placeholder."""
        with pytest.raises(ValueError, match="missing"):
            e.render_template("Hi {missing}")

    @staticmethod
    def test_metrics_record_snapshot_and_clear_roundtrip() -> None:
        # Arrange — start from a clean slate (public API only).
        """Test metrics record snapshot and clear roundtrip."""
        e.clear_metrics()
        # Act
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.NotFoundError)
        snapshot = e.resolve_metrics_snapshot()
        # Assert — public snapshot reflects recorded occurrences.
        tm.that(snapshot.total_exceptions, eq=3)
        tm.that(snapshot.has_exceptions is True, truth=True)
        # Act — clearing resets the observable totals.
        e.clear_metrics()
        cleared = e.resolve_metrics_snapshot()
        # Assert
        tm.that(cleared.total_exceptions, eq=0)
        tm.that(cleared.has_exceptions is False, truth=True)

    @staticmethod
    def test_fail_conflict_returns_structured_failure() -> None:
        """Test fail conflict returns structured failure."""
        result: p.Result[bool] = e.fail_conflict(
            c.ContextScope.USER, "u-1", "already active"
        )
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that(result.error_code, eq=c.ErrorCode.ALREADY_EXISTS)
        tm.that(result.error_data, none=False)
        tm.that(
            cast("dict[str, object]", result.error_data)["resource_type"],
            eq=c.ContextScope.USER,
        )
        tm.that(cast("dict[str, object]", result.error_data)["resource_id"], eq="u-1")

    @staticmethod
    def test_fail_auth_returns_structured_failure() -> None:
        """Test fail auth returns structured failure."""
        result: p.Result[bool] = e.fail_auth("ldap", "user1")
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("authenticate user user1" in str(result.error), truth=True)
        tm.that(result.error_code, eq=c.ErrorCode.AUTHENTICATION_ERROR)
        tm.that(result.error_data, none=False)
        tm.that(cast("dict[str, object]", result.error_data)["auth_method"], eq="ldap")
        tm.that(
            cast("dict[str, object]", result.error_data)[c.ContextKey.USER_ID],
            eq="user1",
        )

    @staticmethod
    def test_fail_authz_returns_structured_failure() -> None:
        """Test fail authz returns structured failure."""
        result: p.Result[bool] = e.fail_authz("u-1", "admin.panel", "write")
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that(result.error_code, eq=c.ErrorCode.AUTHORIZATION_ERROR)
        tm.that(result.error_data, none=False)
        tm.that(
            cast("dict[str, object]", result.error_data)[c.ContextKey.USER_ID], eq="u-1"
        )
        tm.that(
            cast("dict[str, object]", result.error_data)["resource"], eq="admin.panel"
        )
        tm.that(cast("dict[str, object]", result.error_data)["permission"], eq="write")

    @staticmethod
    def test_fail_connection_returns_structured_failure() -> None:
        """Test fail connection returns structured failure."""
        result: p.Result[bool] = e.fail_connection("ldap.example.com")
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("connect to ldap.example.com" in str(result.error), truth=True)
        tm.that(result.error_code, eq=c.ErrorCode.CONNECTION_ERROR)
        tm.that(result.error_data, none=False)
        tm.that(
            cast("dict[str, object]", result.error_data)["host"], eq="ldap.example.com"
        )

    @staticmethod
    def test_fail_timeout_returns_structured_failure() -> None:
        """Test fail timeout returns structured failure."""
        result: p.Result[bool] = e.fail_timeout(30.0, "fetch_users")
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("fetch_users" in str(result.error), truth=True)
        tm.that(result.error_code, eq=c.ErrorCode.TIMEOUT_ERROR)
        tm.that(result.error_data, none=False)
        tm.that(
            cast("dict[str, object]", result.error_data)[c.ContextScope.OPERATION],
            eq="fetch_users",
        )
        tm.that(
            cast("dict[str, object]", result.error_data)["timeout_seconds"],
            eq=pytest.approx(30.0),
        )

    @staticmethod
    def test_failure_result_short_circuits_map_and_recovers() -> None:
        # Arrange
        """Test failure result short circuits map and recovers."""
        result: p.Result[bool] = e.fail_timeout(5.0, "ping")
        # Act — map over a failure is a no-op; unwrap_or supplies the default.
        mapped = result.map(lambda _value: True)
        # Assert
        tm.that(mapped.failure, truth=True)
        tm.that(mapped.error, eq=result.error)
        # unwrap_or on failure returns the default
        unwrap_default: Final[bool] = False
        default_result = result.unwrap_or(unwrap_default)
        tm.that(default_result is unwrap_default, truth=True)

    @staticmethod
    @pytest.mark.parametrize("violation", [e.MroViolation, e.SmellViolation])
    def test_enforcement_violations_are_raisable_exception_types(
        violation: type[Exception],
    ) -> None:
        # Assert — enforcement violations are exposed on the facade and raise.
        """Test enforcement violations are raisable exception types."""
        tm.that(violation, where=Exception.__subclasscheck__)
        message = "enforcement breach"
        with pytest.raises(violation):
            raise violation(message)

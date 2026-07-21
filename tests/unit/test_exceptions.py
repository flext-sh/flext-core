"""Public contract tests for the FlextExceptions MRO facade.

These exercise observable behavior of the `FlextExceptions` facade (exposed as
`e`): the composed typed exception hierarchy, error-domain routing, template
rendering, occurrence metrics, the fail-DSL factories, and the enforcement
violation types — never private attributes or internal collaborators.
"""

from __future__ import annotations

import pytest

from flext_tests import e
from tests import c, p


class TestsFlextCoreExceptions:
    """Behavioral contract of the FlextExceptions facade."""

    @pytest.mark.parametrize(
        "subclass",
        [
            e.ValidationError,
            e.ConfigurationError,
            e.ConnectionError,
            e.TimeoutError,
            e.AuthenticationError,
            e.AuthorizationError,
            e.NotFoundError,
            e.ConflictError,
            e.RateLimitError,
            e.CircuitBreakerError,
            e.TypeError,
            e.OperationError,
            e.AttributeAccessError,
        ],
    )
    def test_facade_exposes_typed_errors_as_base_error_subclasses(
        self, subclass: type[e.BaseError]
    ) -> None:
        # Act / Assert — every typed error is reachable on the facade and is a
        # catchable member of the BaseError hierarchy.
        assert issubclass(subclass, e.BaseError)
        assert issubclass(subclass, Exception)

    @pytest.mark.parametrize(
        ("error", "expected_domain"),
        [
            (e.ValidationError("bad", field="email"), c.ErrorDomain.VALIDATION),
            (e.AuthenticationError("nope"), c.ErrorDomain.AUTH),
            (e.AuthorizationError("nope"), c.ErrorDomain.AUTH),
            (e.NotFoundError("missing"), c.ErrorDomain.NOT_FOUND),
            (e.ConnectionError("down"), c.ErrorDomain.NETWORK),
            (e.TimeoutError("slow"), c.ErrorDomain.TIMEOUT),
        ],
    )
    def test_error_domain_routes_typed_errors(
        self, error: e.BaseError, expected_domain: str
    ) -> None:
        # Assert — the public routing domain is derived from the error code.
        assert error.error_domain == expected_domain
        assert error.matches_error_domain(expected_domain) is True
        assert error.matches_error_domain("UNRELATED_DOMAIN") is False

    def test_error_message_property_exposes_message(self) -> None:
        # Arrange / Act
        error = e.ValidationError("field is required", field="name")
        # Assert — public message accessor and str formatting.
        assert error.error_message == "field is required"
        assert "field is required" in str(error)

    def test_typed_error_raises_and_carries_public_fields(self) -> None:
        # Act / Assert
        message = "not found"
        with pytest.raises(e.BaseError) as excinfo:
            raise e.NotFoundError(message, resource_type="user", resource_id="u-1")
        raised = excinfo.value
        assert isinstance(raised, e.NotFoundError)
        assert raised.resource_type == "user"
        assert raised.resource_id == "u-1"

    def test_auto_correlation_generates_public_correlation_id(self) -> None:
        # Act
        error = e.ValidationError("bad", field="email", auto_correlation=True)
        # Assert — correlation id is generated and exposed publicly.
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")

    @pytest.mark.parametrize(
        ("declared", "expected"), [("int", int), (str, str), ("dict", dict)]
    )
    def test_type_error_resolves_type_names_to_types(
        self, declared: type | str, expected: type
    ) -> None:
        # Act — TypeError accepts either a type or its name and resolves both.
        error = e.TypeError(
            "type mismatch", expected_type=declared, actual_type=declared
        )
        # Assert
        assert error.expected_type is expected
        assert error.actual_type is expected

    def test_render_template_substitutes_provided_values(self) -> None:
        # Act
        rendered = e.render_template("Hello {name} from {place}", name="A", place="B")
        # Assert
        assert rendered == "Hello A from B"

    def test_render_template_fails_fast_on_missing_placeholder(self) -> None:
        # Assert — a missing placeholder is a loud ValueError, never silent.
        with pytest.raises(ValueError, match="missing"):
            e.render_template("Hi {missing}")

    def test_metrics_record_snapshot_and_clear_roundtrip(self) -> None:
        # Arrange — start from a clean slate (public API only).
        e.clear_metrics()
        # Act
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.NotFoundError)
        snapshot = e.resolve_metrics_snapshot()
        # Assert — public snapshot reflects recorded occurrences.
        assert snapshot.total_exceptions == 3
        assert snapshot.has_exceptions is True
        # Act — clearing resets the observable totals.
        e.clear_metrics()
        cleared = e.resolve_metrics_snapshot()
        # Assert
        assert cleared.total_exceptions == 0
        assert cleared.has_exceptions is False

    def test_fail_conflict_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_conflict("user", "u-1", "already active")
        assert result.failure
        assert result.error is not None
        assert result.error_code == c.ErrorCode.ALREADY_EXISTS
        assert result.error_data is not None
        assert result.error_data["resource_type"] == "user"
        assert result.error_data["resource_id"] == "u-1"

    def test_fail_auth_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_auth("ldap", "user1")
        assert result.failure
        assert result.error is not None
        assert "authenticate user user1" in result.error
        assert result.error_code == c.ErrorCode.AUTHENTICATION_ERROR
        assert result.error_data is not None
        assert result.error_data["auth_method"] == "ldap"
        assert result.error_data["user_id"] == "user1"

    def test_fail_authz_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_authz("u-1", "admin.panel", "write")
        assert result.failure
        assert result.error is not None
        assert result.error_code == c.ErrorCode.AUTHORIZATION_ERROR
        assert result.error_data is not None
        assert result.error_data["user_id"] == "u-1"
        assert result.error_data["resource"] == "admin.panel"
        assert result.error_data["permission"] == "write"

    def test_fail_connection_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_connection("ldap.example.com")
        assert result.failure
        assert result.error is not None
        assert "connect to ldap.example.com" in result.error
        assert result.error_code == c.ErrorCode.CONNECTION_ERROR
        assert result.error_data is not None
        assert result.error_data["host"] == "ldap.example.com"

    def test_fail_timeout_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_timeout(30.0, "fetch_users")
        assert result.failure
        assert result.error is not None
        assert "fetch_users" in result.error
        assert result.error_code == c.ErrorCode.TIMEOUT_ERROR
        assert result.error_data is not None
        assert result.error_data["operation"] == "fetch_users"
        assert result.error_data["timeout_seconds"] == pytest.approx(30.0)

    def test_failure_result_short_circuits_map_and_recovers(self) -> None:
        # Arrange
        result: p.Result[bool] = e.fail_timeout(5.0, "ping")
        # Act — map over a failure is a no-op; unwrap_or supplies the default.
        mapped = result.map(lambda _value: True)
        # Assert
        assert mapped.failure
        assert mapped.error == result.error
        assert result.unwrap_or(False) is False

    @pytest.mark.parametrize("violation", [e.MroViolation, e.SmellViolation])
    def test_enforcement_violations_are_raisable_exception_types(
        self, violation: type[Exception]
    ) -> None:
        # Assert — enforcement violations are exposed on the facade and raise.
        assert issubclass(violation, Exception)
        message = "enforcement breach"
        with pytest.raises(violation):
            raise violation(message)

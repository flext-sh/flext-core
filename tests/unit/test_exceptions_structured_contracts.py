"""Structured exception public contract tests.

Behavioral only: every assertion exercises the observable public surface of the
``FlextExceptions`` family and the ``r[T]`` failure contract exposed through the
``flext_tests`` facades. No private attributes, no monkeypatching, no spying on
internal collaborators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence

from tests import c, e, m, p, tm, u

type FailureFactory = Callable[[], p.Result[bool]]

FAILURES: Sequence[
    tuple[str, FailureFactory, str, str, Mapping[str, str | int | None]]
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
            params=m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5),
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
        {"timeout_seconds": 30, c.ContextScope.OPERATION: "dispatch"},
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
        {"auth_method": "token", c.ContextKey.USER_ID: "u-1"},
    ),
    (
        "authz",
        lambda: e.fail_authz("u-1", "admin.panel", "write"),
        "authorize",
        c.ErrorCode.AUTHORIZATION_ERROR,
        {c.ContextKey.USER_ID: "u-1", "resource": "admin.panel", "permission": "write"},
    ),
    (
        "conflict",
        lambda: e.fail_conflict(c.ContextScope.USER, "u-1", "duplicate"),
        "create user",
        c.ErrorCode.ALREADY_EXISTS,
        {
            "resource_type": c.ContextScope.USER,
            "resource_id": "u-1",
            "conflict_reason": "duplicate",
        },
    ),
    (
        c.ContextScope.OPERATION,
        lambda: e.fail_operation("resolve service", "timeout"),
        "resolve service",
        c.ErrorCode.OPERATION_ERROR,
        {c.ContextScope.OPERATION: "resolve service", "reason": "timeout"},
    ),
    (
        "operation_exception",
        lambda: e.fail_operation("register service", ValueError("boom")),
        "register service",
        c.ErrorCode.OPERATION_ERROR,
        {c.ContextScope.OPERATION: "register service", "reason": "boom"},
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
        lambda: e.fail_type_mismatch("FlextUtilitiesLogging", "str"),
        "FlextUtilitiesLogging",
        c.ErrorCode.TYPE_ERROR,
        {"expected_type": "FlextUtilitiesLogging", "actual_type": "str"},
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

type ErrorFactory = Callable[[], e.BaseError]

STRUCTURED_ERRORS: Sequence[
    tuple[str, ErrorFactory, str, str, Mapping[str, str | int | None]]
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
        lambda: e.FlextConnectionError(
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
        lambda: e.FlextTimeoutError(
            "Timed out",
            timeout_seconds=30,
            operation="dispatch",
        ),
        c.ErrorDomain.TIMEOUT.value,
        c.ErrorCode.TIMEOUT_ERROR,
        {"timeout_seconds": 30, c.ContextScope.OPERATION: "dispatch"},
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
        {"auth_method": "token", c.ContextKey.USER_ID: "u-1"},
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
        {c.ContextKey.USER_ID: "u-1", "resource": "admin.panel", "permission": "write"},
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
        {"resource_type": "User", "resource_id": "123", "conflict_reason": "duplicate"},
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
        {
            c.ContextKey.SERVICE_NAME: "payments",
            "failure_count": 5,
            "reset_timeout": 60,
        },
    ),
    (
        "type_error",
        lambda: e.FlextTypeError("Wrong type", expected_type="str", actual_type=int),
        c.ErrorDomain.VALIDATION.value,
        c.ErrorCode.TYPE_ERROR,
        {"expected_type": "str", "actual_type": "int"},
    ),
]


class TestsFlextCoreExceptionsStructuredContracts:
    """Public-contract behavior of structured errors and failure results."""

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_expose_public_contract(
        name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: Mapping[str, str | int | None],
    ) -> None:
        # Arrange / Act
        """Test structured errors expose public contract."""
        error = factory()

        # Assert: structured protocol surface
        tm.that(error, is_=p.StructuredError, msg=name)
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.error_code, eq=expected_code)
        tm.that(error.error_message, eq=error.message)
        tm.that(error.matches_error_domain(expected_domain), truth=True)
        tm.that(error.matches_error_domain("definitely-not-a-domain"), truth=False)

        # Assert: caller-visible payload flows through public metadata
        for key, value in expected_payload.items():
            tm.that(error.metadata.attributes[key], eq=value)

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_are_raisable_and_preserve_contract(
        name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: Mapping[str, str | int | None],
    ) -> None:
        # Act / Assert: the error is a real exception that callers can catch,
        # and its structured contract survives raise/except unchanged.
        """Test structured errors are raisable and preserve contract."""
        tm.that(name, truth=True)
        tm.that(expected_payload, none=False)
        with u.Tests.raises(e.BaseError) as caught:
            raise factory()

        error = caught.value
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.error_code, eq=expected_code)
        tm.that(error.matches_error_domain(expected_domain), truth=True)
        tm.that(error.message in str(error), truth=True)
        tm.that(error.error_message, eq=error.message)
        for key, value in expected_payload.items():
            tm.that(error.metadata.attributes[key], eq=value)

    @staticmethod
    def test_base_error_defaults_to_unknown_domain_protocol_surface() -> None:
        # Arrange / Act
        """Test base error defaults to unknown domain protocol surface."""
        error = e.BaseError("Base failure")

        # Assert
        tm.that(error, is_=p.StructuredError)
        tm.that(error.error_domain, eq=c.ErrorDomain.UNKNOWN.value)
        tm.that(error.error_code, eq=c.ErrorCode.UNKNOWN_ERROR)
        tm.that(error.error_message, eq="Base failure")
        tm.that(error.matches_error_domain(c.ErrorDomain.UNKNOWN.value), truth=True)

    @staticmethod
    def test_specific_errors_are_catchable_as_base_error() -> None:
        # Assert: the family shares a common catchable base contract.
        """Test specific errors are catchable as base error."""
        not_found = e.NotFoundError(
            "User missing",
            resource_type="User",
            resource_id="123",
        )
        with u.Tests.raises(e.BaseError) as caught:
            raise not_found

        error = caught.value
        tm.that(isinstance(error, e.NotFoundError), truth=True)
        tm.that(error.error_code, eq=c.ErrorCode.NOT_FOUND_ERROR)
        tm.that(error.error_domain, eq=c.ErrorDomain.NOT_FOUND.value)

    @staticmethod
    def test_metadata_attributes_preserve_correlation_id() -> None:
        # Arrange / Act
        """Test metadata attributes preserve correlation id."""
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

        # Assert: caller-supplied context is observable on the public surface
        tm.that(error.message, eq="Insert failed")
        tm.that(error.error_code, eq=c.ErrorCode.OPERATION_ERROR)
        tm.that(error.error_domain, eq=c.ErrorDomain.INTERNAL.value)
        tm.that(error.correlation_id, eq="corr-123")
        expected_attributes: Mapping[str, str | int] = {
            "scope": "users",
            c.ContextScope.OPERATION: "insert_user",
            "reason": "constraint",
            "attempt": 2,
        }
        for key, value in expected_attributes.items():
            tm.that(error.metadata.attributes[key], eq=value)

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_factories_return_public_result_contract(
        name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: Mapping[str, str | int | None],
    ) -> None:
        # Arrange / Act
        """Test failure factories return public result contract."""
        result = factory()
        result_data = result.error_data or {}

        # Assert: r[T] failure contract
        tm.that(result.failure, truth=True, msg=name)
        tm.that(result.success, truth=False)
        tm.that(result.error, none=False)
        tm.that(expected_fragment in str(result.error), truth=True)
        tm.that(result.error_code, eq=expected_code)
        tm.that(result.error_data, none=False)
        for key, value in expected_data.items():
            tm.that(result_data[key], eq=value)

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_results_honor_combinator_contract(
        name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: Mapping[str, str | int | None],
    ) -> None:
        # Arrange
        """Test failure results honor combinator contract."""
        result = factory()

        # Act / Assert: map short-circuits, preserving the failure, its code,
        # message and data.
        mapped = result.map(operator.not_)
        tm.that(mapped.failure, truth=True, msg=name)
        tm.that(mapped.error_code, eq=expected_code)
        tm.that(expected_fragment in str(mapped.error), truth=True)
        tm.that(
            {key: (mapped.error_data or {})[key] for key in expected_data},
            eq=dict(expected_data),
        )

        # Strict extraction preserves the failure instead of inventing a value.
        if result.exception is not None:
            with u.Tests.raises(type(result.exception)) as raised:
                result.unwrap()
            tm.that(raised.value is result.exception, truth=True)
        else:
            with u.Tests.raises(RuntimeError) as logical:
                result.unwrap()
            tm.that(
                str(logical.value),
                eq=c.ERR_RESULT_CANNOT_UNWRAP.format(error=result.error),
            )

        # recover converts a failure into a caller-defined success value.
        recovered = result.recover(lambda _error: False)
        tm.ok(recovered, eq=False)

        # tap_error observes the error without altering the failure channel.
        observed: list[bool] = []
        tapped = result.tap_error(lambda _err: observed.append(True))
        tm.that(tapped.failure, truth=True)
        tm.that(len(observed), eq=1)

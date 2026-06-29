"""Structured exception contract fixtures."""

from __future__ import annotations

from collections.abc import Callable

from flext_tests import e

from tests.constants import c
from tests.typings import t

type ErrorFactory = Callable[[], e.BaseError]

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

"""Failure factory contract fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from flext_tests import e
from tests.constants import c
from tests.models import m
from tests.protocols import p

if TYPE_CHECKING:
    from tests.typings import t

type FailureFactory = Callable[[], p.Result[bool]]

FAILURES: t.SequenceOf[
    tuple[str, FailureFactory, str, str, dict[str, str | int | None]]
] = [
    (
        "config",
        lambda: e.fail_config_error(
            "API_KEY", "environment", options=m.ExceptionFactoryOptions(error="missing")
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
        {"timeout_seconds": 30, "operation": "dispatch"},
    ),
    (
        "auth",
        lambda: e.fail_auth(
            "token", "u-1", options=m.ExceptionFactoryOptions(error="denied")
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
        {"resource_type": "user", "resource_id": "u-1", "conflict_reason": "duplicate"},
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
        lambda: e.fail_type_mismatch("FlextUtilitiesLogging", "str"),
        "FlextUtilitiesLogging",
        c.ErrorCode.TYPE_ERROR,
        {"expected_type": "FlextUtilitiesLogging", "actual_type": "str"},
    ),
    (
        "validation",
        lambda: e.fail_validation(
            m.ValidationErrorParams(field="email", value="bad"), error="invalid"
        ),
        "validate email",
        c.ErrorCode.VALIDATION_ERROR,
        {"field": "email", "value": "bad"},
    ),
]

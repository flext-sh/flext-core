"""Exception example sections kept below the module LOC cap."""

from __future__ import annotations

from flext_core import e

from .models import m
from .shared import ExamplesFlextShared


class Ex07FlextExceptionSubclasses(ExamplesFlextShared):
    """Exercise structured exception subclasses."""

    def _exercise_specific_exceptions(self) -> None:
        self.section("subclasses")
        try:
            raise e.ValidationError(
                m.Examples.ErrorMessages.INVALID, field="email", value="bad"
            )
        except e.ValidationError as exc:
            self.audit_check("ValidationError.field", exc.field or "")
            self.audit_check("ValidationError.value", str(exc.value or ""))
        try:
            raise e.ConfigurationError(
                m.Examples.ErrorMessages.BAD_CFG,
                config_key="db.host",
                config_source="env",
            )
        except e.ConfigurationError as exc:
            self.audit_check("ConfigurationError.config_key", exc.config_key or "")
            self.audit_check(
                "ConfigurationError.config_source", exc.config_source or ""
            )
        try:
            raise e.ConnectionError(
                m.Examples.ErrorMessages.DOWN, host="127.0.0.1", port=5432, timeout=3.5
            )
        except e.ConnectionError as exc:
            self.audit_check("ConnectionError.host", exc.host or "")
            self.audit_check("ConnectionError.port", exc.port or 0)
            self.audit_check("ConnectionError.timeout", exc.timeout or 0.0)
        try:
            raise e.TimeoutError(
                m.Examples.ErrorMessages.LATE, timeout_seconds=2.0, operation="sync"
            )
        except e.TimeoutError as exc:
            self.audit_check("TimeoutError.timeout_seconds", exc.timeout_seconds or 0.0)
            self.audit_check("TimeoutError.operation", exc.operation or "")
        try:
            raise e.AuthenticationError(
                m.Examples.ErrorMessages.AUTH_FAIL, auth_method="token", user_id="u-1"
            )
        except e.AuthenticationError as exc:
            self.audit_check("AuthenticationError.auth_method", exc.auth_method or "")
            self.audit_check("AuthenticationError.user_id", exc.user_id or "")
        try:
            raise e.AuthorizationError(
                m.Examples.ErrorMessages.NOPE,
                user_id="u-2",
                resource="invoice:7",
                permission="read",
            )
        except e.AuthorizationError as exc:
            self.audit_check("AuthorizationError.user_id", exc.user_id or "")
            self.audit_check("AuthorizationError.resource", exc.resource or "")
            self.audit_check("AuthorizationError.permission", exc.permission or "")
        try:
            raise e.NotFoundError(
                m.Examples.ErrorMessages.MISSING,
                resource_type="User",
                resource_id="404",
            )
        except e.NotFoundError as exc:
            self.audit_check("NotFoundError.resource_type", exc.resource_type or "")
            self.audit_check("NotFoundError.resource_id", exc.resource_id or "")
        try:
            raise e.ConflictError(
                m.Examples.ErrorMessages.CONFLICT,
                resource_type="User",
                resource_id="13",
                conflict_reason="duplicate",
            )
        except e.ConflictError as exc:
            self.audit_check("ConflictError.resource_type", exc.resource_type or "")
            self.audit_check("ConflictError.resource_id", exc.resource_id or "")
            self.audit_check("ConflictError.conflict_reason", exc.conflict_reason or "")
        try:
            raise e.RateLimitError(
                m.Examples.ErrorMessages.SLOW_DOWN,
                limit=100,
                window_seconds=60,
                retry_after=1.5,
            )
        except e.RateLimitError as exc:
            self.audit_check("RateLimitError.limit", exc.limit or 0)
            self.audit_check("RateLimitError.window_seconds", exc.window_seconds or 0)
            self.audit_check("RateLimitError.retry_after", exc.retry_after or 0.0)
        try:
            raise e.CircuitBreakerError(
                m.Examples.ErrorMessages.OPEN,
                service_name="billing",
                failure_count=5,
                reset_timeout=30.0,
            )
        except e.CircuitBreakerError as exc:
            self.audit_check("CircuitBreakerError.service_name", exc.service_name or "")
            self.audit_check(
                "CircuitBreakerError.failure_count", exc.failure_count or 0
            )
            self.audit_check(
                "CircuitBreakerError.reset_timeout", exc.reset_timeout or 0.0
            )
        try:
            raise e.TypeError(
                m.Examples.ErrorMessages.WRONG_TYPE, expected_type=str, actual_type=int
            )
        except e.TypeError as exc:
            self.audit_check(
                "TypeError.expected_type",
                exc.expected_type.__name__ if exc.expected_type else "",
            )
            self.audit_check(
                "TypeError.actual_type",
                exc.actual_type.__name__ if exc.actual_type else "",
            )
        try:
            raise e.OperationError(
                m.Examples.ErrorMessages.FAILED_OP, operation="publish", reason="quota"
            )
        except e.OperationError as exc:
            self.audit_check("OperationError.operation", exc.operation or "")
            self.audit_check("OperationError.reason", exc.reason or "")
        try:
            raise e.AttributeAccessError(
                m.Examples.ErrorMessages.BAD_ATTR,
                attribute_name="secret",
                attribute_context="UserModel",
            )
        except e.AttributeAccessError as exc:
            self.audit_check(
                "AttributeAccessError.attribute_name", exc.attribute_name or ""
            )
            self.audit_check(
                "AttributeAccessError.attribute_context",
                str(exc.attribute_context or ""),
            )

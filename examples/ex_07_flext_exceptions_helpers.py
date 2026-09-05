"""Exception example sections kept below the module LOC cap."""

from __future__ import annotations

from flext_core import e

from .models import m
from .shared import ExamplesFlextShared


def _raise_attribute_access_error() -> None:
    """Raise one AttributeAccessError example for handler exercise."""
    raise e.AttributeAccessError(
        m.Examples.ErrorMessages.BAD_ATTR,
        attribute_name="secret",
        attribute_context="UserModel",
    )


def _raise_authentication_error() -> None:
    """Raise one AuthenticationError example for handler exercise."""
    raise e.AuthenticationError(
        m.Examples.ErrorMessages.AUTH_FAIL, auth_method="token", user_id="u-1"
    )


def _raise_authorization_error() -> None:
    """Raise one AuthorizationError example for handler exercise."""
    raise e.AuthorizationError(
        m.Examples.ErrorMessages.NOPE,
        user_id="u-2",
        resource="invoice:7",
        permission="read",
    )


def _raise_circuit_breaker_error() -> None:
    """Raise one CircuitBreakerError example for handler exercise."""
    raise e.CircuitBreakerError(
        m.Examples.ErrorMessages.OPEN,
        service_name="billing",
        failure_count=5,
        reset_timeout=30.0,
    )


def _raise_configuration_error() -> None:
    """Raise one ConfigurationError example for handler exercise."""
    raise e.ConfigurationError(
        m.Examples.ErrorMessages.BAD_CFG,
        config_key="db.host",
        config_source="env",
    )


def _raise_conflict_error() -> None:
    """Raise one ConflictError example for handler exercise."""
    raise e.ConflictError(
        m.Examples.ErrorMessages.CONFLICT,
        resource_type="User",
        resource_id="13",
        conflict_reason="duplicate",
    )


def _raise_flext_connection_error() -> None:
    """Raise one FlextConnectionError example for handler exercise."""
    raise e.FlextConnectionError(
        m.Examples.ErrorMessages.DOWN, host="127.0.0.1", port=5432, timeout=3.5
    )


def _raise_flext_timeout_error() -> None:
    """Raise one FlextTimeoutError example for handler exercise."""
    raise e.FlextTimeoutError(
        m.Examples.ErrorMessages.LATE, timeout_seconds=2.0, operation="sync"
    )


def _raise_flext_type_error() -> None:
    """Raise one FlextTypeError example for handler exercise."""
    raise e.FlextTypeError(
        m.Examples.ErrorMessages.WRONG_TYPE, expected_type=str, actual_type=int
    )


def _raise_not_found_error() -> None:
    """Raise one NotFoundError example for handler exercise."""
    raise e.NotFoundError(
        m.Examples.ErrorMessages.MISSING,
        resource_type="User",
        resource_id="404",
    )


def _raise_operation_error() -> None:
    """Raise one OperationError example for handler exercise."""
    raise e.OperationError(
        m.Examples.ErrorMessages.FAILED_OP, operation="publish", reason="quota"
    )


def _raise_rate_limit_error() -> None:
    """Raise one RateLimitError example for handler exercise."""
    raise e.RateLimitError(
        m.Examples.ErrorMessages.SLOW_DOWN,
        limit=100,
        window_seconds=60,
        retry_after=1.5,
    )


def _raise_validation_error() -> None:
    """Raise one ValidationError example for handler exercise."""
    raise e.ValidationError(
        m.Examples.ErrorMessages.INVALID, field="email", value="bad"
    )


class Ex07FlextExceptionSubclasses(ExamplesFlextShared):
    """Exercise structured exception subclasses."""

    def _exercise_specific_exceptions(self) -> None:
        self.section("subclasses")
        try:
            _raise_validation_error()
        except e.ValidationError as exc:
            self.audit_check("ValidationError.field", exc.field or "")
            self.audit_check("ValidationError.value", str(exc.value or ""))
        try:
            _raise_configuration_error()
        except e.ConfigurationError as exc:
            self.audit_check("ConfigurationError.config_key", exc.config_key or "")
            self.audit_check(
                "ConfigurationError.config_source", exc.config_source or ""
            )
        try:
            _raise_flext_connection_error()
        except e.FlextConnectionError as exc:
            self.audit_check("ConnectionError.host", exc.host or "")
            self.audit_check("ConnectionError.port", exc.port or 0)
            self.audit_check("ConnectionError.timeout", exc.timeout or 0.0)
        try:
            _raise_flext_timeout_error()
        except e.FlextTimeoutError as exc:
            self.audit_check("TimeoutError.timeout_seconds", exc.timeout_seconds or 0.0)
            self.audit_check("TimeoutError.operation", exc.operation or "")
        try:
            _raise_authentication_error()
        except e.AuthenticationError as exc:
            self.audit_check("AuthenticationError.auth_method", exc.auth_method or "")
            self.audit_check("AuthenticationError.user_id", exc.user_id or "")
        try:
            _raise_authorization_error()
        except e.AuthorizationError as exc:
            self.audit_check("AuthorizationError.user_id", exc.user_id or "")
            self.audit_check("AuthorizationError.resource", exc.resource or "")
            self.audit_check("AuthorizationError.permission", exc.permission or "")
        try:
            _raise_not_found_error()
        except e.NotFoundError as exc:
            self.audit_check("NotFoundError.resource_type", exc.resource_type or "")
            self.audit_check("NotFoundError.resource_id", exc.resource_id or "")
        try:
            _raise_conflict_error()
        except e.ConflictError as exc:
            self.audit_check("ConflictError.resource_type", exc.resource_type or "")
            self.audit_check("ConflictError.resource_id", exc.resource_id or "")
            self.audit_check("ConflictError.conflict_reason", exc.conflict_reason or "")
        try:
            _raise_rate_limit_error()
        except e.RateLimitError as exc:
            self.audit_check("RateLimitError.limit", exc.limit or 0)
            self.audit_check("RateLimitError.window_seconds", exc.window_seconds or 0)
            self.audit_check("RateLimitError.retry_after", exc.retry_after or 0.0)
        try:
            _raise_circuit_breaker_error()
        except e.CircuitBreakerError as exc:
            self.audit_check("CircuitBreakerError.service_name", exc.service_name or "")
            self.audit_check(
                "CircuitBreakerError.failure_count", exc.failure_count or 0
            )
            self.audit_check(
                "CircuitBreakerError.reset_timeout", exc.reset_timeout or 0.0
            )
        try:
            _raise_flext_type_error()
        except e.FlextTypeError as exc:
            self.audit_check(
                "TypeError.expected_type",
                exc.expected_type.__name__ if exc.expected_type else "",
            )
            self.audit_check(
                "TypeError.actual_type",
                exc.actual_type.__name__ if exc.actual_type else "",
            )
        try:
            _raise_operation_error()
        except e.OperationError as exc:
            self.audit_check("OperationError.operation", exc.operation or "")
            self.audit_check("OperationError.reason", exc.reason or "")
        try:
            _raise_attribute_access_error()
        except e.AttributeAccessError as exc:
            self.audit_check(
                "AttributeAccessError.attribute_name", exc.attribute_name or ""
            )
            self.audit_check(
                "AttributeAccessError.attribute_context",
                str(exc.attribute_context or ""),
            )

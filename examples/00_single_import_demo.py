"""Flext advanced single-import demonstration.

Direct framework access using Python 3.13+ advanced features, minimal code bloat,
maximal functionality via c centralized StrEnum/Literals, railway patterns,
context management, runtime type checking with Pydantic v2, FlextModels DDD patterns,
dependency injection, dispatcher CQRS, and comprehensive flext-core integration.
Uses PEP 695 type aliases, advanced collections.abc patterns, Pydantic 2 with StrEnum,
and strict Python 3.13+ only - no backward compatibility.

**Expected Output:**
- Single import demonstration of all FLEXT components
- Railway-oriented programming patterns
- Context management and correlation tracking
- Domain-driven design with FlextModels
- Dependency injection patterns
- Validation and transformation utilities
- Service orchestration examples

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Set as AbstractSet
from itertools import starmap
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from twisted.python.log import msg

from flext_core import FlextContext, FlextLogger, c, e, r, t, u


class UserProfile(BaseModel):
    """Domain entity with centralized types and business logic - no None types."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    unique_id: str = Field(description="Unique identifier for the user")
    status: c.Status = Field(description="Current status of the user")

    def activate(self) -> r[bool]:
        """Railway pattern for business operations - no None returns."""
        if self.status == c.Status.ACTIVE:
            return r[bool].fail("Already active")
        return r[bool].ok(True)


def validate_transform_user(data: t.ConfigMap) -> r[UserProfile]:
    """Railway pattern using centralized utilities - no None types, strict validation."""
    name_value = data.get("name")
    email_value = data.get("email")
    if not isinstance(name_value, str) or not name_value:
        return r[UserProfile].fail("Name is required and must be a string")
    if not isinstance(email_value, str) or not email_value:
        return r[UserProfile].fail("Email is required and must be a string")
    name: str = name_value
    email: str = email_value
    return r.traverse(
        [
            u.validate_length(name, min_length=c.MIN_USERNAME_LENGTH),
            u.validate_pattern(email, c.PATTERN_EMAIL, "email"),
        ],
        identity,
    ).flat_map(
        lambda _: r.ok(
            UserProfile(
                unique_id=u.generate("correlation"),
                name=name.upper(),
                email=email.lower(),
                status=c.Status.ACTIVE,
            ),
        ),
    )


def process_user_data(*, user_data: t.ConfigMap, operation: c.Action) -> r[str]:
    """Decorated railway with centralized StrEnum constraints - direct functional composition."""
    return validate_transform_user(user_data).map(
        lambda profile: (
            f"{operation.value.upper()}D: {profile.name} ({profile.status.value})"
        ),
    )


class UserService:
    """Advanced service implementing SRP with comprehensive context integration and railway patterns."""

    __slots__ = ("logger", "operation_count")

    def __init__(self) -> None:
        """Initialize with centralized logger and metrics."""
        self.logger = FlextLogger.create_module_logger(__name__)
        self.operation_count = 0

    @staticmethod
    def _activate_user(user: UserProfile) -> r[UserProfile]:
        """Activate user using domain business logic - railway pattern."""

        def return_user(_: bool) -> UserProfile:
            """Return the user after activation."""
            return user

        return user.activate().map(return_user)

    @staticmethod
    def _validate_data(data: t.ConfigMap) -> r[bool]:
        """Validate input data using u (DRY) - no None types."""
        required_fields: AbstractSet[str] = frozenset({"name", "email"})
        present_fields: AbstractSet[str] = frozenset(data.keys())
        if not required_fields <= present_fields:
            missing = required_fields - present_fields
            return r[bool].fail(f"Missing required fields: {missing}")
        return r[bool].ok(value=True)

    def create_user(self, user_data: t.ConfigMap) -> r[UserProfile]:
        """Create user with advanced context tracing and railway pattern - direct functional composition."""
        with FlextContext.Correlation.new_correlation():
            correlation_raw = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            correlation_id = str(correlation_raw) if correlation_raw else "unknown"
            self.operation_count += 1
            self.logger.info(
                "Creating user",
                correlation_id=correlation_id,
                operation_count=self.operation_count,
                user_data_keys=str(tuple(user_data.keys())),
            )
            return (
                self
                ._validate_data(user_data)
                .flat_map(lambda _ok: self._validate_and_transform(user_data, _ok))
                .map(self._log_success)
            )

    def _log_final_result(
        self,
        correlation_id: str,
    ) -> Callable[[UserProfile], UserProfile]:
        """Create logging function for final result - advanced functional pattern."""

        def log_result(user: UserProfile) -> UserProfile:
            self.logger.info(
                "User created successfully",
                user_id=user.unique_id,
                correlation_id=correlation_id,
                user_status=user.status.value,
            )
            return user

        return log_result

    def _log_success(self, user: UserProfile) -> UserProfile:
        """Log success and return user (railway pattern)."""
        self.logger.debug(f"User {user.name} activated successfully")
        return user

    def _validate_and_transform(
        self,
        user_data: t.ConfigMap,
        _: bool,
    ) -> r[UserProfile]:
        """Validate and transform user data for flat_map."""
        return validate_transform_user(user_data)


def demonstrate_utilities() -> None:
    """Advanced utilities demonstration using comprehensive flext-core patterns - direct functional composition."""
    correlation_id = u.generate("correlation")
    test_obj: t.ConfigMap = t.ConfigMap(
        root={"unique_id": correlation_id, "test": True},
    )
    cache_result = u.clear_object_cache(test_obj)
    validation_results = [
        u.validate_length("test", min_length=1, max_length=10),
        u.validate_pattern("test@example.com", c.PATTERN_EMAIL, "email"),
    ]
    result = (
        r
        .traverse(validation_results, lambda r: r)
        .flat_map(lambda _: cache_result)
        .map(
            lambda cache_cleared: "\n".join([
                f"Cache cleared: {cache_cleared}",
                f"Generated ID: {correlation_id[:12]}",
                f"All validations passed: {len(validation_results)} checks",
            ]),
        )
    )
    _ = result.map(print)


def demonstrate_exceptions() -> None:
    """Structured exception handling using centralized constants and railway patterns - functional composition."""
    error_scenarios: Sequence[tuple[str, str, str]] = (
        ("Invalid status", "status", "invalid"),
        ("Empty name", "name", ""),
        ("Invalid email", "email", "not-an-email"),
    )

    def create_error_result(msg: str, field: str, value: str) -> r[str]:
        return r[str].fail(
            e.ValidationError(
                msg,
                field=field,
                value=value,
                error_code=c.VALIDATION_ERROR,
            ).message,
            error_code=c.VALIDATION_ERROR,
        )

    def format_error_message(field: str, value: str) -> str:
        return f"Error: {field}={value}, code: {c.VALIDATION_ERROR}, railway: True"

    def format_exception_message(error: str) -> str:
        return f"Converted exception to result: {error}"

    def process_scenario(field: str, value: str) -> r[str]:
        msg_str = str(msg)
        field_str = str(field)
        value_str = str(value)

        return create_error_result(msg_str, field_str, value_str).map(
            lambda _val: format_error_message(field_str, value_str),
        )

    def process_exception(error: str) -> r[str]:
        error_str = str(error)
        return r.ok(format_exception_message(error_str))

    _ = r.traverse(
        list(starmap(process_scenario, error_scenarios))
        + [process_exception("Standard exception")],
        identity,
    ).map(print)


def identity_result(r_obj: r[str]) -> r[str]:
    """Identity function for result objects."""
    return r_obj


def ignore_and_return_none(_: UserProfile) -> r[bool]:
    """Ignore input and return success."""
    return r[bool].ok(True)


def identity(x: r[str]) -> r[str]:
    """Identity function for traverse operation."""
    return x


def execute_validation_chain(user_data: t.ConfigMap) -> None:
    """Execute validation chain with railway pattern - SRP focused on chaining operations."""
    _ = (
        validate_transform_user(user_data)
        .map(
            lambda user: (
                f"User: {user.name} ({user.status.value}) - ID: {user.unique_id[:8]}"
            ),
        )
        .flat_map(r.ok)
        .flat_map(
            lambda output: process_user_data(
                user_data=user_data,
                operation=c.Action.CREATE,
            ).map(lambda result: f"{output}\nProcess: {result}"),
        )
        .map(print)
        .lash(lambda error: r[None].ok(print(f"Validation failed: {error}") or None))
    )


def execute_service_operations(service: UserService, user_data: t.ConfigMap) -> None:
    """Execute service operations - SRP focused on service interaction."""
    result = service.create_user(user_data)
    if result.is_success:
        user = result.value
        print(f"Service: SUCCESS - {user.name}")
    else:
        print(f"Service: FAILED - {result.error}")


def execute_demonstrations(service: UserService, user_data: t.ConfigMap) -> None:
    """Execute utility demonstrations - SRP focused on side effect execution."""
    _ = service.create_user(user_data).flat_map(ignore_and_return_none)
    demonstrate_utilities()
    demonstrate_exceptions()


def main() -> None:
    """Advanced FLEXT demo with railway patterns and context management - functional composition."""
    logger = FlextLogger.create_module_logger(__name__)
    with FlextContext.Correlation.new_correlation():
        correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
        logger.info("Starting demonstration", correlation_id=str(correlation_id or ""))
        user_data: t.ConfigMap = t.ConfigMap(
            root={"name": "Demo", "email": "demo@example.com"},
        )
        service = UserService()
        execute_validation_chain(user_data)
        execute_service_operations(service, user_data)
        execute_demonstrations(service, user_data)
        logger.info("Comprehensive demonstration completed successfully")


if __name__ == "__main__":
    main()

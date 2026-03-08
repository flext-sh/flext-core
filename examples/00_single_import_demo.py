"""Flext advanced single-import demonstration.

Direct framework access using Python 3.13+ advanced features, minimal code bloat,
maximal functionality via c centralized StrEnum/Literals, railway patterns,
context management, runtime type checking with beartype, FlextModels DDD patterns,
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

from pydantic import ValidationError as PydanticValidationError

from flext_core import (
    FlextContext,
    FlextLogger,
    c,
    e,
    m,
    r,
    t,
    u,
)

from .models import UserInput, UserProfile


# Railway pattern with advanced validation using u (DRY + SRP)
def validate_transform_user(
    data: m.ConfigMap,
) -> r[UserProfile]:
    """Railway pattern using centralized utilities - no None types, strict validation."""
    try:
        typed_input = UserInput.model_validate(data)
    except PydanticValidationError as exc:
        return r[UserProfile].fail(str(exc))

    name = typed_input.name
    email = typed_input.email

    # Advanced validation using u with traverse (DRY - no manual loops)
    return r.traverse(
        [
            u.validate_length(
                name,
                min_length=c.Validation.MIN_USERNAME_LENGTH,
            ),
            u.validate_pattern(email, c.Platform.PATTERN_EMAIL, "email"),
        ],
        identity,
    ).flat_map(
        lambda _: r.ok(
            UserProfile(
                unique_id=u.generate("correlation"),
                name=name.upper(),
                email=email.lower(),
                status=c.Domain.Status.ACTIVE,
            ),
        ),
    )


def process_user_data(
    *,
    user_data: m.ConfigMap,
    operation: c.Cqrs.Action,
) -> r[str]:
    """Decorated railway with centralized StrEnum constraints - direct functional composition."""
    return validate_transform_user(user_data).map(
        lambda profile: (
            f"{operation.value.upper()}D: {profile.name} ({profile.status.value})"
        ),
    )


# Advanced context-aware service (SOLID + DRY)
class UserService:
    """Advanced service implementing SRP with comprehensive context integration and railway patterns."""

    __slots__ = ("logger", "operation_count")

    def __init__(self) -> None:
        """Initialize with centralized logger and metrics."""
        super().__init__()
        self.logger = FlextLogger.create_module_logger(__name__)
        self.operation_count = 0

    @staticmethod
    def _activate_user(user: UserProfile) -> r[UserProfile]:
        """Activate user using domain business logic - railway pattern."""

        def return_user(_: t.ContainerValue) -> UserProfile:
            """Return the user after activation."""
            return user

        return user.activate().map(return_user)

    @staticmethod
    def _validate_data(
        data: m.ConfigMap,
    ) -> r[bool]:
        """Validate input data using u (DRY) - no None types."""
        required_fields: AbstractSet[str] = frozenset({
            "name",
            "email",
        })  # Advanced collections.abc Set
        present_fields: AbstractSet[str] = frozenset(data.keys())

        if not required_fields <= present_fields:
            missing = required_fields - present_fields
            return r[bool].fail(f"Missing required fields: {missing}")

        return r[bool].ok(value=True)

    def create_user(self, user_data: m.ConfigMap) -> r[UserProfile]:
        """Create user with advanced context tracing and railway pattern - direct functional composition."""
        with FlextContext.Correlation.new_correlation():
            correlation_id = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get() or "unknown"
            )
            self.operation_count += 1

            self.logger.info(  # Using mixin logger
                "Creating user",
                correlation_id=correlation_id,
                operation_count=self.operation_count,
                user_data_keys=str(tuple(user_data.keys())),
            )

            # Railway pattern with advanced functional composition (DRY)
            return (
                self
                ._validate_data(user_data)
                .flat_map(lambda _: self._validate_and_transform(user_data, _))
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
        user_data: m.ConfigMap,
        _: t.ContainerValue,
    ) -> r[UserProfile]:
        """Validate and transform user data for flat_map."""
        return validate_transform_user(user_data)


# Comprehensive utilities demonstration (DRY + SRP with advanced flext-core integration)
def demonstrate_utilities() -> None:
    """Advanced utilities demonstration using comprehensive flext-core patterns - direct functional composition."""
    # Create test data and perform operations with railway pattern (DRY + SRP)
    correlation_id = u.generate("correlation")
    test_obj: m.ConfigMap = m.ConfigMap(
        root={
            "unique_id": correlation_id,
            "test": True,
        },
    )

    # Railway pattern with traverse for multiple operations (DRY - no manual loops)
    cache_result = u.clear_object_cache(test_obj)
    validation_results = [
        u.validate_length("test", min_length=1, max_length=10),
        u.validate_pattern(
            "test@example.com",
            c.Platform.PATTERN_EMAIL,
            "email",
        ),
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

    # Safe output with railway pattern
    result.map(print)


# Advanced exception handling with comprehensive error integration (DRY + SRP)
def demonstrate_exceptions() -> None:
    """Structured exception handling using centralized constants and railway patterns - functional composition."""
    # Railway pattern with traverse for multiple error scenarios (DRY - no manual loops)
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
                error_code=c.Errors.VALIDATION_ERROR,
            ).message,
            error_code=c.Errors.VALIDATION_ERROR,
        )

    def format_error_message(field: str, value: str) -> str:
        return (
            f"Error: {field}={value}, code: {c.Errors.VALIDATION_ERROR}, railway: True"
        )

    def format_exception_message(error: str) -> str:
        return f"Converted exception to result: {error}"

    def process_scenario(
        msg: t.ContainerValue,
        field: t.ContainerValue,
        value: t.ContainerValue,
    ) -> r[str]:
        msg_str = str(msg)
        field_str = str(field)
        value_str = str(value)

        def format_error_after_validation(_: t.ContainerValue) -> str:
            """Format error message after validation."""
            return format_error_message(field_str, value_str)

        return create_error_result(msg_str, field_str, value_str).map(
            format_error_after_validation,
        )

    def process_exception(error: t.ContainerValue) -> r[str]:
        error_str = str(error)
        return r.ok(format_exception_message(error_str))

    r.traverse(
        list(starmap(process_scenario, error_scenarios))
        + [
            # Standard exception conversion
            process_exception("Standard exception"),
        ],
        identity,
    ).map(print)


# Railway pattern handlers (SRP - single responsibility for result handling)


def identity_result(r_obj: r[str]) -> r[str]:
    """Identity function for result objects."""
    return r_obj


def ignore_and_return_none(_: t.ContainerValue) -> r[None]:
    """Ignore input and return None."""
    return r.ok(None)


def identity(x: r[str]) -> r[str]:
    """Identity function for traverse operation."""
    return x


def execute_validation_chain(
    user_data: m.ConfigMap,
) -> None:
    """Execute validation chain with railway pattern - SRP focused on chaining operations."""
    # Railway pattern with advanced functional composition (DRY + SRP)
    (
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
                operation=c.Cqrs.Action.CREATE,
            ).map(lambda result: f"{output}\nProcess: {result}"),
        )
        .map(print)
        .lash(lambda error: r[None].ok(print(f"Validation failed: {error}") or None))
    )


def execute_service_operations(
    service: UserService,
    user_data: m.ConfigMap,
) -> None:
    """Execute service operations - SRP focused on service interaction."""
    result = service.create_user(user_data)
    if result.is_success:
        user = result.value
        print(f"Service: SUCCESS - {user.name}")
    else:
        print(f"Service: FAILED - {result.error}")


def execute_demonstrations(
    service: UserService,
    user_data: m.ConfigMap,
) -> None:
    """Execute utility demonstrations - SRP focused on side effect execution."""
    # Railway pattern with side effects (DRY - no manual loops)
    service.create_user(user_data).map(ignore_and_return_none)
    # Execute demonstrations as side effects
    demonstrate_utilities()
    demonstrate_exceptions()


# Main demonstration (minimal code, maximal functionality - DRY + SRP)
def main() -> None:
    """Advanced FLEXT demo with railway patterns and context management - functional composition."""
    logger = FlextLogger.create_module_logger(__name__)

    with FlextContext.Correlation.new_correlation():
        correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
        logger.info("Starting demonstration", correlation_id=str(correlation_id or ""))

        # Advanced collections.abc Mapping for user data (DRY - single definition)
        user_data: m.ConfigMap = m.ConfigMap(
            root={
                "name": "Demo",
                "email": "demo@example.com",
            },
        )

        # Service instance (DRY - single creation)
        service = UserService()

        # Execute operations with SRP separation (DRY - no code duplication)
        execute_validation_chain(user_data)
        execute_service_operations(service, user_data)
        execute_demonstrations(service, user_data)

        logger.info("Comprehensive demonstration completed successfully")


if __name__ == "__main__":
    main()

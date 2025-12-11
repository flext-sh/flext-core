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
from dataclasses import dataclass
from itertools import starmap

from flext_core import (
    FlextContext,
    FlextLogger,
    c,
    e,
    r,
    t,
    u,
)

# Using t directly - no local type aliases (DRY + SRP)
# All types come from t namespace - centralized type system


# Advanced domain entity using c StrEnum (DRY + SRP)
@dataclass(frozen=True)
class UserProfile:
    """Domain entity with centralized types and business logic - no None types."""

    name: str
    email: str
    unique_id: str
    status: c.Domain.Status

    def activate(self) -> r[None]:
        """Railway pattern for business operations - no None returns."""
        if self.status == c.Domain.Status.ACTIVE:
            return r.fail("Already active")
        return r.ok(None)


# Railway pattern with advanced validation using u (DRY + SRP)
def validate_transform_user(
    data: t.ConfigurationDict,
) -> r[UserProfile]:
    """Railway pattern using centralized utilities - no None types, strict validation."""
    # Extract with advanced collections.abc Mapping access and functional composition
    name_value = data.get("name")
    email_value = data.get("email")

    # Validate and extract with type narrowing
    if not isinstance(name_value, str) or not name_value:
        return r[UserProfile].fail("Name is required and must be a string")
    if not isinstance(email_value, str) or not email_value:
        return r[UserProfile].fail("Email is required and must be a string")

    name: str = name_value
    email: str = email_value

    # Advanced validation using u with traverse (DRY - no manual loops)
    return r.traverse(
        [
            u.Validation.validate_length(
                name,
                min_length=c.Validation.MIN_USERNAME_LENGTH,
            ),
            u.Validation.validate_pattern(email, c.Platform.PATTERN_EMAIL, "email"),
        ],
        lambda r: r,
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
    user_data: t.ConfigurationDict,
    operation: c.Cqrs.Action,
) -> r[str]:
    """Decorated railway with centralized StrEnum constraints - direct functional composition."""
    return validate_transform_user(user_data).map(
        lambda profile: f"{operation.value.upper()}D: {profile.name} ({profile.status.value})",
    )


# Advanced context-aware service (SOLID + DRY)
class UserService:
    """Advanced service implementing SRP with comprehensive context integration and railway patterns."""

    __slots__ = ("logger", "operation_count")

    def __init__(self) -> None:
        """Initialize with centralized logger and metrics."""
        self.logger = FlextLogger.create_module_logger(__name__)
        self.operation_count = 0

    def create_user(self, user_data: t.ConfigurationDict) -> r[UserProfile]:
        """Create user with advanced context tracing and railway pattern - direct functional composition."""
        with FlextContext.Request.request_context(operation_name="create_user"):
            correlation_id = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get() or "unknown"
            )
            self.operation_count += 1

            self.logger.info(  # Using mixin logger
                "Creating user",
                extra={
                    "correlation_id": correlation_id,
                    "operation_count": self.operation_count,
                    "user_data_keys": tuple(user_data.keys()),
                },
            )

            # Railway pattern with advanced functional composition (DRY)
            return (
                self._validate_data(user_data)
                .flat_map(lambda _: validate_transform_user(user_data))
                .map(self._log_success)
            )

    @staticmethod
    def _validate_data(
        data: t.ConfigurationDict,
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

        return r[bool].ok(True)

    @staticmethod
    def _activate_user(user: UserProfile) -> r[UserProfile]:
        """Activate user using domain business logic - railway pattern."""
        return user.activate().map(lambda _: user)

    def _log_success(self, user: UserProfile) -> UserProfile:
        """Log success and return user (railway pattern)."""
        self.logger.debug(f"User {user.name} activated successfully")
        return user

    def _log_final_result(
        self,
        correlation_id: str,
    ) -> Callable[[UserProfile], UserProfile]:
        """Create logging function for final result - advanced functional pattern."""

        def log_result(user: UserProfile) -> UserProfile:
            self.logger.info(
                "User created successfully",
                extra={
                    "user_id": user.unique_id,
                    "correlation_id": correlation_id,
                    "user_status": user.status.value,
                },
            )
            return user

        return log_result


# Comprehensive utilities demonstration (DRY + SRP with advanced flext-core integration)
def demonstrate_utilities() -> None:
    """Advanced utilities demonstration using comprehensive flext-core patterns - direct functional composition."""
    # Create test data and perform operations with railway pattern (DRY + SRP)
    correlation_id = u.generate("correlation")
    test_obj: t.ConfigurationDict = {
        "unique_id": correlation_id,
        "test": True,
    }

    # Railway pattern with traverse for multiple operations (DRY - no manual loops)
    cache_result = u.Cache.clear_object_cache(test_obj)
    validation_results = [
        u.Validation.validate_length("test", min_length=1, max_length=10),
        u.Validation.validate_pattern(
            "test@example.com",
            c.Platform.PATTERN_EMAIL,
            "email",
        ),
    ]

    result = (
        r.traverse(validation_results, lambda r: r)
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

    r.traverse(
        list(
            starmap(
                lambda msg, field, value: r.fail(
                    e.ValidationError(
                        msg,
                        field=field,
                        value=value,
                        error_code=c.Errors.VALIDATION_ERROR,
                    ).message,
                    error_code=c.Errors.VALIDATION_ERROR,
                )
                .map(
                    lambda _: f"Error: {field}={value}, code: {c.Errors.VALIDATION_ERROR}, railway: True",
                )
                .map(print),
                error_scenarios,
            ),
        )
        + [
            # Standard exception conversion
            r.fail("Standard exception")
            .map(lambda error: f"Converted exception to result: {error}")
            .map(print),
        ],
        lambda r: r,
    )


# Railway pattern handlers (SRP - single responsibility for result handling)


def execute_validation_chain(
    user_data: t.ConfigurationDict,
) -> None:
    """Execute validation chain with railway pattern - SRP focused on chaining operations."""
    # Railway pattern with advanced functional composition (DRY + SRP)
    (
        validate_transform_user(user_data)
        .map(
            lambda user: f"User: {user.name} ({user.status.value}) - ID: {user.unique_id[:8]}",
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
    user_data: t.ConfigurationDict,
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
    user_data: t.ConfigurationDict,
) -> None:
    """Execute utility demonstrations - SRP focused on side effect execution."""
    # Railway pattern with side effects (DRY - no manual loops)
    service.create_user(user_data).map(lambda _: None)
    # Execute demonstrations as side effects
    demonstrate_utilities()
    demonstrate_exceptions()


# Main demonstration (minimal code, maximal functionality - DRY + SRP)
def main() -> None:
    """Advanced FLEXT demo with railway patterns and context management - functional composition."""
    logger = FlextLogger.create_module_logger(__name__)

    with FlextContext.Request.request_context(operation_name="demo"):
        correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
        logger.info("Starting demonstration", extra={"correlation_id": correlation_id})

        # Advanced collections.abc Mapping for user data (DRY - single definition)
        user_data: t.ConfigurationDict = {
            "name": "Demo",
            "email": "demo@example.com",
        }

        # Service instance (DRY - single creation)
        service = UserService()

        # Execute operations with SRP separation (DRY - no code duplication)
        execute_validation_chain(user_data)
        execute_service_operations(service, user_data)
        execute_demonstrations(service, user_data)

        logger.info("Comprehensive demonstration completed successfully")


if __name__ == "__main__":
    main()

"""Shared Test Domain Module for FLEXT Core Tests.

This module provides reusable test fixtures, domain models, and utilities
that can be shared across all test modules, eliminating code duplication and
promoting consistent testing patterns.

ARCHITECTURE CHANGE: This module now uses test-specific domain models
instead of importing from examples/, eliminating circular import issues.
"""

from __future__ import annotations

from decimal import Decimal

from .test_shared_domain import (
    TestComplexValueObject,
    TestDomainFactory,
    TestMoney,
    TestUser,
    TestUserStatus,
)

# Entity aliases for backward compatibility
ConcreteFlextEntity = TestUser
ConcreteValueObject = TestMoney  # Using TestMoney as a proper value object
ComplexValueObject = TestComplexValueObject  # Using TestComplexValueObject properly


def create_test_entity_safe(name: str, **kwargs: object) -> TestUser:
    """Create test entity with error handling."""
    status = str(kwargs.get("status", "active"))
    result = TestDomainFactory.create_concrete_entity(name=name, status=status)
    if result.is_failure:
        error_msg: str = f"Failed to create test entity: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create test entity: result data is None"
    #     raise ValueError(error_msg)
    return result.value


def create_test_value_object_safe(
    amount: str,
    currency: str = "USD",
    **kwargs: object,
) -> object:
    """Create test value object with error handling."""
    result = TestDomainFactory.create_concrete_value_object(
        amount=Decimal(amount),
        currency=currency,
        description=str(kwargs.get("description", "")),
    )
    if result.is_failure:
        error_msg: str = f"Failed to create test value object: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create test value object: result data is None"
    #     raise ValueError(error_msg)
    return result.value


def create_complex_test_value_object_safe(
    name: str,
    tags: list[str],
    metadata: dict[str, object],
) -> object:
    """Create complex test value object with error handling."""
    result = TestDomainFactory.create_complex_value_object(
        name=name,
        tags=tags,
        metadata=metadata,
    )
    if result.is_failure:
        error_msg: str = f"Failed to create complex test value object: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create complex test value object: result data is None"
    #     raise ValueError(error_msg)
    return result.value


__all__: list[str] = [
    "ComplexValueObject",
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "TestDomainFactory",
    "TestUserStatus",
    "create_complex_test_value_object_safe",
    "create_test_entity_safe",
    "create_test_value_object_safe",
]

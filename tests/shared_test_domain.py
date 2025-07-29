"""Shared Test Domain Module for FLEXT Core Tests.

This module provides reusable test fixtures, domain models, and utilities
that can be shared across all test modules, eliminating code duplication and
promoting consistent testing patterns.

Features:
- Common test fixtures and domain models
- Shared test utilities and helper functions
- Consistent domain model creation patterns
- Integration with examples shared_domain
- Factory patterns for test object creation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add examples directory to path to import shared domain
examples_path = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_path))

try:
    from shared_domain import (
        ComplexValueObject,
        ConcreteFlextEntity,
        ConcreteValueObject,
        TestDomainFactory,
    )
except ImportError:
    # Fallback: Create minimal test domain objects locally
    from flext_core import FlextEntity, FlextValueObject

    class ConcreteFlextEntity(FlextEntity):
        """Concrete test entity for testing FlextEntity functionality."""

        name: str
        value: int = 0

    class ConcreteValueObject(FlextValueObject):
        """Concrete test value object for testing FlextValueObject functionality."""

        data: str
        count: int = 1

    class ComplexValueObject(FlextValueObject):
        """Complex test value object with multiple fields for comprehensive testing."""

        name: str
        age: int
        active: bool = True

    class TestDomainFactory:
        """Factory for creating test domain objects safely."""

        @staticmethod
        def create_entity(name: str = "test", **kwargs: object) -> ConcreteFlextEntity:
            """Create a test entity instance."""
            return ConcreteFlextEntity(name=name, **kwargs)

        @staticmethod
        def create_value_object(
            data: str = "test", **kwargs: object,
        ) -> ConcreteValueObject:
            """Create a test value object instance."""
            return ConcreteValueObject(data=data, **kwargs)


def create_test_entity_safe(name: str, **kwargs: object) -> ConcreteFlextEntity:  # type: ignore[misc]
    """Create test entity with error handling."""
    result = TestDomainFactory.create_concrete_entity(name=name, **kwargs)
    if result.is_failure:
        error_msg = f"Failed to create test entity: {result.error}"
        raise ValueError(error_msg)
    return result.data


def create_test_value_object_safe(  # type: ignore[misc]
    amount: str,
    currency: str = "USD",
    **kwargs: object,
) -> ConcreteValueObject:
    """Create test value object with error handling."""
    from decimal import Decimal

    result = TestDomainFactory.create_concrete_value_object(
        amount=Decimal(amount),
        currency=currency,
        **kwargs,
    )
    if result.is_failure:
        error_msg = f"Failed to create test value object: {result.error}"
        raise ValueError(error_msg)
    return result.data


def create_complex_test_value_object_safe(  # type: ignore[misc]
    name: str,
    tags: list[str],
    metadata: dict[str, object],
) -> ComplexValueObject:
    """Create complex test value object with error handling."""
    result = TestDomainFactory.create_complex_value_object(
        name=name,
        tags=tags,
        metadata=metadata,
    )
    if result.is_failure:
        error_msg = f"Failed to create complex test value object: {result.error}"
        raise ValueError(error_msg)
    return result.data


__all__ = [
    "ComplexValueObject",
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "TestDomainFactory",
    "create_complex_test_value_object_safe",
    "create_test_entity_safe",
    "create_test_value_object_safe",
]

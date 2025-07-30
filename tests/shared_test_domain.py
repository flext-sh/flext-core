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
from decimal import Decimal
from pathlib import Path

# Add examples directory to path to import shared domain
examples_path = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_path))

# Import must be after path modification
# ruff: noqa: E402
from shared_domain import (
    ComplexValueObject,
    ConcreteFlextEntity,
    ConcreteValueObject,
    TestDomainFactory,
)


def create_test_entity_safe(name: str, **kwargs: object) -> ConcreteFlextEntity:
    """Create test entity with error handling."""
    status = str(kwargs.get("status", "active"))
    result = TestDomainFactory.create_concrete_entity(name=name, status=status)
    if result.is_failure:
        error_msg = f"Failed to create test entity: {result.error}"
        raise ValueError(error_msg)
    if result.data is None:
        none_error_msg = "Failed to create test entity: result data is None"
        raise ValueError(none_error_msg)
    return result.data


def create_test_value_object_safe(
    amount: str,
    currency: str = "USD",
    **kwargs: object,
) -> ConcreteValueObject:
    """Create test value object with error handling."""
    result = TestDomainFactory.create_concrete_value_object(
        amount=Decimal(amount),
        currency=currency,
        description=str(kwargs.get("description", "")),
    )
    if result.is_failure:
        error_msg = f"Failed to create test value object: {result.error}"
        raise ValueError(error_msg)
    if result.data is None:
        none_error_msg = "Failed to create test value object: result data is None"
        raise ValueError(none_error_msg)
    return result.data


def create_complex_test_value_object_safe(
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
    if result.data is None:
        none_error_msg = (
            "Failed to create complex test value object: result data is None"
        )
        raise ValueError(none_error_msg)
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

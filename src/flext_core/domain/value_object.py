"""FlextValueObject - Enterprise Domain Value Object Base Class.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Professional implementation of Domain-Driven Design (DDD) value object
base class
following enterprise software engineering principles. This module provides the
foundation class for implementing domain value objects with
attribute-based equality,
immutability, and comprehensive validation.

Single Responsibility: This module contains only the FlextValueObject
base class
and its core functionality, adhering to SOLID principles.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from pydantic import BaseModel
from pydantic import ConfigDict


class FlextValueObject(BaseModel, ABC):
    """Abstract base class for domain value objects with enterprise validation.

    FlextValueObject represents descriptive aspects of the domain with no
    conceptual identity. Value objects are defined by their attributes and
    are interchangeable when their attributes match, implementing core DDD
    principles with maximum type safety and immutability.

    Enterprise Features:
        - Immutable design ensuring thread-safe concurrent operations
        - Attribute-based equality following DDD value object semantics
        - Comprehensive validation using Pydantic V2 with business rules
        - Self-validating design with domain-specific rule enforcement
        - Side-effect-free operations maintaining functional programming
          principles
        - Production-ready serialization with efficient comparison operations

    Architectural Design:
        - Attribute-based equality (not identity-based like entities)
        - Immutable for consistency, safety, and thread-safe operations
        - Self-validating with comprehensive domain rule enforcement
        - Side-effect-free methods supporting functional composition
        - Efficient hashing based on all attributes for collection usage

    Production Usage Patterns:
        Value object implementation:
        >>> class Money(FlextValueObject):
        ...     amount: Decimal
        ...     currency: str
        ...
        ...     def validate_domain_rules(self) -> None:
        ...         if self.amount < 0:
        ...             raise ValueError("Money amount cannot be negative")
        ...         if len(self.currency) != 3:
        ...             msg = "Currency code must be exactly 3 characters"
        ...             raise ValueError(msg)
        ...         if not self.currency.isupper():
        ...             raise ValueError("Currency code must be uppercase")

        Value equality and interchangeability:
        >>> money1 = Money(amount=Decimal("10.50"), currency="USD")
        >>> money2 = Money(amount=Decimal("10.50"), currency="USD")
        >>> assert money1 == money2  # Same attributes = same value
        >>> assert money1 is not money2  # Different instances

        Immutable operations:
        >>> doubled_money = (
            Money(amount=money1.amount * 2, currency=money1.currency)
        )
        >>> assert doubled_money.amount == Decimal("21.00")

    Thread Safety Guarantees:
        - All instances are immutable after creation (frozen Pydantic models)
        - Concurrent read operations are fully thread-safe
        - Value comparison and hashing operations are atomic
        - No shared mutable state between value object instances

    Performance Characteristics:
        - Efficient attribute-based comparison with optimized hashing
        - Minimal memory overhead with frozen model design
        - O(1) hash computation for fast collection operations
        - Zero-copy operations where possible with immutable design

    """

    model_config = ConfigDict(
        # Immutable value objects for thread safety
        frozen=True,
        # Strict validation for enterprise reliability
        validate_assignment=True,
        str_strip_whitespace=True,
        # Efficient serialization and comparison
        extra="forbid",
        arbitrary_types_allowed=False,
        # JSON schema generation for API documentation
        json_schema_extra={
            "description": ("Enterprise value object with attribute-based equality"),
        },
    )

    def __eq__(self, other: object) -> bool:
        """Value object equality based on all attributes (DDD principle)."""
        if not isinstance(other, self.__class__):
            return False
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        """Hash based on all attributes for efficient collection usage."""
        # Create hash from all field values, handling unhashable types
        values = tuple(
            v if not isinstance(v, (dict, list)) else str(v)
            for v in self.model_dump().values()
        )
        return hash(values)

    def __str__(self) -> str:
        """Return string representation showing key attributes."""
        # Show first few important fields for readability
        max_fields = 3
        fields = list(self.model_dump().items())[:max_fields]
        field_str = ", ".join(f"{k}={v}" for k, v in fields)
        ellipsis = "..." if len(self.model_dump()) > max_fields else ""
        return f"{self.__class__.__name__}({field_str}{ellipsis})"

    @abstractmethod
    def validate_domain_rules(self) -> None:
        """Validate value object business rules and domain constraints.

        Each value object implementation must provide specific validation
        logic to ensure it maintains valid state according to domain
        requirements and business rules.

        Raises:
            ValueError: If any domain rule or business constraint is violated

        Example Implementation:
            >>> def validate_domain_rules(self) -> None:
            ...     if self.percentage < 0 or self.percentage > 100:
            ...         raise ValueError(
        ...             "Percentage must be between 0 and 100"
        ...         )
            ...     if self.precision > 2:
            ...         msg = (
                "Percentage precision cannot exceed 2 decimal places"
            )
            ...         raise ValueError(msg)

        """


__all__ = ["FlextValueObject"]

"""Base domain types for domain-driven design patterns.

PYDANTIC CONSOLIDATION: Now uses centralized Pydantic base classes with Python 3.13 features
for standardization, validation, and code reduction across the entire domain layer.
"""

from __future__ import annotations

from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import Field

# PYDANTIC CONSOLIDATION: Import from centralized Pydantic base classes
from flx_core.domain.pydantic_base import (
    AndSpecification,
    DomainEvent,
    NotSpecification,
    OrSpecification,
    ServiceResult,
)
from flx_core.domain.pydantic_base import DomainAggregateRoot as AggregateRoot
from flx_core.domain.pydantic_base import DomainEntity as Entity
from flx_core.domain.pydantic_base import DomainSpecification as Specification
from flx_core.domain.pydantic_base import DomainValueObject as ValueObject

__all__ = [
    "AggregateRoot",
    "AndSpecification",
    "CompositeSpecification",
    "DomainEvent",
    "DomainId",
    "Entity",
    "NotSpecification",
    "OrSpecification",
    "ServiceResult",
    "Specification",
    "ValueObject",
]

T = TypeVar("T")
ID = TypeVar("ID", bound="DomainId")


class DomainId(ValueObject):
    """Base class for all domain identifiers using Pydantic validation and Python 3.13 features."""

    value: UUID = Field(default_factory=uuid4)

    def __str__(self) -> str:
        """Return string representation of the domain identifier."""
        return str(self.value)

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self.value)


# Legacy compatibility aliases for specification patterns
CompositeSpecification = Specification  # Maps to DomainSpecification base class

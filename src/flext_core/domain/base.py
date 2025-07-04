"""Base domain types for domain-driven design patterns.

PYDANTIC CONSOLIDATION: Now uses centralized Pydantic base classes with Python 3.13 features
for standardization, validation, and code reduction across the entire domain layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from pydantic import Field

from flext_core.domain.advanced_types import (
    ServiceResult,
)

# PYDANTIC CONSOLIDATION: Import from centralized Pydantic base classes
from flext_core.domain.pydantic_base import (
    AndSpecification,
    DomainEvent,
    NotSpecification,
    OrSpecification,
)
from flext_core.domain.pydantic_base import DomainAggregateRoot as AggregateRoot
from flext_core.domain.pydantic_base import DomainEntity as Entity
from flext_core.domain.pydantic_base import DomainSpecification as Specification
from flext_core.domain.pydantic_base import DomainValueObject as ValueObject

if TYPE_CHECKING:
    from uuid import UUID

__all__ = [
    "AggregateRoot",
    "AndSpecification",
    # "CompositeSpecification",  # Not imported, using individual And/Or/Not specifications
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


# Domain ID types
class PipelineId(ValueObject):
    """Pipeline identifier value object."""

    value: str = Field(description="Pipeline unique identifier")


class UserId(ValueObject):
    """User identifier value object."""

    value: str = Field(description="User unique identifier")


class PluginId(ValueObject):
    """Plugin identifier value object."""

    value: str = Field(description="Plugin unique identifier")


def create_pipeline_id(value: str) -> PipelineId:
    """Create a pipeline ID value object."""
    return PipelineId(value=value)


def create_user_id(value: str) -> UserId:
    """Create a user ID value object."""
    return UserId(value=value)


def create_plugin_id(value: str) -> PluginId:
    """Create a plugin ID value object."""
    return PluginId(value=value)

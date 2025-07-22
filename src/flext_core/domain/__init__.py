"""Domain Layer - Pure Business Logic.

This layer contains the core business logic and domain models.
It has no dependencies on external frameworks or infrastructure.

🎯 DOMAIN PRINCIPLE:
This layer defines the business entities, value objects, domain events,
and business rules. It's the heart of the application and should be
completely independent of external concerns.
"""

from __future__ import annotations

# Essential domain building blocks
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject

# Legacy alias for backward compatibility
BaseAggregateRoot = DomainAggregateRoot

__all__ = [
    "BaseAggregateRoot",  # Legacy alias
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainEntity",
    "DomainEvent",
    "DomainValueObject",
]

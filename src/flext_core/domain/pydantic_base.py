"""REFACTORED Base Pydantic models for FLEXT ecosystem.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the foundation for all Pydantic models across FLEXT modules.
Uses modern Python 3.13 + advanced mixins + protocols for maximum code reduction.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# Single shared configuration - KISS principle
DEFAULT_CONFIG = ConfigDict(
    validate_assignment=True,
    use_enum_values=True,
    extra="forbid",
    str_strip_whitespace=True,
)


# ==============================================================================
# REFACTORED BASE MODELS - USING MIXINS FOR CODE REDUCTION
# ==============================================================================


class DomainBaseModel(BaseModel):
    """Base for ALL domain models - Single configuration."""

    model_config = DEFAULT_CONFIG


class DomainValueObject(DomainBaseModel):
    """Immutable value objects."""

    model_config = ConfigDict(**DEFAULT_CONFIG, frozen=True)


class DomainEntity(DomainBaseModel):
    """Entities with identity."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainEvent(DomainValueObject):
    """Domain events."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainAggregateRoot(DomainEntity):
    """Aggregate root with events."""

    events: list[DomainEvent] = Field(default_factory=list, exclude=True)

    def add_event(self, event: DomainEvent) -> None:
        """Add domain event."""
        self.events.append(event)

    def clear_events(self) -> None:
        """Clear events."""
        self.events.clear()

    def get_events(self) -> list[DomainEvent]:
        """Get and clear all events."""
        events = self.events.copy()
        self.events.clear()
        return events


# ==============================================================================
# API BASE MODELS - FOR FLEXT-API PROJECT
# ==============================================================================


class APIRequest(DomainBaseModel):
    """Base class for API request models."""


class APIResponse(DomainBaseModel):
    """Base class for API response models."""


# Export essential components only - KISS principle
__all__ = [
    "DEFAULT_CONFIG",
    "APIRequest",
    "APIResponse",
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainEntity",
    "DomainEvent",
    "DomainValueObject",
    "Field",  # Re-export for convenience
]

"""REFACTORED Base Pydantic models for FLEXT ecosystem.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the foundation for all Pydantic models across FLEXT modules.
Uses modern Python 3.13 + advanced mixins + protocols for maximum code reduction.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import Any
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from flext_core.domain.mixins import IdentifierMixin
from flext_core.domain.mixins import TimestampMixin

# Base configurations - SIMPLIFIED
STRICT_CONFIG = ConfigDict(
    validate_assignment=True,
    use_enum_values=True,
    arbitrary_types_allowed=True,
    extra="forbid",
    str_strip_whitespace=True,
    validate_default=True,
    populate_by_name=True,
)

API_CONFIG = ConfigDict(
    validate_assignment=True,
    use_enum_values=True,
    arbitrary_types_allowed=True,
    extra="ignore",  # Allow extra fields from external APIs
    str_strip_whitespace=True,
    validate_default=True,
    populate_by_name=True,
)

SETTINGS_CONFIG = ConfigDict(
    validate_assignment=True,
    use_enum_values=True,
    extra="forbid",
    validate_default=True,
)


# ==============================================================================
# REFACTORED BASE MODELS - USING MIXINS FOR CODE REDUCTION
# ==============================================================================


class DomainBaseModel(BaseModel):
    """Base for ALL domain models - SINGLE CONFIGURATION."""

    model_config = STRICT_CONFIG


class DomainValueObject(DomainBaseModel):
    """Immutable value objects for domain modeling."""

    model_config = ConfigDict(**STRICT_CONFIG, frozen=True)


class DomainEntity(DomainBaseModel):
    """Entities with identity - USES MIXIN FOR CODE REDUCTION."""

    # Required entity attributes with defaults
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainEvent(DomainValueObject):
    """Base domain event - USES MIXIN FOR CODE REDUCTION."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainAggregateRoot(DomainEntity):
    """Aggregate root with domain events - USES MIXIN FOR CODE REDUCTION."""

    events: list[DomainEvent] = Field(default_factory=list, exclude=True)

    def add_event(self, event: DomainEvent) -> None:
        """Add domain event to aggregate."""
        self.events.append(event)

    def clear_events(self) -> None:
        """Clear domain events."""
        self.events.clear()

    def get_events(self) -> list[DomainEvent]:
        """Get and clear all domain events.

        Returns:
            A list of domain events.

        """
        events = self.events.copy()
        self.events.clear()
        return events


# ==============================================================================
# API MODELS - USING MIXINS FOR CODE REDUCTION
# ==============================================================================


class APIBaseModel(BaseModel):
    """Base for API request/response models - USES MIXIN FOR CODE REDUCTION."""

    model_config = API_CONFIG


class APIRequest(APIBaseModel):
    """Base for API request models."""


class APIResponse(APIBaseModel):
    """Base for API response models."""

    success: bool = Field(
        default=True,
        description="Whether the request was successful",
    )
    message: str | None = Field(None, description="Optional message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp",
    )


class APIPaginatedResponse(APIResponse):
    """Base for paginated API responses."""

    data: list[Any] = Field(default_factory=list)
    total: int = Field(0, description="Total number of items")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Items per page")

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages.

        Returns:
            The total number of pages.

        """
        return (
            (self.total + self.page_size - 1) // self.page_size
            if self.page_size > 0
            else 0
        )


# ==============================================================================
# CONFIGURATION MODELS - USING MIXINS FOR CODE REDUCTION
# ==============================================================================


class BaseSettings(BaseModel):
    """Base for settings/configuration models."""

    model_config = SETTINGS_CONFIG


# Export commonly used Pydantic components
__all__ = [
    "API_CONFIG",
    "SETTINGS_CONFIG",
    "STRICT_CONFIG",
    "APIPaginatedResponse",
    "APIResponse",
    "BaseModel",  # Export BaseModel for compatibility
    "BaseSettings",
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainEntity",
    "DomainEvent",
    "DomainValueObject",
    "Field",  # Export Field for type safety
    "IdentifierMixin",
    "TimestampMixin",
]

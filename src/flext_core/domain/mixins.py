"""Advanced mixin system for FLEXT.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides reusable mixins for maximum code reduction across all modules.
Each mixin focuses on a single responsibility following SOLID principles.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from pydantic import computed_field

from flext_core.domain.shared_types import EntityId
from flext_core.domain.shared_types import EntityStatus

if TYPE_CHECKING:
    from uuid import UUID

T = TypeVar("T")


@runtime_checkable
class TimestampMixin(Protocol):
    """Mixin for entities with timestamp tracking."""

    created_at: datetime
    updated_at: datetime

    def mark_updated(self) -> None:
        """Mark entity as updated with current timestamp."""
        self.updated_at = datetime.now(UTC)

    def is_recently_created(self, seconds: int = 300) -> bool:
        """Check if entity was created within the specified seconds.

        Arguments:
            seconds: The number of seconds to check.

        Returns:
            True if the entity was created within the specified seconds,
            False otherwise.

        """
        return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)


@runtime_checkable
class StatusMixin(Protocol):
    """Mixin for entities with status tracking."""

    status: EntityStatus

    def is_active(self) -> bool:
        """Check if entity is active.

        Returns:
            True if the entity is active, False otherwise.

        """
        return self.status.value == "active"

    def is_inactive(self) -> bool:
        """Check if entity is inactive.

        Returns:
            True if the entity is inactive, False otherwise.

        """
        return self.status.value == "inactive"

    def activate(self) -> None:
        """Activate the entity."""
        self.status = EntityStatus.ACTIVE

    def deactivate(self) -> None:
        """Deactivate the entity."""
        self.status = EntityStatus.INACTIVE


@runtime_checkable
class ConfigurationMixin(Protocol):
    """Mixin for configuration management."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            A dictionary containing the configuration.

        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def get_subsection(self, prefix: str) -> dict[str, Any]:
        """Get configuration subsection by prefix.

        Arguments:
            prefix: The prefix to use to filter the configuration.

        Returns:
            A dictionary containing the filtered configuration.

        """
        data = self.to_dict()
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}

    def merge(self, other: dict[str, Any]) -> None:
        """Merge other configuration into this one.

        Arguments:
            other: The other configuration to merge into this one.

        """
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, value)


@runtime_checkable
class IdentifierMixin(Protocol):
    """Mixin for entities with unique identifiers."""

    id: EntityId

    @computed_field
    def uuid_str(self) -> str:
        """Get string representation of UUID.

        Returns:
            A string representation of the UUID.

        """
        return str(self.id)

    @computed_field
    def short_id(self) -> str:
        """Get short ID (first 8 characters).

        Returns:
            A string representation of the short ID.

        """
        return str(self.id)[:8]

    def equals_id(self, other_id: EntityId | UUID | str) -> bool:
        """Check if this entity has the same ID as the provided ID.

        Arguments:
            other_id: The other ID to compare to.

        Returns:
            True if the IDs are the same, False otherwise.

        """
        if isinstance(other_id, str):
            return str(self.id) == other_id
        return self.id == other_id


__all__ = [
    "ConfigurationMixin",
    "IdentifierMixin",
    "StatusMixin",
    "TimestampMixin",
]

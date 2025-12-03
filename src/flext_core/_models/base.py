"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: ZERO imports de flext_core (evita ciclos via __init__.py).
Usa apenas: stdlib (uuid, datetime) + pydantic.

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from flext_core.constants import c
from flext_core.typings import t


class FlextModelsBase:
    """Container for base model classes - Tier 0, 100% standalone."""

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class FrozenStrictModel(BaseModel):
        """Immutable base model with strict validation."""

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            frozen=True,
        )

    class FrozenValueModel(FrozenStrictModel):
        """Value model with equality/hash by value."""

        def __eq__(self, other: object) -> bool:
            """Compare by value using model_dump."""
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.model_dump() == other.model_dump()

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts."""
            data = self.model_dump()
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))

    class IdentifiableMixin(BaseModel):
        """Mixin for unique identifiers."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        unique_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Unique identifier for the model instance",
        )

    class TimestampableMixin(BaseModel):
        """Mixin for timestamps."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the model was created (UTC timezone)",
        )
        updated_at: datetime | None = Field(
            default=None,
            description="Timestamp when the model was last updated (UTC timezone)",
        )

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:  # noqa: PLR6301
            """Serialize timestamps to ISO 8601 format for JSON.

            Note: Pydantic @field_serializer requires instance method signature
            even if self is not used in the implementation.
            """
            return value.isoformat() if value else None

        @computed_field
        def is_modified(self) -> bool:
            """Check if the model has been modified after creation."""
            return self.updated_at is not None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

    class VersionableMixin(BaseModel):
        """Mixin for versioning (usa constante inline, não c)."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        version: int = Field(
            default=c.Performance.DEFAULT_VERSION,
            ge=c.Performance.MIN_VERSION,
            description="Version number for optimistic locking",
        )

        @computed_field
        def is_initial_version(self) -> bool:
            """Check if this is the initial version (version 1)."""
            return self.version == c.Performance.DEFAULT_VERSION

        def increment_version(self) -> None:
            """Increment the version number for optimistic locking."""
            self.version += 1

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""

    class Metadata(BaseModel):
        """Immutable metadata model - zero-dependency foundation.

        This is the SINGLE source of truth for metadata across the entire FLEXT ecosystem.
        All other modules import from here, creating a clear dependency hierarchy.
        """

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

        created_by: str | None = Field(
            default=None,
            description="User/service that created this metadata",
        )
        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="UTC timestamp of creation",
        )
        modified_by: str | None = Field(
            default=None,
            description="User/service that last modified this metadata",
        )
        modified_at: datetime | None = Field(
            default=None,
            description="UTC timestamp of last modification",
        )
        tags: list[str] = Field(
            default_factory=list,
            description="Tags for categorization and filtering",
        )
        attributes: dict[str, t.MetadataAttributeValue] = Field(
            default_factory=dict,
            description="Additional metadata attributes (JSON-serializable)",
        )


__all__ = ["FlextModelsBase"]

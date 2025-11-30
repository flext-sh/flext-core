"""Metadata model - Zero-dependency base metadata for FLEXT ecosystem.

CRITICAL: This module has ZERO imports from other flext_core modules to avoid circular dependencies.
It can be imported anywhere in the codebase without causing import cycles.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

type MetadataAttributeValue = (
    str
    | int
    | float
    | bool
    | list[str | int | float | bool | None]
    | dict[str, str | int | float | bool | None]
    | None
)


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
    attributes: dict[str, MetadataAttributeValue] = Field(
        default_factory=dict,
        description="Additional metadata attributes (JSON-serializable)",
    )


__all__ = ["Metadata"]

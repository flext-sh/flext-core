"""Metadata model for FlextRuntime — isolated for lazy pydantic loading.

This module is imported lazily by FlextRuntime.__getattr__ to defer
pydantic dependency loading until Metadata is first accessed.

Used by exceptions.py and other low-level modules that cannot import
from _models.base to maintain proper architecture layering.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from flext_core import t


class Metadata(BaseModel):
    """Minimal metadata model - implements p.Log.Metadata protocol.

    Used by exceptions.py and other low-level modules that cannot import
    from _models.base to maintain proper architecture layering.
    Tier 0.5 can be imported by Tier 1 modules like exceptions.py.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp recording when this metadata object was created.",
        title="Created At",
        examples=["2026-01-01T00:00:00Z"],
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp indicating the last metadata update time.",
        title="Updated At",
        examples=["2026-01-01T00:05:00Z"],
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version for the metadata schema payload.",
        title="Schema Version",
        examples=["1.0.0"],
    )
    attributes: Mapping[str, t.MetadataValue] = Field(
        default_factory=dict,
        description="Flexible key-value metadata attributes attached to the owning entity.",
        title="Attributes",
        examples=[{"service": "billing", "region": "us-east-1"}],
    )

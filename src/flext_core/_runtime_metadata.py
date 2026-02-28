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

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="1.0.0")
    attributes: Mapping[str, t.MetadataAttributeValue] = Field(default_factory=dict)

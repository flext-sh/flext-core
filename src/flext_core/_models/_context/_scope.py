"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import BeforeValidator, Field

from flext_core import FlextModelFoundation, c, t
from flext_core._models._context import FlextModelsContextData


class FlextModelsContextScope:
    """Namespace for context scope and statistics models."""

    class ContextScopeData(FlextModelFoundation.ArbitraryTypesModel):
        """Scope-specific data container for context management."""

        scope_name: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Name of the scope",
            ),
        ] = ""
        scope_type: Annotated[
            str,
            Field(default="", description="Type/category of scope"),
        ] = ""
        data: Annotated[
            Mapping[str, t.ValueOrModel],
            BeforeValidator(lambda v: FlextModelsContextData.normalize_to_mapping(v)),
            Field(default_factory=dict, description="Scope data"),
        ]
        metadata: Annotated[
            Mapping[str, t.NormalizedValue],
            BeforeValidator(lambda v: FlextModelsContextData.normalize_to_mapping(v)),
            Field(default_factory=dict, description="Scope metadata"),
        ]

    class ContextStatistics(FlextModelFoundation.ArbitraryTypesModel):
        """Statistics tracking for context operations."""

        sets: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of set operations"),
        ] = c.ZERO
        gets: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of get operations"),
        ] = c.ZERO
        removes: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of remove operations"),
        ] = c.ZERO
        clears: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of clear operations"),
        ] = c.ZERO
        operations: Annotated[
            Mapping[str, t.NormalizedValue],
            BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                )
            ),
            Field(
                default_factory=dict,
                description="Additional metric counters and timing values grouped by metric key.",
            ),
        ] = Field(default_factory=dict)


__all__ = ["FlextModelsContextScope"]

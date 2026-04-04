"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import BeforeValidator, Field

from flext_core import FlextModelsBase, FlextModelsContextData, c, t


class FlextModelsContextScope:
    """Namespace for context scope and statistics models."""

    class ContextScopeData(FlextModelsBase.ArbitraryTypesModel):
        """Scope-specific data container for context management."""

        scope_name: Annotated[
            t.NonEmptyStr,
            Field(description="Name of the scope"),
        ] = ""
        scope_type: Annotated[
            str,
            Field(default="", description="Type/category of scope"),
        ] = ""
        data: Annotated[
            Mapping[str, t.ValueOrModel],
            BeforeValidator(lambda v: FlextModelsContextData.normalize_to_mapping(v)),
            Field(description="Scope data"),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            t.ContainerMapping,
            BeforeValidator(lambda v: FlextModelsContextData.normalize_to_mapping(v)),
            Field(description="Scope metadata"),
        ] = Field(default_factory=dict)

    class ContextStatistics(FlextModelsBase.ArbitraryTypesModel):
        """Statistics tracking for context operations."""

        sets: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of set operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        gets: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of get operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        removes: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of remove operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        clears: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of clear operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        operations: Annotated[
            t.ContainerMapping,
            BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
            Field(
                description="Additional metric counters and timing values grouped by metric key.",
            ),
        ] = Field(default_factory=dict)


__all__ = ["FlextModelsContextScope"]

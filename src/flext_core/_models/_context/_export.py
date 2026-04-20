"""Context export and snapshot models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from types import MappingProxyType
from typing import Annotated

from pydantic import BeforeValidator, Field

from flext_core import (
    FlextModelsBase as m,
    FlextModelsContainers,
    FlextModelsContextData,
    FlextModelsEntity,
    t,
)


class FlextModelsContextExport:
    """Namespace for context export models."""

    class ContextExport(
        FlextModelsContextData.SerializableDataValidatorMixin,
        FlextModelsEntity.Value,
    ):
        """Typed snapshot returned by export_snapshot."""

        data: Annotated[
            Mapping[str, t.RuntimeData],
            Field(
                description="All context data from all scopes",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.Dict | None,
            BeforeValidator(
                lambda v: FlextModelsContextData.normalize_metadata_before(v),
            ),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        statistics: Annotated[
            Mapping[str, t.Container],
            BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
            Field(
                description="Usage statistics (operation counts, timing info)",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))


__all__: list[str] = ["FlextModelsContextExport"]

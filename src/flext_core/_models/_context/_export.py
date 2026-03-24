"""Context export and snapshot models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import (
    BeforeValidator,
    Field,
    computed_field,
)

from flext_core import (
    FlextModelFoundation,
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
            Mapping[str, t.ValueOrModel],
            Field(
                default_factory=dict,
                description="All context data from all scopes",
            ),
        ]
        metadata: Annotated[
            FlextModelFoundation.Metadata | t.Dict | None,
            BeforeValidator(
                lambda v: FlextModelsContextData.normalize_metadata_before(v),
            ),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        statistics: Annotated[
            t.ContainerMapping,
            BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
            Field(
                default_factory=dict,
                description="Usage statistics (operation counts, timing info)",
            ),
        ] = Field(default_factory=dict)

        @computed_field
        def has_statistics(self) -> bool:
            return bool(self.statistics)

        @computed_field
        def total_data_items(self) -> int:
            return len(self.data)


__all__ = ["FlextModelsContextExport"]

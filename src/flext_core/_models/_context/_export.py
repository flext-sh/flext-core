"""Context export and snapshot models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, Field
from typing_extensions import TypeForm

from flext_core import t
from flext_core._models._context._data import FlextModelsContextData
from flext_core._models.base import FlextModelsBase
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.pydantic import FlextModelsPydantic


class FlextModelsContextExport:
    """Namespace for context export models."""

    class ContextExport(
        FlextModelsContextData.SerializableDataValidatorMixin,
        FlextModelsEntity.Value,
    ):
        """Typed snapshot returned by export_snapshot."""

        # Why assigned-value form for the specifier calls (not ``Annotated``
        # metadata): pyright's ``dataclass_transform`` synthesis recognizes
        # ``default_factory`` default-ness only from the specifier call
        # assigned to the class variable; a specifier inside ``Annotated``
        # metadata synthesizes a REQUIRED ``__init__`` parameter.
        data: t.MappingKV[str, t.JsonPayload] = Field(
            default_factory=FlextModelsPydantic.empty(
                TypeForm(t.MappingKV[str, t.JsonPayload]),
            ),
            description="All context data from all scopes",
        )
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.Dict | None,
            BeforeValidator(FlextModelsContextData.normalize_metadata_before),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        statistics: Annotated[
            t.JsonMapping,
            BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
        ] = Field(
            default_factory=FlextModelsPydantic.empty(TypeForm(t.JsonMapping)),
            description="Usage statistics (operation counts, timing info)",
        )


__all__: t.MutableSequenceOf[str] = ["FlextModelsContextExport"]

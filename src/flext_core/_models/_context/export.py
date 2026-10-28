"""Context export and snapshot models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Mapping, MutableSequence
from types import MappingProxyType
from typing import Annotated

from pydantic import BeforeValidator, Field

from flext_core import t
from flext_core._models import FlextModelsBase, FlextModelsContainers, FlextModelsEntity
from flext_core._models._context import FlextModelsContextData


class FlextModelsContextExport:
    """Namespace for context export models."""

    class ContextExport(
        FlextModelsEntity.Value,
    ):
        """Typed snapshot returned by export_snapshot."""

        # Why assigned-value form for the specifier calls (not ``Annotated``
        # metadata): pyright's ``dataclass_transform`` synthesis recognizes
        # ``default_factory`` default-ness only from the specifier call
        # assigned to the class variable; a specifier inside ``Annotated``
        # metadata synthesizes a REQUIRED ``__init__`` parameter.
        data: Annotated[
            Mapping[str, t.JsonPayload],
            BeforeValidator(
                FlextModelsContextData.ContextData.validate_dict_serializable
            ),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, t.JsonPayload]({}),
            validate_default=True,
            description="All context data from all scopes",
        )
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.Dict | None,
            BeforeValidator(
                FlextModelsContextData.ContextData.validate_metadata_before
            ),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        statistics: Annotated[
            t.JsonMapping,
            BeforeValidator(
                FlextModelsContextData.ContextData.validate_dict_serializable
            ),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, t.JsonValue]({}),
            validate_default=True,
            description="Usage statistics (operation counts, timing info)",
        )


__all__: MutableSequence[str] = ["FlextModelsContextExport"]

"""FlextModelsConfig - declarative config record models (ADR-005).

Frozen Pydantic v2 record for a loaded config document: its parsed data plus
optional schema and source-path references. flext-core owns only this minimal
record; flext-cli returns instances of it from its advanced loader.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import FlextTypes as t
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp


class FlextModelsConfig:
    """Container for declarative config record models (ADR-005)."""

    class ConfigDocument(m.FrozenModel):
        """A loaded, parsed config document with optional schema/source refs."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        data: Annotated[
            t.JsonMapping,
            mp.Field(description="Parsed config mapping (execution parametrization)."),
        ]
        source_path: Annotated[
            str | None,
            mp.Field(default=None, description="Absolute path of the config source."),
        ] = None
        schema_ref: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Path of the JSON Schema validating this document.",
            ),
        ] = None


__all__: list[str] = ["FlextModelsConfig"]

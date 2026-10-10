"""FlextModelsOptions - call option envelopes below the base model presets.

The base model presets take their timestamp defaults from the generators
utility, so an option envelope that the generators utility consumes is built
on the pydantic surface alone: built on a base preset it would close an import
cycle between the presets and the utility they depend on.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase


class FlextModelsOptions:
    """Container for option envelopes consumed below the base model presets."""

    class GenerateOptions(FlextModelsPydantic.BaseModel):
        """Typed options envelope for public ID generation."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(extra="forbid")
        )

        prefix: Annotated[
            str | None,
            FlextModelsPydantic.Field(description="Custom ID prefix"),
        ] = None
        parts: Annotated[
            FlextTypingBase.VariadicTuple[FlextTypingBase.JsonValue] | None,
            FlextModelsPydantic.Field(
                description="Optional parts inserted between prefix and random suffix",
            ),
        ] = None
        length: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                description="Optional random suffix length override",
            ),
        ] = None
        include_timestamp: Annotated[
            bool,
            FlextModelsPydantic.Field(
                description="Whether to prepend a UTC timestamp to parts",
            ),
        ] = False
        separator: Annotated[
            str,
            FlextModelsPydantic.Field(
                description="Separator used for custom formatted IDs",
            ),
        ] = "_"


__all__: list[str] = ["FlextModelsOptions"]

"""Handler state models - Pydantic v2, state-only surface.

Only fields, validators, and computed properties that are consumed by
``src/`` (handlers, registry, utilities). Orchestration mutations live in
``FlextUtilitiesHandler``. All helper methods, dead factories, dict-style
accessors, and redundant wrapper classes have been removed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import (
    FlextConstants as c,
    FlextProtocols as p,
    FlextTypes as t,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp

from .flextmodelshandler_part_01 import (
    FlextModelsHandler as FlextModelsHandlerPart01,
)


class FlextModelsHandler(FlextModelsHandlerPart01):
    class DecoratorConfig(m.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        command: Annotated[
            type,
            mp.Field(description="Command type this handler processes"),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            mp.Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            t.SequenceOf[type[p.Middleware]],
            mp.Field(description="Middleware types to apply to this handler"),
        ] = mp.Field(default_factory=tuple)

    class CombinedRailwayOptions(m.ImmutableValueModel):
        """Railway configuration consumed by @d.combined()."""

        enabled: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether combined() applies railway wrapping.",
            ),
        ] = False
        error_code: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Error code passed to railway() when railway wrapping is enabled.",
            ),
        ] = None


__all__: list[str] = ["FlextModelsHandler"]

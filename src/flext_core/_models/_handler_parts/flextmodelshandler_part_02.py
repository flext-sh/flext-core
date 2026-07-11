"""Handler state models - Pydantic v2, state-only surface.

Only fields, validators, and computed properties that are consumed by
``src/`` (handlers, registry, utilities). Orchestration mutations live in
``FlextUtilitiesHandler``. All helper methods, dead factories, dict-style
accessors, and redundant wrapper classes have been removed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
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
    class HandlerRuntimeState(m.ArbitraryTypesModel):
        """Aggregate runtime state for the active handler pipeline."""

        execution_context: Annotated[
            FlextModelsHandlerPart01.ExecutionContext,
            mp.Field(description="Execution context for the active handler"),
        ]
        context_stack: Annotated[
            MutableSequence[FlextModelsHandlerPart01.ExecutionContext],
            mp.Field(
                description="Stack of nested execution contexts.",
            ),
        ] = mp.Field(default_factory=list)
        accepted_message_types: Annotated[
            tuple[t.MessageTypeSpecifier, ...],
            mp.Field(
                description="Accepted message types computed for dispatch routing",
            ),
        ] = mp.Field(default_factory=tuple)
        revalidate_pydantic_messages: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether Pydantic messages must be revalidated on dispatch",
            ),
        ] = False
        type_warning_emitted: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether the handler already emitted a type warning",
            ),
        ] = False

        @mp.computed_field()
        @property
        def handler_name(self) -> str:
            """Active handler name taken from the execution context."""
            return self.execution_context.handler_name

        @mp.computed_field()
        @property
        def handler_mode(self) -> c.HandlerType:
            """Active handler mode taken from the execution context."""
            return self.execution_context.handler_mode

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

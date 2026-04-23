"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
)
from typing import Annotated

from flext_core import (
    FlextModelsBase as m,
    FlextModelsEntity,
    FlextModelsHandler,
    FlextUtilitiesPydantic as up,
    p,
    t,
)


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistryState(m.ArbitraryTypesModel):
        """Validated registry runtime state shared by public registry methods."""

        dispatcher: Annotated[
            p.Dispatcher | None,
            up.Field(
                default=None,
                description="Dispatcher used for handler registration and execution.",
            ),
        ] = None
        registered_keys: Annotated[
            frozenset[str],
            up.Field(
                default_factory=frozenset,
                description="Keys registered in the instance scope of the registry.",
            ),
        ]

        @up.computed_field()
        def configured(self) -> bool:
            """Whether a dispatcher has been materialized for the registry."""
            return self.dispatcher is not None

    class RegistrySummary(FlextModelsEntity.Value):
        """Aggregated outcome for batch handler registration tracking."""

        registered: Annotated[
            MutableSequence[FlextModelsHandler.RegistrationDetails],
            up.Field(
                description="Successfully registered handlers with registration details.",
            ),
        ] = up.Field(default_factory=list[FlextModelsHandler.RegistrationDetails])
        skipped: Annotated[
            t.StrSequence,
            up.Field(
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = up.Field(default_factory=tuple)
        errors: Annotated[
            MutableSequence[str],
            up.Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = up.Field(default_factory=list[str])

        @up.computed_field()
        def failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            return bool(self.errors)

        @up.computed_field()
        def success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors


__all__: t.MutableSequenceOf[str] = ["FlextModelsRegistry"]

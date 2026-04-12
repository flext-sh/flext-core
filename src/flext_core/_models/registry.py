"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Annotated

from flext_core import FlextModelsBase, FlextModelsEntity, FlextModelsHandler, p, t
from flext_core._utilities.pydantic import FlextUtilitiesPydantic


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistryState(FlextModelsBase.ArbitraryTypesModel):
        """Validated registry runtime state shared by public registry methods."""

        dispatcher: Annotated[
            p.Dispatcher | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Dispatcher used for handler registration and execution.",
            ),
        ] = None
        registered_keys: Annotated[
            frozenset[str],
            FlextUtilitiesPydantic.Field(
                default_factory=frozenset,
                description="Keys registered in the instance scope of the registry.",
            ),
        ]

        @FlextUtilitiesPydantic.computed_field()
        def configured(self) -> bool:
            """Whether a dispatcher has been materialized for the registry."""
            return self.dispatcher is not None

    class RegistrySummary(FlextModelsEntity.Value):
        """Aggregated outcome for batch handler registration tracking."""

        registered: Annotated[
            MutableSequence[FlextModelsHandler.RegistrationDetails],
            FlextUtilitiesPydantic.Field(
                description="Successfully registered handlers with registration details.",
            ),
        ] = FlextUtilitiesPydantic.Field(
            default_factory=list[FlextModelsHandler.RegistrationDetails]
        )
        skipped: Annotated[
            t.StrSequence,
            FlextUtilitiesPydantic.Field(
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=list[str])
        errors: Annotated[
            MutableSequence[str],
            FlextUtilitiesPydantic.Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=list[str])

        @FlextUtilitiesPydantic.computed_field()
        def failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            return bool(self.errors)

        @FlextUtilitiesPydantic.computed_field()
        def success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors


__all__: list[str] = ["FlextModelsRegistry"]

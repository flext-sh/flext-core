"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Annotated

from flext_core import t
from flext_core._models.base import FlextModelsBase
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._protocols import FlextProtocolsHandler
from flext_core._utilities import FlextUtilitiesPydantic


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistryState(FlextModelsBase.ArbitraryTypesModel):
        """Validated registry runtime state shared by public registry methods."""

        dispatcher: Annotated[
            FlextProtocolsHandler.Dispatcher | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Dispatcher used for handler registration and execution.",
            ),
        ] = None
        registered_keys: Annotated[
            frozenset[str],
            FlextModelsPydantic.Field(
                description="Keys registered in the instance scope of the registry.",
            ),
        ] = FlextModelsPydantic.Field(default_factory=frozenset)

        @FlextUtilitiesPydantic.computed_field
        @property
        def configured(self) -> bool:
            """Whether a dispatcher has been materialized for the registry."""
            return self.dispatcher is not None

    class RegistrySummary(FlextModelsEntity.Value):
        """Aggregated outcome for batch handler registration tracking."""

        registered: Annotated[
            MutableSequence[FlextModelsHandler.RegistrationDetails],
            FlextModelsPydantic.Field(
                description=(
                    "Successfully registered handlers with registration details."
                ),
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=list[FlextModelsHandler.RegistrationDetails],
        )
        skipped: Annotated[
            t.StrSequence,
            FlextModelsPydantic.Field(
                description=(
                    "Handler identifiers that were skipped (already registered)"
                ),
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = FlextModelsPydantic.Field(default_factory=tuple)
        errors: Annotated[
            MutableSequence[str],
            FlextModelsPydantic.Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = FlextModelsPydantic.Field(default_factory=list[str])

        @FlextUtilitiesPydantic.computed_field
        @property
        def failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            return bool(self.errors)

        @FlextUtilitiesPydantic.computed_field
        @property
        def success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors


__all__: t.MutableSequenceOf[str] = ["FlextModelsRegistry"]

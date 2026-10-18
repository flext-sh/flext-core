"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Annotated, ClassVar

from typing_extensions import TypeForm

from flext_core import p, t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsEntity,
    FlextModelsHandler,
    FlextModelsPydantic,
)
from flext_core._utilities import FlextUtilitiesPydantic


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistryOptions(FlextModelsBase.ArbitraryTypesModel):
        """Validated runtime dependency and plugin ownership of a registry."""

        registry_runtime: t.Port[p.ServiceRuntime] = FlextModelsPydantic.Field(
            exclude=True,
            description="Explicit runtime shared by registration and execution.",
        )
        _registered_keys: set[str] = FlextModelsPydantic.PrivateAttr(
            default_factory=FlextModelsPydantic.empty(TypeForm(set[str])),
        )
        _class_plugin_storage: ClassVar[dict[str, t.RegisterableService]] = {}

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
            default_factory=FlextModelsPydantic.empty(
                TypeForm(MutableSequence[FlextModelsHandler.RegistrationDetails]),
            ),
        )
        skipped: Annotated[
            t.StrSequence,
            FlextModelsPydantic.Field(
                description=(
                    "Handler identifiers that were skipped (already registered)"
                ),
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=FlextModelsPydantic.empty(TypeForm(t.StrSequence)),
        )
        errors: Annotated[
            MutableSequence[str],
            FlextModelsPydantic.Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=FlextModelsPydantic.empty(TypeForm(MutableSequence[str])),
        )

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

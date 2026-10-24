"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from threading import RLock
from typing import Annotated, ClassVar

from flext_core import p, t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsEntity,
    FlextModelsHandler,
    FlextModelsPydantic,
)


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistryServiceState(FlextModelsBase.ArbitraryTypesModel):
        """Instance-owned synchronized registry access for composed services."""

        _registry: p.Registry | None = FlextModelsPydantic.PrivateAttr(None)
        _registry_lock: p.Lock = FlextModelsPydantic.PrivateAttr(default_factory=RLock)

    class RegistryOptions(FlextModelsBase.ArbitraryTypesModel):
        """Validated runtime dependency and plugin ownership of a registry."""

        registry_runtime: t.Port[p.ServiceRuntime] = FlextModelsPydantic.Field(
            exclude=True,
            description="Explicit runtime shared by registration and execution.",
        )
        _registered_keys: set[str] = FlextModelsPydantic.PrivateAttr(
            default_factory=set[str],
        )
        _class_plugin_storage: ClassVar[
            t.MutableMappingKV[str, p.RegisterableService]
        ] = {}

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
            validate_default=True,
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
            default=(),
            validate_default=True,
        )
        errors: Annotated[
            MutableSequence[str],
            FlextModelsPydantic.Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=list[str],
            validate_default=True,
        )

        @FlextModelsPydantic.computed_field
        @property
        def failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            return bool(self.errors)

        @FlextModelsPydantic.computed_field
        @property
        def success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors


__all__: t.MutableSequenceOf[str] = ["FlextModelsRegistry"]

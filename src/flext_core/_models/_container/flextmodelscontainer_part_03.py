"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._container import flextmodelscontainer_part_02
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsSettings,
)
from flext_core._typings import FlextTypesPydantic, FlextTypesServices


class FlextModelsContainerPart03(
    flextmodelscontainer_part_02.FlextModelsContainerPart02
):
    class ServiceRegistrationSpec(FlextModelsBase.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                strict=True,
                arbitrary_types_allowed=True,
            )
        )

        settings: Annotated[
            FlextProtocolsSettings.Settings | None,
            FlextModelsPydantic.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            FlextProtocolsContext.Context | None,
            FlextModelsPydantic.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        dispatcher: FlextTypesPydantic.Port[FlextProtocolsHandler.Dispatcher | None] = (
            FlextModelsPydantic.Field(
                default=None,
                exclude=True,
                description=(
                    "Explicit command bus retained by this scope; "
                    "otherwise a fresh bus is built."
                ),
            )
        )
        services: Annotated[
            FlextProtocolsContainer.ServiceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            ),
        ] = None
        factories: Annotated[
            FlextProtocolsContainer.FactoryMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            ),
        ] = None
        resources: Annotated[
            FlextProtocolsContainer.ResourceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            ),
        ] = None
        user_overrides: (
            FlextProtocolsBase.Model | FlextTypesServices.UserOverridesMapping | None
        ) = FlextModelsPydantic.Field(
            default=None,
            title="User Overrides",
            description="User-level configuration overrides applied after defaults.",
            validate_default=True,
        )
        container_config: FlextProtocolsBase.Model | None = FlextModelsPydantic.Field(
            default=None,
            title="Container Config",
            description="Container configuration model controlling DI behavior.",
            validate_default=True,
        )


__all__: list[str] = ["FlextModelsContainerPart03"]

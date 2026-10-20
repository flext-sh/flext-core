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

from flext_core import p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._container_parts import flextmodelscontainer_part_02


class FlextModelsContainer(flextmodelscontainer_part_02.FlextModelsContainer):
    t = t
    p = p

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
            p.Settings | None,
            FlextModelsPydantic.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            p.Context | None,
            FlextModelsPydantic.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        dispatcher: t.Port[p.Dispatcher | None] = FlextModelsPydantic.Field(
            default=None,
            exclude=True,
            description=(
                "Explicit command bus retained by this scope; "
                "otherwise a fresh bus is built."
            ),
        )
        services: Annotated[
            p.ServiceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            ),
        ] = None
        factories: Annotated[
            p.FactoryMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            ),
        ] = None
        resources: Annotated[
            p.ResourceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            ),
        ] = None
        user_overrides: t.UserOverridesMapping | None = FlextModelsPydantic.Field(
            default=None,
            title="User Overrides",
            description="User-level configuration overrides applied after defaults.",
            validate_default=True,
        )
        container_config: p.Model | None = FlextModelsPydantic.Field(
            default=None,
            title="Container Config",
            description="Container configuration model controlling DI behavior.",
            validate_default=True,
        )


__all__: list[str] = ["FlextModelsContainer"]

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
from flext_core._models._container import FlextModelsContainerPart02


class FlextModelsContainerPart03(FlextModelsContainerPart02):
    class ServiceRegistrationSpec(FlextModelsBase.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                strict=True,
            )
        )

        settings: Annotated[
            t.Port[p.Settings] | None,
            FlextModelsPydantic.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            t.Port[p.Context] | None,
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
        user_overrides: t.Port[p.Model] | t.UserOverridesMapping | None = (
            FlextModelsPydantic.Field(
                default=None,
                title="User Overrides",
                description="User-level configuration overrides applied after defaults.",
                validate_default=True,
            )
        )
        container_config: t.Port[p.Model] | None = FlextModelsPydantic.Field(
            default=None,
            title="Container Config",
            description="Container configuration model controlling DI behavior.",
            validate_default=True,
        )


__all__: list[str] = ["FlextModelsContainerPart03"]

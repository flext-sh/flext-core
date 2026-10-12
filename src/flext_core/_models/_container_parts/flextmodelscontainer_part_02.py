"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._container_parts import flextmodelscontainer_part_01
from flext_core.constants import c
from flext_core.typings import t


class FlextModelsContainer(flextmodelscontainer_part_01.FlextModelsContainer):
    class ResourceRegistration(
        flextmodelscontainer_part_01.FlextModelsContainer.RegistrationEnvelope,
    ):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories invoked on every resolve for
        connection-style dependencies (DB/HTTP).
        """

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Resource identifier/name"),
        ]
        factory: Annotated[
            t.ResourceCallable,
            FlextModelsPydantic.Field(
                ...,
                description="Factory returning the lifecycle-managed resource",
            ),
        ]

    class ContainerConfig(FlextModelsBase.FlexibleInternalModel):
        """Model for container configuration.

        Replaces: m.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        enable_singleton: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Enable singleton pattern for factories",
            ),
        ] = True
        enable_factory_caching: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Enable caching of factory-created instances",
            ),
        ] = True
        max_services: Annotated[
            t.PositiveInt,
            FlextModelsPydantic.Field(
                c.DEFAULT_SIZE,
                le=c.MAX_ITEMS,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.DEFAULT_SIZE
        max_factories: Annotated[
            t.PositiveInt,
            FlextModelsPydantic.Field(
                c.DEFAULT_MAX_FACTORIES,
                le=c.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = c.DEFAULT_MAX_FACTORIES
        enable_auto_registration: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=False,
                description="Enable automatic service registration from decorators",
            ),
        ] = False
        enable_lifecycle_hooks: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Enable lifecycle hooks (on_register, on_get, etc.)",
            ),
        ] = True
        lazy_loading: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True, description="Enable lazy loading of services"
            ),
        ] = True


__all__: list[str] = ["FlextModelsContainer"]

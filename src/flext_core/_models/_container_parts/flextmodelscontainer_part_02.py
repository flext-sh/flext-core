"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from flext_core import FlextConstants as c, FlextTypes as t
from flext_core._models._container_parts.flextmodelscontainer_part_01 import (
    FlextModelsContainer as FlextModelsContainerPart01,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._models.containers import FlextModelsContainers
from flext_core._runtime._container import FlextRuntimeContainer as FlextRuntime
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug


class FlextModelsContainer(FlextModelsContainerPart01):
    class ResourceRegistration(m.ArbitraryTypesModel):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        name: Annotated[
            t.NonEmptyStr, mp.Field(..., description="Resource identifier/name")
        ]
        factory: Annotated[
            t.ResourceCallable,
            tp.SkipValidation,
            mp.Field(
                ..., description="Factory returning the lifecycle-managed resource"
            ),
        ]
        registration_time: Annotated[
            datetime,
            mp.Field(
                description="Timestamp when resource was registered (configured timezone)"
            ),
        ] = mp.Field(default_factory=ug.now)
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value, m.Metadata
                )
            ),
            mp.Field(
                None, description="Additional resource metadata (JSON-serializable)"
            ),
        ] = None

    class ContainerConfig(m.FlexibleInternalModel):
        """Model for container configuration.

        Replaces: p.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        enable_singleton: Annotated[
            bool, mp.Field(True, description="Enable singleton pattern for factories")
        ] = True
        enable_factory_caching: Annotated[
            bool,
            mp.Field(True, description="Enable caching of factory-created instances"),
        ] = True
        max_services: Annotated[
            t.PositiveInt,
            mp.Field(
                c.DEFAULT_SIZE,
                le=c.MAX_ITEMS,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.DEFAULT_SIZE
        max_factories: Annotated[
            t.PositiveInt,
            mp.Field(
                c.DEFAULT_MAX_FACTORIES,
                le=c.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = c.DEFAULT_MAX_FACTORIES
        enable_auto_registration: Annotated[
            bool,
            mp.Field(
                False,
                description="Enable automatic service registration from decorators",
            ),
        ] = False
        enable_lifecycle_hooks: Annotated[
            bool,
            mp.Field(
                True, description="Enable lifecycle hooks (on_register, on_get, etc.)"
            ),
        ] = True
        lazy_loading: Annotated[
            bool, mp.Field(True, description="Enable lazy loading of services")
        ] = True


__all__: list[str] = ["FlextModelsContainer"]

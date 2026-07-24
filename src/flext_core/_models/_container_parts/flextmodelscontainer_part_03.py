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

from flext_core import FlextProtocols as p, FlextTypes as t
from flext_core._models._container_parts.flextmodelscontainer_part_02 import (
    FlextModelsContainer as FlextModelsContainerPart02,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextModelsContainer(FlextModelsContainerPart02):
    class ServiceRegistrationSpec(m.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            strict=True, arbitrary_types_allowed=True
        )

        settings: Annotated[
            p.Settings | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            p.Context | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        services: Annotated[
            t.MappingKV[
                str, FlextModelsContainer.ServiceRegistration | t.RegisterableService
            ]
            | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            ),
        ] = None
        factories: Annotated[
            t.MappingKV[
                str, FlextModelsContainer.FactoryRegistration | t.FactoryCallable
            ]
            | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            ),
        ] = None
        resources: Annotated[
            t.MappingKV[
                str, FlextModelsContainer.ResourceRegistration | t.ResourceCallable
            ]
            | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            ),
        ] = None
        user_overrides: (
            FlextModelsContainers.ConfigMap
            | t.MappingKV[
                str, FlextModelsContainers.ConfigMap | t.ScalarList | t.Scalar
            ]
            | None
        ) = mp.Field(
            None,
            title="User Overrides",
            description="User-level configuration overrides applied after defaults.",
            validate_default=True,
        )
        container_config: FlextModelsContainer.ContainerConfig | None = mp.Field(
            None,
            title="Container Config",
            description="Container configuration model controlling DI behavior.",
            validate_default=True,
        )


__all__: list[str] = ["FlextModelsContainer"]

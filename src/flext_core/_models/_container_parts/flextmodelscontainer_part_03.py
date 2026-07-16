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
from flext_core._models._container_parts.flextmodelscontainer_part_02 import (
    FlextModelsContainer as FlextModelsContainerPart02,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.context import FlextProtocolsContext as pc
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up


class FlextModelsContainer(FlextModelsContainerPart02):
    class ServiceRegistrationSpec(m.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[t.ConfigDict] = t.ConfigDict(
            strict=True, arbitrary_types_allowed=True
        )

        settings: Annotated[
            ps.Settings | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            pc.Context | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        services: t.MappingKV[str, FlextModelsContainer.ServiceRegistration] | None = (
            mp.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            )
        )
        factories: t.MappingKV[str, FlextModelsContainer.FactoryRegistration] | None = (
            mp.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            )
        )
        resources: (
            t.MappingKV[str, FlextModelsContainer.ResourceRegistration] | None
        ) = mp.Field(
            None,
            title="Resources",
            description="Pre-registered resource factories for bootstrap.",
            validate_default=True,
        )
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

        @up.field_validator("services", mode="before")
        @classmethod
        def validate_services(
            cls,
            value: (
                t.MappingKV[
                    str,
                    FlextModelsContainer.ServiceRegistration | t.RegisterableService,
                ]
                | None
            ),
        ) -> t.MappingKV[str, FlextModelsContainer.ServiceRegistration] | None:
            if value is None:
                return None
            return {
                name: (
                    registration
                    if isinstance(
                        registration, FlextModelsContainer.ServiceRegistration
                    )
                    else FlextModelsContainer.ServiceRegistration(
                        name=name,
                        service=registration,
                        service_type=registration.__class__.__name__,
                    )
                )
                for name, registration in value.items()
            }

        @staticmethod
        def _norm_callable_reg[Reg: p.ArbitraryTypesModel](
            value: t.MappingKV[str, Reg | t.FactoryCallable] | None, reg_cls: type[Reg]
        ) -> t.MappingKV[str, Reg] | None:
            """Normalize a name→(Registration|callable) dict to name→Registration."""
            if value is None:
                return None
            return {
                name: (
                    registration
                    if isinstance(registration, reg_cls)
                    else reg_cls.model_validate({"name": name, "factory": registration})
                )
                for name, registration in value.items()
            }

        @up.field_validator("factories", mode="before")
        @classmethod
        def validate_factories(
            cls,
            value: (
                t.MappingKV[
                    str, FlextModelsContainer.FactoryRegistration | t.FactoryCallable
                ]
                | None
            ),
        ) -> t.MappingKV[str, FlextModelsContainer.FactoryRegistration] | None:
            return cls._norm_callable_reg(
                value, FlextModelsContainer.FactoryRegistration
            )

        @up.field_validator("resources", mode="before")
        @classmethod
        def validate_resources(
            cls,
            value: (
                t.MappingKV[
                    str, FlextModelsContainer.ResourceRegistration | t.ResourceCallable
                ]
                | None
            ),
        ) -> t.MappingKV[str, FlextModelsContainer.ResourceRegistration] | None:
            return cls._norm_callable_reg(
                value, FlextModelsContainer.ResourceRegistration
            )


__all__: list[str] = ["FlextModelsContainer"]

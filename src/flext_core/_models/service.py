"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Annotated

from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up
from flext_core.protocols import FlextProtocols as p
from flext_core.typings import FlextTypes as t


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class ServiceRuntime(m.ArbitraryTypesModel):
        """Shared runtime state for services and infrastructure collaborators.

        Represents the core service runtime with configuration, context,
        dependency injection container, and the internal dispatcher/registry
        implementations exposed only through protocol-typed fields.
        """

        settings: Annotated[
            p.Settings,
            tp.SkipValidation,
            mp.Field(
                description="Service configuration settings for runtime behavior.",
            ),
        ]
        context: Annotated[
            p.Context,
            tp.SkipValidation,
            mp.Field(
                description="Execution context carrying correlation and tracing metadata.",
            ),
        ]
        container: Annotated[
            p.Container,
            tp.SkipValidation,
            mp.Field(
                description="Dependency injection container for service resolution.",
            ),
        ]
        dispatcher: Annotated[
            p.Dispatcher | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Dispatcher resolved for CQRS routing in this runtime.",
            ),
        ] = None
        registry: Annotated[
            p.Registry | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Registry bound to the runtime when one is materialized.",
            ),
        ] = None

    class RuntimeBootstrapOptions(m.ArbitraryTypesModel):
        """Options for runtime bootstrapping."""

        settings: Annotated[
            p.Settings | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Pre-built settings instance used directly for the runtime.",
                validate_default=True,
            ),
        ] = None
        settings_type: type | None = mp.Field(
            None,
            description="FlextSettings class used to load runtime settings.",
            validate_default=True,
        )
        settings_overrides: t.ScalarMapping | None = mp.Field(
            None,
            description="Key-value overrides applied on top of the loaded configuration.",
            validate_default=True,
        )
        context: Annotated[
            p.Context | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Pre-built execution context to inject into the runtime.",
                validate_default=True,
            ),
        ] = None
        dispatcher: Annotated[
            p.Dispatcher | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Pre-built dispatcher injected into the runtime DSL.",
                validate_default=True,
            ),
        ] = None
        registry: Annotated[
            p.Registry | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Pre-built registry injected into the runtime DSL.",
                validate_default=True,
            ),
        ] = None
        subproject: str | None = mp.Field(
            None,
            description="Subproject name used to scope configuration and wiring.",
            validate_default=True,
        )
        services: Annotated[
            t.MappingKV[str, Annotated[t.RegisterableService, tp.SkipValidation]]
            | None,
            mp.Field(
                None,
                description="Named services to register in the dependency container.",
                validate_default=True,
            ),
        ] = None
        factories: Annotated[
            t.MappingKV[str, Annotated[t.FactoryCallable, tp.SkipValidation]] | None,
            mp.Field(
                None,
                description="Named factory callables to register in the dependency container.",
                validate_default=True,
            ),
        ] = None
        resources: Annotated[
            t.MappingKV[str, Annotated[t.ResourceCallable, tp.SkipValidation]] | None,
            mp.Field(
                None,
                description="Named lifecycle resources to register in the dependency container.",
                validate_default=True,
            ),
        ] = None
        container_overrides: t.ScalarMapping | None = mp.Field(
            None,
            description="Provider overrides applied to the dependency container.",
            validate_default=True,
        )
        wire_modules: t.SequenceOf[ModuleType | str] | None = mp.Field(
            None,
            description="Modules to wire for dependency-injector resolution.",
            validate_default=True,
        )
        wire_packages: t.StrSequence | None = mp.Field(
            None,
            description="Package names to consider for dependency wiring.",
            validate_default=True,
        )
        wire_classes: t.SequenceOf[type] | None = mp.Field(
            None,
            description="Classes whose modules are wired for dependency resolution.",
            validate_default=True,
        )

        @up.field_validator("wire_packages", mode="before")
        @classmethod
        def validate_wire_packages(
            cls,
            value: t.JsonPayload | None,
        ) -> t.JsonPayload | None:
            if not isinstance(value, Sequence) or isinstance(
                value,
                (str, bytes, bytearray),
            ):
                return value
            normalized = tuple(item for item in value if isinstance(item, str))
            return normalized if len(normalized) == len(value) else None


__all__: t.MutableSequenceOf[str] = ["FlextModelsService"]

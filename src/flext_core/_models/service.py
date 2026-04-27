"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from types import ModuleType
from typing import Annotated

from flext_core import (
    FlextModelsBase as m,
    FlextModelsPydantic as mp,
    p,
    t,
)


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
            mp.Field(
                description="Service configuration settings for runtime behavior."
            ),
        ]
        context: Annotated[
            p.Context,
            mp.Field(
                description="Execution context carrying correlation and tracing metadata.",
            ),
        ]
        container: Annotated[
            p.Container,
            mp.Field(
                description="Dependency injection container for service resolution."
            ),
        ]
        dispatcher: Annotated[
            p.Dispatcher | None,
            mp.Field(
                None,
                description="Dispatcher resolved for CQRS routing in this runtime.",
            ),
        ] = None
        registry: Annotated[
            p.Registry | None,
            mp.Field(
                None,
                description="Registry bound to the runtime when one is materialized.",
            ),
        ] = None

    class RuntimeBootstrapOptions(m.ArbitraryTypesModel):
        """Options for runtime bootstrapping."""

        settings: p.Settings | None = mp.Field(
            None,
            description="Pre-built settings instance used directly for the runtime.",
            validate_default=True,
        )
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
        context: p.Context | None = mp.Field(
            None,
            description="Pre-built execution context to inject into the runtime.",
            validate_default=True,
        )
        dispatcher: p.Dispatcher | None = mp.Field(
            None,
            description="Pre-built dispatcher injected into the runtime DSL.",
            validate_default=True,
        )
        registry: p.Registry | None = mp.Field(
            None,
            description="Pre-built registry injected into the runtime DSL.",
            validate_default=True,
        )
        subproject: str | None = mp.Field(
            None,
            description="Subproject name used to scope configuration and wiring.",
            validate_default=True,
        )
        services: Mapping[str, t.RegisterableService] | None = mp.Field(
            None,
            description="Named services to register in the dependency container.",
            validate_default=True,
        )
        factories: Mapping[str, t.FactoryCallable] | None = mp.Field(
            None,
            description="Named factory callables to register in the dependency container.",
            validate_default=True,
        )
        resources: Mapping[str, t.ResourceCallable] | None = mp.Field(
            None,
            description="Named lifecycle resources to register in the dependency container.",
            validate_default=True,
        )
        container_overrides: t.ScalarMapping | None = mp.Field(
            None,
            description="Provider overrides applied to the dependency container.",
            validate_default=True,
        )
        wire_modules: Sequence[ModuleType | str] | None = mp.Field(
            None,
            description="Modules to wire for dependency-injector resolution.",
            validate_default=True,
        )
        wire_packages: t.StrSequence | None = mp.Field(
            None,
            description="Package names to consider for dependency wiring.",
            validate_default=True,
        )
        wire_classes: Sequence[type] | None = mp.Field(
            None,
            description="Classes whose modules are wired for dependency resolution.",
            validate_default=True,
        )

    class DependencyContainerCreationOptions(m.ArbitraryTypesModel):
        """Options used to create and populate dependency container instances."""

        settings: Mapping[str, t.JsonPayload] | None = mp.Field(
            None,
            title="Configuration",
            description="Optional configuration mapping bound to dependency container providers.",
            validate_default=True,
        )
        services: Mapping[str, t.RegisterableService] | None = mp.Field(
            None,
            title="Services",
            description="Object providers registered before optional wiring.",
            validate_default=True,
        )
        factories: Mapping[str, t.FactoryCallable] | None = mp.Field(
            None,
            title="Factories",
            description="Factory providers registered with singleton/factory semantics.",
            validate_default=True,
        )
        resources: Mapping[str, t.ResourceCallable] | None = mp.Field(
            None,
            title="Resources",
            description="Lifecycle resource providers registered before wiring.",
            validate_default=True,
        )
        wire_modules: Sequence[ModuleType] | None = mp.Field(
            None,
            title="Wire Modules",
            description="Modules wired for dependency-injector @inject resolution.",
            validate_default=True,
        )
        wire_packages: t.StrSequence | None = mp.Field(
            None,
            title="Wire Packages",
            description="Package names considered for dependency wiring.",
            validate_default=True,
        )
        wire_classes: Sequence[type] | None = mp.Field(
            None,
            title="Wire Classes",
            description="Classes whose modules are wired for dependency resolution.",
            validate_default=True,
        )
        factory_cache: bool = mp.Field(
            True,
            title="Factory Cache",
            description="Whether registered factories use singleton caching semantics.",
            validate_default=True,
        )


__all__: t.MutableSequenceOf[str] = ["FlextModelsService"]

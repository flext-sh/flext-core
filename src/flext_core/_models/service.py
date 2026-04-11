"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Annotated

from pydantic import Field

from flext_core import FlextModelsBase, p, t


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Runtime triple (settings, context, container) for services.

        Represents the core service runtime with configuration, context,
        and dependency injection container. CQRS components (dispatcher,
        registry) should be used directly - not through FlextService.
        """

        settings: Annotated[
            p.Settings,
            Field(description="Service configuration settings for runtime behavior."),
        ]
        context: Annotated[
            p.Context,
            Field(
                description="Execution context carrying correlation and tracing metadata.",
            ),
        ]
        container: Annotated[
            p.Container,
            Field(description="Dependency injection container for service resolution."),
        ]

    class RuntimeBootstrapOptions(FlextModelsBase.ArbitraryTypesModel):
        """Options for runtime bootstrapping."""

        settings: p.Settings | None = Field(
            default=None,
            description="Pre-built settings instance used directly for the runtime.",
        )
        settings_type: type | None = Field(
            default=None,
            description="FlextSettings class used to load runtime settings.",
        )
        settings_overrides: t.ScalarMapping | None = Field(
            default=None,
            description="Key-value overrides applied on top of the loaded configuration.",
        )
        context: p.Context | None = Field(
            default=None,
            description="Pre-built execution context to inject into the runtime.",
        )
        subproject: str | None = Field(
            default=None,
            description="Subproject name used to scope configuration and wiring.",
        )
        services: Mapping[str, t.RegisterableService] | None = Field(
            default=None,
            description="Named services to register in the dependency container.",
        )
        factories: Mapping[str, t.FactoryCallable] | None = Field(
            default=None,
            description="Named factory callables to register in the dependency container.",
        )
        resources: Mapping[str, t.ResourceCallable] | None = Field(
            default=None,
            description="Named lifecycle resources to register in the dependency container.",
        )
        container_overrides: t.ScalarMapping | None = Field(
            default=None,
            description="Provider overrides applied to the dependency container.",
        )
        wire_modules: Sequence[ModuleType | str] | None = Field(
            default=None,
            description="Modules to wire for dependency-injector resolution.",
        )
        wire_packages: t.StrSequence | None = Field(
            default=None,
            description="Package names to consider for dependency wiring.",
        )
        wire_classes: Sequence[type] | None = Field(
            default=None,
            description="Classes whose modules are wired for dependency resolution.",
        )

    class DependencyContainerCreationOptions(FlextModelsBase.ArbitraryTypesModel):
        """Options used to create and populate dependency container instances."""

        settings: t.ConfigMap | None = Field(
            default=None,
            title="Configuration",
            description="Optional configuration mapping bound to dependency container providers.",
        )
        services: Mapping[str, t.RegisterableService] | None = Field(
            default=None,
            title="Services",
            description="Object providers registered before optional wiring.",
        )
        factories: Mapping[str, t.FactoryCallable] | None = Field(
            default=None,
            title="Factories",
            description="Factory providers registered with singleton/factory semantics.",
        )
        resources: Mapping[str, t.ResourceCallable] | None = Field(
            default=None,
            title="Resources",
            description="Lifecycle resource providers registered before wiring.",
        )
        wire_modules: Sequence[ModuleType] | None = Field(
            default=None,
            title="Wire Modules",
            description="Modules wired for dependency-injector @inject resolution.",
        )
        wire_packages: t.StrSequence | None = Field(
            default=None,
            title="Wire Packages",
            description="Package names considered for dependency wiring.",
        )
        wire_classes: Sequence[type] | None = Field(
            default=None,
            title="Wire Classes",
            description="Classes whose modules are wired for dependency resolution.",
        )
        factory_cache: bool = Field(
            default=True,
            title="Factory Cache",
            description="Whether registered factories use singleton caching semantics.",
        )


__all__ = ["FlextModelsService"]

"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + _models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, ClassVar

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    SkipValidation,
    field_validator,
)

from flext_core import FlextModelsBase, FlextRuntime, c, p, t


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(
        FlextModelsBase.ArbitraryTypesModel,
    ):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: t.ConfigMap for service tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            Field(
                ...,
                description="Service identifier/name",
            ),
        ]
        service: Annotated[
            t.RegisterableService,
            SkipValidation,
            Field(..., description="Service instance (protocols, models, callables)"),
        ]
        registration_time: Annotated[
            datetime,
            Field(
                description="UTC timestamp when service was registered",
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            Field(
                default=None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None
        service_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Service type name (e.g., 'DatabaseService')",
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            Field(
                description="Service tags for categorization",
            ),
        ] = Field(default_factory=list)

        @field_validator("service", mode="before")
        @classmethod
        def validate_service(
            cls,
            value: t.RegisterableService,
        ) -> t.RegisterableService | t.ConfigMap | t.ObjectList:
            return FlextRuntime.normalize_registerable_service(value)

    class FactoryRegistration(
        FlextModelsBase.ArbitraryTypesModel,
    ):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: t.ConfigMap for factory tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            Field(
                ...,
                description="Factory identifier/name",
            ),
        ]
        factory: Annotated[
            t.FactoryCallable,
            SkipValidation,
            Field(..., description="Factory function that creates service instances"),
        ]
        registration_time: Annotated[
            datetime,
            Field(
                description="UTC timestamp when factory was registered",
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        is_singleton: Annotated[
            bool,
            Field(
                default=False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        cached_instance: Annotated[
            t.RegisterableService | None,
            SkipValidation,
            Field(
                default=None,
                description="Cached singleton instance (if is_singleton=True)",
            ),
        ] = None
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            Field(
                default=None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of times factory has been invoked",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES

    class ResourceRegistration(
        FlextModelsBase.ArbitraryTypesModel,
    ):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        name: Annotated[
            t.NonEmptyStr,
            Field(
                ...,
                description="Resource identifier/name",
            ),
        ]
        factory: Annotated[
            t.ResourceCallable,
            SkipValidation,
            Field(..., description="Factory returning the lifecycle-managed resource"),
        ]
        registration_time: Annotated[
            datetime,
            Field(
                description="UTC timestamp when resource was registered",
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            Field(
                default=None,
                description="Additional resource metadata (JSON-serializable)",
            ),
        ] = None

    class ContainerConfig(FlextModelsBase.FlexibleInternalModel):
        """Model for container configuration.

        Replaces: t.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        enable_singleton: Annotated[
            bool,
            Field(
                default=True,
                description="Enable singleton pattern for factories",
            ),
        ] = True
        enable_factory_caching: Annotated[
            bool,
            Field(
                default=True,
                description="Enable caching of factory-created instances",
            ),
        ] = True
        max_services: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_SIZE,
                le=c.MAX_ITEMS,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.DEFAULT_SIZE
        max_factories: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_MAX_FACTORIES,
                le=c.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = c.DEFAULT_MAX_FACTORIES
        enable_auto_registration: Annotated[
            bool,
            Field(
                default=False,
                description="Enable automatic service registration from decorators",
            ),
        ] = False
        enable_lifecycle_hooks: Annotated[
            bool,
            Field(
                default=True,
                description="Enable lifecycle hooks (on_register, on_get, etc.)",
            ),
        ] = True
        lazy_loading: Annotated[
            bool,
            Field(
                default=True,
                description="Enable lazy loading of services",
            ),
        ] = True

    class ServiceRegistrationSpec(FlextModelsBase.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            arbitrary_types_allowed=True,
        )

        settings: Annotated[
            p.Settings | None,
            SkipValidation,
            Field(
                default=None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ]
        context: Annotated[
            p.Context | None,
            SkipValidation,
            Field(
                default=None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ]
        services: Mapping[str, FlextModelsContainer.ServiceRegistration] | None = Field(
            default=None,
            title="Services",
            description="Pre-registered service instances for bootstrap.",
        )
        factories: Mapping[str, FlextModelsContainer.FactoryRegistration] | None = (
            Field(
                default=None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
            )
        )
        resources: Mapping[str, FlextModelsContainer.ResourceRegistration] | None = (
            Field(
                default=None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
            )
        )
        user_overrides: (
            t.ConfigMap | Mapping[str, t.ConfigMap | t.ScalarList | t.Scalar] | None
        ) = Field(
            default=None,
            title="User Overrides",
            description="User-level configuration overrides applied after defaults.",
        )
        container_config: FlextModelsContainer.ContainerConfig | None = Field(
            default=None,
            title="Container Config",
            description="Container configuration model controlling DI behavior.",
        )

    class FactoryDecoratorConfig(FlextModelsBase.ImmutableValueModel):
        """Configuration extracted from @d.factory() decorator.

        Used by factory discovery to auto-register factories with FlextContainer.
        Stores metadata about factory name, singleton behavior, and lazy loading.

        Attributes:
            name: The name to register this factory under in the container.
            singleton: Whether the factory creates singleton instances. Default: False.
            lazy: Whether to defer factory invocation until first use. Default: True.

        Examples:
            >>> settings = FlextModelsContainer.FactoryDecoratorConfig(
            ...     name="database_service",
            ...     singleton=True,
            ...     lazy=False,
            ... )
            >>> settings.name
            'database_service'
            >>> settings.singleton
            True

        """

        name: Annotated[
            t.NonEmptyStr,
            Field(
                ...,
                description="Name to register this factory under in the container",
            ),
        ]
        singleton: Annotated[
            bool,
            Field(
                default=False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        lazy: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to defer factory invocation until first use",
            ),
        ] = True


__all__ = ["FlextModelsContainer"]

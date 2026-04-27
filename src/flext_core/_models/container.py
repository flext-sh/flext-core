"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from datetime import UTC, datetime
from typing import Annotated, ClassVar, no_type_check

from flext_core import (
    FlextModelsBase as m,
    FlextModelsContainers,
    FlextModelsPydantic as mp,
    FlextRuntime,
    FlextTypesPydantic as tp,
    FlextUtilitiesPydantic as up,
    c,
    p,
    t,
)


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(
        m.ArbitraryTypesModel,
    ):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: m.ConfigMap for service tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                ...,
                description="Service identifier/name",
            ),
        ]
        service: Annotated[
            t.RegisterableService,
            tp.SkipValidation,
            mp.Field(
                ..., description="Service instance (protocols, models, callables)"
            ),
        ]
        registration_time: Annotated[
            datetime,
            mp.Field(
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when service was registered",
            ),
        ]
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    m.Metadata,
                ),
            ),
            mp.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None
        service_type: Annotated[
            str | None,
            mp.Field(
                None,
                description="Service type name (e.g., 'DatabaseService')",
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple, description="Service tags for categorization"
            ),
        ]

        @up.field_validator("service", mode="before")
        @classmethod
        def validate_service(
            cls,
            value: t.RegisterableService,
        ) -> (
            t.RegisterableService
            | FlextModelsContainers.ConfigMap
            | FlextModelsContainers.ObjectList
        ):
            return FlextRuntime.normalize_registerable_service(value)

    class FactoryRegistration(
        m.ArbitraryTypesModel,
    ):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: m.ConfigMap for factory tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                ...,
                description="Factory identifier/name",
            ),
        ]
        factory: Annotated[
            t.FactoryCallable,
            tp.SkipValidation,
            mp.Field(
                ..., description="Factory function that creates service instances"
            ),
        ]
        registration_time: Annotated[
            datetime,
            mp.Field(
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when factory was registered",
            ),
        ]
        is_singleton: Annotated[
            bool,
            mp.Field(
                False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        cached_instance: Annotated[
            t.RegisterableService | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Cached singleton instance (if is_singleton=True)",
            ),
        ] = None
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    m.Metadata,
                ),
            ),
            mp.Field(
                None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            t.NonNegativeInt,
            mp.Field(
                c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of times factory has been invoked",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES

    class ResourceRegistration(
        m.ArbitraryTypesModel,
    ):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                ...,
                description="Resource identifier/name",
            ),
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
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when resource was registered",
            ),
        ]
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    m.Metadata,
                ),
            ),
            mp.Field(
                None,
                description="Additional resource metadata (JSON-serializable)",
            ),
        ] = None

    class ContainerConfig(m.FlexibleInternalModel):
        """Model for container configuration.

        Replaces: m.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        enable_singleton: Annotated[
            bool,
            mp.Field(
                True,
                description="Enable singleton pattern for factories",
            ),
        ] = True
        enable_factory_caching: Annotated[
            bool,
            mp.Field(
                True,
                description="Enable caching of factory-created instances",
            ),
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
                True,
                description="Enable lifecycle hooks (on_register, on_get, etc.)",
            ),
        ] = True
        lazy_loading: Annotated[
            bool,
            mp.Field(
                True,
                description="Enable lazy loading of services",
            ),
        ] = True

    class ServiceRegistrationSpec(m.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            strict=True,
            arbitrary_types_allowed=True,
        )

        settings: Annotated[
            p.Settings | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ]
        context: Annotated[
            p.Context | None,
            tp.SkipValidation,
            mp.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ]
        services: Mapping[str, FlextModelsContainer.ServiceRegistration] | None = (
            mp.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            )
        )
        factories: Mapping[str, FlextModelsContainer.FactoryRegistration] | None = (
            mp.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            )
        )
        resources: Mapping[str, FlextModelsContainer.ResourceRegistration] | None = (
            mp.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            )
        )
        user_overrides: (
            FlextModelsContainers.ConfigMap
            | Mapping[str, FlextModelsContainers.ConfigMap | t.ScalarList | t.Scalar]
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

    class FactoryDecoratorConfig(m.ImmutableValueModel):
        """Configuration extracted from @d.factory() decorator.

        Used by factory discovery to auto-register factories with FlextContainer.
        Stores metadata about factory name, singleton behavior, and lazy loading.

        Attributes:
            name: The name to register this factory under in the container.
            singleton: Whether the factory creates singleton instances. Default: False.
            lazy: Whether to defer factory invocation until first use. Default: True.

        Examples:
            >>> settings = mc.FactoryDecoratorConfig(
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
            mp.Field(
                ...,
                description="Name to register this factory under in the container",
            ),
        ]
        singleton: Annotated[
            bool,
            mp.Field(
                False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        lazy: Annotated[
            bool,
            mp.Field(
                True,
                description="Whether to defer factory invocation until first use",
            ),
        ] = True


mc = FlextModelsContainer
_ = no_type_check(FlextModelsContainer)

__all__: list[str] = ["FlextModelsContainer", "mc"]

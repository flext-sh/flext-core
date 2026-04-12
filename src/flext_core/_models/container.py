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

from flext_core import (
    FlextConstantsPydantic,
    FlextModelsBase,
    FlextModelsPydantic,
    FlextTypesPydantic,
    FlextUtilitiesPydantic,
    c,
    p,
    t,
)
from flext_core.runtime import FlextRuntime


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
            FlextUtilitiesPydantic.Field(
                ...,
                description="Service identifier/name",
            ),
        ]
        service: Annotated[
            t.RegisterableService,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                ..., description="Service instance (protocols, models, callables)"
            ),
        ]
        registration_time: Annotated[
            datetime,
            FlextUtilitiesPydantic.Field(
                description="UTC timestamp when service was registered",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: datetime.now(UTC))
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            FlextModelsPydantic.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            FlextUtilitiesPydantic.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None
        service_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                None,
                description="Service type name (e.g., 'DatabaseService')",
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            FlextUtilitiesPydantic.Field(
                description="Service tags for categorization",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=list)

        @FlextUtilitiesPydantic.field_validator("service", mode="before")
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
            FlextUtilitiesPydantic.Field(
                ...,
                description="Factory identifier/name",
            ),
        ]
        factory: Annotated[
            t.FactoryCallable,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                ..., description="Factory function that creates service instances"
            ),
        ]
        registration_time: Annotated[
            datetime,
            FlextUtilitiesPydantic.Field(
                description="UTC timestamp when factory was registered",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: datetime.now(UTC))
        is_singleton: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        cached_instance: Annotated[
            t.RegisterableService | None,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                None,
                description="Cached singleton instance (if is_singleton=True)",
            ),
        ] = None
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            FlextModelsPydantic.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            FlextUtilitiesPydantic.Field(
                None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                c.DEFAULT_MAX_COMMAND_RETRIES,
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
            FlextUtilitiesPydantic.Field(
                ...,
                description="Resource identifier/name",
            ),
        ]
        factory: Annotated[
            t.ResourceCallable,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                ..., description="Factory returning the lifecycle-managed resource"
            ),
        ]
        registration_time: Annotated[
            datetime,
            FlextUtilitiesPydantic.Field(
                description="UTC timestamp when resource was registered",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: datetime.now(UTC))
        metadata: Annotated[
            FlextModelsBase.Metadata | t.ConfigMap | None,
            FlextModelsPydantic.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            FlextUtilitiesPydantic.Field(
                None,
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
            FlextUtilitiesPydantic.Field(
                True,
                description="Enable singleton pattern for factories",
            ),
        ] = True
        enable_factory_caching: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                True,
                description="Enable caching of factory-created instances",
            ),
        ] = True
        max_services: Annotated[
            t.PositiveInt,
            FlextUtilitiesPydantic.Field(
                c.DEFAULT_SIZE,
                le=c.MAX_ITEMS,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.DEFAULT_SIZE
        max_factories: Annotated[
            t.PositiveInt,
            FlextUtilitiesPydantic.Field(
                c.DEFAULT_MAX_FACTORIES,
                le=c.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = c.DEFAULT_MAX_FACTORIES
        enable_auto_registration: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                False,
                description="Enable automatic service registration from decorators",
            ),
        ] = False
        enable_lifecycle_hooks: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                True,
                description="Enable lifecycle hooks (on_register, on_get, etc.)",
            ),
        ] = True
        lazy_loading: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                True,
                description="Enable lazy loading of services",
            ),
        ] = True

    class ServiceRegistrationSpec(FlextModelsBase.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[FlextConstantsPydantic.ConfigDict] = (
            FlextConstantsPydantic.ConfigDict(
                strict=True,
                arbitrary_types_allowed=True,
            )
        )

        settings: Annotated[
            p.Settings | None,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ]
        context: Annotated[
            p.Context | None,
            FlextTypesPydantic.SkipValidation,
            FlextUtilitiesPydantic.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ]
        services: Mapping[str, FlextModelsContainer.ServiceRegistration] | None = (
            FlextUtilitiesPydantic.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            )
        )
        factories: Mapping[str, FlextModelsContainer.FactoryRegistration] | None = (
            FlextUtilitiesPydantic.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            )
        )
        resources: Mapping[str, FlextModelsContainer.ResourceRegistration] | None = (
            FlextUtilitiesPydantic.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            )
        )
        user_overrides: (
            t.ConfigMap | Mapping[str, t.ConfigMap | t.ScalarList | t.Scalar] | None
        ) = FlextUtilitiesPydantic.Field(
            None,
            title="User Overrides",
            description="User-level configuration overrides applied after defaults.",
            validate_default=True,
        )
        container_config: FlextModelsContainer.ContainerConfig | None = (
            FlextUtilitiesPydantic.Field(
                None,
                title="Container Config",
                description="Container configuration model controlling DI behavior.",
                validate_default=True,
            )
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
            FlextUtilitiesPydantic.Field(
                ...,
                description="Name to register this factory under in the container",
            ),
        ]
        singleton: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        lazy: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                True,
                description="Whether to defer factory invocation until first use",
            ),
        ] = True


__all__: list[str] = ["FlextModelsContainer"]

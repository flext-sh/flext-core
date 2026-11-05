"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, ClassVar

import flext_core
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)

if TYPE_CHECKING:
    pass


class FlextModelsContainer:
    """Public facade for FlextModelsContainer."""

    class RegistrationEnvelope(FlextModelsBase.ArbitraryTypesModel):
        """Shared registration timestamp and validated metadata boundary."""

        registration_time: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description=(
                    "Timestamp when service was registered (configured timezone)"
                ),
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: datetime.now(UTC),
        )
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.ConfigMap | None,
            FlextModelsPydantic.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None

        @FlextModelsPydantic.field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata_model_input(
            cls, value: flext_core.p.MetadataInput
        ) -> flext_core.p.Metadata | None:
            """Keep metadata identity and validate raw registration attributes.

            Returns:
                The validated metadata or None.

            """
            if value is None:
                return None
            if isinstance(value, FlextModelsBase.Metadata):
                return value
            return FlextModelsBase.Metadata.model_validate({
                flext_core.c.FIELD_ATTRIBUTES: (
                    value.model_dump(mode="json")
                    if isinstance(value, flext_core.p.HasModelDump)
                    else value
                )
            })

    class ServiceRegistration(RegistrationEnvelope):
        """Service registry entry following the declared service contract."""

        name: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Service identifier/name"),
        ]
        service: Annotated[
            flext_core.t.SkipJsonSchema[flext_core.p.RegisterableService],
            FlextModelsPydantic.Field(
                ...,
                description="Service instance (protocols, models, callables)",
                exclude=True,
            ),
        ]
        tags: Annotated[
            Sequence[str],
            FlextModelsPydantic.Field(
                description="Service tags for categorization", validate_default=True
            ),
        ] = ()

        @FlextModelsPydantic.computed_field
        @property
        def service_type(self) -> str:
            """Type name of the registered service, derived from the service."""
            return type(self.service).__name__

    class FactoryRegistration(RegistrationEnvelope):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: p.ConfigMap for factory tracking.
        """

        name: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Factory identifier/name"),
        ]
        factory: Annotated[
            flext_core.p.FactoryCallable,
            FlextModelsPydantic.Field(
                ...,
                description="Factory function that creates service instances",
            ),
        ]

    class ResourceRegistration(
        RegistrationEnvelope,
    ):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories invoked on every resolve for
        connection-style dependencies (DB/HTTP).
        """

        name: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Resource identifier/name"),
        ]
        factory: Annotated[
            flext_core.p.ResourceCallable,
            FlextModelsPydantic.Field(
                ...,
                description="Factory returning the lifecycle-managed resource",
            ),
        ]

    class ContainerConfig(FlextModelsBase.FlexibleInternalModel):
        """Model for container configuration.

        Replaces: p.ConfigMap for container configuration storage.
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
            flext_core.t.PositiveInt,
            FlextModelsPydantic.Field(
                flext_core.c.DEFAULT_SIZE,
                le=flext_core.c.MAX_ITEMS,
                description="Maximum number of services allowed in registry",
            ),
        ] = flext_core.c.DEFAULT_SIZE
        max_factories: Annotated[
            flext_core.t.PositiveInt,
            FlextModelsPydantic.Field(
                flext_core.c.DEFAULT_MAX_FACTORIES,
                le=flext_core.c.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = flext_core.c.DEFAULT_MAX_FACTORIES
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

    class ServiceRegistrationSpec(FlextModelsBase.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                strict=True,
            )
        )

        settings: Annotated[
            flext_core.t.Port[flext_core.p.Settings] | None,
            FlextModelsPydantic.Field(
                None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            flext_core.t.Port[flext_core.p.Context] | None,
            FlextModelsPydantic.Field(
                None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        dispatcher: flext_core.t.Port[flext_core.p.Dispatcher | None] = (
            FlextModelsPydantic.Field(
                default=None,
                exclude=True,
                description=(
                    "Explicit command bus retained by this scope; "
                    "otherwise a fresh bus is built."
                ),
            )
        )
        services: Annotated[
            flext_core.p.ServiceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
                validate_default=True,
            ),
        ] = None
        factories: Annotated[
            flext_core.p.FactoryMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
                validate_default=True,
            ),
        ] = None
        resources: Annotated[
            flext_core.p.ResourceMap | None,
            FlextModelsPydantic.Field(
                None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
                validate_default=True,
            ),
        ] = None
        user_overrides: (
            flext_core.t.Port[flext_core.p.BaseModel]
            | flext_core.t.UserOverridesMapping
            | None
        ) = FlextModelsPydantic.Field(
            default=None,
            title="User Overrides",
            description=("User-level configuration overrides applied after defaults."),
            validate_default=True,
        )
        container_config: flext_core.t.Port[flext_core.p.BaseModel] | None = (
            FlextModelsPydantic.Field(
                default=None,
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
            >>> settings = FlextModelsContainers.FactoryDecoratorConfig(
            ...     name="database_service", singleton=True, lazy=False
            ... )
            >>> settings.name
            'database_service'
            >>> settings.singleton
            True

        """

        name: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(
                ...,
                description="Name to register this factory under in the container",
            ),
        ]
        singleton: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        lazy: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Whether to defer factory invocation until first use",
            ),
        ] = True


__all__: list[str] = ["FlextModelsContainer"]

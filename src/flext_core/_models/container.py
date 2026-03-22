"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + _models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, ClassVar, TypeIs

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from flext_core import (
    FlextConstants as c,
    FlextModelFoundation,
    FlextProtocols as p,
    FlextTypes as t,
)


def _generate_datetime_utc() -> datetime:
    return datetime.now(UTC)


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    @staticmethod
    def _is_metadata_instance(
        v: t.MetadataInput,
    ) -> TypeIs[FlextModelFoundation.Metadata]:
        return isinstance(v, FlextModelFoundation.Metadata)

    @staticmethod
    def _normalize_metadata(value: t.MetadataInput) -> FlextModelFoundation.Metadata:
        if value is None:
            return FlextModelFoundation.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {},
            })
        if FlextModelsContainer._is_metadata_instance(value):
            return value
        if not isinstance(value, Mapping):
            msg = f"metadata must be None, dict, or FlextModelFoundation.Metadata, got {value.__class__.__name__}"
            raise TypeError(msg)
        return FlextModelFoundation.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: dict(value.items()),
        })

    class _MetadataValidatorMixin:
        """Mixin to provide metadata field coercion/normalization for models."""

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: t.MetadataInput) -> FlextModelFoundation.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode)."""
            return FlextModelsContainer._normalize_metadata(v)

    class ServiceRegistration(_MetadataValidatorMixin, BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: t.ConfigMap for service tracking.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
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
                default_factory=_generate_datetime_utc,
                description="UTC timestamp when service was registered",
            ),
        ] = Field(default_factory=_generate_datetime_utc)
        metadata: Annotated[
            FlextModelFoundation.Metadata | t.ConfigMap | None,
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
            list[str],
            Field(
                default_factory=list,
                description="Service tags for categorization",
            ),
        ] = Field(default_factory=list)

        @field_validator("service", mode="before")
        @classmethod
        def validate_service_type(
            cls,
            v: t.RegisterableService,
        ) -> t.RegisterableService | t.ConfigMap | t.ObjectList:
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            if isinstance(v, (BaseModel, Path)):
                return v
            if callable(v):
                return v
            if isinstance(v, Mapping):
                normalized_mapping: dict[str, t.ValueOrModel] = {}
                for key, item in v.items():
                    if isinstance(item, datetime):
                        normalized_mapping[str(key)] = (
                            item.replace(tzinfo=UTC) if item.tzinfo is None else item
                        )
                    elif isinstance(item, Path):
                        normalized_mapping[str(key)] = str(item)
                    elif isinstance(
                        item,
                        (
                            str,
                            int,
                            float,
                            bool,
                            list,
                            dict,
                            tuple,
                            type(None),
                            BaseModel,
                        ),
                    ):
                        normalized_mapping[str(key)] = item
                    else:
                        msg = f"Invalid type in Mapping: {type(item)}"
                        raise TypeError(msg)
                return t.ConfigMap(root=normalized_mapping)
            if isinstance(v, Sequence) and (not isinstance(v, (str, bytes, bytearray))):
                normalized_sequence: list[t.Container] = []
                for item in v:
                    if isinstance(item, datetime):
                        item = item.replace(tzinfo=UTC) if item.tzinfo is None else item
                    elif isinstance(item, Path):
                        item = str(item)
                    elif not isinstance(
                        item,
                        (
                            str,
                            int,
                            float,
                            bool,
                            list,
                            dict,
                            tuple,
                            type(None),
                            BaseModel,
                        ),
                    ):
                        msg = f"Invalid type in Sequence: {type(item)}"
                        raise TypeError(msg)

                    container_item: t.Container = str(item)
                    normalized_sequence.append(container_item)
                return t.ObjectList(root=normalized_sequence)
            if hasattr(v, "__dict__"):
                return v
            if hasattr(v, "bind") and hasattr(v, "info"):
                return v
            msg = f"Service must be a RegisterableService type, got {type(v).__name__}"
            raise ValueError(msg)

    class FactoryRegistration(_MetadataValidatorMixin, BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: t.ConfigMap for factory tracking.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
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
                default_factory=_generate_datetime_utc,
                description="UTC timestamp when factory was registered",
            ),
        ] = Field(default_factory=_generate_datetime_utc)
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
            FlextModelFoundation.Metadata | t.ConfigMap | None,
            Field(
                default=None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.ZERO,
                description="Number of times factory has been invoked",
            ),
        ] = c.ZERO

    class ResourceRegistration(_MetadataValidatorMixin, BaseModel):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
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
                default_factory=_generate_datetime_utc,
                description="UTC timestamp when resource was registered",
            ),
        ] = Field(default_factory=_generate_datetime_utc)
        metadata: Annotated[
            FlextModelFoundation.Metadata | t.ConfigMap | None,
            Field(
                default=None,
                description="Additional resource metadata (JSON-serializable)",
            ),
        ] = None

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: t.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=False, validate_assignment=True
        )
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
                default=c.DEFAULT_MAX_SERVICES,
                le=c.MAX_BATCH_SIZE,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.DEFAULT_MAX_SERVICES
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

    class ServiceRegistrationSpec(FlextModelFoundation.ArbitraryTypesModel):
        """Bootstrap specification for container registration.

        Holds pre-registered services, factories, resources, and configuration.
        Deferred to TIER 1 to avoid circular imports with p/t.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True, arbitrary_types_allowed=True
        )

        config: Annotated[
            p.Settings | None,
            SkipValidation,
            Field(
                default=None,
                title="Config",
                description="Settings instance bound to the container runtime.",
            ),
        ] = None
        context: Annotated[
            p.Context | None,
            SkipValidation,
            Field(
                default=None,
                title="Context",
                description="Execution context attached to the container.",
            ),
        ] = None
        services: Annotated[
            Mapping[str, FlextModelsContainer.ServiceRegistration] | None,
            SkipValidation,
            Field(
                default=None,
                title="Services",
                description="Pre-registered service instances for bootstrap.",
            ),
        ] = None
        factories: Annotated[
            Mapping[str, FlextModelsContainer.FactoryRegistration] | None,
            SkipValidation,
            Field(
                default=None,
                title="Factories",
                description="Pre-registered factory callables for bootstrap.",
            ),
        ] = None
        resources: Annotated[
            Mapping[str, FlextModelsContainer.ResourceRegistration] | None,
            SkipValidation,
            Field(
                default=None,
                title="Resources",
                description="Pre-registered resource factories for bootstrap.",
            ),
        ] = None
        user_overrides: Annotated[
            t.ConfigMap
            | Mapping[
                str,
                t.ConfigMap | Sequence[t.Scalar] | bool | datetime | float | int | str,
            ]
            | None,
            SkipValidation,
            Field(
                default=None,
                title="User Overrides",
                description="User-level configuration overrides applied after defaults.",
            ),
        ] = None
        container_config: Annotated[
            FlextModelsContainer.ContainerConfig | None,
            SkipValidation,
            Field(
                default=None,
                title="Container Config",
                description="Container configuration model controlling DI behavior.",
            ),
        ] = None

    class FactoryDecoratorConfig(BaseModel):
        """Configuration extracted from @d.factory() decorator.

        Used by factory discovery to auto-register factories with FlextContainer.
        Stores metadata about factory name, singleton behavior, and lazy loading.

        Attributes:
            name: The name to register this factory under in the container.
            singleton: Whether the factory creates singleton instances. Default: False.
            lazy: Whether to defer factory invocation until first use. Default: True.

        Examples:
            >>> config = FlextModelsContainer.FactoryDecoratorConfig(
            ...     name="database_service",
            ...     singleton=True,
            ...     lazy=False,
            ... )
            >>> config.name
            'database_service'
            >>> config.singleton
            True

        """

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
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

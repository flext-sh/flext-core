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
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from flext_core import c, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.containers import FlextModelsContainers
from flext_core.runtime import FlextRuntime

_MetadataInput = (
    FlextModelFoundation.Metadata
    | FlextModelsContainers.ConfigMap
    | Mapping[str, t.Scalar]
    | None
)


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    @staticmethod
    def _is_metadata_instance(
        v: _MetadataInput,
    ) -> TypeGuard[FlextModelFoundation.Metadata]:
        return isinstance(v, FlextModelFoundation.Metadata)

    @staticmethod
    def _normalize_metadata(value: _MetadataInput) -> FlextModelFoundation.Metadata:
        if value is None:
            return FlextModelFoundation.Metadata.model_validate({"attributes": {}})
        if FlextModelsContainer._is_metadata_instance(value):
            return value
        if not isinstance(value, Mapping):
            msg = f"metadata must be None, dict, or FlextModelFoundation.Metadata, got {value.__class__.__name__}"
            raise TypeError(msg)
        return FlextModelFoundation.Metadata.model_validate({
            "attributes": dict(value.items())
        })

    class ServiceRegistration(BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: m.ConfigMap for service tracking.
        """

        model_config = ConfigDict(
            frozen=False, validate_assignment=True, arbitrary_types_allowed=True
        )
        name: Annotated[
            str,
            Field(
                ...,
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Service identifier/name",
            ),
        ]
        service: Annotated[
            t.RegisterableService | t.DispatchableService,
            SkipValidation,
            Field(..., description="Service instance (protocols, models, callables)"),
        ]
        registration_time: Annotated[
            datetime,
            Field(
                default_factory=FlextRuntime.generate_datetime_utc,
                description="UTC timestamp when service was registered",
            ),
        ] = Field(default_factory=FlextRuntime.generate_datetime_utc)
        metadata: Annotated[
            FlextModelFoundation.Metadata | FlextModelsContainers.ConfigMap | None,
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

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode)."""
            return FlextModelsContainer._normalize_metadata(v)

        @field_validator("service", mode="before")
        @classmethod
        def validate_service_type(
            cls,
            v: t.RegisterableService,
        ) -> (
            t.RegisterableService
            | FlextModelsContainers.ConfigMap
            | FlextModelsContainers.ObjectList
        ):
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            if isinstance(v, (BaseModel, Path)):
                return v
            if callable(v):
                return v
            if isinstance(v, Mapping):
                normalized_mapping: dict[str, t.NormalizedValue | BaseModel] = {}
                for key, item in v.items():
                    normalized_mapping[str(key)] = FlextRuntime.normalize_to_container(
                        item
                    )
                return FlextModelsContainers.ConfigMap(root=normalized_mapping)
            if isinstance(v, Sequence) and (not isinstance(v, (str, bytes, bytearray))):
                normalized_sequence: list[t.Container] = []
                for item in v:
                    normalized_item = FlextRuntime.normalize_to_container(item)
                    container_item: t.Container = str(normalized_item)
                    normalized_sequence.append(container_item)
                return FlextModelsContainers.ObjectList(root=normalized_sequence)
            if hasattr(v, "__dict__"):
                return v
            if hasattr(v, "bind") and hasattr(v, "info"):
                return v
            msg = f"Service must be a RegisterableService type, got {type(v).__name__}"
            raise ValueError(msg)

    class FactoryRegistration(BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: m.ConfigMap for factory tracking.
        """

        model_config = ConfigDict(
            frozen=False, validate_assignment=True, arbitrary_types_allowed=True
        )
        name: Annotated[
            str,
            Field(
                ...,
                min_length=c.Reliability.RETRY_COUNT_MIN,
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
                default_factory=FlextRuntime.generate_datetime_utc,
                description="UTC timestamp when factory was registered",
            ),
        ] = Field(default_factory=FlextRuntime.generate_datetime_utc)
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
            FlextModelFoundation.Metadata | FlextModelsContainers.ConfigMap | None,
            Field(
                default=None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            int,
            Field(
                default=c.ZERO,
                ge=c.ZERO,
                description="Number of times factory has been invoked",
            ),
        ] = c.ZERO

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode)."""
            return FlextModelsContainer._normalize_metadata(v)

    class ResourceRegistration(BaseModel):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        model_config = ConfigDict(
            frozen=False, validate_assignment=True, arbitrary_types_allowed=True
        )
        name: Annotated[
            str,
            Field(
                ...,
                min_length=c.Reliability.RETRY_COUNT_MIN,
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
                default_factory=FlextRuntime.generate_datetime_utc,
                description="UTC timestamp when resource was registered",
            ),
        ] = Field(default_factory=FlextRuntime.generate_datetime_utc)
        metadata: Annotated[
            FlextModelFoundation.Metadata | FlextModelsContainers.ConfigMap | None,
            Field(
                default=None,
                description="Additional resource metadata (JSON-serializable)",
            ),
        ] = None

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Normalize resource metadata to Metadata model."""
            return FlextModelsContainer._normalize_metadata(v)

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: m.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        model_config = ConfigDict(frozen=False, validate_assignment=True)
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
            int,
            Field(
                default=c.Container.DEFAULT_MAX_SERVICES,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Performance.MAX_BATCH_SIZE,
                description="Maximum number of services allowed in registry",
            ),
        ] = c.Container.DEFAULT_MAX_SERVICES
        max_factories: Annotated[
            int,
            Field(
                default=c.Container.DEFAULT_MAX_FACTORIES,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Container.MAX_FACTORIES,
                description="Maximum number of factories allowed in registry",
            ),
        ] = c.Container.DEFAULT_MAX_FACTORIES
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

        model_config = ConfigDict(frozen=True)
        name: Annotated[
            str,
            Field(
                ...,
                min_length=c.Reliability.RETRY_COUNT_MIN,
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

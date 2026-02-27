"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + _models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Mapping, Sequence, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from flext_core._models.base import FlextModelFoundation
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

_MetadataInput = (
    FlextModelFoundation.Metadata | t.ConfigMap | Mapping[str, t.ScalarValue] | None
)


def _is_metadata_instance(
    v: _MetadataInput,
) -> TypeGuard[FlextModelFoundation.Metadata]:
    return (
        v is not None
        and hasattr(v, "model_dump")
        and FlextModelFoundation.Metadata in v.__class__.__mro__
    )


def _normalize_metadata(value: _MetadataInput) -> FlextModelFoundation.Metadata:
    if value is None:
        return FlextModelFoundation.Metadata(attributes={})
    if _is_metadata_instance(value):
        return value
    if not FlextRuntime.is_dict_like(value):
        msg = (
            f"metadata must be None, dict, or FlextModelFoundation.Metadata, "
            f"got {value.__class__.__name__}"
        )
        raise TypeError(msg)
    normalized_attrs: dict[str, t.MetadataAttributeValue] = {
        str(key): FlextRuntime.normalize_to_metadata_value(raw_value)
        for key, raw_value in (
            value.root.items() if isinstance(value, t.ConfigMap) else value.items()
        )
    }
    attrs: Mapping[str, t.MetadataAttributeValue] = normalized_attrs
    return FlextModelFoundation.Metadata(attributes=attrs)


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: t.ConfigMap for service tracking.
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        name: str = Field(
            ...,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Service identifier/name",
        )
        # Service instance - uses RegisterableService Protocol union type
        # ARCHITECTURAL NOTE: DI containers accept any registerable service.
        # SkipValidation needed because Protocol types can't be validated by Pydantic.
        # Type safety is enforced at container API level via get_typed().
        service: Annotated[t.RegisterableService, SkipValidation] = Field(
            ...,
            description="Service instance (protocols, models, callables)",
        )
        registration_time: datetime = Field(
            default_factory=FlextRuntime.generate_datetime_utc,
            description="UTC timestamp when service was registered",
        )
        metadata: FlextModelFoundation.Metadata | t.ConfigMap | None = Field(
            default=None,
            description="Additional service metadata (JSON-serializable)",
        )
        service_type: str | None = Field(
            default=None,
            description="Service type name (e.g., 'DatabaseService')",
        )
        tags: list[str] = Field(
            default_factory=list,
            description="Service tags for categorization",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode)."""
            return _normalize_metadata(v)

        @field_validator("service", mode="before")
        @classmethod
        def validate_service_type(cls, v: object) -> object:
            """Validate service is a RegisterableService type.

            RegisterableService includes: str, int, float, bool, datetime, None,
            BaseModel, Path, Sequence, Mapping, callables, and objects with __dict__.
            """
            # Scalars
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            # Models and paths
            if isinstance(v, (BaseModel, Path)):
                return v
            # Callables
            if callable(v):
                return v
            # Collections
            if isinstance(v, Mapping):
                return v
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                return v
            # Objects with __dict__ or protocol-like attributes
            if hasattr(v, "__dict__"):
                return v
            if hasattr(v, "bind") and hasattr(v, "info"):
                return v
            # Reject invalid types
            msg = f"Service must be a RegisterableService type, got {type(v).__name__}"
            raise ValueError(msg)

    class FactoryRegistration(BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: t.ConfigMap for factory tracking.
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        name: str = Field(
            ...,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Factory identifier/name",
        )
        # Factory returns RegisterableService for type-safe factory resolution
        # Supports all registerable types: PayloadValue, protocols, callables
        # SkipValidation needed because Pydantic can't validate callable types
        factory: Annotated[t.FactoryCallable, SkipValidation] = Field(
            ...,
            description="Factory function that creates service instances",
        )
        registration_time: datetime = Field(
            default_factory=FlextRuntime.generate_datetime_utc,
            description="UTC timestamp when factory was registered",
        )
        is_singleton: bool = Field(
            default=False,
            description="Whether factory creates singleton instances",
        )
        cached_instance: t.RegisterableService | None = Field(
            default=None,
            description="Cached singleton instance (if is_singleton=True)",
        )
        metadata: FlextModelFoundation.Metadata | t.ConfigMap | None = Field(
            default=None,
            description="Additional factory metadata (JSON-serializable)",
        )
        invocation_count: int = Field(
            default=c.ZERO,
            ge=c.ZERO,
            description="Number of times factory has been invoked",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode)."""
            return _normalize_metadata(v)

    class ResourceRegistration(BaseModel):
        """Model for lifecycle-managed resource registrations.

        Captures resource factories that dependency-injector should wrap via
        ``providers.Resource`` for connection-style dependencies (DB/HTTP).
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        name: str = Field(
            ...,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Resource identifier/name",
        )
        factory: Annotated[t.ResourceCallable, SkipValidation] = Field(
            ...,
            description="Factory returning the lifecycle-managed resource",
        )
        registration_time: datetime = Field(
            default_factory=FlextRuntime.generate_datetime_utc,
            description="UTC timestamp when resource was registered",
        )
        metadata: FlextModelFoundation.Metadata | t.ConfigMap | None = Field(
            default=None,
            description="Additional resource metadata (JSON-serializable)",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: _MetadataInput) -> FlextModelFoundation.Metadata:
            """Normalize resource metadata to Metadata model."""
            return _normalize_metadata(v)

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: t.ConfigMap for container configuration storage.
        Provides type-safe configuration for DI container behavior.
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
        )

        enable_singleton: bool = Field(
            default=True,
            description="Enable singleton pattern for factories",
        )
        enable_factory_caching: bool = Field(
            default=True,
            description="Enable caching of factory-created instances",
        )
        max_services: int = Field(
            default=c.Container.DEFAULT_MAX_SERVICES,
            ge=c.Reliability.RETRY_COUNT_MIN,
            le=c.Performance.MAX_BATCH_SIZE,
            description="Maximum number of services allowed in registry",
        )
        max_factories: int = Field(
            default=c.Container.DEFAULT_MAX_FACTORIES,
            ge=c.Reliability.RETRY_COUNT_MIN,
            le=c.Container.MAX_FACTORIES,
            description="Maximum number of factories allowed in registry",
        )
        enable_auto_registration: bool = Field(
            default=False,
            description="Enable automatic service registration from decorators",
        )
        enable_lifecycle_hooks: bool = Field(
            default=True,
            description="Enable lifecycle hooks (on_register, on_get, etc.)",
        )
        lazy_loading: bool = Field(
            default=True,
            description="Enable lazy loading of services",
        )

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

        name: str = Field(
            ...,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Name to register this factory under in the container",
        )
        singleton: bool = Field(
            default=False,
            description="Whether factory creates singleton instances",
        )
        lazy: bool = Field(
            default=True,
            description="Whether to defer factory invocation until first use",
        )


__all__ = ["FlextModelsContainer"]

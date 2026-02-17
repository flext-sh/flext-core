"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + _models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from flext_core._models.base import FlextModelsBase
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    # Re-export for external access - use centralized ValidationLevel
    # from FlextConstants
    ValidationLevel = c.Cqrs.ValidationLevel

    class ServiceRegistration(BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: dict[str, t.GeneralValueType] for service tracking.
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
        service: Annotated[object, SkipValidation] = Field(
            ...,
            description="Service instance (protocols, models, callables)",
        )
        registration_time: datetime = Field(
            default_factory=FlextRuntime.generate_datetime_utc,
            description="UTC timestamp when service was registered",
        )
        metadata: FlextModelsBase.Metadata | t.ConfigurationMapping | None = Field(
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
        def validate_metadata(cls, v: t.GeneralValueType) -> FlextModelsBase.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode).

            Accepts: None, dict, or Metadata. Always returns Metadata.
            Uses FlextUtilitiesModel.normalize_to_metadata() for centralized normalization.
            """
            return FlextUtilitiesModel.normalize_to_metadata(v)

    class FactoryRegistration(BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: dict[str, t.GeneralValueType] for factory tracking.
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
        # Supports all registerable types: GeneralValueType, protocols, callables
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
        cached_instance: t.GeneralValueType | BaseModel | None = Field(
            default=None,
            description="Cached singleton instance (if is_singleton=True)",
        )
        metadata: FlextModelsBase.Metadata | t.ConfigurationMapping | None = Field(
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
        def validate_metadata(cls, v: t.GeneralValueType) -> FlextModelsBase.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode).

            Accepts: None, dict, or Metadata. Always returns Metadata.
            Uses FlextUtilitiesModel.normalize_to_metadata() for centralized normalization.
            """
            return FlextUtilitiesModel.normalize_to_metadata(v)

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
        # Factory returns GeneralValueType for type-safe resource resolution
        factory: Callable[[], t.GeneralValueType] = Field(
            ...,
            description="Factory returning the lifecycle-managed resource",
        )
        registration_time: datetime = Field(
            default_factory=FlextRuntime.generate_datetime_utc,
            description="UTC timestamp when resource was registered",
        )
        metadata: FlextModelsBase.Metadata | t.ConfigurationMapping | None = Field(
            default=None,
            description="Additional resource metadata (JSON-serializable)",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: t.GeneralValueType) -> FlextModelsBase.Metadata:
            """Normalize resource metadata to Metadata model."""
            return FlextUtilitiesModel.normalize_to_metadata(v)

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: dict[str, t.GeneralValueType] for container configuration storage.
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
        validation_mode: c.Cqrs.ValidationLevel = Field(
            default=c.Cqrs.ValidationLevel.STRICT,
            description="Validation mode: 'strict' or 'lenient'",
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

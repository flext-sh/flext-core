"""Container models - Dependency Injection registry models.

TIER 0.5: Usa apenas stdlib + pydantic + _models/metadata.py (evita ciclos via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.typings import t
from flext_core.utilities import u


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    # Re-export for external access - use centralized ValidationLevel from FlextConstants
    ValidationLevel = c.Cqrs.ValidationLevel

    class ServiceRegistration(BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: t.Types.ServiceRegistrationDict for service tracking.
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
        # Note: Using 'object' as base type allows arbitrary service instances
        # while maintaining type safety (all classes inherit from object)
        # This is essential for DI container to accept any service type
        service: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType] | object
        ) = Field(
            ...,
            description="Service instance (primitives, BaseModel, callable, or any object)",
        )
        registration_time: datetime = Field(
            default_factory=u.Generators.generate_datetime_utc,
            description="UTC timestamp when service was registered",
        )
        metadata: FlextModelsBase.Metadata | t.Types.ServiceMetadataMapping | None = (
            Field(
                default=None,
                description="Additional service metadata (JSON-serializable)",
            )
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
            return u.Model.normalize_to_metadata(v)

    class FactoryRegistration(BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: t.Types.FactoryRegistrationDict for factory tracking.
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
        factory: Callable[
            [],
            (t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]),
        ] = Field(
            ...,
            description="Factory function that creates service instances",
        )
        registration_time: datetime = Field(
            default_factory=u.Generators.generate_datetime_utc,
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
        metadata: FlextModelsBase.Metadata | t.Types.ServiceMetadataMapping | None = (
            Field(
                default=None,
                description="Additional factory metadata (JSON-serializable)",
            )
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
            return u.Model.normalize_to_metadata(v)

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: t.Types.ConfigurationDict for container configuration storage.
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


__all__ = ["FlextModelsContainer"]

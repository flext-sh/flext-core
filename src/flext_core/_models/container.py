"""Container models - Dependency Injection registry models.

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.metadata import Metadata
from flext_core.runtime import FlextRuntime
from flext_core.utilities import FlextUtilities


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(BaseModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: dict[str, object] for service tracking.
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        name: str = Field(
            ...,
            min_length=1,
            description="Service identifier/name",
        )
        service: object = Field(
            ...,
            description="Service instance (actual object)",
        )
        registration_time: datetime = Field(
            default_factory=FlextUtilities.Generators.generate_datetime_utc,
            description="UTC timestamp when service was registered",
        )
        metadata: Metadata | dict[str, object] | None = Field(
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
        def validate_metadata(cls, v: object) -> Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode).

            Accepts: None, dict, or Metadata. Always returns Metadata.
            Maintains advanced capability to convert dict → Metadata.
            """
            if v is None:
                return Metadata(attributes={})
            if FlextRuntime.is_dict_like(v):
                return Metadata(attributes=v)
            if isinstance(v, Metadata):
                return v
            msg = f"metadata must be None, dict, or Metadata, got {type(v).__name__}"
            raise TypeError(msg)

    class FactoryRegistration(BaseModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: dict[str, object] for factory tracking.
        """

        model_config = ConfigDict(
            frozen=False,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        name: str = Field(
            ...,
            min_length=1,
            description="Factory identifier/name",
        )
        factory: Callable[[], object] = Field(
            ...,
            description="Factory function that creates service instances",
        )
        registration_time: datetime = Field(
            default_factory=FlextUtilities.Generators.generate_datetime_utc,
            description="UTC timestamp when factory was registered",
        )
        is_singleton: bool = Field(
            default=False,
            description="Whether factory creates singleton instances",
        )
        cached_instance: object | None = Field(
            default=None,
            description="Cached singleton instance (if is_singleton=True)",
        )
        metadata: Metadata | dict[str, object] | None = Field(
            default=None,
            description="Additional factory metadata (JSON-serializable)",
        )
        invocation_count: int = Field(
            default=0,
            ge=0,
            description="Number of times factory has been invoked",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: object) -> Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode).

            Accepts: None, dict, or Metadata. Always returns Metadata.
            Maintains advanced capability to convert dict → Metadata.
            """
            if v is None:
                return Metadata(attributes={})
            if FlextRuntime.is_dict_like(v):
                return Metadata(attributes=v)
            if isinstance(v, Metadata):
                return v
            msg = f"metadata must be None, dict, or Metadata, got {type(v).__name__}"
            raise TypeError(msg)

    class ContainerConfig(BaseModel):
        """Model for container configuration.

        Replaces: dict[str, object] for container configuration storage.
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
            default=1000,
            ge=1,
            le=10000,
            description="Maximum number of services allowed in registry",
        )
        max_factories: int = Field(
            default=500,
            ge=1,
            le=5000,
            description="Maximum number of factories allowed in registry",
        )
        validation_mode: str = Field(
            default="strict",
            pattern="^(strict|lenient)$",
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

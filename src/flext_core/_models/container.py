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
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.base import FlextModelsBase
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u

# NOTE: Use uGenerators.generate_datetime_utc() directly - no inline helpers per FLEXT standards


# Tier 0.5 inline enum (defined outside class to avoid forward reference issues)
class _ContainerValidationLevel(StrEnum):
    """Validation level for container operations."""

    STRICT = "strict"
    LENIENT = "lenient"


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    # Re-export for external access
    ValidationLevel = _ContainerValidationLevel

    @staticmethod
    def _is_dict_like(value: t.GeneralValueType) -> bool:
        """Check if value is dict-like."""
        return isinstance(value, Mapping)

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
            Maintains advanced capability to convert dict → Metadata.
            """
            if v is None:
                return FlextModelsBase.Metadata(attributes={})
            if FlextModelsContainer._is_dict_like(v):
                # Use FlextRuntime.normalize_to_metadata_value directly - no wrapper needed
                # Type narrowing: v is Mapping after _is_dict_like check
                if isinstance(v, Mapping):
                    attributes: dict[str, t.MetadataAttributeValue] = {}
                    for key, value in v.items():
                        attributes[str(key)] = FlextRuntime.normalize_to_metadata_value(
                            value
                        )
                    return FlextModelsBase.Metadata(attributes=attributes)
                return FlextModelsBase.Metadata(attributes={})
            if isinstance(v, FlextModelsBase.Metadata):
                return v
            msg = f"metadata must be None, dict, or FlextModelsBase.Metadata, got {type(v).__name__}"
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
            default=0,
            ge=0,
            description="Number of times factory has been invoked",
        )

        @field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata(cls, v: t.GeneralValueType) -> FlextModelsBase.Metadata:
            """Validate and normalize metadata to Metadata (STRICT mode).

            Accepts: None, dict, or Metadata. Always returns Metadata.
            Maintains advanced capability to convert dict → Metadata.
            """
            if v is None:
                return FlextModelsBase.Metadata(attributes={})
            if FlextModelsContainer._is_dict_like(v):
                # Use FlextRuntime.normalize_to_metadata_value directly - no wrapper needed
                # Type narrowing: v is Mapping after _is_dict_like check
                if isinstance(v, Mapping):
                    attributes: dict[str, t.MetadataAttributeValue] = {}
                    for key, value in v.items():
                        attributes[str(key)] = FlextRuntime.normalize_to_metadata_value(
                            value
                        )
                    return FlextModelsBase.Metadata(attributes=attributes)
                return FlextModelsBase.Metadata(attributes={})
            if isinstance(v, FlextModelsBase.Metadata):
                return v
            msg = f"metadata must be None, dict, or FlextModelsBase.Metadata, got {type(v).__name__}"
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
        validation_mode: _ContainerValidationLevel = Field(
            default=_ContainerValidationLevel.STRICT,
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

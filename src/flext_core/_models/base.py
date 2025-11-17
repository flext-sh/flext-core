"""Base utility patterns extracted from FlextModels.

This module contains the FlextModelsBase class with all base utility patterns
as nested classes. It should NOT be imported directly - use FlextModels.Base instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import Field, HttpUrl, computed_field

from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import FlextConstants


class FlextModelsBase:
    """Base utility pattern container class.

    This class acts as a namespace container for base utility patterns.
    All nested classes are accessed via FlextModels.Base.* in the main models.py.
    """

    class Metadata(FlextModelsEntity.FrozenStrictModel):
        """Immutable metadata model."""

        created_by: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        modified_by: str | None = None
        modified_at: datetime | None = None
        tags: list[str] = Field(default_factory=list)
        attributes: dict[str, object] = Field(default_factory=dict)

    class Payload[T](
        FlextModelsEntity.ArbitraryTypesModel,
        FlextModelsEntity.IdentifiableMixin,
        FlextModelsEntity.TimestampableMixin,
    ):
        """Enhanced payload model with computed field.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        data: T = Field(...)  # Required field, no default
        metadata: dict[str, str | int | float] = Field(default_factory=dict)
        expires_at: datetime | None = None
        correlation_id: str | None = None
        source_service: str | None = None
        message_type: str | None = None

        @computed_field
        def is_expired(self) -> bool:
            """Computed property to check if payload is expired."""
            if self.expires_at is None:
                return False
            return datetime.now(UTC) > self.expires_at

    class Url(FlextModelsEntity.Value):
        """Enhanced URL value object using Pydantic v2 HttpUrl validation."""

        url: HttpUrl = Field(description="HTTP/HTTPS URL validated by Pydantic v2")

    class LogOperation(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced log operation model."""

        level: str = Field(default_factory=lambda: "INFO")
        message: str
        context: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        source: str | None = None
        operation: str | None = None
        obj: object | None = None

    class TimestampConfig(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced timestamp configuration."""

        obj: object
        use_utc: bool = Field(default_factory=lambda: True)
        auto_update: bool = Field(default_factory=lambda: True)
        format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
        timezone: str | None = None
        created_at_field: str = Field("created_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        updated_at_field: str = Field("updated_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        field_names: dict[str, str] = Field(
            default_factory=lambda: {
                "created_at": "created_at",
                "updated_at": "updated_at",
            }
        )

    class SerializationRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced serialization request."""

        data: object
        format: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.SerializationFormat.JSON
        )
        encoding: str = Field(default_factory=lambda: "utf-8")
        compression: str | None = None
        pretty_print: bool = False
        use_model_dump: bool = True
        indent: int | None = None
        sort_keys: bool = False
        ensure_ascii: bool = False

    class ConditionalExecutionRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Conditional execution request."""

        condition: Callable[[object], bool]
        true_action: Callable[..., object] | None = (
            None  # Optional for test convenience
        )
        false_action: Callable[..., object] | None = None
        context: dict[str, object] = Field(default_factory=dict)

        @classmethod
        def validate_condition(
            cls, v: Callable[..., object] | None
        ) -> Callable[..., object] | None:
            """Validate callables are properly defined (Pydantic v2 mode='after')."""
            return v

    class StateInitializationRequest(FlextModelsEntity.ArbitraryTypesModel):
        """State initialization request."""

        data: object
        state_key: str
        initial_value: object
        ttl_seconds: int | None = None
        persistence_level: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.PersistenceLevel.MEMORY
        )
        field_name: str = "state"
        state: object


__all__ = ["FlextModelsBase"]

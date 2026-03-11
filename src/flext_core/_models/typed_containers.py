"""Typed container models replacing generic container values.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, field_validator

from flext_core import t
from flext_core._models.entity import FlextModelsEntity


class ConfigValue(FlextModelsEntity.Value):
    """Single configuration value."""

    value: t.Scalar | None = Field(description="The configuration value")
    source: str = Field(default="default", description="Source of the value")


class ConfigMapping(FlextModelsEntity.Value):
    """Structured config mapping."""

    data: Mapping[str, ConfigValue] = Field(
        default_factory=dict,
        description="Mapping of config keys to values",
    )

    @field_validator("data")
    @classmethod
    def _validate_data(
        cls, value: Mapping[str, ConfigValue]
    ) -> Mapping[str, ConfigValue]:
        return value

    def get(self, key: str) -> ConfigValue | None:
        """Get a typed config value by key."""
        return self.data.get(key)

    def to_dict(self) -> dict[str, t.Scalar | None]:
        """Convert typed mapping to scalar dictionary."""
        return {key: item.value for key, item in self.data.items()}


class LogContext(FlextModelsEntity.Value):
    """Structured logging context."""

    correlation_id: str = Field(description="Unique correlation ID for tracing")
    operation: str = Field(description="Operation being performed")
    metadata: ConfigMapping | None = Field(
        default=None,
        description="Additional context metadata",
    )


class HandlerInput(FlextModelsEntity.Value):
    """Generic handler input for typed payload + context."""

    payload: ConfigMapping = Field(description="Input payload data")
    context: LogContext | None = Field(default=None, description="Execution context")


class HandlerOutput(FlextModelsEntity.Value):
    """Generic handler output with result or error."""

    success: bool = Field(description="Whether operation succeeded")
    data: ConfigMapping | None = Field(
        default=None, description="Output data on success"
    )
    error: str | None = Field(default=None, description="Error message on failure")


class FlextModelsTypedContainers:
    """Namespace container for typed container models."""

    ConfigValue = ConfigValue
    ConfigMapping = ConfigMapping
    LogContext = LogContext
    HandlerInput = HandlerInput
    HandlerOutput = HandlerOutput


__all__ = [
    "ConfigMapping",
    "ConfigValue",
    "FlextModelsTypedContainers",
    "HandlerInput",
    "HandlerOutput",
    "LogContext",
]

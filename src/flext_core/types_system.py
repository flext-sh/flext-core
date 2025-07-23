"""Type System - Validated type definitions with Pydantic V2.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a comprehensive type system for the FLEXT ecosystem
using Pydantic V2 for runtime validation and Python 3.13 type features.

Design Principles:
- Single Responsibility: Each type has one clear purpose
- Open/Closed: Types are extensible through composition
- Liskov Substitution: Type aliases maintain substitutability
- Interface Segregation: Minimal, focused type definitions
- Dependency Inversion: Abstract types, no concrete dependencies

KISS: Simple, clear type definitions with validation
DRY: Single source of truth for all FLEXT type definitions
"""

from __future__ import annotations

import re
from typing import Annotated
from typing import NewType
from uuid import uuid4

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


def flext_validate_non_empty_string(value: str) -> str:
    """Validate that a string is non-empty after stripping whitespace."""
    stripped = value.strip()
    if not stripped:
        msg = "Cannot be empty or whitespace-only"
        raise ValueError(msg)

    return stripped


def flext_validate_identifier(value: str) -> str:
    """Validate string is valid identifier (alphanumeric, dash, underscore)."""
    value = flext_validate_non_empty_string(value)

    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        msg = "Must contain only letters, numbers, underscores, and dashes"
        raise ValueError(msg)

    max_identifier_length = 255
    if len(value) > max_identifier_length:
        msg = "Must be 255 characters or less"
        raise ValueError(msg)

    return value


def flext_validate_service_name(value: str) -> str:
    """Validate service names for dependency injection."""
    value = flext_validate_identifier(value)

    # Additional service name rules
    if value.startswith("-") or value.endswith("-"):
        msg = "Cannot start or end with a dash"
        raise ValueError(msg)

    if "--" in value or "__" in value:
        msg = "Cannot contain consecutive dashes or underscores"
        raise ValueError(msg)

    return value


def flext_validate_config_key(value: str) -> str:
    """Validate configuration keys (dot notation allowed)."""
    value = flext_validate_non_empty_string(value)

    if not re.match(r"^[a-zA-Z0-9_.-]+$", value):
        msg = "Must contain only letters, numbers, underscores, dots, dashes"
        raise ValueError(msg)

    if value.startswith(".") or value.endswith("."):
        msg = "Cannot start or end with a dot"
        raise ValueError(msg)

    if ".." in value:
        msg = "Cannot contain consecutive dots"
        raise ValueError(msg)

    return value


def flext_validate_event_type(value: str) -> str:
    """Validate event type names (reverse domain notation)."""
    value = flext_validate_non_empty_string(value)

    if not re.match(r"^[a-zA-Z0-9_.-]+$", value):
        msg = "Must contain only letters, numbers, underscores, dots, dashes"
        raise ValueError(msg)

    parts = value.split(".")
    min_parts_for_event_type = 2
    if len(parts) < min_parts_for_event_type:
        msg = "Must contain at least one dot (e.g., 'domain.event')"
        raise ValueError(msg)

    for part in parts:
        if not part:
            msg = "Cannot have empty parts between dots"
            raise ValueError(msg)

    return value


# Core validated types using Pydantic's Annotated system
FlextEntityId = Annotated[
    str,
    AfterValidator(flext_validate_identifier),
    Field(
        description="Unique identifier for domain entities",
        examples=["user-123", "order_456", "product-abc"],
        min_length=1,
        max_length=255,
    ),
]

FlextServiceName = Annotated[
    str,
    AfterValidator(flext_validate_service_name),
    Field(
        description="Service identifier for dependency injection",
        examples=["database", "user-service", "email_sender"],
        min_length=1,
        max_length=100,
    ),
]

FlextConfigKey = Annotated[
    str,
    AfterValidator(flext_validate_config_key),
    Field(
        description="Configuration key in dot notation",
        examples=["database.host", "api.timeout", "logging.level"],
        min_length=1,
        max_length=200,
    ),
]

FlextEventType = Annotated[
    str,
    AfterValidator(flext_validate_event_type),
    Field(
        description="Event type in reverse domain notation",
        examples=["user.created", "order.completed", "system.started"],
        min_length=3,
        max_length=150,
    ),
]

# Less constrained types for data payloads
FlextResourceId = NewType("FlextResourceId", str)
FlextContextData = NewType("FlextContextData", str)
FlextTraceId = NewType("FlextTraceId", str)


# FlextPayload is now available from flext_core.payload
# This eliminates duplication and follows Single Responsibility
# Principle


class FlextIdentifier(BaseModel):
    """Strongly-typed identifier with validation and generation.

    This class provides a robust way to handle identifiers throughout
    the system with automatic validation and UUID generation support.

    Features:
        - Automatic UUID generation if not provided
        - Validation of custom identifiers
        - Immutable after creation
        - String representation for easy usage
        - Type-safe operations

    Examples:
        Auto-generated UUID:
        >>> id1 = FlextIdentifier()
        >>> assert len(str(id1)) == 36  # UUID length

        Custom identifier:
        >>> id2 = FlextIdentifier(value="user-123")
        >>> assert str(id2) == "user-123"

        Validation:
        >>> FlextIdentifier(value="")  # Raises ValidationError

    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    value: str = Field(
        default_factory=lambda: str(uuid4()),
        description="The identifier value",
        min_length=1,
        max_length=255,
    )

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Validate identifier value is non-empty."""
        if not v.strip():
            msg = "Identifier cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def __str__(self) -> str:
        """Return string representation of the identifier."""
        return self.value

    def __hash__(self) -> int:
        """Hash based on identifier value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Equality based on identifier value."""
        if isinstance(other, FlextIdentifier):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False


class FlextTypedDict(BaseModel):
    """Type-safe dictionary with validation.

    Provides a validated dictionary container for structured data
    that needs type safety and validation at runtime.

    Examples:
        >>> data = FlextTypedDict(name="Alice", age=30, active=True)
        >>> assert data.get("name") == "Alice"
        >>> assert data.has("age")

    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        validate_assignment=True,
    )

    def get(self, key: str, default: object = None) -> object:
        """Get a value by key with optional default."""
        return getattr(self, key, default)

    def has(self, key: str) -> bool:
        """Check if key exists in the dictionary."""
        return hasattr(self, key)

    def keys(self) -> list[str]:
        """Get all keys in the dictionary."""
        extra = self.__pydantic_extra__
        if extra is None:
            return []
        return list(extra.keys())

    def items(self) -> list[tuple[str, object]]:
        """Get all key-value pairs."""
        extra = self.__pydantic_extra__
        if extra is None:
            return []
        return list(extra.items())


# Export all types for library usage
__all__ = [
    "FlextConfigKey",
    "FlextContextData",
    # Validated string types
    "FlextEntityId",
    "FlextEventType",
    "FlextIdentifier",
    # Less constrained types
    "FlextResourceId",
    "FlextServiceName",
    "FlextTraceId",
    "FlextTypedDict",
    "flext_validate_config_key",
    "flext_validate_event_type",
    "flext_validate_identifier",
    # Validation functions
    "flext_validate_non_empty_string",
    "flext_validate_service_name",
]

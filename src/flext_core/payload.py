"""FlextPayload - Validated payload container.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Validated payload container for structured data transfer.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class FlextPayload(BaseModel):
    """Validated payload container for structured data transfer.

    This class provides a type-safe way to handle data payloads
    throughout the FLEXT ecosystem with automatic validation and
    serialization.

    Features:
        - Strict validation of all fields
        - Automatic serialization/deserialization
        - Immutable after creation (frozen)
        - Type-safe field access
        - JSON schema generation

    Examples:
        Basic usage:
        >>> payload = FlextPayload(
        ...     user_id="123", action="login", timestamp="2025-01-01T00:00:00Z"
        ... )
        >>> assert payload.user_id == "123"

        With nested data:
        >>> data = {"name": "Alice", "age": 30}
        >>> payload = FlextPayload(user_data=data, event_type="user.updated")

        Validation:
        >>> payload = FlextPayload()  # Empty payload is valid
        >>> payload = FlextPayload(key="value", count=42, active=True)

    """

    model_config = ConfigDict(
        # Allow any additional fields (payloads are flexible)
        extra="allow",
        # Frozen for immutability
        frozen=True,
        # Validate on assignment
        validate_assignment=True,
        # String processing
        str_strip_whitespace=True,
        # JSON schema generation
        json_schema_extra={
            "description": "Flexible payload container for structured data",
            "examples": [
                {"user_id": "123", "action": "login"},
                {
                    "data": {"key": "value"},
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            ],
        },
    )

    def __getattr__(self, name: str) -> object:
        """Allow dynamic attribute access for payload fields."""
        extra = self.__pydantic_extra__
        if extra is not None and name in extra:
            return extra[name]
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def has(self, field_name: str) -> bool:
        """Check if a field exists in the payload.

        Args:
            field_name: Name of the field to check

        Returns:
            True if field exists, False otherwise

        Examples:
            >>> payload = FlextPayload(user_id="123")
            >>> assert payload.has("user_id")
            >>> assert not payload.has("missing_field")

        """
        extra = self.__pydantic_extra__
        return extra is not None and field_name in extra

    def get(self, field_name: str, default: object = None) -> object:
        """Get field value with optional default.

        Args:
            field_name: Name of the field to get
            default: Default value if field doesn't exist

        Returns:
            Field value or default

        Examples:
            >>> payload = FlextPayload(user_id="123")
            >>> assert payload.get("user_id") == "123"
            >>> assert payload.get("missing", "default") == "default"

        """
        extra = self.__pydantic_extra__
        if extra is None:
            return default
        return extra.get(field_name, default)

    def keys(self) -> list[str]:
        """Get all field names in the payload.

        Returns:
            List of field names

        Examples:
            >>> payload = FlextPayload(user_id="123", action="login")
            >>> assert "user_id" in payload.keys()
            >>> assert "action" in payload.keys()

        """
        extra = self.__pydantic_extra__
        if extra is None:
            return []
        return list(extra.keys())

    def items(self) -> list[tuple[str, object]]:
        """Get all field name-value pairs.

        Returns:
            List of (name, value) tuples

        Examples:
            >>> payload = FlextPayload(user_id="123", action="login")
            >>> items = payload.items()
            >>> assert ("user_id", "123") in items

        """
        extra = self.__pydantic_extra__
        if extra is None:
            return []
        return list(extra.items())

    def __contains__(self, field_name: str) -> bool:
        """Support 'in' operator for field existence check.

        Args:
            field_name: Name of the field to check

        Returns:
            True if field exists, False otherwise

        Examples:
            >>> payload = FlextPayload(user_id="123")
            >>> assert "user_id" in payload
            >>> assert "missing" not in payload

        """
        return self.has(field_name)


__all__ = ["FlextPayload"]

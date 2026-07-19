"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase


if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26): FlextTypesServices imports this
    # protocol while t is still being composed; t is annotation-only here.
    from datetime import datetime
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core import FlextTypes as t

from .flextprotocolslogging_part_01 import (
    FlextProtocolsLogging as FlextProtocolsLoggingPart01,
)


class FlextProtocolsLogging(FlextProtocolsLoggingPart01):
    @runtime_checkable
    class Metadata(FlextProtocolsBase.BaseModel, Protocol):
        """Metadata protocol."""

        @property
        def attributes(self) -> t.JsonMapping:
            """Metadata attributes."""
            ...

        @property
        def created_at(self) -> datetime:
            """Creation timestamp."""
            ...

        @property
        def updated_at(self) -> datetime:
            """Update timestamp."""
            ...

        @property
        def version(self) -> str:
            """Version string."""
            ...

    @runtime_checkable
    class Connection(FlextProtocolsBase.Base, Protocol):
        """External system connection protocol."""

        def close_connection(self) -> None:
            """Close connection."""
            ...

        @property
        def connection_string(self) -> str:
            """Connection string."""
            ...

        def test_connection(self) -> FlextProtocolsResult.Result[bool]:
            """Test connection."""
            ...

    @runtime_checkable
    class ValidatorSpec(FlextProtocolsBase.Base, Protocol):
        """Protocol for validator specifications with operator composition.

        Validators implement __call__ to validate values and support composition
        via __and__ (both must pass), __or__ (either passes), and __invert__ (negation).

        Example:
            validator = V.string.non_empty & V.string.max_length(100)
            is_valid = validator("hello")  # True

        """

        def __call__(self, value: t.JsonValue) -> bool:
            """Validate value, return True if valid."""
            ...

        def __and__(
            self, other: FlextProtocolsLogging.ValidatorSpec
        ) -> FlextProtocolsLogging.ValidatorSpec:
            """Compose with AND - both validators must pass."""
            ...

        def __invert__(self) -> FlextProtocolsLogging.ValidatorSpec:
            """Negate validator - passes when original fails."""
            ...

        def __or__(
            self, other: FlextProtocolsLogging.ValidatorSpec
        ) -> FlextProtocolsLogging.ValidatorSpec:
            """Compose with OR - at least one validator must pass."""
            ...

    @runtime_checkable
    class Entry(FlextProtocolsBase.Base, Protocol):
        """Entry protocol (read-only)."""

        @property
        def attributes(self) -> t.MappingKV[str, t.StrSequence]:
            """Entry attributes as immutable mapping."""
            ...

        @property
        def dn(self) -> str:
            """Distinguished name."""
            ...

        def add_attribute(self, name: str, values: t.StrSequence) -> Self:
            """Add attribute values, returning self for chaining."""
            ...

        def remove_attribute(self, name: str) -> Self:
            """Remove attribute, returning self for chaining."""
            ...

        def update_attribute(self, name: str, values: t.StrSequence) -> Self:
            """Update attribute values, returning self for chaining."""
            ...

        def to_dict(self) -> t.ScalarMapping:
            """Convert to dictionary representation."""
            ...

        def to_ldif(self) -> str:
            """Convert to LDIF format."""
            ...


__all__: list[str] = ["FlextProtocolsLogging"]

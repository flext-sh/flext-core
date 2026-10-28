"""FlextProtocolsLoggingPart02 - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols import FlextProtocolsBase
from flext_core._protocols._logging.flextprotocolslogging_part_01 import (
    FlextProtocolsLoggingPart01,
)

if TYPE_CHECKING:
    from datetime import datetime

    from flext_core import t
    from flext_core._protocols import FlextProtocolsResult


class FlextProtocolsLoggingPart02(FlextProtocolsLoggingPart01):
    @runtime_checkable
    class Metadata(Protocol):
        """Metadata protocol."""

        @property
        def attributes(self) -> t.JsonMapping:
            """JSON attributes validated by the canonical metadata model."""
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

        @classmethod
        def model_validate(
            cls,
            obj: Mapping[
                str,
                t.Scalar
                | Mapping[str, t.Scalar | Sequence[t.Scalar]]
                | Sequence[t.Scalar]
                | None,
            ]
            | Self,
            *,
            strict: bool | None = None,
            from_attributes: bool | None = None,
            context: Mapping[str, t.Scalar] | None = None,
        ) -> Self:
            """Validate and create metadata from input data."""
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
            self,
            other: FlextProtocolsLoggingPart02.ValidatorSpec,
        ) -> FlextProtocolsLoggingPart02.ValidatorSpec:
            """Compose with AND - both validators must pass."""
            ...

        def __invert__(self) -> FlextProtocolsLoggingPart02.ValidatorSpec:
            """Negate validator - passes when original fails."""
            ...

        def __or__(
            self,
            other: FlextProtocolsLoggingPart02.ValidatorSpec,
        ) -> FlextProtocolsLoggingPart02.ValidatorSpec:
            """Compose with OR - at least one validator must pass."""
            ...

    @runtime_checkable
    class Entry(FlextProtocolsBase.Base, Protocol):
        """Entry protocol (read-only)."""

        @property
        def attributes(self) -> Mapping[str, Sequence[str]]:
            """Entry attributes as immutable mapping."""
            ...

        @property
        def dn(self) -> str:
            """Distinguished name."""
            ...

        def add_attribute(self, name: str, values: Sequence[str]) -> Self:
            """Add attribute values, returning self for chaining."""
            ...

        def remove_attribute(self, name: str) -> Self:
            """Remove attribute, returning self for chaining."""
            ...

        def update_attribute(self, name: str, values: Sequence[str]) -> Self:
            """Update attribute values, returning self for chaining."""
            ...

        def to_dict(self) -> Mapping[str, t.Scalar]:
            """Convert to dictionary representation."""
            ...

        def to_ldif(self) -> str:
            """Convert to LDIF format."""
            ...


__all__: list[str] = ["FlextProtocolsLoggingPart02"]

"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import BaseModel
from structlog.typing import BindableLogger

from flext_core import FlextProtocolsBase, FlextProtocolsResult, t

if TYPE_CHECKING:
    from flext_core import r


class FlextProtocolsLogging:
    """Protocols for logging, connection, validation, and entries."""

    @runtime_checkable
    class Logger(BindableLogger, Protocol):
        """Protocol for structlog logger with all logging methods.

        Extends BindableLogger to add explicit method signatures for
        logging methods (debug, info, warning, error, etc.) that are
        available via __getattr__ at runtime.
        """

        def critical(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log critical message."""
            ...

        def debug(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log debug message."""
            ...

        def error(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log error message."""
            ...

        def exception(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log exception with traceback."""
            ...

        def info(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log info message."""
            ...

        def warning(
            self,
            msg: str,
            *args: t.RuntimeData,
            **kw: t.RuntimeData | Exception,
        ) -> r[bool] | None:
            """Log warning message."""
            ...

    @runtime_checkable
    class Metadata(Protocol):
        """Metadata protocol."""

        @property
        def attributes(self) -> t.ConfigMap:
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

        def get_connection_string(self) -> str:
            """Get connection string."""
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

        def __call__(self, value: t.Container) -> bool:
            """Validate value, return True if valid."""
            ...

        def __and__(
            self,
            other: FlextProtocolsLogging.ValidatorSpec,
        ) -> FlextProtocolsLogging.ValidatorSpec:
            """Compose with AND - both validators must pass."""
            ...

        def __invert__(self) -> FlextProtocolsLogging.ValidatorSpec:
            """Negate validator - passes when original fails."""
            ...

        def __or__(
            self,
            other: FlextProtocolsLogging.ValidatorSpec,
        ) -> FlextProtocolsLogging.ValidatorSpec:
            """Compose with OR - at least one validator must pass."""
            ...

    @runtime_checkable
    class Entry(FlextProtocolsBase.Base, Protocol):
        """Entry protocol (read-only)."""

        @property
        def attributes(self) -> Mapping[str, t.StrSequence]:
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

        def set_attribute(self, name: str, values: t.StrSequence) -> Self:
            """Set attribute values, returning self for chaining."""
            ...

        def to_dict(self) -> t.ScalarMapping:
            """Convert to dictionary representation."""
            ...

        def to_ldif(self) -> str:
            """Convert to LDIF format."""
            ...

    @runtime_checkable
    class TextStream(Protocol):
        """Protocol for text-based output streams (stdout, stderr, file handles)."""

        mode: str
        name: str
        encoding: str

        def write(self, msg: str) -> int: ...

        def flush(self) -> None: ...

    type AccessibleData = (
        t.ConfigMap
        | Mapping[str, t.ValueOrModel]
        | t.NormalizedValue
        | BaseModel
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsLogging.ValidatorSpec
    )


__all__ = ["FlextProtocolsLogging"]

"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from flext_core import FlextTypes as t


class FlextProtocolsLogging:
    """Protocols for logging, connection, validation, and entries."""

    @runtime_checkable
    class Logger(Protocol):
        """Protocol for structlog logger with all logging methods.

        Extends BindableLogger to add explicit method signatures for
        logging methods (debug, info, warning, error, etc.) that are
        available via __getattr__ at runtime.
        """

        @property
        def name(self) -> str:
            """Logger name exposed by the public adapter."""
            ...

        def bind(self, **new_values: t.JsonPayload) -> Self:
            """Bind context and return a logger preserving the public protocol."""
            ...

        def new(self, **new_values: t.JsonPayload) -> Self:
            """Replace bound context and return a logger preserving the protocol."""
            ...

        def unbind(self, *keys: str, safe: bool = False) -> Self:
            """Remove bound keys and optionally ignore missing values."""
            ...

        def try_unbind(self, *keys: str) -> Self:
            """Remove bound keys while ignoring missing values."""
            ...

        def build_exception_context(
            self,
            *,
            exception: Exception | None,
            exc_info: bool,
            context: t.MappingKV[str, t.JsonPayload | Exception],
        ) -> t.JsonMapping:
            """Build normalized structured exception context."""
            ...

        def critical(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log critical message."""
            ...

        def debug(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log debug message."""
            ...

        def error(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log error message."""
            ...

        def exception(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log exception with traceback."""
            ...

        def info(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log info message."""
            ...

        def log(
            self,
            level: str,
            message: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log a message at an arbitrary level."""
            ...

        def trace(
            self,
            message: str,
            *args: t.LogValue,
            **kwargs: t.JsonPayload,
        ) -> t.LogResult:
            """Log a trace/debug-level diagnostic message."""
            ...

        def warning(
            self,
            msg: str,
            *args: t.LogValue,
            **kw: t.LogValue,
        ) -> t.LogResult:
            """Log warning message."""
            ...

    @runtime_checkable
    class HasLogger(Protocol):
        """Protocol for values that expose a canonical logger attribute."""

        logger: FlextProtocolsLogging.Logger

    @runtime_checkable
    class OutputLogger(Protocol):
        """Protocol for raw structlog wrapped loggers returned by logger factories."""

        def critical(self, message: str) -> None: ...

        def debug(self, message: str) -> None: ...

        def error(self, message: str) -> None: ...

        def exception(self, message: str) -> None: ...

        def info(self, message: str) -> None: ...

        def msg(self, message: str) -> None: ...

        def warn(self, message: str) -> None: ...

        def warning(self, message: str) -> None: ...


__all__: list[str] = ["FlextProtocolsLogging"]

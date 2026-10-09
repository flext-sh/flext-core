"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from structlog.types import BindableLogger

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsLogging:
    """Protocols for logging, connection, validation, and entries."""

    @runtime_checkable
    class Logger(BindableLogger, Protocol):
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

        def debug(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
            """Log debug message."""
            ...

        def error(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
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

        def info(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
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

        def warning(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
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

    @runtime_checkable
    class LoggingStage(Protocol):
        """Transform one typed event before the terminal renderer runs."""

        def __call__(
            self,
            logger: FlextProtocolsLogging.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
            """Return the next event; stage exceptions propagate unchanged."""
            ...

    @runtime_checkable
    class StructlogOptions(Protocol):
        """Declared options consumed by the logging composition owner."""

        @property
        def log_level(self) -> int | None: ...

        @property
        def console_renderer(self) -> bool: ...

        @property
        def processing_stages(
            self,
        ) -> t.SequenceOf[FlextProtocolsLogging.LoggingStage]: ...

        @property
        def wrapper_class_factory(self) -> t.LoggerWrapperFactory | None: ...

        @property
        def logger_factory(self) -> t.LoggerFactory: ...

        @property
        def cache_logger_on_first_use(self) -> bool: ...

        @property
        def async_logging(self) -> bool: ...


__all__: list[str] = ["FlextProtocolsLogging"]

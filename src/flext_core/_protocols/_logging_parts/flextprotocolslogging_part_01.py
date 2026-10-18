"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from structlog.types import BindableLogger

from flext_core._protocols import FlextProtocolsSettings

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsLogging:
    """Protocols for logging, connection, validation, and entries."""

    class AsyncLogJob(Protocol):
        """Immutable work submitted to the native asynchronous stream."""

        @property
        def text(self) -> str | None: ...

        @property
        def completion(self) -> t.LoggingWriteCompletion: ...

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
            exception: BaseException | None,
            exc_info: bool,
            context: t.MappingKV[str, t.JsonPayload | Exception],
        ) -> t.LoggingEvent:
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
            **kwargs: t.LogValue,
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
    class OutputLoggerFactory(Protocol):
        """Construct an output logger from native positional logger names."""

        def __call__(self, *args: str) -> FlextProtocolsLogging.OutputLogger: ...

    @runtime_checkable
    class NativeLoggingWrapper(Protocol):
        """Native construction, context binding, and level emission."""

        def __init__(
            self,
            logger: FlextProtocolsLogging.OutputLogger,
            processors: t.SequenceOf[t.LoggingProcessor],
            context: t.LoggingContext,
        ) -> None: ...

        @property
        def _context(self) -> t.LoggingContext: ...

        def bind(self, **new_values: t.LoggingEventValue) -> Self: ...

        def new(self, **new_values: t.LoggingEventValue) -> Self: ...

        def unbind(self, *keys: str) -> Self: ...

        def try_unbind(self, *keys: str) -> Self: ...

        def critical(self, event: str, **context: t.LoggingEventValue) -> None: ...

        def debug(self, event: str, **context: t.LoggingEventValue) -> None: ...

        def error(self, event: str, **context: t.LoggingEventValue) -> None: ...

        def exception(self, event: str, **context: t.LoggingEventValue) -> None: ...

        def info(self, event: str, **context: t.LoggingEventValue) -> None: ...

        def warning(self, event: str, **context: t.LoggingEventValue) -> None: ...

    @runtime_checkable
    class StructlogOptions(Protocol):
        """Declared options consumed by the logging composition owner."""

        @property
        def log_level(self) -> int | None: ...

        @property
        def console_renderer(self) -> bool | None: ...

        @property
        def processing_stages(
            self,
        ) -> t.SequenceOf[FlextProtocolsLogging.LoggingStage]: ...

        @property
        def observers(self) -> t.SequenceOf[FlextProtocolsLogging.LoggingObserver]: ...

        @property
        def handlers(self) -> t.SequenceOf[FlextProtocolsLogging.LoggingHandler]: ...

        @property
        def redacted_keys(self) -> t.SequenceOf[str]: ...

        @property
        def redacted_values(self) -> t.SequenceOf[str]: ...

        @property
        def wrapper_class_factory(self) -> t.LoggerWrapperFactory | None: ...

        @property
        def logger_factory(self) -> t.LoggerFactory: ...

        @property
        def cache_logger_on_first_use(self) -> bool: ...

        @property
        def async_logging(self) -> bool | None: ...

    @runtime_checkable
    class LoggingSettings(FlextProtocolsSettings.Settings, Protocol):
        """Actual application settings elected by the Core composition root."""

        @property
        def debug(self) -> bool: ...

        @property
        def trace(self) -> bool: ...

        @property
        def async_logging(self) -> bool: ...

        @property
        def logging_profile(self) -> str: ...

        @property
        def model_fields_set(self) -> set[str]: ...

    class LoggingRecord(Protocol):
        """Sanitized event snapshot, independent of destination presentation."""

        @property
        def message(self) -> str: ...

        @property
        def level(self) -> str: ...

        @property
        def context(self) -> t.JsonMapping: ...

        @property
        def exception(self) -> BaseException | None: ...

    @runtime_checkable
    class LoggingObserver(Protocol):
        """Receive one sanitized snapshot before native output fanout."""

        def __call__(self, event: FlextProtocolsLogging.LoggingRecord) -> None: ...

    @runtime_checkable
    class LoggingHandler(Protocol):
        """Native handler contract used by configuration and stdlib fanout."""

        @property
        def level(self) -> int: ...

        @property
        def formatter(self) -> t.LoggingNativeFormatter | None: ...

        def handle(
            self, record: t.LoggingNativeRecord
        ) -> bool | t.LoggingNativeRecord: ...

        def handleError(self, record: t.LoggingNativeRecord) -> None: ...

        def setFormatter(self, fmt: t.LoggingNativeFormatter | None) -> None: ...

        def setLevel(self, level: int | str) -> None: ...

        def flush(self) -> None: ...

        def close(self) -> None: ...


__all__: list[str] = ["FlextProtocolsLogging"]

"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from structlog.types import BindableLogger
from structlog.typing import Processor

from flext_core._protocols import FlextProtocolsSettings
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._typings import FlextTypesServices

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsLogging:
    """Protocols for logging, connection, validation, and entries."""

    type LogArgument = FlextTypesServices.LogArgument[FlextProtocolsBase.Model]
    type LogValue = FlextTypesServices.LogValue[FlextProtocolsBase.Model]
    type LoggingEventValue = FlextTypesServices.LoggingEventValue[
        FlextProtocolsBase.Model
    ]
    type LoggingEvent = FlextTypesServices.LoggingEvent[FlextProtocolsBase.Model]
    type LoggingContext = FlextTypesServices.LoggingContext[FlextProtocolsBase.Model]
    type LogResult = FlextProtocolsResult.Result[bool]

    @runtime_checkable
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
        ) -> FlextProtocolsLogging.LoggingEvent:
            """Build normalized structured exception context."""
            ...

        def critical(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log critical message."""
            ...

        def debug(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log debug message."""
            ...

        def error(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log error message."""
            ...

        def exception(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log exception with traceback."""
            ...

        def info(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log info message."""
            ...

        def log(
            self,
            level: str,
            message: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log a message at an arbitrary level."""
            ...

        def trace(
            self,
            message: str,
            *args: FlextProtocolsLogging.LogValue,
            **kwargs: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log a trace/debug-level diagnostic message."""
            ...

        def warning(
            self,
            msg: str,
            *args: FlextProtocolsLogging.LogValue,
            **kw: FlextProtocolsLogging.LogValue,
        ) -> FlextProtocolsLogging.LogResult:
            """Log warning message."""
            ...

    @runtime_checkable
    class HasLogger(Protocol):
        """Protocol for values that expose a canonical logger attribute."""

        logger: FlextProtocolsLogging.Logger

    @runtime_checkable
    class OutputLogger(Protocol):
        """Raw wrapped loggers consumed with positional messages by structlog."""

        def critical(self, message: str, /) -> None: ...

        def debug(self, message: str, /) -> None: ...

        def error(self, message: str, /) -> None: ...

        def exception(self, message: str, /) -> None: ...

        def info(self, message: str, /) -> None: ...

        def warn(self, message: str, /) -> None: ...

        def warning(self, message: str, /) -> None: ...

    @runtime_checkable
    class LoggingStage(Protocol):
        """Transform one typed event before the terminal renderer runs."""

        def __call__(
            self,
            logger: FlextProtocolsLogging.OutputLogger,
            method_name: str,
            event_dict: FlextProtocolsLogging.LoggingEvent,
        ) -> FlextProtocolsLogging.LoggingEvent:
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
            processors: t.SequenceOf[Processor],
            context: FlextProtocolsLogging.LoggingContext,
        ) -> None: ...

        @property
        def _context(self) -> FlextProtocolsLogging.LoggingContext: ...

        def bind(
            self, **new_values: FlextProtocolsLogging.LoggingEventValue
        ) -> Self: ...

        def new(
            self, **new_values: FlextProtocolsLogging.LoggingEventValue
        ) -> Self: ...

        def unbind(self, *keys: str) -> Self: ...

        def try_unbind(self, *keys: str) -> Self: ...

        def critical(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

        def debug(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

        def error(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

        def exception(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

        def info(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

        def warning(
            self, event: str, **context: FlextProtocolsLogging.LoggingEventValue
        ) -> None: ...

    @runtime_checkable
    class LoggerWrapperFactory(Protocol):
        """Construct the native logging wrapper class at runtime composition."""

        def __call__(self) -> type[FlextProtocolsLogging.NativeLoggingWrapper]: ...

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
        def wrapper_class_factory(
            self,
        ) -> FlextProtocolsLogging.LoggerWrapperFactory | None: ...

        @property
        def logger_factory(
            self,
        ) -> FlextProtocolsLogging.OutputLoggerFactory | None: ...

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

    @runtime_checkable
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

        def __call__(self, event: FlextProtocolsLogging.LoggingRecord, /) -> None: ...

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

        def flush(self) -> None: ...

        def close(self) -> None: ...


__all__: list[str] = ["FlextProtocolsLogging"]

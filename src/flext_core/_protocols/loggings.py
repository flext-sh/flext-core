"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from structlog.types import BindableLogger
from structlog.typing import Processor

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsResult,
    FlextProtocolsSettings,
)

if TYPE_CHECKING:
    from datetime import datetime


class FlextProtocolsLogging:
    """Public facade for FlextProtocolsLogging."""

    type LogArgument[T] = t.LogArgument[T]
    type LogValue = t.LogValue[FlextProtocolsBase.Model]
    type LoggingEventValue = t.LoggingEventValue[FlextProtocolsBase.Model]
    type LoggingEvent = t.LoggingEvent[FlextProtocolsBase.Model]
    type LoggingContext = t.LoggingContext[FlextProtocolsBase.Model]
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
            context: Mapping[str, t.JsonPayload | Exception],
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
            *args: t.LogPositionalValue[FlextProtocolsBase.Model],
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
            processors: Sequence[Processor],
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

        def __call__(
            self,
        ) -> type[FlextProtocolsLogging.NativeLoggingWrapper]: ...

    @runtime_checkable
    class StructlogOptions(Protocol):
        """Declared options consumed by the logging composition owner."""

        log_level: int | None

        console_renderer: bool | None

        processing_stages: Sequence[FlextProtocolsLogging.LoggingStage]
        observers: Sequence[FlextProtocolsLogging.LoggingObserver]
        handlers: Sequence[FlextProtocolsLogging.LoggingHandler]
        redacted_keys: Sequence[str]

        redacted_values: Sequence[str]

        wrapper_class_factory: FlextProtocolsLogging.LoggerWrapperFactory | None
        logger_factory: FlextProtocolsLogging.OutputLoggerFactory | None
        cache_logger_on_first_use: bool

        async_logging: bool | None

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

        @property
        def created_by(self) -> str | None:
            """Identifier of the actor that originally created this metadata."""
            ...

        @property
        def modified_by(self) -> str | None:
            """Identifier of the actor that last modified this metadata."""
            ...

        @property
        def tags(self) -> Sequence[str]:
            """Normalized labels used to classify and filter this metadata."""
            ...

        @property
        def metadata_value(self) -> t.Scalar | None:
            """Scalar metadata value."""
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

    @runtime_checkable
    class TextStream(Protocol):
        """Protocol for text-based output streams (stdout, stderr, file handles)."""

        mode: str
        name: str
        encoding: str

        def write(self, msg: str) -> int: ...

        def flush(self) -> None: ...

    type AccessibleData = (
        t.JsonPayload
        | FlextProtocolsBase.Model
        | Mapping[str, t.JsonPayload | FlextProtocolsBase.Model | None]
        | FlextProtocolsResult.HasModelDump
        | ValidatorSpec
    )


__all__: list[str] = ["FlextProtocolsLogging"]

"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import traceback
from typing import ClassVar, Self, TYPE_CHECKING

import structlog

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextProtocols as p,
    FlextTypes as t,
)
from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.logging import FlextConstantsLogging as cl
from flext_core._exceptions.factories import FlextExceptionsFactories as ef
from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug
from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext as ulc
from flext_core import FlextResult as r


if TYPE_CHECKING:
    from structlog.typing import Context
    import types


# NOTE (multi-agent): mro-i6nq.12 — consolidated _loggings_parts/part_01..05
# into this single facade module.
class FlextUtilitiesLogging(ulc):
    """Context-aware utility logger tuned for dispatcher-centric CQRS flows.

    Composed via MRO from:
    - FlextUtilitiesLoggingConfig — structlog configuration, async writer, processors
    - ulc — context binding, value normalization, source paths
    """

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    _level_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    _structlog_instance: p.Logger | None = None

    _structlog_configured: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        *,
        settings: p.Settings | None = None,
        _bound_logger: p.Logger | None = None,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> None:
        """Initialize FlextUtilitiesLogging with name and optional context."""
        super().__init__()
        if not name:
            msg = "logger name is required"
            raise ValueError(msg)
        resolved_name = name
        self.name = resolved_name
        if _bound_logger is not None:
            self._structlog_instance = _bound_logger
            return
        resolved_context: t.MutableJsonMapping = {}
        if context is not None:
            resolved_context = dict(
                FlextUtilitiesLogging.to_container_context({
                    key: value for key, value in context.items() if value is not None
                })
            )
        if settings is not None:
            service_name = getattr(settings, c.ContextKey.SERVICE_NAME, None)
            service_version = getattr(settings, c.ContextKey.SERVICE_VERSION, None)
            correlation_id = getattr(settings, c.ContextKey.CORRELATION_ID, None)
            if isinstance(service_name, str) and service_name:
                resolved_context[c.ContextKey.SERVICE_NAME] = service_name
            if isinstance(service_version, str) and service_version:
                resolved_context[c.ContextKey.SERVICE_VERSION] = service_version
            if isinstance(correlation_id, str) and correlation_id:
                resolved_context[c.ContextKey.CORRELATION_ID] = correlation_id
        base_logger = type(self).resolve_bound_logger(resolved_name)
        self._structlog_instance = (
            base_logger.bind(
                **FlextUtilitiesLogging._to_scalar_context(resolved_context)
            )
            if resolved_context
            else base_logger
        )

    def __call__(self) -> Self:
        """Return self to support factory-style DI registration."""
        return self

    @property
    def _context(self) -> Context:
        """Context mapping for BindableLogger protocol compliance."""
        return {}

    @property
    def logger(self) -> p.Logger:
        """Wrapped structlog logger instance."""
        instance = self._structlog_instance
        if instance is None:
            instance = type(self).resolve_bound_logger(getattr(self, "name", __name__))
            self._structlog_instance = instance
        return instance

    @classmethod
    def resolve_bound_logger(cls, name: str) -> p.Logger:
        """Fetch the underlying bound structlog logger for internal use."""
        cls.ensure_structlog_configured()
        if not name:
            msg = "logger name is required"
            raise ValueError(msg)
        logger: p.Logger = structlog.get_logger(name)
        return logger

    def bind(self, **context: t.JsonPayload) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**self.to_container_context(context))
        return self.__class__(self.name, _bound_logger=bound_logger)

    def new(self, **context: t.JsonPayload) -> Self:
        """Create new logger with context — implements BindableLogger protocol."""
        return self.bind(**context)

    def _exception_context_from_inputs(
        self,
        resolved_exception: Exception | None,
        raw_exception: t.LogValue | None,
        exc_info_value: t.LogValue,
        extra_context: t.MappingKV[str, t.LogValue],
    ) -> t.JsonDict:
        include_stack_trace = self._should_include_stack_trace()
        context_dict: t.JsonDict = {}
        if resolved_exception is not None:
            context_dict["exception_type"] = resolved_exception.__class__.__name__
            context_dict["exception_message"] = str(resolved_exception)
            if include_stack_trace:
                context_dict["stack_trace"] = "".join(
                    traceback.format_exception(
                        resolved_exception.__class__,
                        resolved_exception,
                        resolved_exception.__traceback__,
                    )
                )
        elif exc_info_value and include_stack_trace:
            context_dict["stack_trace"] = traceback.format_exc()
        for key, value in extra_context.items():
            if key in {"exception", "exc_info"}:
                continue
            if not isinstance(value, BaseException):
                context_dict[key] = FlextUtilitiesLogging._to_container_value(value)
        if resolved_exception is None and isinstance(raw_exception, BaseException):
            context_dict["exception_type"] = raw_exception.__class__.__name__
            context_dict["exception_message"] = str(raw_exception)
        return context_dict

    def exception(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log exception with conditional stack trace (DEBUG only)."""
        message = msg
        filtered_args: tuple[t.JsonValue, ...] = tuple(
            FlextUtilitiesLogging._to_container_value(arg)
            for arg in args
            if not isinstance(arg, BaseException)
        )
        try:
            resolved_exception: Exception | None = (
                args[0] if args and isinstance(args[0], Exception) else None
            )
            context_dict = self._exception_context_from_inputs(
                resolved_exception, kw.get("exception"), kw.get("exc_info", True), kw
            )
            _ = self.logger.error(
                message,
                *filtered_args,
                **FlextUtilitiesLogging._to_scalar_context(context_dict),
            )
            return r[bool].ok(True)
        except c.EXC_BROAD_RUNTIME as exc:
            FlextUtilitiesLogging._report_internal_logging_failure("exception", exc)
            return e.fail_operation("exception logging", exc)

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: t.MappingKV[str, t.JsonPayload | Exception],
    ) -> t.JsonMapping:
        """Build normalized structured exception context for logging."""
        result: t.JsonDict = {
            k: str(v)
            if isinstance(v, Exception)
            else FlextUtilitiesLogging._to_container_value(v)
            for k, v in context.items()
        }
        if exception is not None:
            result["exception_type"] = exception.__class__.__name__
            result["exception_message"] = str(exception)
        if exc_info and self._should_include_stack_trace():
            result["stack_trace"] = traceback.format_exc()
        return result

    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Unbind keys from logger — implements BindableLogger protocol."""
        bound_logger = (
            self.logger.try_unbind(*keys) if safe else self.logger.unbind(*keys)
        )
        return self.__class__(self.name, _bound_logger=bound_logger)

    def try_unbind(self, *keys: str) -> Self:
        """Unbind keys while ignoring missing values."""
        return self.unbind(*keys, safe=True)

    def warning(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log warning message."""
        return self._log_standard_level(cl.LogLevel.WARNING, msg, *args, **kw)

    @staticmethod
    def _resolve_level_name(level: cl.LogLevel | str) -> str:
        match level:
            case cl.LogLevel() as enum_level:
                level_raw: str = enum_level.value
            case _:
                level_raw = level
        return level_raw.lower()

    @staticmethod
    def _resolve_log_context(
        args: t.SequenceOf[t.LogValue], context: t.MappingKV[str, t.LogValue]
    ) -> t.JsonMapping:
        resolved_context: dict[str, t.LogValue] = dict(context)
        if "source" not in resolved_context and (
            source_path := FlextUtilitiesLogging._caller_source_path()
        ):
            resolved_context["source"] = source_path
        for idx, arg in enumerate(args):
            resolved_context[f"arg_{idx}"] = arg
        return FlextUtilitiesLogging._to_scalar_context(resolved_context)

    def _log(
        self,
        level: cl.LogLevel | str,
        event: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Consolidate all log level methods into one internal logging path."""
        try:
            level_str = FlextUtilitiesLogging._resolve_level_name(level)
            scalar_context = FlextUtilitiesLogging._resolve_log_context(args, context)
            getattr(self.logger, level_str)(event, **scalar_context)
            return r[bool].ok(True)
        except ce.EXC_BROAD_RUNTIME as exc:
            return ef.fail_operation("logging", exc)

    def _log_standard_level(
        self, level: cl.LogLevel, msg: str, *args: t.LogValue, **kw: t.LogValue
    ) -> t.LogResult:
        return self._log(level, msg, *args, **kw)

    def critical(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log critical message."""
        return self._log_standard_level(cl.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log debug message."""
        return self._log_standard_level(cl.LogLevel.DEBUG, msg, *args, **kw)

    def error(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log error message."""
        return self._log_standard_level(cl.LogLevel.ERROR, msg, *args, **kw)

    def info(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Log info message."""
        return self._log_standard_level(cl.LogLevel.INFO, msg, *args, **kw)

    def log(
        self, level: str, message: str, *args: t.LogValue, **context: t.LogValue
    ) -> t.LogResult:
        """Log message with specified level."""
        level_enum: cl.LogLevel = cl.LogLevel(level.upper())
        converted_args: tuple[t.JsonValue, ...] = tuple(
            FlextUtilitiesLogging._to_container_value(arg) for arg in args
        )
        return self._log(level_enum, message, *converted_args, **context)

    def trace(
        self, message: str, *args: t.LogValue, **kwargs: t.JsonPayload
    ) -> t.LogResult:
        """Log trace message."""
        try:
            try:
                formatted_message = message % args if args else message
            except ce.EXC_TYPE_VALIDATION:
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(
                formatted_message, **FlextUtilitiesLogging._to_scalar_context(kwargs)
            )
            return r[bool].ok(True)
        except ce.EXC_BROAD_RUNTIME as exc:
            FlextUtilitiesLogging._report_internal_logging_failure("trace", exc)
            return ef.fail_operation("trace logging", exc)

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start tracking."""
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Log timing; the context-manager traceback is intentionally unused."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = "success" if success else "failed"
            context: mc.ConfigMap = mc.ConfigMap(
                root={
                    c.MetadataKey.DURATION_SECONDS: elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                }
            )
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            if success:
                _ = self.logger.info(
                    f"{self._operation_name} {status}",
                    **FlextUtilitiesLogging.to_container_context(context.root),
                )
            else:
                _ = self.logger.error(
                    f"{self._operation_name} {status}",
                    **FlextUtilitiesLogging.to_container_context(context.root),
                )

    class Integration:
        """Application-layer integration helpers using structlog directly."""

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Set up complete service infrastructure."""
            sl = FlextUtilitiesLogging.structlog()
            _ = sl.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = sl.contextvars.bind_contextvars(service_version=service_version)
            if enable_context_correlation:
                correlation_id = f"flext-{ug.generate_id().replace('-', '')[:12]}"
                _ = sl.contextvars.bind_contextvars(correlation_id=correlation_id)
            sl.fetch_logger(__name__).info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: mc.ConfigMap | None = None,
        ) -> None:
            """Track domain event with context correlation."""
            sl = FlextUtilitiesLogging.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            sl.fetch_logger(__name__).info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

        @staticmethod
        def track_service_resolution(
            service_name: str,
            *,
            resolved: bool = True,
            error_message: str | None = None,
        ) -> None:
            """Track service resolution with context correlation."""
            sl = FlextUtilitiesLogging.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            logger = sl.fetch_logger(__name__)
            if resolved:
                logger.info(
                    "Service resolved",
                    service_name=service_name,
                    correlation_id=correlation_id,
                )
            else:
                logger.error(
                    "Service resolution failed",
                    service_name=service_name,
                    error=error_message,
                    correlation_id=correlation_id,
                )

    @classmethod
    def fetch_logger(cls, name: str) -> p.Logger:
        """Fetch the canonical public logger wrapper."""
        return cls.create_module_logger(name)

    @classmethod
    def create_module_logger(
        cls, name: str, *, context: t.MappingKV[str, t.JsonPayload | None] | None = None
    ) -> p.Logger:
        """Create a logger instance for a module."""
        cls.ensure_structlog_configured()
        merged_context: t.MutableJsonMapping = {}
        if context is not None:
            merged_context.update(
                cls.to_container_context({
                    key: value for key, value in context.items() if value is not None
                })
            )
        logger: p.Logger = cls(name, context=merged_context)
        return logger


__all__: t.StrSequence = ("FlextUtilitiesLogging",)

"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import time
import traceback
import types
from collections.abc import Mapping, MutableMapping
from contextlib import suppress
from typing import ClassVar, Self

import structlog
from structlog.typing import Context

from flext_core import (
    FlextRuntime,
    FlextUtilitiesGenerators,
    c,
    m,
    p,
    r,
    t,
)
from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext


class FlextLogger(FlextUtilitiesLoggingContext):
    """Context-aware logger tuned for dispatcher-centric CQRS flows.

    Composed via MRO from:
    - FlextUtilitiesLoggingConfig — structlog configuration, async writer, processors
    - FlextUtilitiesLoggingContext — context binding, value normalization, source paths
    """

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry] = {}
    _level_contexts: ClassVar[t.ScopedContainerRegistry] = {}
    _structlog_instance: p.Logger | None = None
    _structlog_configured: ClassVar[bool] = False

    def __init__(
        self,
        name: str | None = None,
        *,
        settings: p.Settings | None = None,
        _bound_logger: p.Logger | None = None,
        _level: c.LogLevel | str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with name and optional context."""
        super().__init__()
        resolved_name = name or type(self).__name__
        self.name = resolved_name
        if _bound_logger is not None:
            self._structlog_instance = _bound_logger
            return
        if settings is not None:
            _level = getattr(settings, "level", _level)
            _service_name = getattr(settings, c.ContextKey.SERVICE_NAME, _service_name)
            _service_version = getattr(
                settings,
                c.ContextKey.SERVICE_VERSION,
                _service_version,
            )
            _correlation_id = getattr(
                settings,
                c.ContextKey.CORRELATION_ID,
                _correlation_id,
            )
            _force_new = getattr(settings, "force_new", _force_new)
        context: t.MutableStrMapping = {}
        if _service_name:
            context[c.ContextKey.SERVICE_NAME] = _service_name
        if _service_version:
            context[c.ContextKey.SERVICE_VERSION] = _service_version
        if _correlation_id:
            context[c.ContextKey.CORRELATION_ID] = _correlation_id
        base_logger = type(self).resolve_bound_logger(resolved_name)
        self._structlog_instance = (
            base_logger.bind(**context) if context else base_logger
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
            instance = type(self).resolve_bound_logger(
                getattr(self, "name", __name__),
            )
            self._structlog_instance = instance
        return instance

    @classmethod
    def resolve_bound_logger(cls, name: str | None = None) -> p.Logger:
        """Fetch the underlying bound structlog logger for internal use."""
        cls.ensure_structlog_configured()
        if name is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                name = __name__
        logger: p.Logger = structlog.get_logger(name)
        return logger

    @classmethod
    def fetch_logger(cls, name: str | None = None) -> p.Logger:
        """Fetch the canonical public logger wrapper."""
        resolved_name = name
        if resolved_name is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                resolved_name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                resolved_name = __name__
        return cls.create_module_logger(resolved_name)

    @classmethod
    def create_bound_logger(cls, name: str, bound_logger: p.Logger) -> Self:
        """Internal factory for creating logger with pre-bound structlog instance."""
        return cls(name, _bound_logger=bound_logger)

    @classmethod
    def create_module_logger(
        cls,
        name: str = "flext",
        *,
        settings: p.Settings | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
    ) -> p.Logger:
        """Create a logger instance for a module."""
        cls.ensure_structlog_configured()
        return cls(
            name,
            settings=settings,
            _service_name=service_name,
            _service_version=service_version,
            _correlation_id=correlation_id,
        )

    @classmethod
    def for_container(
        cls,
        container: p.Container,
        level: str | None = None,
        **context: t.RuntimeData,
    ) -> Self:
        """Create logger configured for a specific container."""
        if level is None:
            settings: p.Settings | None
            try:
                settings = container.settings
            except (AttributeError, RuntimeError, TypeError, ValueError):
                settings = None
            level = getattr(settings, "log_level", "INFO")
        logger = cls(f"container_{id(container)}")
        if context:
            _ = logger.bind_global_context(**context)
        return logger

    def bind(self, **context: t.RuntimeData) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**self._to_container_context(context))
        return self.__class__.create_bound_logger(self.name, bound_logger)

    @classmethod
    def to_container_context(
        cls,
        context: Mapping[str, t.LogValue | t.Container | t.ValueOrModel],
    ) -> t.FlatContainerMapping:
        """Public wrapper for context normalization used outside helper mixins."""
        return cls._to_container_context(context)

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: Mapping[str, t.RuntimeData | Exception],
    ) -> t.FlatContainerMapping:
        """Build normalized context payload for exception/error logging."""
        include_stack_trace = self._should_include_stack_trace()
        context_dict: dict[str, t.Container] = {}
        if exception is not None:
            context_dict["exception_type"] = exception.__class__.__name__
            context_dict["exception_message"] = str(exception)
            if include_stack_trace:
                context_dict["stack_trace"] = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    ),
                )
        elif exc_info and include_stack_trace:
            context_dict["stack_trace"] = traceback.format_exc()
        for key, value in context.items():
            if not isinstance(value, BaseException):
                context_dict[key] = FlextRuntime.to_plain_container(
                    FlextRuntime.normalize_to_container(value),
                )
        return context_dict

    def critical(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log critical message."""
        return self._log_standard_level(c.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log debug message."""
        return self._log_standard_level(c.LogLevel.DEBUG, msg, *args, **kw)

    def error(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log error message."""
        return self._log_standard_level(c.LogLevel.ERROR, msg, *args, **kw)

    def exception(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log exception with conditional stack trace (DEBUG only)."""
        message = str(msg)
        filtered_args: tuple[t.Scalar, ...] = tuple(
            FlextLogger._to_scalar_value(arg)
            for arg in args
            if not isinstance(arg, BaseException)
        )
        try:
            resolved_exception: Exception | None = (
                args[0] if args and isinstance(args[0], Exception) else None
            )
            raw_exception = kw.get("exception")
            exc_info_value = kw.get("exc_info", True)
            context_input: MutableMapping[str, t.Scalar | Exception] = {}
            for key, value in kw.items():
                if key in {"exception", "exc_info"}:
                    continue
                if isinstance(value, Exception):
                    context_input[key] = value
                else:
                    context_input[key] = FlextLogger._to_scalar_value(value)
            built_context = self.build_exception_context(
                exception=resolved_exception,
                exc_info=bool(exc_info_value),
                context=context_input,
            )
            context_dict: dict[str, t.Container] = dict(built_context)
            if resolved_exception is None and isinstance(raw_exception, BaseException):
                context_dict["exception_type"] = raw_exception.__class__.__name__
                context_dict["exception_message"] = str(raw_exception)
            _ = self.logger.error(
                message,
                *filtered_args,
                **FlextLogger._to_scalar_context(context_dict),
            )
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            FlextLogger._report_internal_logging_failure("exception", exc)
            return self._fail_logging_operation("exception logging", exc)

    def info(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log info message."""
        return self._log_standard_level(c.LogLevel.INFO, msg, *args, **kw)

    def log(
        self,
        level: str,
        message: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Log message with specified level."""
        level_enum: c.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = c.LogLevel(level.upper())
        converted_args: tuple[t.Container, ...] = tuple(
            FlextLogger._to_scalar_value(arg) for arg in args
        )
        return self._log(level_enum, message, *converted_args, **context)

    def new(self, **context: t.RuntimeData) -> Self:
        """Create new logger with context — implements BindableLogger protocol."""
        return self.bind(**context)

    def trace(
        self,
        message: str,
        *args: t.LogValue,
        **kwargs: t.RuntimeData,
    ) -> t.LogResult:
        """Log trace message."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(
                formatted_message,
                **FlextLogger._to_scalar_context(kwargs),
            )
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            FlextLogger._report_internal_logging_failure("trace", exc)
            return self._fail_logging_operation("trace logging", exc)

    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Unbind keys from logger — implements BindableLogger protocol."""
        if safe:
            with suppress(KeyError, ValueError, AttributeError):
                bound_logger = self.logger.unbind(*keys)
                return self.__class__.create_bound_logger(self.name, bound_logger)
            return self
        bound_logger = self.logger.unbind(*keys)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    def warning(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log warning message."""
        return self._log_standard_level(c.LogLevel.WARNING, msg, *args, **kw)

    def _log(
        self,
        _level: c.LogLevel | str,
        event: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Internal logging method — consolidates all log level methods."""
        try:
            if "source" not in context and (
                source_path := FlextLogger._caller_source_path()
            ):
                context["source"] = source_path
            for idx, arg in enumerate(args):
                context[f"arg_{idx}"] = arg
            match _level:
                case c.LogLevel() as enum_level:
                    level_raw: str = enum_level.value
                case _:
                    level_raw = str(_level)
            level_str = level_raw.lower()
            scalar_context = FlextLogger._to_scalar_context(context)
            getattr(self.logger, level_str)(event, **scalar_context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return self._fail_logging_operation("logging", exc)

    def _log_standard_level(
        self,
        level: c.LogLevel,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        return self._log(level, msg, *args, **kw)

    class Integration:
        """Application-layer integration helpers using structlog directly."""

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Setup complete service infrastructure."""
            sl = FlextLogger.structlog()
            _ = sl.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = sl.contextvars.bind_contextvars(
                    service_version=service_version,
                )
            if enable_context_correlation:
                correlation_id = f"flext-{FlextUtilitiesGenerators.generate_id().replace('-', '')[:12]}"
                _ = sl.contextvars.bind_contextvars(
                    correlation_id=correlation_id,
                )
            sl.get_logger(__name__).info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: m.ConfigMap | None = None,
        ) -> None:
            """Track domain event with context correlation."""
            sl = FlextLogger.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            sl.get_logger(__name__).info(
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
            sl = FlextLogger.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            logger = sl.get_logger(__name__)
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
            exc_tb: types.TracebackType | None,
        ) -> None:
            """Log operation result with timing."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = "success" if success else "failed"
            context: m.ConfigMap = m.ConfigMap(
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
                    **FlextLogger.to_container_context(context.root),
                )
            else:
                _ = self.logger.error(
                    f"{self._operation_name} {status}",
                    **FlextLogger.to_container_context(context.root),
                )


__all__: t.StrSequence = ["FlextLogger"]

"""Bound-logger instance family for the FLEXT logging facade.

``FlextBoundLogger`` carries the stateful per-name logger protocol
(``bind``/``new``/level methods) that ``u.fetch_logger`` hands out; the
``FlextUtilitiesLogging`` facade keeps the stateless class-level surface so
the Utilities-layer method-shape census holds.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from typing import Self

import structlog

from flext_core import c, p, r, t
from flext_core._utilities import FlextUtilitiesLoggingContext


class FlextBoundLogger(FlextUtilitiesLoggingContext):
    """Context-aware bound logger instance handed out by the logging facade.

    Composed via MRO from ``FlextUtilitiesLoggingContext`` (context binding,
    value normalization, source paths); each instance owns one logger name and
    its structlog binding.
    """

    def __init__(
        self,
        name: str,
        *,
        settings: p.Settings | None = None,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> None:
        """Initialize the bound logger with name and optional context.

        Raises:
            ValueError: If logger name is required.

        """
        super().__init__()
        if not name:
            msg = "logger name is required"
            raise ValueError(msg)
        resolved_name = name
        self.name = resolved_name
        resolved_context: t.MutableJsonMapping = {}
        if context is not None:
            resolved_context = dict(
                self.to_container_context({
                    key: value for key, value in context.items() if value is not None
                }),
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
        resolved_context.setdefault(c.ContextKey.LOGGER_NAME, name)
        self._native = structlog.wrap_logger(
            None,
            logger_factory_args=(name,),
            cache_logger_on_first_use=False,
            **resolved_context,
        )

    def __call__(self) -> Self:
        """Return this logger for native factory-style DI registration."""
        return self

    @property
    def _context(self) -> t.LoggingContext:
        """Context mapping for BindableLogger protocol compliance."""
        return structlog.get_context(self._native)

    @property
    def logger(self) -> p.NativeLoggingWrapper:
        """Native lazy composition, including the latest factory and processors."""
        self.ensure_structlog_configured()
        return self._native

    def bind(self, **context: t.JsonPayload) -> Self:
        """Return a new logger with additional native bound context."""
        bound_logger = self.logger.bind(**context)
        return self.__class__(self.name, context=structlog.get_context(bound_logger))

    def new(self, **context: t.JsonPayload) -> Self:
        """Replace bound context without mutating this logger."""
        return self.__class__(self.name, context=context)

    def exception(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit native exception context and preserve selected failures."""
        if args and isinstance(args[0], BaseException):
            kw["exception"], args = args[0], args[1:]
        kw.setdefault("exc_info", True)
        return self._log(c.LogLevel.ERROR, msg, *args, **kw)

    def build_exception_context(
        self,
        *,
        exception: BaseException | None,
        exc_info: bool,
        context: t.MappingKV[str, t.JsonPayload | Exception],
    ) -> t.LoggingEvent:
        """Adapt causal data to native exc_info without formatting a stack."""
        result: t.LoggingEvent = {
            k: self._to_container_value(v) for k, v in context.items()
        }
        if exception is not None:
            result["exception_type"] = exception.__class__.__name__
            result["exception_message"] = str(exception)
        if exception is not None or exc_info:
            result["exc_info"] = exception if exception is not None else exc_info
        return result

    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Return a native binding without the selected keys."""
        bound_logger = (
            self.logger.try_unbind(*keys) if safe else self.logger.unbind(*keys)
        )
        return self.__class__(self.name, context=structlog.get_context(bound_logger))

    def try_unbind(self, *keys: str) -> Self:
        """Unbind keys while ignoring missing values."""
        return self.unbind(*keys, safe=True)

    def warning(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit WARNING through the native pipeline."""
        return self._log(c.LogLevel.WARNING, msg, *args, **kw)

    def _log(
        self,
        level: t.LogLevel | str,
        event: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Emit once, preserving original business identity and saved traceback."""
        exception = context.get("exception")
        exc_info = context.get("exc_info", False)
        business_info = self.resolve_logging_exception_info(True)
        try:
            selected_info = self.resolve_logging_exception_info(
                exception if isinstance(exception, BaseException) else exc_info
            )
            if selected_info is not None:
                business_info = selected_info
            level_str = str(logging.getLevelName(self.level_number(level))).lower()
            native_context: t.LoggingEvent = dict(
                self.to_container_context({
                    key: value
                    for key, value in context.items()
                    if key not in {"exc_info", "exception"}
                })
            )
            if selected_info is not None:
                native_context["exc_info"] = selected_info
            if args:
                native_context["positional_args"] = args
            getattr(self.logger, level_str)(event, **native_context)
        except BaseException as logging_error:
            if business_info is not None and business_info[1] is not logging_error:
                business_error = business_info[1]
                business_error.add_note(
                    f"Logging also failed: {type(logging_error).__name__}"
                )
                raise business_error.with_traceback(business_info[2])
            raise
        return r[bool].ok(value=True)

    def critical(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit CRITICAL through the native pipeline."""
        return self._log(c.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit DEBUG through the native pipeline."""
        return self._log(c.LogLevel.DEBUG, msg, *args, **kw)

    def error(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit ERROR through the native pipeline."""
        return self._log(c.LogLevel.ERROR, msg, *args, **kw)

    def info(self, msg: str, *args: t.LogValue, **kw: t.LogValue) -> t.LogResult:
        """Emit ordinary INFO through the native pipeline."""
        return self._log(c.LogLevel.INFO, msg, *args, **kw)

    def log(
        self,
        level: str,
        message: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Emit at the requested standard severity."""
        return self._log(level, message, *args, **context)

    def trace(
        self,
        message: str,
        *args: t.LogValue,
        **kwargs: t.LogValue,
    ) -> t.LogResult:
        """Emit opt-in high-volume diagnostics at DEBUG."""
        if self.logging_settings().trace:
            return self._log(c.LogLevel.DEBUG, message, *args, **kwargs)
        return r[bool].ok(value=True)

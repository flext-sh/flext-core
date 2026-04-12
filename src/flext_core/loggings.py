"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import atexit
import inspect
import io
import logging
import queue
import sys
import threading
import time
import traceback
import types
import typing
import warnings
from collections.abc import (
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import suppress
from pathlib import Path
from typing import ClassVar, Self, override

import structlog
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level
from structlog.types import Processor
from structlog.typing import Context

from flext_core import c, e, p, r, t
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core.runtime import FlextRuntime


class FlextLogger(FlextRuntime):
    """Context-aware logger tuned for dispatcher-centric CQRS flows.

    FlextLogger layers structured logging on ``structlog`` with scoped contexts,
    dependency-injector factories, performance tracking helpers, and adapters for
    ``r`` so command/query handlers emit consistent telemetry without
    bespoke wrappers.
    """

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry] = {}
    _level_contexts: ClassVar[t.ScopedContainerRegistry] = {}
    _structlog_instance: p.Logger | None = None
    _structlog_configured: ClassVar[bool] = False
    type _LogArg = t.RuntimeData | Exception

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread."""

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._stream_mode: str = str(getattr(stream, "mode", "w"))
            self._stream_name: str = str(getattr(stream, "name", "<async-log-writer>"))
            self._stream_encoding: str = str(
                getattr(stream, "encoding", c.DEFAULT_ENCODING),
            )
            self._stream_errors: str | None = getattr(stream, "errors", None)
            self._stream_newlines: str | tuple[str, ...] | None = getattr(
                stream,
                "newlines",
                None,
            )
            self.queue: queue.Queue[str | None] = queue.Queue(
                maxsize=c.MAX_ITEMS,
            )
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="flext-async-log-writer",
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)
            self._writer_logger: p.Logger | None = None

        @property
        def _writer_log(self) -> p.Logger:
            """Logger for async log writer."""
            logger = getattr(self, "_writer_logger", None)
            if logger is None:
                logger = structlog.get_logger(__name__)
                self._writer_logger = logger
            return logger

        @property
        def mode(self) -> str:
            """Expose text stream mode for TextIO compatibility."""
            return self._stream_mode

        @property
        def name(self) -> str:
            """Expose text stream name for TextIO compatibility."""
            return self._stream_name

        @property
        def buffer(self) -> typing.BinaryIO:
            """Return underlying binary buffer."""
            buf = getattr(self.stream, "buffer", None)
            if buf is not None:
                return buf
            return io.BytesIO()

        @property
        def line_buffering(self) -> bool:
            """Return whether line buffering is enabled."""
            return bool(getattr(self.stream, "line_buffering", False))

        @override
        def flush(self) -> None:
            """Flush stream (best effort)."""
            flush_fn = getattr(self.stream, "flush", None)
            if flush_fn is None or not callable(flush_fn):
                return
            try:
                flush_fn()
            except (OSError, ValueError, TypeError, AttributeError):
                return

        def shutdown(self) -> None:
            """Stop worker thread and flush remaining messages."""
            if self.stop_event.is_set():
                return
            self.stop_event.set()
            with suppress(Exception):
                self.queue.put_nowait(None)
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
            self.flush()

        @override
        def write(self, s: str, /) -> int:
            """Write message to queue (non-blocking)."""
            with suppress(Exception):
                self.queue.put(s, block=c.ASYNC_BLOCK_ON_FULL)
            return len(s)

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get(timeout=0.1)
                    if msg is None:
                        break
                    _ = self.stream.write(msg)
                    _ = self.stream.flush()
                    self.queue.task_done()
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                except (OSError, ValueError, TypeError) as exc:
                    self._writer_log.warning(
                        "Async log writer stream operation failed",
                        exc_info=exc,
                    )
                    with suppress(OSError, ValueError, TypeError):
                        _ = self.stream.write("Error in async log writer\n")

    _async_writer: ClassVar[_AsyncLogWriter | None] = None

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if not cls._structlog_configured:
            cls.configure_structlog()
            cls._structlog_configured = True

    @classmethod
    def structlog_configured(cls) -> bool:
        """Check if structlog has been configured."""
        return cls._structlog_configured

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
            instance = type(self).resolve_bound_logger(getattr(self, "name", __name__))
            self._structlog_instance = instance
        return instance

    @staticmethod
    def _structlog_processor(
        value: typing.Callable[..., t.ValueOrModel] | t.Container | None,
    ) -> typing.TypeIs[Processor]:
        return callable(value)

    @staticmethod
    def structlog() -> types.ModuleType:
        """Return the imported structlog module."""
        return structlog

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
    def fetch_logger(cls, name: str | None = None) -> Self:
        """Fetch the canonical public logger wrapper using shared FLEXT configuration."""
        resolved_name = name
        if resolved_name is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                resolved_name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                resolved_name = __name__
        return cls.create_module_logger(resolved_name)

    @staticmethod
    def _resolve_structlog_params(
        settings: t.ModelCarrier | None,
        *,
        log_level: int | None,
        console_renderer: bool,
        additional_processors: Sequence[t.StructlogProcessor] | None,
        wrapper_class_factory: t.LoggerWrapperFactory | None,
        logger_factory: t.LoggerFactory,
        cache_logger_on_first_use: bool,
    ) -> tuple[
        int,
        bool,
        Sequence[t.StructlogProcessor] | None,
        t.LoggerWrapperFactory | None,
        t.LoggerFactory,
        bool,
        bool,
    ]:
        """Extract structlog params from settings model or pass-through args."""
        async_logging = True
        if settings is not None:
            log_level = getattr(settings, "log_level", log_level)
            console_renderer = getattr(settings, "console_renderer", console_renderer)
            cfg_processors = getattr(settings, "additional_processors", None)
            if cfg_processors:
                additional_processors = cfg_processors
            wrapper_class_factory = getattr(
                settings,
                "wrapper_class_factory",
                wrapper_class_factory,
            )
            logger_factory = getattr(settings, "logger_factory", logger_factory)
            cache_logger_on_first_use = getattr(
                settings,
                "cache_logger_on_first_use",
                cache_logger_on_first_use,
            )
            async_logging = getattr(settings, "async_logging", True)
        level = log_level if log_level is not None else logging.INFO
        return (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        )

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        additional_processors: Sequence[t.StructlogProcessor] | None,
    ) -> list[Processor]:
        """Assemble the structlog processor chain."""
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
        ]
        if additional_processors:
            processors.extend(
                proc for proc in additional_processors if cls._structlog_processor(proc)
            )
        if console_renderer:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(JSONRenderer())
        return processors

    @classmethod
    def _resolve_logger_factory(
        cls,
        *,
        logger_factory: t.LoggerFactory,
        async_logging: bool,
    ) -> t.LoggerFactory | None:
        """Resolve the logger factory, enabling async output when requested."""
        if logger_factory is not None:
            return logger_factory
        if async_logging:
            with suppress(AttributeError):
                print_logger_factory = structlog.PrintLoggerFactory
                if callable(print_logger_factory):
                    return cls._build_async_logger_factory(print_logger_factory)
            with suppress(AttributeError):
                write_logger_factory = structlog.WriteLoggerFactory
                if callable(write_logger_factory):
                    return cls._build_async_logger_factory(write_logger_factory)
        return None

    @classmethod
    def _build_async_logger_factory(
        cls,
        factory_builder: typing.Callable[..., t.LoggerFactory],
    ) -> t.LoggerFactory:
        """Build a structlog logger factory bound to the shared async writer."""
        if cls._async_writer is None:
            cls._async_writer = cls._AsyncLogWriter(sys.stdout)
        return factory_builder(file=cls._async_writer)

    @classmethod
    def configure_structlog(
        cls,
        *,
        settings: t.ModelCarrier | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[t.StructlogProcessor] | None = None,
        wrapper_class_factory: t.LoggerWrapperFactory | None = None,
        logger_factory: t.LoggerFactory = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults."""
        if cls._structlog_configured:
            return
        (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        ) = cls._resolve_structlog_params(
            settings,
            log_level=log_level,
            console_renderer=console_renderer,
            additional_processors=additional_processors,
            wrapper_class_factory=wrapper_class_factory,
            logger_factory=logger_factory,
            cache_logger_on_first_use=cache_logger_on_first_use,
        )
        processors = cls._build_structlog_processors(
            console_renderer=console_renderer,
            additional_processors=additional_processors,
        )
        wrapper_arg = (
            wrapper_class_factory()
            if wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(level)
        )
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory,
            async_logging=async_logging,
        )
        configure_fn = getattr(structlog, "configure", None)
        if configure_fn is not None and callable(configure_fn):
            _ = configure_fn(
                processors=processors,
                wrapper_class=wrapper_arg,
                logger_factory=factory_to_use if callable(factory_to_use) else None,
                cache_logger_on_first_use=cache_logger_on_first_use,
            )
        cls._structlog_configured = True

    @classmethod
    def reconfigure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[t.StructlogProcessor] | None = None,
    ) -> None:
        """Force reconfigure structlog (ignores is_configured checks)."""
        structlog.reset_defaults()
        if cls._async_writer is not None:
            cls._async_writer.shutdown()
            cls._async_writer = None
        cls._structlog_configured = False
        cls.configure_structlog(
            log_level=log_level,
            console_renderer=console_renderer,
            additional_processors=additional_processors,
        )

    @classmethod
    def reset_structlog_state_for_testing(cls) -> None:
        """Reset structlog configuration state for testing purposes."""
        structlog.reset_defaults()
        cls._structlog_configured = False

    @staticmethod
    def level_based_context_filter(
        _logger: p.Logger | None,
        method_name: str,
        event_dict: t.ScalarMapping,
    ) -> t.ScalarMapping:
        """Filter context variables based on log level."""
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            c.WarningLevel.ERROR: 40,
            "critical": 50,
        }
        current_level = level_hierarchy.get(method_name.lower(), 20)
        filtered_dict: t.MutableConfigurationMapping = {}
        for key, value in event_dict.items():
            if key.startswith("_level_"):
                parts = key.split("_", c.DEFAULT_MAX_WORKERS)
                if len(parts) >= c.DEFAULT_MAX_WORKERS:
                    required_level_name = parts[2]
                    actual_key = parts[3]
                    required_level = level_hierarchy.get(
                        required_level_name.lower(),
                        10,
                    )
                    if current_level >= required_level:
                        normalized_key = (
                            "settings" if actual_key == "config" else actual_key
                        )
                        filtered_dict[normalized_key] = value
                else:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value
        return filtered_dict

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
            _ = structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = structlog.contextvars.bind_contextvars(
                    service_version=service_version,
                )
            if enable_context_correlation:
                correlation_id = f"flext-{FlextUtilitiesGenerators.generate_id().replace('-', '')[:12]}"
                _ = structlog.contextvars.bind_contextvars(
                    correlation_id=correlation_id,
                )
            structlog.get_logger(__name__).info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: t.ConfigMap | None = None,
        ) -> None:
            """Track domain event with context correlation."""
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            structlog.get_logger(__name__).info(
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
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            logger = structlog.get_logger(__name__)
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

    @staticmethod
    def resolve_log_level_from_settings() -> int:
        """Resolve log level from default constant."""
        default_log_level = c.DEFAULT_LEVEL.upper()
        return int(
            getattr(logging, default_log_level)
            if hasattr(logging, default_log_level)
            else logging.INFO,
        )

    @classmethod
    def _global_context(cls) -> t.ConfigMap:
        """Get current global context (internal use only)."""
        try:
            context_vars = cls.structlog().contextvars.get_contextvars()
            context_map: t.FlatContainerMapping = (
                {
                    str(k): cls._to_container_value(v)
                    for k, v in dict(context_vars).items()
                }
                if context_vars
                else {}
            )
            context_obj: Mapping[str, t.ValueOrModel] = dict(context_map)
            return t.ConfigMap(root=dict(context_obj))
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return t.ConfigMap(root={})

    @classmethod
    def bind_context(cls, scope: str, **context: t.RuntimeData) -> r[bool]:
        """Bind context variables to a specific scope.

        Args:
            scope: Scope name. Use c.SCOPE_* constants:
                   - SCOPE_APPLICATION: Persists for entire app lifetime
                   - SCOPE_REQUEST: Persists for single request/command
                   - SCOPE_OPERATION: Persists for single operation
            **context: Context variables to bind

        Returns:
            r[bool]: Success with True if context bound, failure with error message otherwise.

        """
        try:
            cls._scoped_contexts.setdefault(scope, {})
            current_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value)
                for key, value in cls._scoped_contexts[scope].items()
            }
            incoming_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value) for key, value in context.items()
            }
            current_context_obj: t.RecursiveContainerMapping = dict(
                current_context.items(),
            )
            incoming_context_obj: t.RecursiveContainerMapping = dict(
                incoming_context.items(),
            )
            merge_result = FlextUtilitiesCollection.merge_mappings(
                incoming_context_obj,
                current_context_obj,
                strategy="deep",
            )
            merged_value = merge_result.unwrap_or(current_context_obj)
            merged_context: t.MutableFlatContainerMapping = {}
            for key, value in merged_value.items():
                merged_context[str(key)] = cls._to_container_value(value)
            cls._scoped_contexts[scope] = merged_context
            cls.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation(f"bind context for scope '{scope}'", exc)

    @classmethod
    def bind_context_for_level(cls, level: str, **context: t.RuntimeData) -> r[bool]:
        """Bind context variables that only appear in logs at specified level or higher.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - case insensitive
            **context: Context variables to bind

        Returns:
            r[bool]: Success with True if bound, failure with error details

        """
        try:
            level_lower = level.lower()
            level_normalized = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                c.WarningLevel.ERROR: c.WarningLevel.ERROR,
                "critical": "critical",
            }.get(level_lower, level_lower)
            cls._level_contexts.setdefault(level_normalized, {})
            normalized_context = cls._to_container_context(context)
            prefixed_context = {
                f"_level_{level_normalized}_{key}": value
                for key, value in normalized_context.items()
            }
            cls._level_contexts[level_normalized].update(normalized_context)
            cls.structlog().contextvars.bind_contextvars(**prefixed_context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation(f"bind context for level {level}", exc)

    @classmethod
    def bind_global_context(cls, **context: t.RuntimeData) -> r[bool]:
        """Bind context globally using structlog contextvars.

        Args:
            **context: Context variables to bind globally

        Returns:
            r[bool]: Success with True if context bound, failure with error message otherwise.

        """
        try:
            normalized_context = cls._to_container_context(context)
            cls.structlog().contextvars.bind_contextvars(**normalized_context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("bind global context", exc)

    @classmethod
    def clear_global_context(cls) -> r[bool]:
        """Clear global logging context and cached scoped bindings."""
        try:
            cls.structlog().contextvars.clear_contextvars()
            cls._scoped_contexts.clear()
            cls._level_contexts.clear()
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("clear global context", exc)

    @classmethod
    def clear_scope(cls, scope: str) -> r[bool]:
        """Clear all context variables for a specific scope.

        Args:
            scope: Scope to clear (use c.SCOPE_* constants)

        Returns:
            r[bool]: Success with True if scope cleared, failure with error message otherwise.

        """
        try:
            if scope in cls._scoped_contexts:
                keys = list(cls._scoped_contexts[scope].keys())
                if keys:
                    cls.structlog().contextvars.unbind_contextvars(*keys)
                cls._scoped_contexts[scope].clear()
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation(f"clear scope '{scope}'", exc)

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
    ) -> Self:
        """Create a logger instance for a module.

        Args:
            name: Module name (typically __name__). Defaults to "flext".
            settings: Optional settings model used to seed logger context.
            service_name: Optional service name bound at construction time.
            service_version: Optional service version bound at construction time.
            correlation_id: Optional correlation identifier bound at construction time.

        Returns:
            FlextLogger: Logger instance for the module

        """
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
        """Create logger configured for a specific container.

        Args:
            container: Container instance to bind logger to.
            level: Optional log level override. If not provided, uses container's
                settings log_level.
            **context: Additional context variables to bind.

        Returns:
            FlextLogger: Logger instance configured for the container.

        """
        if level is None:
            settings: p.Settings | None
            try:
                settings = container.settings
            except (AttributeError, RuntimeError, TypeError, ValueError):
                settings = None
            level = getattr(settings, "log_level", "INFO")
        logger = cls.create_module_logger(f"container_{id(container)}")
        if context:
            _ = logger.bind_global_context(**context)
        return logger

    @classmethod
    def unbind_global_context(cls, *keys: str) -> r[bool]:
        """Unbind specific keys from global context.

        Args:
            *keys: Context keys to unbind

        """
        try:
            unbind_keys: t.StrSequence = [str(key) for key in keys]
            cls.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("unbind global context", exc)

    @staticmethod
    def _convert_to_relative_path(filename: str) -> str:
        """Convert absolute path to relative path from workspace root."""
        try:
            abs_path = Path(filename).resolve()
            workspace_root = FlextLogger._find_workspace_root(abs_path)
            if workspace_root:
                try:
                    return str(abs_path.relative_to(workspace_root))
                except ValueError:
                    return Path(filename).name
            return Path(filename).name
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure(
                "convert_to_relative_path",
                exc,
            )
            return Path(filename).name

    @staticmethod
    def _extract_class_name(frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname."""
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if hasattr(self_obj, "__class__"):
                class_name: str = self_obj.__class__.__name__
                return class_name
        if hasattr(frame.f_code, "co_qualname"):
            qualname = frame.f_code.co_qualname
            if "." in qualname:
                parts = qualname.rsplit(".", 1)
                if len(parts) == c.LEVEL_PREFIX_PARTS_COUNT:
                    potential_class = parts[0]
                    if potential_class and potential_class[0].isupper():
                        return potential_class
        return None

    @staticmethod
    def _find_workspace_root(abs_path: Path) -> Path | None:
        """Find workspace root by looking for common markers."""
        current = abs_path.parent
        markers = ["pyproject.toml", ".git", "poetry.lock"]
        for _ in range(10):
            if any((current / marker).exists() for marker in markers):
                return current
            if current == current.parent:
                break
            current = current.parent
        return None

    @staticmethod
    def _format_log_message(message: str, *args: _LogArg) -> str:
        """Format log message with % arguments."""
        try:
            return message % args if args else message
        except (TypeError, ValueError):
            return f"{message} | args={args!r}"

    @staticmethod
    def _to_container_value(
        value: _LogArg | t.Container | t.ValueOrModel,
    ) -> t.Container:
        """Normalize value to Container (internal helper)."""
        if isinstance(value, Exception):
            return str(value)
        if value is None:
            return ""
        if FlextUtilitiesGuardsTypeCore.scalar(value) or isinstance(value, Path):
            return value
        if FlextUtilitiesGuardsTypeModel.pydantic_model(value):
            return value.model_dump_json()
        normalized = FlextRuntime.normalize_to_container(value)
        if FlextUtilitiesGuardsTypeCore.scalar(normalized) or isinstance(
            normalized,
            Path,
        ):
            return normalized
        return normalized.model_dump_json()

    @staticmethod
    def _to_scalar_value(
        value: _LogArg | t.Container | t.ValueOrModel | None,
    ) -> t.Scalar:
        if value is None:
            return ""
        if isinstance(value, Exception):
            return str(value)
        if isinstance(value, (list, tuple, dict, Mapping)):
            return str(value)
        if FlextUtilitiesGuardsTypeCore.scalar(value):
            return value
        return str(value)

    @staticmethod
    def _to_container_context(
        context: Mapping[str, _LogArg | t.Container | t.ValueOrModel],
    ) -> t.FlatContainerMapping:
        """Convert mapping to container context using normalization."""
        return {
            key: FlextLogger._to_container_value(value)
            for key, value in context.items()
        }

    @classmethod
    def _to_scalar_context(
        cls,
        context: Mapping[str, _LogArg | t.Container | t.ValueOrModel | None],
    ) -> t.ScalarMapping:
        return {key: cls._to_scalar_value(value) for key, value in context.items()}

    @staticmethod
    def _caller_source_path() -> str | None:
        """Get source file path with line, class and method context."""
        try:
            caller_frame = FlextLogger._calling_frame()
            if caller_frame is None:
                return None
            filename = caller_frame.f_code.co_filename
            abs_path = Path(filename).resolve()
            workspace_root = FlextLogger._find_workspace_root(abs_path)
            if workspace_root is None:
                return None
            relative_path = abs_path.relative_to(workspace_root)
            if relative_path.parts and relative_path.parts[0] == ".venv":
                return None
            file_path = str(relative_path)
            line_number = caller_frame.f_lineno + 1
            method_name = caller_frame.f_code.co_name
            class_name = FlextLogger._extract_class_name(caller_frame)
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != "<module>":
                source_parts.append(method_name)
            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure("get_caller_source_path", exc)
            return None

    @staticmethod
    def _calling_frame() -> types.FrameType | None:
        """Get the calling frame 4 levels up the stack."""
        frame = inspect.currentframe()
        if not frame:
            return None
        for _ in range(4):
            frame = frame.f_back
            if not frame:
                return None
        return frame

    @staticmethod
    def _report_internal_logging_failure(operation: str, exc: Exception) -> None:
        with suppress(AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            FlextLogger.structlog().get_logger("flext_core.loggings").warning(
                "Internal logger operation failed",
                operation=operation,
                error=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _fail_logging_operation(operation: str, exc: Exception) -> r[bool]:
        """Return the canonical failed result for a logger operation."""
        return e.fail_operation(operation, exc)

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            return logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure(
                "should_include_stack_trace",
                exc,
            )
            return True

    def bind(self, **context: t.RuntimeData) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**self._to_container_context(context))
        return self.__class__.create_bound_logger(self.name, bound_logger)

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: Mapping[str, t.RuntimeData | Exception],
    ) -> t.ConfigMap:
        """Build normalized context payload for exception/error logging."""
        include_stack_trace = self._should_include_stack_trace()
        context_dict: t.ConfigMap = t.ConfigMap(root={})
        if exception is not None:
            exception_data: t.ConfigMap = t.ConfigMap(
                root={
                    "exception_type": exception.__class__.__name__,
                    "exception_message": str(exception),
                },
            )
            merged_root: MutableMapping[str, t.ValueOrModel] = dict(context_dict.root)
            merged_root.update(dict(exception_data.root))
            context_dict = t.ConfigMap(root=merged_root)
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
                context_dict[key] = FlextRuntime.normalize_to_container(value)
        return context_dict

    def critical(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log critical message - Logger.Log implementation.

        Business Rule: Logs a critical-level message with optional context. Uses _log
        method for actual logging. Uses u for centralized logging management.

        Audit Implication: Critical logging ensures audit trail completeness by recording
        critical messages about severe failures. Critical messages are always included
        in production logs and critical for audit trail reconstruction and emergency
        response. All critical messages go through this method, ensuring consistent
        log formatting and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log debug message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.DEBUG, msg, *args, **kw)

    def error(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log error message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.ERROR, msg, *args, **kw)

    def exception(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
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
            context_dict = self.build_exception_context(
                exception=resolved_exception,
                exc_info=bool(exc_info_value),
                context=context_input,
            )
            if resolved_exception is None and isinstance(raw_exception, BaseException):
                context_dict["exception_type"] = raw_exception.__class__.__name__
                context_dict["exception_message"] = str(raw_exception)
            _ = self.logger.error(
                message,
                *filtered_args,
                **FlextLogger._to_scalar_context(context_dict.root),
            )
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            FlextLogger._report_internal_logging_failure("exception", exc)
            return self._fail_logging_operation("exception logging", exc)

    def log(
        self,
        level: str,
        message: str,
        *args: _LogArg,
        **context: t.RuntimeData,
    ) -> r[bool]:
        """Log message with specified level - Logger.Log implementation.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            *args: Positional args for printf-style message formatting
            **context: Keyword context to include in structured log

        """
        level_enum: c.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = c.LogLevel(level.upper())
        converted_args: tuple[t.Container, ...] = tuple(
            FlextLogger._to_scalar_value(arg) for arg in args
        )
        return self._log(level_enum, message, *converted_args, **context)

    def new(self, **context: t.RuntimeData) -> Self:
        """Create new logger with context - implements BindableLogger protocol."""
        return self.bind(**context)

    def trace(
        self,
        message: str,
        *args: _LogArg,
        **kwargs: t.RuntimeData,
    ) -> r[bool]:
        """Log trace message - Logger.Log implementation."""
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
        """Unbind keys from logger - implements BindableLogger protocol."""
        if safe:
            with suppress(KeyError, ValueError, AttributeError):
                bound_logger = self.logger.unbind(*keys)
                return self.__class__.create_bound_logger(self.name, bound_logger)
            return self
        bound_logger = self.logger.unbind(*keys)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    def try_unbind(self, *keys: str) -> Self:
        """Unbind keys in safe mode (deprecated compatibility helper)."""
        warnings.warn(
            "FlextLogger.try_unbind is deprecated; use unbind(*keys, safe=True). "
            "Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unbind(*keys, safe=True)

    def info(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log info message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.INFO, msg, *args, **kw)

    def warning(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log warning message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.WARNING, msg, *args, **kw)

    def _log(
        self,
        _level: c.LogLevel | str,
        event: str,
        *args: t.RuntimeData,
        **context: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Internal logging method - consolidates all log level methods."""
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
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        return self._log(level, msg, *args, **kw)

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
            context: t.ConfigMap = t.ConfigMap(
                root={
                    c.MetadataKey.DURATION_SECONDS: elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                },
            )
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            if success:
                _ = self.logger.info(
                    f"{self._operation_name} {status}",
                    **FlextLogger._to_container_context(context.root),
                )
            else:
                _ = self.logger.error(
                    f"{self._operation_name} {status}",
                    **FlextLogger._to_container_context(context.root),
                )


__all__: t.StrSequence = ["FlextLogger"]

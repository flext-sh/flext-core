"""Structlog configuration and processor chain building.

Extracted from FlextLogger as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import io
import logging
import queue
import sys
import threading
import types
import typing
from collections.abc import Sequence
from contextlib import suppress
from typing import ClassVar, override

import structlog
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level
from structlog.types import Processor

from flext_core import c, p, t


class FlextUtilitiesLoggingConfig:
    """Structlog configuration, async writer, and processor chain assembly."""

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry]
    _level_contexts: ClassVar[t.ScopedContainerRegistry]
    _structlog_instance: p.Logger | None
    _structlog_configured: ClassVar[bool]

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread."""

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._stream_mode: str = str(getattr(stream, "mode", "w"))
            self._stream_name: str = str(
                getattr(stream, "name", "<async-log-writer>"),
            )
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

    @staticmethod
    def _structlog_processor(
        value: typing.Callable[..., t.ValueOrModel] | t.Container | None,
    ) -> typing.TypeIs[Processor]:
        return callable(value)

    @staticmethod
    def structlog() -> types.ModuleType:
        """Return the imported structlog module."""
        return structlog

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


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]

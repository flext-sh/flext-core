"""Facade for FlextUtilitiesLoggingConfig.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import logging
import queue
import sys
import threading
from collections.abc import MutableMapping, MutableSequence, Sequence
from concurrent.futures import Future
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, override

import structlog
from structlog.processors import (
    CallsiteParameterAdder,
    ExceptionRenderer,
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
)
from structlog.stdlib import PositionalArgumentsFormatter, add_log_level
from structlog.tracebacks import ExceptionDictTransformer
from structlog.typing import Processor

from flext_core import FlextSettings, c, m, p, t
from flext_core._utilities import FlextUtilitiesMetadata
from flext_core._utilities._logging_config.logging_stream import (
    FlextUtilitiesLoggingStream,
)

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingConfig:
    """Compose logging configuration through its internal part hierarchy."""

    _log_threshold: ClassVar[int]
    _runtime_settings: ClassVar[p.LoggingSettings | None] = None
    _options: ClassVar[p.StructlogOptions | None] = None

    class _StreamHandler(logging.StreamHandler[t.TextStream]):
        """Retain native rendering/locking while surfacing the native sink error."""

        include_callsite: bool = True

        @override
        def handleError(self, record: t.LoggingNativeRecord) -> None:
            error = sys.exception()
            if error is not None:
                raise error
            msg = "Logging handler failed without an active exception"
            raise RuntimeError(msg)

    @staticmethod
    def resolve_logging_exception_info(
        value: p.LoggingEventValue | tuple[p.AttributeProbe, ...] | None,
    ) -> t.LoggingExceptionInfo | None:
        """Resolve native exception inputs once, retaining a saved traceback.

        Returns:
            The exception triple or None when exception capture is disabled.

        Raises:
            TypeError: If the native exception input is invalid.

        """
        if value is True:
            value = sys.exception()
        if isinstance(value, BaseException):
            return type(value), value, value.__traceback__
        if value is None or value is False:
            return None
        if not isinstance(value, tuple) or len(value) != len(sys.exc_info()):
            msg = "exc_info must be an exception, saved tuple, boolean or None"
            raise TypeError(msg)
        if (
            isinstance(value[0], type)
            and issubclass(value[0], BaseException)
            and isinstance(value[1], BaseException)
            and (value[2] is None or isinstance(value[2], TracebackType))
        ):
            return value[0], value[1], value[2]
        msg = "exc_info must be an exception, saved tuple, boolean or None"
        raise TypeError(msg)

    @staticmethod
    def log_threshold() -> int:
        """Return the elected minimum severity applied before enrichment."""
        return FlextUtilitiesLoggingConfig._log_threshold

    @staticmethod
    def structlog() -> types.ModuleType:
        """Return the imported structlog module for owner-internal access.

        Returns:
            The imported structlog module for owner-internal access.

        """
        return structlog

    class _AsyncLogWriter(FlextUtilitiesLoggingStream.TextStream):
        """Real text stream whose writes are queued to a background thread."""

        def __init__(self, stream: t.TextStream) -> None:
            super().__init__(stream)
            self._failure: BaseException | None = None
            self._admission = threading.Condition()
            self.queue: queue.Queue[p.AsyncLogJob] = queue.Queue(maxsize=c.MAX_ITEMS)
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="flext-async-log-writer",
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)

        @override
        def close(self) -> None:
            """Finish owned work without closing the caller's output stream."""
            if self.closed:
                return
            self.shutdown()
            with self._admission:
                self._closed = True
                self._admission.notify_all()

        @override
        def flush(self) -> None:
            """Wait for submitted writes and propagate the first native failure."""
            completion = self._admit(None)
            _ = completion.result()

        def shutdown(self) -> None:
            """Stop worker thread and flush remaining messages.

            Raises:
                TimeoutError: If the worker exceeds its shutdown deadline.

            """
            with self._admission:
                if self.closed:
                    return
                self.stop_event.set()
                self._admission.notify_all()
            if self.thread.is_alive():
                self.thread.join(timeout=c.LOG_SHUTDOWN_JOIN_SECONDS)
            if self.thread.is_alive():
                msg = "Async log writer did not finish before shutdown timeout"
                raise TimeoutError(msg)
            atexit.unregister(self.shutdown)
            with self._admission:
                self._raise_failure()

        @override
        def write(self, s: str, /) -> int:
            """Submit a write; queue pressure and prior failures escape.

            Returns:
                The number of characters admitted to the output queue.

            """
            _ = self._admit(s)
            return len(s)

        def _admit(self, text: str | None) -> t.LoggingWriteCompletion:
            """Admit typed work atomically with worker failure and termination.

            Returns:
                The native completion for the admitted write or flush.

            Raises:
                Full: If the configured queue rejects further work.
                ValueError: If the writer has stopped.

            """
            completion: t.LoggingWriteCompletion = Future()
            with self._admission:
                self._raise_failure()
                self._check_open()
                if self.stop_event.is_set():
                    msg = "I/O operation on stopped async log writer"
                    raise ValueError(msg)
                while self.queue.full():
                    if not c.ASYNC_BLOCK_ON_FULL:
                        raise queue.Full
                    self._admission.wait()
                    self._raise_failure()
                    self._check_open()
                    if self.stop_event.is_set():
                        msg = "I/O operation on stopped async log writer"
                        raise ValueError(msg)
                job = m.AsyncLogJob(text=text, completion=completion)
                self.queue.put_nowait(job)
            return completion

        def _raise_failure(self) -> None:
            failure = self._failure
            if failure is not None:
                raise failure

        def _worker(self) -> None:
            """Write serially and transfer the original failure to each Future."""
            while True:
                try:
                    job = self.queue.get(
                        timeout=c.LOG_WORKER_POLL_SECONDS,
                    )
                except queue.Empty:
                    with self._admission:
                        if self.stop_event.is_set() and self.queue.empty():
                            break
                    continue
                with self._admission:
                    failure = self._failure
                    self._admission.notify_all()
                if failure is not None:
                    job.completion.set_exception(failure)
                    self.queue.task_done()
                    continue
                try:
                    written = self._write_job(job)
                # A user-owned sink can raise any Exception. Deliver that exact
                # object to the caller's Future, rather than logging or hiding it.
                except Exception as failure:  # ruff: ignore[blind-except]
                    with self._admission:
                        self._failure = failure
                        self.stop_event.set()
                        self._admission.notify_all()
                    atexit.unregister(self.shutdown)
                    job.completion.set_exception(failure)
                else:
                    job.completion.set_result(written)
                self.queue.task_done()

        def _write_job(self, job: p.AsyncLogJob) -> int:
            """Let a native Future retain every sink failure for the caller.

            Returns:
                The number of characters written before flushing.

            """
            written = self._target_stream.write(job.text) if job.text is not None else 0
            self._target_stream.flush()
            return written

    _async_writer: ClassVar[_AsyncLogWriter | None] = None

    @staticmethod
    def selected_logging_settings() -> p.LoggingSettings | None:
        """Return the explicitly elected logging settings."""
        return FlextUtilitiesLoggingConfig._runtime_settings

    @staticmethod
    def selected_logging_options() -> p.StructlogOptions | None:
        """Return the elected composition, if configured."""
        return FlextUtilitiesLoggingConfig._options

    @staticmethod
    def configure_logging_state(
        settings: p.LoggingSettings,
        options: p.StructlogOptions,
        threshold: int,
    ) -> None:
        """Publish the state owned by the shared logging configuration."""
        FlextUtilitiesLoggingConfig._runtime_settings = settings
        FlextUtilitiesLoggingConfig._options = options
        FlextUtilitiesLoggingConfig._log_threshold = threshold

    @staticmethod
    def _resolve_logger_factory(
        *,
        logger_factory: p.OutputLoggerFactory | None,
        async_logging: bool,
    ) -> p.OutputLoggerFactory | None:
        """Keep the selected native factory or the acknowledged stderr writer.

        Returns:
            The selected output factory.

        """
        if (
            not async_logging or logger_factory is not None
        ) and FlextUtilitiesLoggingConfig._async_writer is not None:
            # Existing runtime loggers still own bindings to this transport.
            # A new composition drains it; process shutdown closes it.
            FlextUtilitiesLoggingConfig._async_writer.flush()
        if logger_factory is not None:
            return logger_factory
        if not async_logging:
            return structlog.PrintLoggerFactory(
                file=FlextUtilitiesLoggingStream.TextStream(sys.stderr)
            )
        if FlextUtilitiesLoggingConfig._async_writer is None:
            FlextUtilitiesLoggingConfig._async_writer = (
                FlextUtilitiesLoggingConfig._AsyncLogWriter(sys.stderr)
            )
        return structlog.PrintLoggerFactory(
            file=FlextUtilitiesLoggingConfig._async_writer
        )

    @staticmethod
    def level_based_context_filter(
        _logger: p.OutputLogger | None,
        method_name: str,
        event_dict: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Filter level-tagged context using the native severity mapping.

        Returns:
            The event without context above its severity.

        """
        level_hierarchy = logging.getLevelNamesMapping()
        current_level = level_hierarchy[method_name.upper()]
        filtered_dict: p.LoggingEvent = {}
        for key, value in event_dict.items():
            if not key.startswith("_level_"):
                filtered_dict[key] = value
                continue
            required, separator, actual_key = key.removeprefix("_level_").partition("_")
            if not separator:
                filtered_dict[key] = value
                continue
            required_level = level_hierarchy[required.upper()]
            if current_level < required_level:
                continue
            normalized_key = "settings" if actual_key == "config" else actual_key
            filtered_dict[normalized_key] = value
        return filtered_dict

    @staticmethod
    def drop_below_threshold(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: p.LoggingEvent,
        *,
        threshold: int | None = None,
    ) -> p.LoggingEvent:
        """Apply the elected threshold before native enrichment and hooks.

        Returns:
            The admitted event.

        Raises:
            DropEvent: If the event falls below the elected threshold.

        """
        _ = logger, method_name
        level = logging.getLevelNamesMapping()[str(event_dict["level"]).upper()]
        admitted_level = (
            FlextUtilitiesLoggingConfig.log_threshold()
            if threshold is None
            else threshold
        )
        if level < admitted_level:
            raise structlog.DropEvent
        return event_dict

    @classmethod
    def _console_event(
        cls,
        logger: p.OutputLogger | None,
        method_name: str,
        event: p.LoggingEvent,
        *,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> p.LoggingEvent:
        """Select console detail without changing an observer or sibling sink.

        Returns:
            A presentation-only copy of the sanitized event.

        """
        _ = logger, method_name
        selected = runtime_settings or cls.logging_settings()
        hidden = (
            frozenset()
            if selected.debug or selected.trace
            else frozenset(
                item.value for item in structlog.processors.CallsiteParameter
            )
        )
        return {key: value for key, value in event.items() if key not in hidden}

    @staticmethod
    def _exception_metadata(
        logger: p.OutputLogger,
        method_name: str,
        event: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Keep causal metadata before the native processor consumes exc_info.

        Returns:
            The event with its original exception metadata.

        """
        _ = logger, method_name
        info = FlextUtilitiesLoggingConfig.resolve_logging_exception_info(
            event.get("exc_info")
        )
        if info is not None:
            exception = info[1]
            event["_original_exception"] = exception
            event["exception_type"] = type(exception).__name__
            event["exception_message"] = str(exception)
        return event

    @staticmethod
    def _observe_event(
        logger: p.OutputLogger,
        method_name: str,
        event: p.LoggingEvent,
        *,
        options: p.StructlogOptions,
    ) -> p.LoggingEvent:
        """Sanitize the SDK boundary, then notify typed subscribers once.

        Returns:
            The sanitized event delivered to subscribers.

        """
        _ = logger, method_name
        original_exception = event.pop("_original_exception", None)

        # ConsoleRenderer accepts text in `exception`; native structured stacks
        # remain ordinary structured context, shared with JSON and observers.
        if isinstance(event.get("exception"), list):
            event["stack_trace"] = event.pop("exception")
        record = m.LoggingRecord.model_validate({
            "message": str(event[c.HandlerType.EVENT]),
            "level": str(event["level"]),
            "context": dict(event),
        })
        sanitized: p.LoggingEvent = dict(
            FlextUtilitiesMetadata.redact_metadata(
                record.context,
                keys=tuple(
                    c.SENSITIVE_ERROR_DATA_KEYS | frozenset(options.redacted_keys)
                ),
                values=options.redacted_values,
                replacement="[REDACTED]",
            )
        )
        record = m.LoggingRecord.model_validate({
            "message": sanitized[c.HandlerType.EVENT],
            "level": sanitized["level"],
            "context": sanitized,
            "exception": original_exception,
        })
        for observer in options.observers:
            observer(record)
        return sanitized

    @staticmethod
    def _format_exception(exc_info: t.LoggingExceptionInfo) -> str:
        """Render native exception information without expanding its traceback.

        Returns:
            The exception type and message.

        """
        return f"{exc_info[0].__name__}: {exc_info[1]}"

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        options: p.StructlogOptions,
        trace: bool,
        runtime_settings: p.LoggingSettings | None = None,
        debug: bool = False,
    ) -> Sequence[Processor]:
        """Compose one shared event pipeline and a native terminal renderer.

        Returns:
            The ordered native processors.

        """
        processors: MutableSequence[Processor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.drop_below_threshold
            if runtime_settings is None
            else partial(
                cls.drop_below_threshold,
                threshold=cls.level_number(runtime_settings.log_level),
            ),
            PositionalArgumentsFormatter(),
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
            cls._exception_metadata,
        ]
        if any((
            debug,
            trace,
            not console_renderer,
            any(
                not isinstance(handler, cls._StreamHandler) or handler.include_callsite
                for handler in options.handlers
            ),
            options.observers,
            options.processing_stages,
        )):
            processors.insert(
                -2,
                CallsiteParameterAdder(
                    additional_ignores=[
                        "flext_core._utilities._logging_context.bound_logger",
                        "flext_core._utilities.logging",
                    ]
                ),
            )
        processors.extend(options.processing_stages)
        processors.extend((
            ExceptionRenderer(
                ExceptionDictTransformer(show_locals=False, max_frames=sys.maxsize)
                if trace
                else cls._format_exception
            ),
            partial(cls._observe_event, options=options),
        ))
        if options.handlers:
            processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)
        elif console_renderer:
            processors.append(
                partial(cls._console_event, runtime_settings=runtime_settings)
            )
            processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))
        else:
            processors.append(JSONRenderer())
        return processors

    @staticmethod
    def resolve_effective_log_level(
        *,
        debug: bool,
        log_level: t.LogLevel,
        log_level_explicit: bool = True,
    ) -> t.LogLevel:
        """Keep trace detail independent of the configured logging threshold.

        Returns:
            The effective severity without changing trace detail.

        """
        return c.LogLevel.DEBUG if debug and not log_level_explicit else log_level

    @classmethod
    def logging_settings(cls) -> p.LoggingSettings:
        """Return the elected application provider, or the actual Core settings."""
        selected = cls.selected_logging_settings()
        return selected if selected is not None else FlextSettings.fetch_global()

    @classmethod
    def logging_options(cls) -> p.StructlogOptions:
        """Return the active composition for scoped lifecycle restoration."""
        selected = cls.selected_logging_options()
        return selected if selected is not None else m.StructlogOptions()

    @classmethod
    def resolve_logging_settings(
        cls,
        *,
        runtime_settings: p.LoggingSettings | None = None,
        log_level: str | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
        logging_profile: str | None = None,
    ) -> p.LoggingSettings:
        """Validate optional overrides without I/O or process-wide mutation.

        Returns:
            The validated logging settings snapshot.

        """
        current = (
            runtime_settings if runtime_settings is not None else cls.logging_settings()
        )
        overrides: MutableMapping[str, p.SettingsOverride | None] = {
            key: value
            for key, value in (
                ("log_level", log_level),
                ("debug", debug),
                ("trace", trace),
                ("logging_profile", logging_profile),
            )
            if value is not None
        }
        selected = current.clone(**overrides)
        level = cls.resolve_effective_log_level(
            log_level=c.LogLevel(
                logging.getLevelName(cls.level_number(selected.log_level))
            ),
            debug=selected.debug,
            log_level_explicit=(
                log_level is not None
                or (debug is None and "log_level" in current.model_fields_set)
            ),
        )
        return selected.clone(log_level=level.value)

    @classmethod
    def compose_native_logger(
        cls,
        name: str,
        *,
        runtime_settings: p.LoggingSettings,
        options: p.StructlogOptions,
    ) -> p.NativeLoggingWrapper:
        """Compose a scoped logger without replacing the process configuration.

        Returns:
            A binding whose processors, threshold and sink remain runtime-owned.

        Raises:
            ValueError: If no output factory can be elected.

        """
        console = (
            options.console_renderer
            if options.console_renderer is not None
            else runtime_settings.logging_profile == "console"
        )
        wrapper = (
            options.wrapper_class_factory()
            if options.wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(logging.NOTSET)
        )
        factory = options.logger_factory
        if options.handlers:
            factory = cls._handler_factory(options)
        if factory is None:
            factory = cls._resolve_logger_factory(
                logger_factory=None,
                async_logging=runtime_settings.async_logging
                if options.async_logging is None
                else options.async_logging,
            )
        if factory is None:
            msg = "Logging composition did not elect an output factory"
            raise ValueError(msg)
        return structlog.wrap_logger(
            factory(name),
            processors=cls._build_structlog_processors(
                console_renderer=console,
                options=options,
                trace=runtime_settings.trace,
                runtime_settings=runtime_settings,
                debug=runtime_settings.debug,
            ),
            wrapper_class=wrapper,
            cache_logger_on_first_use=options.cache_logger_on_first_use,
        ).bind()

    @classmethod
    def _handler_factory(cls, options: p.StructlogOptions) -> p.OutputLoggerFactory:
        """Own one native handler graph per composition.

        Returns:
            The factory bound to that graph's native manager.

        """
        root = logging.RootLogger(logging.NOTSET)
        root.handlers = cls._validated_native_handlers(options)
        manager = logging.Manager(root)

        def factory(*names: str) -> p.OutputLogger:
            return manager.getLogger(names[0] if names else c.LOGGER_NAME_FLEXT_CORE)

        return factory

    @classmethod
    def create_logging_handler(
        cls,
        *,
        stream: t.TextStream | None = None,
        console_renderer: bool = False,
        log_level: str | None = None,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> p.LoggingHandler:
        """Compose native per-destination presentation with fail-loud delivery.

        Returns:
            The configured native handler.

        """
        handler = cls._StreamHandler(stream)
        handler.include_callsite = not console_renderer
        processors: MutableSequence[Processor] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        ]
        if console_renderer:
            processors.append(
                partial(cls._console_event, runtime_settings=runtime_settings)
            )
        processors.append(
            structlog.dev.ConsoleRenderer(colors=False)
            if console_renderer
            else structlog.processors.JSONRenderer()
        )
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=processors,
            )
        )
        if log_level is not None:
            handler.setLevel(cls.level_number(log_level))
        return handler

    @classmethod
    def configure_structlog(
        cls,
        *,
        settings: p.StructlogOptions | None = None,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> None:
        """Initialize defaults once or apply explicitly supplied typed options.

        Cached public loggers lazily consume the same elected composition.
        Caller-supplied handlers retain ownership of their delivery lifecycle.
        """
        owner = FlextUtilitiesLoggingConfig
        if (
            owner.selected_logging_options() is not None
            and settings is None
            and runtime_settings is None
        ):
            return
        options = settings if settings is not None else cls.logging_options()
        selected = cls.resolve_logging_settings(runtime_settings=runtime_settings)
        if runtime_settings is not None and options.log_level is not None:
            if options.log_level != cls.level_number(selected.log_level):
                options = m.StructlogOptions.model_validate(
                    options,
                    from_attributes=True,
                ).model_copy(update={"log_level": None})
        elif options.log_level is not None:
            selected = cls.resolve_logging_settings(
                runtime_settings=selected,
                log_level=logging.getLevelName(options.log_level),
            )
        threshold = cls.level_number(selected.log_level)
        processors = cls._build_structlog_processors(
            console_renderer=(
                options.console_renderer
                if options.console_renderer is not None
                else selected.logging_profile == "console"
            ),
            options=options,
            trace=selected.trace,
            debug=selected.debug,
        )
        # The threshold is enforced by ``drop_below_threshold`` at emit time,
        # so the bound logger itself must not filter statically.
        wrapper_arg = (
            options.wrapper_class_factory()
            if options.wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(logging.NOTSET)
        )
        logger_factory = options.logger_factory
        if options.handlers:
            logger_factory = cls._handler_factory(options)
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory,
            async_logging=(
                options.async_logging
                if options.async_logging is not None
                else selected.async_logging
            ),
        )
        structlog.configure(
            processors=processors,
            wrapper_class=wrapper_arg,
            logger_factory=factory_to_use,
            cache_logger_on_first_use=options.cache_logger_on_first_use,
        )
        owner.configure_logging_state(
            runtime_settings if runtime_settings is not None else selected,
            options,
            threshold,
        )

    @staticmethod
    def _validated_native_handlers(
        options: p.StructlogOptions,
    ) -> t.LoggingNativeHandlers:
        """Validate native sink contracts before replacing the active composition.

        Returns:
            The admitted native handlers.

        Raises:
            TypeError: If a handler or formatter has an incompatible type.
            ValueError: If factories conflict or sink failures are not propagated.

        """
        if options.handlers and options.logger_factory is not None:
            msg = "Select native handlers or a logger factory, not both"
            raise ValueError(msg)
        handlers: t.LoggingNativeHandlers = []
        for handler in options.handlers:
            if not isinstance(handler, logging.Handler):
                msg = (
                    "Selected handlers must implement the native "
                    "logging.Handler boundary"
                )
                raise TypeError(msg)
            if not isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter):
                msg = "Selected handlers require a native ProcessorFormatter"
                raise TypeError(msg)
            if type(handler).handleError is logging.Handler.handleError:
                msg = (
                    "Selected handlers must propagate sink failures; "
                    "use create_logging_handler"
                )
                raise ValueError(msg)
            handlers.append(handler)
        return handlers

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if cls.selected_logging_options() is None:
            cls.configure_structlog()

    @staticmethod
    def level_number(level: int | str) -> int:
        """Validate a standard severity and return its native number.

        Returns:
            The native numeric severity.

        Raises:
            ValueError: If the severity is not a supported standard level.

        """
        value = (
            level
            if isinstance(level, int)
            else logging.getLevelNamesMapping().get(level.strip().upper())
        )
        if value is None or logging.getLevelName(value) not in c.LogLevel:
            msg = f"Invalid log level: {level}"
            raise ValueError(msg)
        return value

    @classmethod
    def apply_log_level(
        cls,
        *,
        log_level: str | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
    ) -> None:
        """Apply the effective level of settings values to every logger.

        Resolution is owned by ``resolve_effective_log_level``; the result
        reaches loggers that were already created and cached.
        """
        selected = cls.resolve_logging_settings(
            log_level=log_level,
            debug=debug,
            trace=trace,
        )
        cls.configure_structlog(runtime_settings=selected)


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]

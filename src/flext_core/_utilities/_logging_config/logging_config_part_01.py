"""Structlog configuration and asynchronous stream lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import logging
import queue
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, override

import structlog

from flext_core import c, m, p, t
from flext_core._utilities._logging_config.logging_stream import (
    FlextUtilitiesLoggingStream,
)

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingConfigPart01:
    """Structlog configuration, async writer, and processor chain assembly."""

    _log_threshold: ClassVar[int]
    _runtime_settings: ClassVar[p.LoggingSettings | None] = None
    _options: ClassVar[p.StructlogOptions | None] = None

    class _StreamHandler(logging.StreamHandler[t.TextStream]):
        """Retain native rendering/locking while surfacing the native sink error."""

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
        return FlextUtilitiesLoggingConfigPart01._log_threshold

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
            """Transfer failures to awaiting callers without replacing them."""
            with ThreadPoolExecutor(max_workers=1) as writer:
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
                    operation = writer.submit(self._write_job, job)
                    failure = operation.exception()
                    if failure is not None:
                        with self._admission:
                            self._failure = failure
                            self.stop_event.set()
                            self._admission.notify_all()
                        atexit.unregister(self.shutdown)
                        job.completion.set_exception(failure)
                    else:
                        job.completion.set_result(operation.result())
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
        return FlextUtilitiesLoggingConfigPart01._runtime_settings

    @staticmethod
    def selected_logging_options() -> p.StructlogOptions | None:
        """Return the elected composition, if configured."""
        return FlextUtilitiesLoggingConfigPart01._options

    @staticmethod
    def configure_logging_state(
        settings: p.LoggingSettings, options: p.StructlogOptions, threshold: int
    ) -> None:
        """Publish the state owned by the shared logging configuration."""
        FlextUtilitiesLoggingConfigPart01._runtime_settings = settings
        FlextUtilitiesLoggingConfigPart01._options = options
        FlextUtilitiesLoggingConfigPart01._log_threshold = threshold

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
        ) and FlextUtilitiesLoggingConfigPart01._async_writer is not None:
            FlextUtilitiesLoggingConfigPart01._async_writer.shutdown()
            FlextUtilitiesLoggingConfigPart01._async_writer = None
        if logger_factory is not None:
            return logger_factory
        if not async_logging:
            return structlog.PrintLoggerFactory(
                file=FlextUtilitiesLoggingStream.TextStream(sys.stderr)
            )
        if FlextUtilitiesLoggingConfigPart01._async_writer is None:
            FlextUtilitiesLoggingConfigPart01._async_writer = (
                FlextUtilitiesLoggingConfigPart01._AsyncLogWriter(sys.stderr)
            )
        return structlog.PrintLoggerFactory(
            file=FlextUtilitiesLoggingConfigPart01._async_writer
        )


__all__: list[str] = ["FlextUtilitiesLoggingConfigPart01"]

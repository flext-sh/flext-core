"""Structlog configuration and asynchronous stream lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import queue
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, ClassVar, override

import structlog

from flext_core import c, p, t
from flext_core._models.config import FlextModelsConfig
from flext_core._utilities._logging_config_parts.logging_stream import (
    FlextUtilitiesLoggingStream,
)

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingConfig:
    """Structlog configuration, async writer, and processor chain assembly."""

    _structlog_configured: ClassVar[bool] = False
    _log_threshold: ClassVar[int]

    @staticmethod
    def log_threshold() -> int:
        """Return the process-wide minimum level number that is emitted.

        Returns:
            The process-wide minimum level number that is emitted.

        """
        return FlextUtilitiesLoggingConfig._log_threshold

    @staticmethod
    def _publish_logging_state(*, configured: bool, threshold: int) -> None:
        """Store the process-wide configuration flag and level threshold."""
        FlextUtilitiesLoggingConfig._structlog_configured = configured
        FlextUtilitiesLoggingConfig._log_threshold = threshold

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
                job = FlextModelsConfig.AsyncLogJob(text=text, completion=completion)
                self.queue.put_nowait(job)
            return completion

        def _raise_failure(self) -> None:
            failure = self._failure
            if failure is not None:
                raise failure

        def _worker(self) -> None:
            """Transfer failures to awaiting callers without replacing them."""
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
                    written = (
                        self._target_stream.write(job.text)
                        if job.text is not None
                        else 0
                    )
                    self._target_stream.flush()
                except BaseException as exc:
                    with self._admission:
                        self._failure = exc
                        self.stop_event.set()
                        self._admission.notify_all()
                    atexit.unregister(self.shutdown)
                    job.completion.set_exception(exc)
                else:
                    job.completion.set_result(written)
                finally:
                    self.queue.task_done()

    _async_writer: ClassVar[_AsyncLogWriter | None] = None


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]

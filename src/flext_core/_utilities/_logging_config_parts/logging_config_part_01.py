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
from flext_core._utilities import FlextUtilitiesTypingAdapterScalars

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingConfig:
    """Structlog configuration, async writer, and processor chain assembly."""

    # Process-wide logging state: written only through ``_publish_logging_state``
    # so a subclass (``u``, ``FlextUtilitiesLogging``) never shadows it.
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

    class _NativeStreamMetadata(TextIO):
        """Native stream identity and metadata, never fabricated.

        Every attribute is read from the wrapped native stream (the live
        ``sys.stdout``/``sys.stderr`` when those are wrapped); a native stream
        lacking an attribute raises its native ``AttributeError``.
        """

        def __init__(self, stream: t.TextStream) -> None:
            self.stream = stream
            self._use_live_stdout = stream is sys.stdout
            self._use_live_stderr = stream is sys.stderr
            self._closed = False

        @property
        def _target_stream(self) -> t.TextStream:
            if self._use_live_stdout:
                return sys.stdout
            if self._use_live_stderr:
                return sys.stderr
            return self.stream

        @property
        @override
        def mode(self) -> str:
            """Native stream mode."""
            return self._target_stream.mode

        @property
        @override
        def name(self) -> str | int:
            """Native stream path or file descriptor.

            Raises:
                TypeError: If the native name is neither a path nor a descriptor.

            """
            name = self._target_stream.name
            if isinstance(name, str | int):
                return name
            msg = "Text stream name must be a path or file descriptor"
            raise TypeError(msg)

        @property
        @override
        def buffer(self) -> t.BinaryStream:
            """Native binary buffer of the wrapped stream."""
            return self._target_stream.buffer

        @property
        @override
        def line_buffering(self) -> int:
            """Native line-buffering flag."""
            return self._target_stream.line_buffering

        @property
        @override
        def encoding(self) -> str:
            """Native text encoding."""
            return self._target_stream.encoding

        @property
        @override
        def errors(self) -> str | None:
            """Native encoding error policy."""
            return self._target_stream.errors

        @property
        @override
        def newlines(self) -> str | t.StrTuple | None:
            """Native newline translation state.

            Raises:
                TypeError: If the native value is not a string, tuple, or None.

            """
            newlines = self._target_stream.newlines
            if newlines is None or isinstance(newlines, str):
                return newlines
            if isinstance(newlines, tuple):
                adapter = FlextUtilitiesTypingAdapterScalars.strict_str_tuple_adapter()
                return adapter.validate_python(newlines)
            msg = "Text stream newlines must be a string or tuple of strings"
            raise TypeError(msg)

        @property
        @override
        def closed(self) -> bool:
            """Whether this adapter is closed; the native stream stays open."""
            return self._closed

    class _NativeStreamOperations(_NativeStreamMetadata):
        """Non-write operations delegated to the native stream.

        Operations that observe the stream first drain admitted writes; closing
        this adapter never closes the caller's native stream.
        """

        @override
        def close(self) -> None:
            """Flush admitted writes, then close this adapter only."""
            if not self._closed:
                self.flush()
                self._closed = True

        @override
        def fileno(self) -> int:
            """Return the native file descriptor.

            Returns:
                The descriptor of the wrapped stream.

            """
            return self._open_stream().fileno()

        @override
        def isatty(self) -> bool:
            """Return whether the native stream is interactive.

            Returns:
                The interactivity of the wrapped stream.

            """
            return self._open_stream().isatty()

        @override
        def readable(self) -> bool:
            """Return whether the native stream is readable.

            Returns:
                The readability of the wrapped stream.

            """
            return self._open_stream().readable()

        @override
        def seekable(self) -> bool:
            """Return whether the native stream is seekable.

            Returns:
                The seekability of the wrapped stream.

            """
            return self._open_stream().seekable()

        @override
        def writable(self) -> bool:
            """Return whether the native stream is writable.

            Returns:
                The writability of the wrapped stream.

            """
            return self._open_stream().writable()

        @override
        def read(self, n: int = -1, /) -> str:
            """Read from the native stream after admitted writes drain.

            Returns:
                The text read from the wrapped stream.

            """
            return self._drained_stream().read(n)

        @override
        def readline(self, limit: int = -1, /) -> str:
            """Read one native line after admitted writes drain.

            Returns:
                The line read from the wrapped stream.

            """
            return self._drained_stream().readline(limit)

        @override
        def readlines(self, hint: int = -1, /) -> list[str]:
            """Read native lines after admitted writes drain.

            Returns:
                The lines read from the wrapped stream.

            """
            return self._drained_stream().readlines(hint)

        @override
        def seek(self, offset: int, whence: int = 0, /) -> int:
            """Move the native position after admitted writes drain.

            Returns:
                The new position of the wrapped stream.

            """
            return self._drained_stream().seek(offset, whence)

        @override
        def tell(self) -> int:
            """Return the native position after admitted writes drain.

            Returns:
                The current position of the wrapped stream.

            """
            return self._drained_stream().tell()

        @override
        def truncate(self, size: int | None = None, /) -> int:
            """Truncate the native stream after admitted writes drain.

            Returns:
                The new size of the wrapped stream.

            """
            return self._drained_stream().truncate(size)

        @override
        def writelines(self, lines: t.IterableOf[str], /) -> None:
            """Queue each line through the same admission as ``write``."""
            for line in lines:
                _ = self.write(line)

        @override
        def __iter__(self) -> t.IteratorOf[str]:
            """Iterate native lines.

            Returns:
                This adapter as its own line iterator.

            """
            return self

        @override
        def __next__(self) -> str:
            """Return the next native line after admitted writes drain.

            Returns:
                The next line of the wrapped stream.

            """
            return next(self._drained_stream())

        @override
        def __enter__(self) -> t.TextStream:
            """Enter a context that closes this adapter on exit.

            Returns:
                This open adapter.

            """
            _ = self._open_stream()
            return self

        @override
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: types.TracebackType | None,
            /,
        ) -> None:
            """Close this adapter without closing the native stream."""
            self.close()

        def _open_stream(self) -> t.TextStream:
            """Return the native stream while this adapter is open.

            Returns:
                The wrapped native stream.

            Raises:
                ValueError: If this adapter is closed.

            """
            if self._closed:
                msg = "I/O operation on closed log stream"
                raise ValueError(msg)
            return self._target_stream

        def _drained_stream(self) -> t.TextStream:
            """Return the native stream once admitted writes reached it.

            Returns:
                The wrapped native stream after a causal flush.

            """
            _ = self._open_stream()
            self.flush()
            return self._target_stream

    class _AsyncLogWriter(_NativeStreamOperations):
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

            Raises:
                ValueError: If this writer has already stopped.

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

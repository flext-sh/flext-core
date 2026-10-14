"""Structlog configuration and asynchronous stream lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import io
import queue
import sys
import threading
import typing
from typing import TYPE_CHECKING, ClassVar, override

import structlog

from flext_core import c, t

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

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread."""

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._use_live_stdout = stream is sys.stdout
            self._use_live_stderr = stream is sys.stderr
            self._stream_mode: str = str(getattr(stream, "mode", "w"))
            self._stream_name: str = str(getattr(stream, "name", "<async-log-writer>"))
            self._stream_encoding: str = str(
                getattr(stream, "encoding", c.DEFAULT_ENCODING),
            )
            self._stream_errors: str | None = getattr(stream, "errors", None)
            self._stream_newlines: str | t.VariadicTuple[str] | None = getattr(
                stream,
                "newlines",
                None,
            )
            self._failure: BaseException | None = None
            self.queue: queue.Queue[str] = queue.Queue(maxsize=c.MAX_ITEMS)
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="flext-async-log-writer",
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)

        @property
        def _target_stream(self) -> typing.TextIO:
            if self._use_live_stdout:
                return sys.stdout
            if self._use_live_stderr:
                return sys.stderr
            return self.stream

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
            """The underlying binary buffer."""
            buf: typing.BinaryIO | None = getattr(self._target_stream, "buffer", None)
            if buf is not None:
                return buf
            return io.BytesIO()

        @property
        def line_buffering(self) -> bool:
            """The whether line buffering is enabled."""
            return bool(getattr(self._target_stream, "line_buffering", False))

        @override
        def flush(self) -> None:
            """Flush the target and propagate its original stream failure."""
            self.queue.join()
            if self._failure is not None:
                raise self._failure
            self._target_stream.flush()

        def shutdown(self) -> None:
            """Stop worker thread and flush remaining messages.

            Raises:
                TimeoutError: If the worker exceeds its shutdown deadline.
            """
            self.queue.shutdown()
            if self.thread.is_alive():
                self.thread.join(timeout=c.LOG_SHUTDOWN_JOIN_SECONDS)
            if self.thread.is_alive():
                msg = "Async log writer did not finish before its shutdown deadline"
                raise TimeoutError(msg)
            self.flush()

        @override
        def write(self, s: str, /) -> int:
            """Admit a message using the declared queue capacity policy.

            Returns:
                The resulting ``int``.

            """
            if self._failure is not None:
                raise self._failure
            failure: BaseException | None = None
            try:
                self.queue.put(s, block=c.ASYNC_BLOCK_ON_FULL)
            except queue.ShutDown as exc:
                failure = self._failure if self._failure is not None else exc
            if failure is not None:
                raise failure
            return len(s)

        def _write_queued_message(self, msg: str) -> None:
            target_stream = self._target_stream
            _ = target_stream.write(msg)
            _ = target_stream.flush()

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get()
                except queue.ShutDown:
                    break
                try:
                    self._write_queued_message(msg)
                finally:
                    failure = sys.exception()
                    if failure is not None:
                        self._failure = failure
                        self.queue.shutdown(immediate=True)
                    self.queue.task_done()

    _async_writer: ClassVar[_AsyncLogWriter | None] = None


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]

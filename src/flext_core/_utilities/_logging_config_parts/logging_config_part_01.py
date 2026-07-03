"""Structlog configuration and processor chain building.

Extracted from FlextLogger as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

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
from contextlib import suppress
from typing import ClassVar, override

import structlog

from flext_core import (
    FlextConstants as c,
    FlextProtocols as p,
    FlextTypes as t,
)


class FlextUtilitiesLoggingConfig:
    """Structlog configuration, async writer, and processor chain assembly."""

    _structlog_configured: ClassVar[bool]

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread."""

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._use_live_stdout = stream is sys.stdout
            self._use_live_stderr = stream is sys.stderr
            self._stream_mode: str = str(getattr(stream, "mode", "w"))
            self._stream_name: str = str(
                getattr(stream, "name", "<async-log-writer>"),
            )
            self._stream_encoding: str = str(
                getattr(stream, "encoding", c.DEFAULT_ENCODING),
            )
            self._stream_errors: str | None = getattr(stream, "errors", None)
            self._stream_newlines: str | t.VariadicTuple[str] | None = getattr(
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
        def _target_stream(self) -> typing.TextIO:
            if self._use_live_stdout:
                return sys.stdout
            if self._use_live_stderr:
                return sys.stderr
            return self.stream

        @property
        def _writer_log(self) -> p.Logger:
            """Logger for async log writer."""
            existing: p.Logger | None = getattr(self, "_writer_logger", None)
            if existing is not None:
                return existing
            created: p.Logger = structlog.get_logger(__name__)
            self._writer_logger = created
            return created

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
            buf: typing.BinaryIO | None = getattr(self._target_stream, "buffer", None)
            if buf is not None:
                return buf
            return io.BytesIO()

        @property
        def line_buffering(self) -> bool:
            """Return whether line buffering is enabled."""
            return bool(getattr(self._target_stream, "line_buffering", False))

        @override
        def flush(self) -> None:
            """Flush stream (best effort)."""
            flush_fn = getattr(self._target_stream, "flush", None)
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

        def _write_queued_message(self, msg: str) -> None:
            target_stream = self._target_stream
            _ = target_stream.write(msg)
            _ = target_stream.flush()
            self.queue.task_done()

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get(timeout=0.1)
                    if msg is None:
                        break
                    self._write_queued_message(msg)
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
                        _ = self._target_stream.write("Error in async log writer\n")

    _async_writer: ClassVar[_AsyncLogWriter | None] = None


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]

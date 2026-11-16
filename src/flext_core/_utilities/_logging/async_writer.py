"""Asynchronous logging writer lifecycle, admission and failure propagation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import atexit
import queue
import threading
from concurrent.futures import Future, wait
from typing import override

from flext_core import c, m, p, t
from flext_core._utilities._logging.stream import FlextUtilitiesLoggingStream


class FlextUtilitiesLoggingAsyncWriter(FlextUtilitiesLoggingStream.TextStream):
    """Real text stream whose writes are queued to a background thread."""

    def __init__(self, stream: t.TextStream) -> None:
        """Start the background writer over the caller-owned stream."""
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

    def drain(self) -> None:
        """Wait for submitted writes within the shutdown deadline.

        Raises:
            TimeoutError: If the stream does not accept them in time.

        """
        completion = self._admit(None)
        if not wait((completion,), timeout=c.LOG_SHUTDOWN_JOIN_SECONDS).done:
            msg = "Async log writer did not finish before shutdown timeout"
            raise TimeoutError(msg)
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
            except Exception as failure:  # ruff: ignore[blind-except] - intentional broad catch to deliver any sink failure to the caller Future
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


__all__: list[str] = ["FlextUtilitiesLoggingAsyncWriter"]

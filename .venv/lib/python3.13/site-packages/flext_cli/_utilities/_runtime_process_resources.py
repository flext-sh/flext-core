"""Pre-spawn and durable-resource ownership for streamed processes."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import BinaryIO

from flext_cli import c, p, r


class FlextCliUtilitiesRuntimeProcessResourcesMixin:
    """Prepare stdin/live descriptors and finalize durable resources."""

    @staticmethod
    def _prepare_streamed_stdin(
        stack: contextlib.ExitStack, input_data: str | bytes | None
    ) -> p.Result[tuple[BinaryIO | None, BinaryIO | None, bytes]]:
        if input_data is None:
            return r[tuple[BinaryIO | None, BinaryIO | None, bytes]].ok((
                None,
                None,
                b"",
            ))
        payload = (
            input_data.encode("utf-8") if isinstance(input_data, str) else input_data
        )
        try:
            reader, writer = (
                FlextCliUtilitiesRuntimeProcessResourcesMixin._anonymous_stdin_pipe(
                    stack
                )
            )
        except c.EXC_OS_VALUE as exc:
            return r[tuple[BinaryIO | None, BinaryIO | None, bytes]].fail(
                f"stdin preparation error: {exc}"
            )
        return r[tuple[BinaryIO | None, BinaryIO | None, bytes]].ok((
            reader,
            writer,
            payload,
        ))

    @staticmethod
    def _anonymous_stdin_pipe(stack: contextlib.ExitStack) -> tuple[BinaryIO, BinaryIO]:
        """Open one memory-only parent-writer/child-reader pipe pair."""
        read_fd, write_fd = os.pipe()
        try:
            reader = os.fdopen(read_fd, "rb", buffering=0)
        except c.EXC_OS_VALUE:
            os.close(read_fd)
            os.close(write_fd)
            raise
        try:
            writer = os.fdopen(write_fd, "wb", buffering=0)
        except c.EXC_OS_VALUE:
            reader.close()
            os.close(write_fd)
            raise
        stack.callback(writer.close)
        stack.callback(reader.close)
        return reader, writer

    @staticmethod
    def _prepare_live_descriptor(
        stack: contextlib.ExitStack, *, live: bool
    ) -> p.Result[tuple[int | None]]:
        if not live:
            return r[tuple[int | None]].ok((None,))
        try:
            live_fd = FlextCliUtilitiesRuntimeProcessResourcesMixin._open_live_fd(stack)
        except c.EXC_OS_VALUE as exc:
            return r[tuple[int | None]].fail(f"live output preparation error: {exc}")
        return r[tuple[int | None]].ok((live_fd,))

    @staticmethod
    def _open_live_fd(stack: contextlib.ExitStack) -> int:
        live_fd = os.dup(sys.stdout.fileno())
        stack.callback(os.close, live_fd)
        if os.name != "nt" or not os.isatty(live_fd):
            was_blocking = os.get_blocking(live_fd)
            os.set_blocking(live_fd, False)
            stack.callback(os.set_blocking, live_fd, was_blocking)
        return live_fd

    @staticmethod
    def _flush_durable_log(durable_log: BinaryIO) -> tuple[str, ...]:
        """Finalize the durable log after reaping; only I/O errors are failures.

        The flush runs post-reaping, after bounded cleanup has deliberately
        consumed its deadline window, so wall-clock position relative to the
        child deadline is not a failure condition.
        """
        errors: list[str] = []
        try:
            durable_log.flush()
            os.fsync(durable_log.fileno())
        except c.EXC_OS_VALUE as exc:
            errors.append(f"durable log flush error: {exc}")
        return tuple(errors)

    @staticmethod
    def _close_process_resources(stack: contextlib.ExitStack) -> tuple[str, ...]:
        try:
            stack.close()
        except c.EXC_OS_VALUE as exc:
            return (f"process resource close error: {exc}",)
        return ()

    @staticmethod
    def _spawn_deadline_exhausted(
        absolute_deadline: float | None, grace_seconds: float
    ) -> bool:
        return (
            absolute_deadline is not None
            and time.monotonic() >= absolute_deadline - grace_seconds
        )


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessResourcesMixin"]

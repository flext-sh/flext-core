"""Signal forwarding and deterministic streamed-process cleanup."""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import IO, BinaryIO

from flext_cli import p
from flext_cli._utilities._runtime_process_monitor import (
    FlextCliUtilitiesRuntimeProcessMonitorMixin,
)
from flext_cli._utilities._runtime_process_threads import (
    FlextCliUtilitiesRuntimeProcessThreadsMixin,
)


class FlextCliUtilitiesRuntimeProcessCleanupMixin(
    FlextCliUtilitiesRuntimeProcessMonitorMixin,
    FlextCliUtilitiesRuntimeProcessThreadsMixin,
):
    """Forward signals, kill descendants, reap root, and drain output."""

    @classmethod
    def _install_forwarding_handlers(
        cls,
        received_signals: list[int],
        forwarded_signals: list[int],
        wake: threading.Event,
    ) -> list[Callable[[], object]]:
        """Capture operator signals before opening the containment window."""
        restore_handlers: list[Callable[[], object]] = []

        def forward(signal_number: int, _frame: FrameType | None) -> None:
            received_signals.append(signal_number)
            wake.set()

        forwarded = (signal.SIGINT, signal.SIGTERM)
        if os.name != "nt" and hasattr(signal, "SIGHUP"):
            forwarded = (*forwarded, signal.SIGHUP)
        try:
            for signal_number in forwarded:
                previous = signal.getsignal(signal_number)
                signal.signal(signal_number, forward)
                forwarded_signals.append(int(signal_number))
                restore_handlers.append(
                    lambda number=int(signal_number), handler=previous: signal.signal(
                        number, handler
                    )
                )
        except (OSError, ValueError):
            for restore in reversed(restore_handlers):
                restore()
            raise
        return restore_handlers

    @staticmethod
    def _restore_forwarding_handlers(
        restore_handlers: list[Callable[[], object]],
    ) -> tuple[str, ...]:
        """Restore parent handlers after child lifecycle completion."""
        failures: list[str] = []
        for restore in reversed(restore_handlers):
            try:
                restore()
            except (OSError, ValueError) as exc:
                failures.append(f"signal handler restore failed: {exc}")
        return tuple(failures)

    @classmethod
    def _reap_and_drain(
        cls,
        process: p.Cli.ProcessHandle,
        waiter: threading.Thread,
        process_done: threading.Event,
        wake: threading.Event,
        stop: threading.Event,
        pump_streams: tuple[tuple[threading.Thread, IO[bytes]], ...],
        input_pump: tuple[threading.Thread, BinaryIO] | None,
        cleanup_errors: list[str],
        job_handle: int,
        absolute_deadline: float | None,
        return_codes: list[int],
    ) -> int | None:
        """Kill the owned boundary, reap root, drain output, and prove empty."""
        cleanup_deadline = (
            absolute_deadline
            if absolute_deadline is not None
            else time.monotonic() + 1.0
        )
        cls._empty_owned_boundary(
            process, process_done, wake, cleanup_errors, job_handle, cleanup_deadline
        )
        waiter.join(cls._remaining(cleanup_deadline))
        if waiter.is_alive():
            cleanup_errors.append("process deadline expired before root reaping")
        if input_pump is not None:
            cls._drain_input(
                input_pump[0], input_pump[1], cleanup_errors, cleanup_deadline
            )
        for pump, source in pump_streams:
            cls._drain_output(pump, stop, source, cleanup_errors, cleanup_deadline)
        return return_codes[0] if return_codes else process.poll()

    @classmethod
    def _drain_input(
        cls,
        pump: threading.Thread,
        sink: BinaryIO,
        cleanup_errors: list[str],
        cleanup_deadline: float,
    ) -> None:
        """Join the input writer after the child boundary has lost every reader."""
        pump.join(cls._remaining(cleanup_deadline))
        if pump.is_alive():
            try:
                sink.close()
            except (OSError, ValueError) as exc:
                cleanup_errors.append(f"process input close error: {exc}")
            pump.join(cls._remaining(cleanup_deadline))
        if pump.is_alive():
            cleanup_errors.append("process deadline expired before input drain")

    @classmethod
    def _empty_owned_boundary(
        cls,
        process: p.Cli.ProcessHandle,
        process_done: threading.Event,
        wake: threading.Event,
        cleanup_errors: list[str],
        job_handle: int,
        cleanup_deadline: float,
    ) -> None:
        boundary = cls._process_boundary_empty(process.pid, job_handle)
        if boundary.success and boundary.value:
            return
        cls._append_signal_error(
            cleanup_errors,
            cls._signal_process_tree(process, signal.SIGTERM, job_handle, force=False),
        )
        process_done.wait(min(0.1, cls._remaining(cleanup_deadline)))
        boundary = cls._process_boundary_empty(process.pid, job_handle)
        if boundary.success and not boundary.value:
            cls._append_signal_error(
                cleanup_errors,
                cls._signal_process_tree(
                    process, signal.SIGKILL, job_handle, force=True
                ),
            )
        while (
            boundary.success
            and not boundary.value
            and cls._remaining(cleanup_deadline) > 0
        ):
            wake.wait(min(0.02, cls._remaining(cleanup_deadline)))
            wake.clear()
            boundary = cls._process_boundary_empty(process.pid, job_handle)
        if boundary.failure:
            cleanup_errors.append(
                boundary.error or "owned process-boundary probe failed"
            )
        elif not boundary.value:
            cleanup_errors.append("owned process boundary was not empty before return")

    @classmethod
    def _drain_output(
        cls,
        pump: threading.Thread,
        stop: threading.Event,
        source: IO[bytes],
        cleanup_errors: list[str],
        cleanup_deadline: float,
    ) -> None:
        pump.join(cls._remaining(cleanup_deadline))
        if pump.is_alive():
            stop.set()
            try:
                source.close()
            except (OSError, ValueError) as exc:
                cleanup_errors.append(f"process output close error: {exc}")
            pump.join(cls._remaining(cleanup_deadline))
        if pump.is_alive():
            cleanup_errors.append("process deadline expired before output drain")

    @staticmethod
    def _append_signal_error(errors: list[str], error: str | None) -> None:
        if error is not None:
            errors.append(error)

    @staticmethod
    def _remaining(absolute_deadline: float) -> float:
        return max(0.0, absolute_deadline - time.monotonic())


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessCleanupMixin"]

"""Thread ownership for streamed process wait and output work."""

from __future__ import annotations

import threading
from typing import IO, BinaryIO

from flext_cli import p
from flext_cli._utilities._runtime_process_stream import (
    FlextCliUtilitiesRuntimeProcessStreamMixin,
)
from flext_cli._utilities._runtime_process_wait import (
    FlextCliUtilitiesRuntimeProcessWaitMixin,
)


class FlextCliUtilitiesRuntimeProcessThreadsMixin(
    FlextCliUtilitiesRuntimeProcessStreamMixin, FlextCliUtilitiesRuntimeProcessWaitMixin
):
    """Start bounded lifecycle threads at their canonical owner."""

    @classmethod
    def _start_input_pump(
        cls, sink: BinaryIO, payload: bytes, failures: list[str], wake: threading.Event
    ) -> threading.Thread:
        """Start the sole non-daemon writer for one anonymous stdin pipe."""
        pump = threading.Thread(
            target=cls._pump_process_input,
            args=(sink, payload, failures, wake),
            name="flext-cli-process-input",
            daemon=False,
        )
        pump.start()
        return pump

    @classmethod
    def _start_root_waiter(
        cls,
        process: p.Cli.ProcessHandle,
        return_codes: list[int],
        failures: list[str],
        process_done: threading.Event,
        wake: threading.Event,
    ) -> threading.Thread:
        waiter = threading.Thread(
            target=cls._wait_for_root_process,
            args=(process, return_codes, failures, process_done, wake),
            name="flext-cli-process-waiter",
            daemon=False,
        )
        waiter.start()
        return waiter

    @classmethod
    def _start_output_pump(
        cls,
        source: IO[bytes],
        durable_log: BinaryIO | None,
        captured_output: bytearray | None,
        live_fd: int | None,
        failures: list[str],
        live_diagnostics: list[str],
        stop: threading.Event,
        wake: threading.Event,
        *,
        thread_name: str,
    ) -> threading.Thread:
        pump = threading.Thread(
            target=cls._pump_process_output,
            args=(
                source,
                durable_log,
                captured_output,
                live_fd,
                failures,
                live_diagnostics,
                stop,
                wake,
            ),
            name=thread_name,
            daemon=False,
        )
        pump.start()
        return pump


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessThreadsMixin"]

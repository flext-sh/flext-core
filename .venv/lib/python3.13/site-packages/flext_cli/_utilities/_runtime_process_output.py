"""Output-pipe ownership for the canonical contained process lifecycle."""

from __future__ import annotations

import contextlib
import threading
from typing import IO, BinaryIO

from flext_cli import p
from flext_cli._utilities._runtime_process_threads import (
    FlextCliUtilitiesRuntimeProcessThreadsMixin,
)


class FlextCliUtilitiesRuntimeProcessOutputMixin(
    FlextCliUtilitiesRuntimeProcessThreadsMixin
):
    """Attach every requested child pipe to exactly one bounded pump."""

    @classmethod
    def _start_process_output(
        cls,
        process: p.Cli.ProcessHandle,
        stack: contextlib.ExitStack,
        durable_log: BinaryIO | None,
        live_fd: int | None,
        failures: list[str],
        live_diagnostics: list[str],
        stop: threading.Event,
        wake: threading.Event,
        stdout_output: bytearray,
        stderr_output: bytearray,
        *,
        capture_output: bool,
    ) -> tuple[tuple[threading.Thread, IO[bytes]], ...]:
        combine_output = durable_log is not None
        pipe_output = combine_output or capture_output
        pump_streams: list[tuple[threading.Thread, IO[bytes]]] = []
        stdout_source = process.stdout
        if pipe_output and stdout_source is None:
            failures.append("process stdout is not available")
        elif stdout_source is not None:
            stack.callback(stdout_source.close)
            stdout_pump = cls._start_output_pump(
                stdout_source,
                durable_log,
                stdout_output if capture_output else None,
                live_fd,
                failures,
                live_diagnostics,
                stop,
                wake,
                thread_name=(
                    "flext-cli-process-output"
                    if combine_output
                    else "flext-cli-process-stdout"
                ),
            )
            pump_streams.append((stdout_pump, stdout_source))
        stderr_source = process.stderr
        if capture_output and stderr_source is None:
            failures.append("process stderr is not available")
        elif stderr_source is not None:
            stack.callback(stderr_source.close)
            stderr_pump = cls._start_output_pump(
                stderr_source,
                None,
                stderr_output,
                None,
                failures,
                live_diagnostics,
                stop,
                wake,
                thread_name="flext-cli-process-stderr",
            )
            pump_streams.append((stderr_pump, stderr_source))
        return tuple(pump_streams)


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessOutputMixin"]

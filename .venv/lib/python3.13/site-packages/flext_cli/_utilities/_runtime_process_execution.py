"""Resource ownership for one portable streamed process lifecycle."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, BinaryIO

from flext_cli import c, p, r, t
from flext_cli._utilities._runtime_process_cleanup import (
    FlextCliUtilitiesRuntimeProcessCleanupMixin,
)
from flext_cli._utilities._runtime_process_outcome import (
    FlextCliUtilitiesRuntimeProcessOutcomeMixin,
)
from flext_cli._utilities._runtime_process_output import (
    FlextCliUtilitiesRuntimeProcessOutputMixin,
)
from flext_cli._utilities._runtime_process_resources import (
    FlextCliUtilitiesRuntimeProcessResourcesMixin,
)
from flext_cli._utilities._runtime_process_start import (
    FlextCliUtilitiesRuntimeProcessStartMixin,
)
from flext_cli._utilities._runtime_process_timing import (
    FlextCliUtilitiesRuntimeProcessTimingMixin,
)


class FlextCliUtilitiesRuntimeProcessExecutionMixin(
    FlextCliUtilitiesRuntimeProcessCleanupMixin,
    FlextCliUtilitiesRuntimeProcessOutcomeMixin,
    FlextCliUtilitiesRuntimeProcessOutputMixin,
    FlextCliUtilitiesRuntimeProcessResourcesMixin,
    FlextCliUtilitiesRuntimeProcessStartMixin,
    FlextCliUtilitiesRuntimeProcessTimingMixin,
):
    """Own one child process and its streaming resources."""

    @classmethod
    def _execute_streamed_process(
        cls,
        cmd: t.StrSequence,
        output_path: Path | None,
        cwd: t.Cli.TextPath | None,
        env: dict[str, str] | None,
        input_data: str | bytes | None,
        *,
        capture_output: bool,
        live: bool,
        timeout: int | None,
        deadline: p.Cli.ProcessDeadline | None,
    ) -> p.Result[p.Cli.CommandBytesOutput]:
        """Own resources and complete one streamed child lifecycle."""
        started = time.monotonic()
        timing_result = cls._resolve_process_timing(
            cmd,
            timeout,
            deadline,
            started,
            capture_output=capture_output,
            has_output_path=output_path is not None,
            live=live,
            on_main_thread=threading.current_thread() is threading.main_thread(),
        )
        if timing_result.failure:
            return r[p.Cli.CommandBytesOutput].fail(
                timing_result.error or "process deadline resolution failed"
            )
        absolute_deadline, grace_seconds, timeout_exit_code = timing_result.unwrap()
        process: p.Cli.ProcessHandle | None = None
        waiter: threading.Thread | None = None
        durable_log: BinaryIO | None = None
        job_handle = 0
        failures: list[str] = []
        cleanup_errors: list[str] = []
        live_diagnostics: list[str] = []
        restore_handlers: list[Callable[[], object]] = []
        forwarded_signals: list[int] = []
        received_signals: list[int] = []
        return_codes: list[int] = []
        stdout_output = bytearray()
        stderr_output = bytearray()
        pump_streams: list[tuple[threading.Thread, IO[bytes]]] = []
        input_pump: tuple[threading.Thread, BinaryIO] | None = None
        pump_stop = threading.Event()
        process_done = threading.Event()
        wake = threading.Event()
        stack = contextlib.ExitStack()
        return_code: int | None = None
        timed_out = False
        final_deadline = absolute_deadline
        cleanup_complete = False

        def execute_lifecycle() -> None:
            nonlocal \
                cleanup_complete, \
                durable_log, \
                final_deadline, \
                job_handle, \
                process, \
                return_code, \
                timed_out, \
                waiter, \
                input_pump
            if threading.current_thread() is threading.main_thread():
                restore_handlers.extend(
                    cls._install_forwarding_handlers(
                        received_signals, forwarded_signals, wake
                    )
                )
            prepared_cmd = tuple(cmd)
            if received_signals:
                wake.set()
                return
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                durable_log = stack.enter_context(output_path.open("wb", buffering=0))
            stdin_result = cls._prepare_streamed_stdin(stack, input_data)
            live_result = cls._prepare_live_descriptor(stack, live=live)
            if stdin_result.failure:
                failures.append(stdin_result.error or "stdin preparation failed")
            elif live_result.failure:
                failures.append(live_result.error or "live output preparation failed")
            elif received_signals:
                wake.set()
            elif cls._spawn_deadline_exhausted(absolute_deadline, grace_seconds):
                failures.append("process deadline exhausted before child spawn")
            else:
                combine_output = output_path is not None
                pipe_output = combine_output or capture_output
                start_result = cls._start_contained_process(
                    prepared_cmd,
                    cwd,
                    env,
                    stdin_result.value[0],
                    capture_output=pipe_output,
                    combine_output=combine_output,
                )
                if start_result.failure:
                    failures.append(start_result.error or "process start failed")
                else:
                    owned_process, job_handle = start_result.unwrap()
                    process = owned_process
                    stdin_reader, stdin_writer, stdin_payload = stdin_result.unwrap()
                    if stdin_reader is not None:
                        try:
                            stdin_reader.close()
                        except (OSError, ValueError) as exc:
                            failures.append(f"parent stdin reader close error: {exc}")
                    waiter = cls._start_root_waiter(
                        owned_process, return_codes, failures, process_done, wake
                    )
                    pump_streams.extend(
                        cls._start_process_output(
                            owned_process,
                            stack,
                            durable_log,
                            live_result.value[0],
                            failures,
                            live_diagnostics,
                            pump_stop,
                            wake,
                            stdout_output,
                            stderr_output,
                            capture_output=capture_output,
                        )
                    )
                    if stdin_writer is not None:
                        input_thread = cls._start_input_pump(
                            stdin_writer, stdin_payload, failures, wake
                        )
                        input_pump = (input_thread, stdin_writer)
                    timed_out, final_deadline = cls._monitor_process(
                        owned_process,
                        process_done,
                        wake,
                        failures,
                        received_signals,
                        job_handle,
                        absolute_deadline,
                        grace_seconds,
                    )
                    return_code = cls._reap_and_drain(
                        owned_process,
                        waiter,
                        process_done,
                        wake,
                        pump_stop,
                        tuple(pump_streams),
                        input_pump,
                        cleanup_errors,
                        job_handle,
                        final_deadline,
                        return_codes,
                    )
                    cleanup_complete = True

        try:
            execute_lifecycle()
        except c.EXC_OS_VALUE as exc:
            failures.append(f"execution error: {exc}")
        finally:
            if process is not None and waiter is not None and not cleanup_complete:
                return_code = cls._reap_and_drain(
                    process,
                    waiter,
                    process_done,
                    wake,
                    pump_stop,
                    tuple(pump_streams),
                    input_pump,
                    cleanup_errors,
                    job_handle,
                    final_deadline,
                    return_codes,
                )
            if durable_log is not None:
                cleanup_errors.extend(cls._flush_durable_log(durable_log))
            close_error = cls._windows_job_close(job_handle)
            if close_error is not None:
                cleanup_errors.append(close_error)
            cleanup_errors.extend(cls._close_process_resources(stack))
            cleanup_errors.extend(cls._restore_forwarding_handlers(restore_handlers))
        return cls._captured_process_result(
            cmd,
            return_code,
            received_signals,
            (*failures, *cleanup_errors),
            stdout_output,
            stderr_output,
            max(0.0, time.monotonic() - started),
            timed_out=timed_out,
            timeout_seconds=timeout,
            timeout_exit_code=timeout_exit_code,
        )


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessExecutionMixin"]

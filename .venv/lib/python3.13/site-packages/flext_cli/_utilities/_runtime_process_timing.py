"""Deadline normalization for the canonical contained process lifecycle."""

from __future__ import annotations

import shlex

from flext_cli import p, r, t


class FlextCliUtilitiesRuntimeProcessTimingMixin:
    """Resolve relative and absolute deadlines through one policy owner."""

    @staticmethod
    def _resolve_process_timing(
        cmd: t.StrSequence,
        timeout: int | None,
        deadline: p.Cli.ProcessDeadline | None,
        started: float,
        *,
        capture_output: bool,
        has_output_path: bool,
        live: bool,
        on_main_thread: bool,
    ) -> p.Result[tuple[float | None, float, int]]:
        if timeout is not None and deadline is not None:
            return r[tuple[float | None, float, int]].fail(
                "timeout and deadline are mutually exclusive"
            )
        if live and not has_output_path:
            return r[tuple[float | None, float, int]].fail(
                "live output requires a durable output path"
            )
        if capture_output and has_output_path:
            return r[tuple[float | None, float, int]].fail(
                "captured and durable output are mutually exclusive"
            )
        if (live or deadline is not None) and not on_main_thread:
            return r[tuple[float | None, float, int]].fail(
                "live/deadline process execution requires the main interpreter thread"
            )
        absolute_deadline: float | None = None
        grace_seconds = 0.0
        timeout_exit_code = 124
        if deadline is not None:
            absolute_deadline = deadline.expires_at_monotonic
            grace_seconds = deadline.termination_grace_seconds
            timeout_exit_code = deadline.timeout_exit_code
        elif timeout is not None:
            if timeout <= 0:
                return r[tuple[float | None, float, int]].fail(
                    f"timeout {timeout}s: {shlex.join(list(cmd))}"
                )
            absolute_deadline = started + timeout
            grace_seconds = min(max(timeout * 0.1, 0.05), timeout * 0.5)
        if absolute_deadline is not None:
            remaining = absolute_deadline - started
            if remaining <= 0 or grace_seconds <= 0 or grace_seconds >= remaining:
                return r[tuple[float | None, float, int]].fail(
                    "process deadline must leave a positive grace reserve"
                )
        return r[tuple[float | None, float, int]].ok((
            absolute_deadline,
            grace_seconds,
            timeout_exit_code,
        ))


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessTimingMixin"]

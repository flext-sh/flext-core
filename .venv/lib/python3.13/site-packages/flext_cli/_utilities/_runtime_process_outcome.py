"""Typed exit normalization for streamed process execution."""

from __future__ import annotations

import shlex

from flext_cli import m, p, r, t


class FlextCliUtilitiesRuntimeProcessOutcomeMixin:
    """Map one completed lifecycle to its public result contract."""

    @staticmethod
    def _process_exit_result(
        cmd: t.StrSequence,
        return_code: int | None,
        received_signals: list[int],
        diagnostics: tuple[str, ...],
        *,
        nonfatal_diagnostics: tuple[str, ...] = (),
        timed_out: bool,
        legacy_timeout: bool,
        legacy_timeout_seconds: int | None,
        timeout_exit_code: int,
    ) -> p.Result[int]:
        """Preserve a primary exit while surfacing additive diagnostics."""
        _ = nonfatal_diagnostics
        if legacy_timeout and timed_out:
            failure = f"timeout {legacy_timeout_seconds}s: {shlex.join(list(cmd))}"
            if diagnostics:
                failure = f"{failure}; {'; '.join(diagnostics)}"
            return r[int].fail(failure)
        if received_signals:
            primary_exit = 128 + abs(received_signals[0])
        elif timed_out:
            primary_exit = timeout_exit_code
        elif return_code is None:
            primary_exit = None
        else:
            primary_exit = 128 + abs(return_code) if return_code < 0 else return_code
        if diagnostics:
            return r[int].fail("; ".join(diagnostics))
        if primary_exit is None:
            return r[int].fail("root process did not expose an exit status")
        return r[int].ok(primary_exit)

    @classmethod
    def _captured_process_result(
        cls,
        cmd: t.StrSequence,
        return_code: int | None,
        received_signals: list[int],
        diagnostics: tuple[str, ...],
        stdout_output: bytearray,
        stderr_output: bytearray,
        duration: float,
        *,
        timed_out: bool,
        timeout_seconds: int | None,
        timeout_exit_code: int,
    ) -> p.Result[p.Cli.CommandBytesOutput]:
        """Attach captured bytes only after the owned process boundary is empty."""
        return cls._process_exit_result(
            cmd,
            return_code,
            received_signals,
            diagnostics,
            timed_out=timed_out,
            legacy_timeout=timeout_seconds is not None,
            legacy_timeout_seconds=timeout_seconds,
            timeout_exit_code=timeout_exit_code,
        ).map(
            lambda exit_code: m.Cli.CommandBytesOutput(
                stdout=bytes(stdout_output),
                stderr=bytes(stderr_output),
                exit_code=exit_code,
                duration=duration,
            )
        )


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessOutcomeMixin"]

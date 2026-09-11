"""Blocking root-process reaping isolated from lifecycle coordination."""

from __future__ import annotations

import threading

from flext_cli import c, p


class FlextCliUtilitiesRuntimeProcessWaitMixin:
    """Reap a root process once and wake event-driven lifecycle monitoring."""

    @staticmethod
    def _wait_for_root_process(
        process: p.Cli.ProcessHandle,
        return_codes: list[int],
        failures: list[str],
        process_done: threading.Event,
        wake: threading.Event,
    ) -> None:
        try:
            return_codes.append(process.wait())
        except c.EXC_OS_VALUE as exc:
            failures.append(f"root process wait error: {exc}")
        finally:
            process_done.set()
            wake.set()


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessWaitMixin"]

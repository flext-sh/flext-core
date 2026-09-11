"""Canonical process-exit classification utilities."""

from __future__ import annotations

from flext_infra import c


class FlextInfraUtilitiesProcess:
    """Normalize external process exits into stable diagnostic classifications."""

    @staticmethod
    def process_exit_classification(exit_code: int) -> str:
        """Classify a process exit without discarding its original status."""
        if exit_code == c.Infra.PROCESS_TIMEOUT_EXIT_CODE:
            return "timeout"
        if exit_code < 0:
            return f"signal={-exit_code}"
        if exit_code > c.Infra.PROCESS_SIGNAL_EXIT_OFFSET:
            return f"signal={exit_code - c.Infra.PROCESS_SIGNAL_EXIT_OFFSET}"
        return "failure"


__all__: list[str] = ["FlextInfraUtilitiesProcess"]

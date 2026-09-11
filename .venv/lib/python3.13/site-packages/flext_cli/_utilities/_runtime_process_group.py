"""Portable process-group lifecycle primitives for ``u.Cli``."""

from __future__ import annotations

import os
import signal

from flext_cli import p, r
from flext_cli._utilities._runtime_windows_job_start import (
    FlextCliUtilitiesRuntimeWindowsJobStartMixin,
)
from flext_cli._utilities._runtime_windows_job_state import (
    FlextCliUtilitiesRuntimeWindowsJobStateMixin,
)


class FlextCliUtilitiesRuntimeProcessGroupMixin(
    FlextCliUtilitiesRuntimeWindowsJobStartMixin,
    FlextCliUtilitiesRuntimeWindowsJobStateMixin,
):
    """Own POSIX process groups and Windows kill-on-close Job Objects."""

    @classmethod
    def _process_boundary_empty(
        cls, process_group_id: int, job_handle: int
    ) -> p.Result[bool]:
        """Prove the owned group/Job has no active members."""
        if os.name == "nt":
            return cls.windows_job_active_count(job_handle).map(
                lambda active_count: active_count == 0
            )
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return r[bool].ok(True)
        except PermissionError as exc:
            return r[bool].fail(f"process-group probe failed: {exc}")
        return r[bool].ok(False)

    @classmethod
    def _signal_process_tree(
        cls,
        process: p.Cli.ProcessHandle,
        signal_number: int,
        job_handle: int,
        *,
        force: bool,
    ) -> str | None:
        """Signal the complete owned process tree."""
        try:
            if os.name == "nt":
                if not force and signal_number == signal.SIGINT:
                    process.send_signal(
                        int(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
                    )
                    return None
                return cls._windows_job_terminate(job_handle, 128 + abs(signal_number))
            os.killpg(process.pid, signal.SIGKILL if force else signal_number)
        except ProcessLookupError:
            return None
        except (OSError, ValueError) as exc:
            return f"process-tree signal error: {exc}"
        return None


__all__: list[str] = ["FlextCliUtilitiesRuntimeProcessGroupMixin"]

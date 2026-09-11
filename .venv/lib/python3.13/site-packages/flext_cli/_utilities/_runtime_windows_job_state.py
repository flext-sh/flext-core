"""Windows Job Object state, termination, and handle cleanup."""

from __future__ import annotations

import os

from flext_cli import p, r


class FlextCliUtilitiesRuntimeWindowsJobStateMixin:
    """Query and close one owned Windows Job Object."""

    @classmethod
    def windows_job_active_count(cls, job_handle: int) -> p.Result[int]:
        """Return active members of one owned Windows Job Object."""
        if os.name != "nt" or job_handle == 0:
            return r[int].ok(0)
        try:
            return cls._windows_job_active_count_native(job_handle)
        except (OSError, TypeError, ValueError) as exc:
            return r[int].fail(f"Windows Job Object query error: {exc}")

    @staticmethod
    def _windows_job_active_count_native(job_handle: int) -> p.Result[int]:
        import ctypes
        from ctypes import wintypes

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = getattr(ctypes, "WinDLL", ctypes.CDLL)(
            "kernel32", use_last_error=True
        )
        query_job = kernel32.QueryInformationJobObject
        query_job.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        )
        query_job.restype = wintypes.BOOL
        info = _BasicAccountingInformation()
        if not query_job(job_handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return r[int].fail(f"QueryInformationJobObject failed: {error}")
        return r[int].ok(int(info.ActiveProcesses))

    @classmethod
    def _windows_job_terminate(cls, job_handle: int, exit_code: int) -> str | None:
        """Terminate every process in a Windows Job Object."""
        if os.name != "nt" or job_handle == 0:
            return None
        try:
            return cls._windows_job_terminate_native(job_handle, exit_code)
        except (OSError, TypeError, ValueError) as exc:
            return f"Windows Job Object termination error: {exc}"

    @staticmethod
    def _windows_job_terminate_native(job_handle: int, exit_code: int) -> str | None:
        import ctypes
        from ctypes import wintypes

        kernel32 = getattr(ctypes, "WinDLL", ctypes.CDLL)(
            "kernel32", use_last_error=True
        )
        terminate_job = kernel32.TerminateJobObject
        terminate_job.argtypes = (wintypes.HANDLE, wintypes.UINT)
        terminate_job.restype = wintypes.BOOL
        if not terminate_job(job_handle, exit_code):
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return f"TerminateJobObject failed: {error}"
        return None

    @classmethod
    def _windows_job_close(cls, job_handle: int) -> str | None:
        """Close a Windows Job Object, enforcing kill-on-close."""
        if os.name != "nt" or job_handle == 0:
            return None
        try:
            return cls._windows_job_close_native(job_handle)
        except (OSError, TypeError, ValueError) as exc:
            return f"Windows Job Object close error: {exc}"

    @staticmethod
    def _windows_job_close_native(job_handle: int) -> str | None:
        import ctypes
        from ctypes import wintypes

        kernel32 = getattr(ctypes, "WinDLL", ctypes.CDLL)(
            "kernel32", use_last_error=True
        )
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        if not close_handle(job_handle):
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return f"CloseHandle failed: {error}"
        return None


__all__: list[str] = ["FlextCliUtilitiesRuntimeWindowsJobStateMixin"]

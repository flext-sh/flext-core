"""Windows Job Object creation and suspended-process startup."""

from __future__ import annotations

import os

from flext_cli import p, r


class FlextCliUtilitiesRuntimeWindowsJobStartMixin:
    """Contain a suspended Windows process before any child code executes."""

    @classmethod
    def _windows_job_create(cls, process_id: int) -> p.Result[int]:
        """Assign a suspended Windows process to a kill-on-close Job Object."""
        if os.name != "nt":
            return r[int].ok(0)
        try:
            return cls._windows_job_create_native(process_id)
        except (OSError, TypeError, ValueError) as exc:
            return r[int].fail(f"Windows Job Object error: {exc}")

    @staticmethod
    def _windows_job_create_native(process_id: int) -> p.Result[int]:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = getattr(ctypes, "WinDLL", ctypes.CDLL)(
            "kernel32", use_last_error=True
        )
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        create_job.restype = wintypes.HANDLE
        set_job = kernel32.SetInformationJobObject
        set_job.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_job.restype = wintypes.BOOL
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        assign_process = kernel32.AssignProcessToJobObject
        assign_process.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        assign_process.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        job_handle = create_job(None, None)
        if not job_handle:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return r[int].fail(f"CreateJobObjectW failed: {error}")
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000
        if not set_job(job_handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            close_handle(job_handle)
            return r[int].fail(f"SetInformationJobObject failed: {error}")
        process_handle = open_process(0x0001 | 0x0100, False, process_id)
        if not process_handle:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            close_handle(job_handle)
            return r[int].fail(f"OpenProcess failed: {error}")
        assigned = assign_process(job_handle, process_handle)
        close_handle(process_handle)
        if not assigned:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            close_handle(job_handle)
            return r[int].fail(f"AssignProcessToJobObject failed: {error}")
        return r[int].ok(int(job_handle))

    @classmethod
    def _windows_process_resume(cls, process_id: int) -> str | None:
        """Resume the initial thread only after Job assignment succeeds."""
        if os.name != "nt":
            return None
        try:
            return cls._windows_process_resume_native(process_id)
        except (OSError, TypeError, ValueError) as exc:
            return f"Windows process resume error: {exc}"

    @staticmethod
    def _windows_process_resume_native(process_id: int) -> str | None:
        import ctypes
        from ctypes import wintypes

        class _ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        kernel32 = getattr(ctypes, "WinDLL", ctypes.CDLL)(
            "kernel32", use_last_error=True
        )
        create_snapshot = kernel32.CreateToolhelp32Snapshot
        create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        create_snapshot.restype = wintypes.HANDLE
        thread_first = kernel32.Thread32First
        thread_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
        thread_first.restype = wintypes.BOOL
        thread_next = kernel32.Thread32Next
        thread_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
        thread_next.restype = wintypes.BOOL
        open_thread = kernel32.OpenThread
        open_thread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_thread.restype = wintypes.HANDLE
        resume_thread = kernel32.ResumeThread
        resume_thread.argtypes = (wintypes.HANDLE,)
        resume_thread.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        snapshot = create_snapshot(0x00000004, 0)
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return f"CreateToolhelp32Snapshot failed: {error}"
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        thread_id = 0
        try:
            found = bool(thread_first(snapshot, ctypes.byref(entry)))
            while found:
                if entry.th32OwnerProcessID == process_id:
                    thread_id = int(entry.th32ThreadID)
                    break
                found = bool(thread_next(snapshot, ctypes.byref(entry)))
        finally:
            close_handle(snapshot)
        if thread_id == 0:
            return f"suspended process {process_id} has no initial thread"
        thread_handle = open_thread(0x0002, False, thread_id)
        if not thread_handle:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return f"OpenThread failed: {error}"
        try:
            prior_suspend_count = resume_thread(thread_handle)
        finally:
            close_handle(thread_handle)
        resume_failed = 0xFFFFFFFF
        if prior_suspend_count == resume_failed:
            error = int(getattr(ctypes, "get_last_error", ctypes.get_errno)())
            return f"ResumeThread failed: {error}"
        return None


__all__: list[str] = ["FlextCliUtilitiesRuntimeWindowsJobStartMixin"]

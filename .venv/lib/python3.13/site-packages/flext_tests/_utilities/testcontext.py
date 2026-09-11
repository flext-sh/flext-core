"""Extracted mixin for flext_tests."""

from __future__ import annotations

import fcntl
import types
from pathlib import Path
from typing import TextIO


class FlextTestsTestContextUtilitiesMixin:
    """Context managers for tests."""

    class FileLock:
        """File-based exclusive lock for pytest-xdist parallel test isolation.

        Centralized SSOT for the LDAP-family / pytest-xdist coordination
        pattern (previously cloned across ``flext-ldap``, ``flext-ldif``,
        ``flext-tap-ldap``).
        """

        def __init__(self, lock_file: Path) -> None:
            self.lock_file = lock_file
            self._fd: int | None = None
            self._file_obj: TextIO | None = None

        def __enter__(self) -> None:
            """Acquire exclusive file lock."""
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_obj = self.lock_file.open("a")
            self._fd = self._file_obj.fileno()
            fcntl.flock(self._fd, fcntl.LOCK_EX)

        # mro-j47u (codex): these dunder arguments are contract-only.
        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Release the lock while preserving its shared inode."""
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            if self._file_obj is not None:
                self._file_obj.close()

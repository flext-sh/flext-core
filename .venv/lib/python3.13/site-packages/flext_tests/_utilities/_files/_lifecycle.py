"""Lifecycle helpers for FlextTestsFiles.

Covers initialization, context-manager entry/exit, cleanup tracking, and
base-directory resolution.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self

from flext_tests import c, t, u


class FlextTestsFilesLifecycleMixin:
    """File-manager lifecycle: init, context manager, cleanup, base_dir."""

    base_dir: Path | None
    _created_files: list[Path] = u.PrivateAttr(default_factory=list)
    _created_dirs: list[Path] = u.PrivateAttr(default_factory=list)

    @classmethod
    def _create_file_manager(cls, base_dir: Path | None) -> Self:
        """Construct the concrete validated file manager."""
        raise NotImplementedError

    def _initialize_file_lifecycle(self) -> None:
        """Initialize file-manager lifecycle state."""
        self._created_files = list[Path]()
        self._created_dirs = list[Path]()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    @property
    def created_dirs(self) -> t.SequenceOf[Path]:
        """The list of created directories."""
        return self._created_dirs

    @property
    def created_files(self) -> t.SequenceOf[Path]:
        """The list of created files."""
        return self._created_files

    def cleanup(self) -> None:
        """Cleanup all tracked files and directories created by this manager."""
        for path in reversed(self._created_files):
            if not path.exists():
                continue
            try:
                path.chmod(c.Tests.PERMISSION_WRITABLE_FILE)
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._created_files.clear()
        for path in reversed(self._created_dirs):
            if not path.exists():
                continue
            try:
                path.chmod(c.Tests.PERMISSION_WRITABLE_DIR)
                shutil.rmtree(path)
            except OSError:
                pass
        self._created_dirs.clear()

    def _resolve_directory(self, directory: Path | None) -> Path:
        """Resolve target directory for file creation."""
        target_dir = directory or self.base_dir
        if target_dir is not None:
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir
        temp_dir = Path(tempfile.mkdtemp())
        self._created_dirs.append(temp_dir)
        return temp_dir


__all__: list[str] = ["FlextTestsFilesLifecycleMixin"]

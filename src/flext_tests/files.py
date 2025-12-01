"""File management utilities for FLEXT ecosystem tests.

Provides generic file operations for testing across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self

from flext_core._models.entity import FlextModelsEntity


class FlextTestsFileManager:
    """Manages test files for FLEXT ecosystem testing.

    Provides generic file creation, management and cleanup operations
    for tests. Projects can extend this with domain-specific methods.

    Example:
        with FlextTestsFileManager() as manager:
            file_path = manager.create_text_file("test content", "test.txt")
            # file is automatically cleaned up after the block

    """

    class FileInfo(FlextModelsEntity.Value):
        """File information model for test files."""

        exists: bool
        size: int = 0
        lines: int = 0
        encoding: str = "utf-8"
        is_empty: bool = False
        first_line: str = ""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize file manager with optional base directory.

        Args:
            base_dir: Optional base directory for file operations.
                     If not provided, temporary directories are used.

        """
        self.base_dir = base_dir
        self.created_files: list[Path] = []
        self.created_dirs: list[Path] = []

    @contextmanager
    def temporary_directory(self) -> Generator[Path]:
        """Create and manage a temporary directory.

        Yields:
            Path to temporary directory that is automatically cleaned up.

        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yield temp_path

    def _create_file(
        self,
        file_path: Path,
        content: str | bytes,
        encoding: str | None = None,
    ) -> Path:
        """Internal method to create a file with content.

        Args:
            file_path: Path where file should be created.
            content: File content (str or bytes).
            encoding: Encoding for text files (None for binary).

        Returns:
            Path to the created file.

        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            file_path.write_text(content, encoding=encoding or "utf-8")
        else:
            file_path.write_bytes(content)

        self.created_files.append(file_path)
        return file_path

    def create_text_file(
        self,
        content: str,
        filename: str = "test.txt",
        directory: Path | None = None,
        encoding: str = "utf-8",
    ) -> Path:
        """Create text file with given content.

        Args:
            content: Text content to write to the file.
            filename: Name for the file (default: test.txt).
            directory: Directory to create file in.
                      If None, uses base_dir or creates temporary directory.
            encoding: File encoding (default: utf-8).

        Returns:
            Path to the created file.

        """
        target_dir = self._resolve_directory(directory)
        file_path = target_dir / filename
        return self._create_file(file_path, content, encoding=encoding)

    def create_binary_file(
        self,
        content: bytes,
        filename: str = "binary_data.bin",
        directory: Path | None = None,
    ) -> Path:
        """Create binary file with given content.

        Args:
            content: Binary content to write to the file.
            filename: Name for the file (default: binary_data.bin).
            directory: Directory to create file in.

        Returns:
            Path to the created file.

        """
        target_dir = self._resolve_directory(directory)
        file_path = target_dir / filename
        return self._create_file(file_path, content)

    def create_empty_file(
        self,
        filename: str = "empty.txt",
        directory: Path | None = None,
    ) -> Path:
        """Create empty file for edge case testing.

        Args:
            filename: Name for the file (default: empty.txt).
            directory: Directory to create file in.

        Returns:
            Path to the created empty file.

        """
        return self.create_text_file("", filename, directory)

    def create_file_set(
        self,
        files: dict[str, str],
        directory: Path | None = None,
        extension: str = ".txt",
    ) -> dict[str, Path]:
        """Create multiple files from content dictionary.

        Args:
            files: Dictionary mapping names to content.
            directory: Directory to create files in.
            extension: Extension to add if not present (default: .txt).

        Returns:
            Dictionary mapping names to created file paths.

        """
        target_dir = self._resolve_directory(directory)
        created: dict[str, Path] = {}

        for name, content in files.items():
            filename = name if "." in name else f"{name}{extension}"
            file_path = self.create_text_file(content, filename, target_dir)
            created[name] = file_path

        return created

    def get_file_info(self, file_path: Path) -> FlextTestsFileManager.FileInfo:
        """Get information about a test file.

        Args:
            file_path: Path to the file.

        Returns:
            FileInfo model with file information.

        """
        if not file_path.exists():
            return self.FileInfo(exists=False)

        stat = file_path.stat()
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1 if content else 0

        return self.FileInfo(
            exists=True,
            size=stat.st_size,
            lines=lines,
            encoding="utf-8",
            is_empty=len(content.strip()) == 0,
            first_line=content.split("\n")[0] if content else "",
        )

    def cleanup(self) -> None:
        """Clean up all created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()

        for dir_path in self.created_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)

        self.created_files.clear()
        self.created_dirs.clear()

    def _resolve_directory(self, directory: Path | None) -> Path:
        """Resolve target directory for file creation.

        Args:
            directory: Optional directory to use.

        Returns:
            Resolved directory path.

        """
        if directory is not None:
            return directory

        if self.base_dir is not None:
            return self.base_dir

        temp_dir = Path(tempfile.mkdtemp())
        self.created_dirs.append(temp_dir)
        return temp_dir

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

    @classmethod
    @contextmanager
    def temporary_files(
        cls,
        files: dict[str, str],
        extension: str = ".txt",
    ) -> Generator[dict[str, Path]]:
        """Context manager for creating temporary files.

        Args:
            files: Dictionary mapping names to content.
            extension: Extension to add if not present.

        Yields:
            Dictionary mapping names to created file paths.

        """
        with cls() as manager:
            created_files = manager.create_file_set(files, extension=extension)
            yield created_files

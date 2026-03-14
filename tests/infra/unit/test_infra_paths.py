"""Tests for FlextInfraPathResolver.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesPaths
from flext_tests import tm


class TestFlextInfraPathResolver:
    """Test suite for FlextInfraPathResolver."""

    def test_workspace_root_with_current_directory(self) -> None:
        """Test resolving workspace root from current directory."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root(".")
        tm.ok(result)
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_workspace_root_with_absolute_path(self, tmp_path: Path) -> None:
        """Test resolving workspace root with absolute path."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root(str(tmp_path))
        tm.ok(result)
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_workspace_root_with_path_object(self, tmp_path: Path) -> None:
        """Test resolving workspace root with Path object."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root(tmp_path)
        tm.ok(result)
        assert isinstance(result.value, Path)
        assert result.value == tmp_path.resolve()

    def test_workspace_root_from_file_in_workspace(self) -> None:
        """Test resolving workspace root from a file in the workspace."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root_from_file(__file__)
        tm.ok(result)
        root = result.value
        assert root.is_absolute()
        assert (root / ".git").exists() or (root / "Makefile").exists()

    def test_workspace_root_from_file_with_path_object(self) -> None:
        """Test workspace root resolution with Path object."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root_from_file(Path(__file__))
        tm.ok(result)
        assert isinstance(result.value, Path)

    def test_workspace_root_from_file_not_found(self, tmp_path: Path) -> None:
        """Test workspace root resolution fails when markers not found."""
        resolver = FlextInfraUtilitiesPaths()
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        result = resolver.workspace_root_from_file(test_file)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "workspace root not found" in result.error

    def test_workspace_root_from_directory_file(self, tmp_path: Path) -> None:
        """Test workspace root resolution from a directory path."""
        resolver = FlextInfraUtilitiesPaths()
        (tmp_path / ".git").mkdir()
        (tmp_path / "Makefile").touch()
        (tmp_path / "pyproject.toml").touch()
        result = resolver.workspace_root_from_file(tmp_path)
        tm.ok(result)
        assert result.value == tmp_path

    def test_workspace_root_from_nested_file(self, tmp_path: Path) -> None:
        """Test workspace root resolution from nested file."""
        resolver = FlextInfraUtilitiesPaths()
        (tmp_path / ".git").mkdir()
        (tmp_path / "Makefile").touch()
        (tmp_path / "pyproject.toml").touch()
        nested_dir = tmp_path / "src" / "module"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "test.py"
        nested_file.write_text("# test")
        result = resolver.workspace_root_from_file(nested_file)
        tm.ok(result)
        assert result.value == tmp_path

    def test_workspace_root_invalid_path(self) -> None:
        """Test workspace root resolution with invalid path."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root("/nonexistent/path/that/does/not/exist")
        tm.ok(result)

    def test_workspace_root_from_file_nonexistent(self) -> None:
        """Test workspace root resolution with nonexistent file."""
        resolver = FlextInfraUtilitiesPaths()
        result = resolver.workspace_root_from_file(
            Path("/nonexistent/impossible/file.py"),
        )
        tm.fail(result)

    def test_workspace_root_with_resolve_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        resolver = FlextInfraUtilitiesPaths()

        def _raise_type_error(self: Path, strict: bool | None = None) -> Path:
            _ = self
            _ = strict
            msg = "invalid path type"
            raise TypeError(msg)

        monkeypatch.setattr(Path, "resolve", _raise_type_error)

        result = resolver.workspace_root(".")
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "failed to resolve" in result.error.lower()

    def test_workspace_root_from_file_with_resolve_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        resolver = FlextInfraUtilitiesPaths()

        def _raise_type_error(self: Path, strict: bool | None = None) -> Path:
            _ = self
            _ = strict
            msg = "invalid path type"
            raise TypeError(msg)

        monkeypatch.setattr(Path, "resolve", _raise_type_error)

        result = resolver.workspace_root_from_file(__file__)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "failed to resolve" in result.error.lower()

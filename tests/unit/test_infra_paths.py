"""Tests for FlextInfraPathResolver.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraPathResolver


class TestFlextInfraPathResolver:
    """Test suite for FlextInfraPathResolver."""

    def test_workspace_root_with_current_directory(self) -> None:
        """Test resolving workspace root from current directory."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root(".")

        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_workspace_root_with_absolute_path(self, tmp_path: Path) -> None:
        """Test resolving workspace root with absolute path."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root(str(tmp_path))

        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value.is_absolute()

    def test_workspace_root_with_path_object(self, tmp_path: Path) -> None:
        """Test resolving workspace root with Path object."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root(tmp_path)

        assert result.is_success
        assert isinstance(result.value, Path)
        assert result.value == tmp_path.resolve()

    def test_workspace_root_from_file_in_workspace(self) -> None:
        """Test resolving workspace root from a file in the workspace."""
        resolver = FlextInfraPathResolver()
        # Use this test file as reference
        result = resolver.workspace_root_from_file(__file__)

        assert result.is_success
        root = result.value
        assert root.is_absolute()
        # Should find markers like .git, Makefile, pyproject.toml
        assert (root / ".git").exists() or (root / "Makefile").exists()

    def test_workspace_root_from_file_with_path_object(self) -> None:
        """Test workspace root resolution with Path object."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root_from_file(Path(__file__))

        assert result.is_success
        assert isinstance(result.value, Path)

    def test_workspace_root_from_file_not_found(self, tmp_path: Path) -> None:
        """Test workspace root resolution fails when markers not found."""
        resolver = FlextInfraPathResolver()
        # Create a temporary file in a directory without workspace markers
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        result = resolver.workspace_root_from_file(test_file)

        assert result.is_failure
        assert "workspace root not found" in result.error

    def test_workspace_root_from_directory_file(self, tmp_path: Path) -> None:
        """Test workspace root resolution from a directory path."""
        resolver = FlextInfraPathResolver()
        # Create workspace markers
        (tmp_path / ".git").mkdir()
        (tmp_path / "Makefile").touch()
        (tmp_path / "pyproject.toml").touch()

        result = resolver.workspace_root_from_file(tmp_path)

        assert result.is_success
        assert result.value == tmp_path

    def test_workspace_root_from_nested_file(self, tmp_path: Path) -> None:
        """Test workspace root resolution from nested file."""
        resolver = FlextInfraPathResolver()
        # Create workspace markers at root
        (tmp_path / ".git").mkdir()
        (tmp_path / "Makefile").touch()
        (tmp_path / "pyproject.toml").touch()

        # Create nested file
        nested_dir = tmp_path / "src" / "module"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "test.py"
        nested_file.write_text("# test")

        result = resolver.workspace_root_from_file(nested_file)

        assert result.is_success
        assert result.value == tmp_path

    def test_workspace_root_invalid_path(self) -> None:
        """Test workspace root resolution with invalid path."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root("/nonexistent/path/that/does/not/exist")

        # Should still succeed (Path.resolve() doesn't validate existence)
        assert result.is_success

    def test_workspace_root_from_file_nonexistent(self) -> None:
        """Test workspace root resolution with nonexistent file."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root_from_file(
            Path("/nonexistent/impossible/file.py")
        )

        assert result.is_failure

    def test_workspace_root_with_invalid_type(self) -> None:
        """Test workspace_root handles TypeError gracefully."""
        resolver = FlextInfraPathResolver()
        # Pass an invalid type that Path() will reject
        result = resolver.workspace_root(None)  # type: ignore[arg-type]

        assert result.is_failure
        assert "failed to resolve" in result.error.lower()

    def test_workspace_root_from_file_with_invalid_type(self) -> None:
        """Test workspace_root_from_file handles TypeError gracefully."""
        resolver = FlextInfraPathResolver()
        result = resolver.workspace_root_from_file(None)  # type: ignore[arg-type]

        assert result.is_failure
        assert "failed to resolve" in result.error.lower()

    def test_execute_returns_current_directory(self) -> None:
        """Test execute method returns current working directory."""
        resolver = FlextInfraPathResolver()
        result = resolver.execute()

        assert isinstance(result, Path)
        assert result == Path.cwd()

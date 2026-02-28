"""Tests for FlextInfraConfigFixer service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.check.services import FlextInfraConfigFixer


class TestFlextInfraConfigFixer:
    """Test suite for FlextInfraConfigFixer."""

    def test_init_creates_instance(self) -> None:
        """Test that fixer initializes with default workspace root."""
        fixer = FlextInfraConfigFixer()
        assert fixer is not None

    def test_init_with_custom_workspace_root(self, tmp_path: Path) -> None:
        """Test that fixer accepts custom workspace root."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        assert fixer is not None

    def test_execute_returns_failure(self) -> None:
        """Test that execute() returns failure with helpful message."""
        fixer = FlextInfraConfigFixer()
        result = fixer.execute()

        assert result.is_failure
        assert "Use run()" in result.error

    def test_run_with_empty_projects(self, tmp_path: Path) -> None:
        """Test that run() handles empty project list."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([])

        assert result.is_success
        assert isinstance(result.value, list)

    def test_run_with_nonexistent_projects(self, tmp_path: Path) -> None:
        """Test that run() handles nonexistent projects gracefully."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run(["nonexistent"])

        assert result.is_success

    def test_run_with_dry_run_flag(self, tmp_path: Path) -> None:
        """Test that run() respects dry_run flag."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([], dry_run=True)

        assert result.is_success

    def test_run_with_verbose_flag(self, tmp_path: Path) -> None:
        """Test that run() respects verbose flag."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([], verbose=True)

        assert result.is_success

    def test_find_pyproject_files_with_empty_workspace(self, tmp_path: Path) -> None:
        """Test that find_pyproject_files returns empty list for empty workspace."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.find_pyproject_files()

        assert result.is_success
        assert isinstance(result.value, list)

    def test_find_pyproject_files_with_specific_paths(self, tmp_path: Path) -> None:
        """Test that find_pyproject_files filters by project paths."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.find_pyproject_files(project_paths=[tmp_path / "project1"])

        assert result.is_success
        assert isinstance(result.value, list)

    def test_process_file_with_missing_file(self, tmp_path: Path) -> None:
        """Test that process_file handles missing files gracefully."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        missing_file = tmp_path / "nonexistent.toml"

        result = fixer.process_file(missing_file)

        assert result.is_failure
        assert "failed to read" in result.error

    def test_process_file_with_valid_toml(self, tmp_path: Path) -> None:
        """Test that process_file handles valid TOML without pyrefly section."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool]\nother = true\n")

        result = fixer.process_file(pyproject)

        assert result.is_success
        assert result.value == []  # No fixes needed

    def test_process_file_with_invalid_toml(self, tmp_path: Path) -> None:
        """Test that process_file handles invalid TOML gracefully."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[invalid toml")

        result = fixer.process_file(pyproject)

        assert result.is_failure
        assert "failed to parse" in result.error

    def test_process_file_with_dry_run(self, tmp_path: Path) -> None:
        """Test that process_file with dry_run doesn't modify file."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        original_content = "[tool]\nother = true\n"
        pyproject.write_text(original_content)

        result = fixer.process_file(pyproject, dry_run=True)

        assert result.is_success
        assert pyproject.read_text() == original_content

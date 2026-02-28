"""Tests for FlextInfraWorkspaceChecker service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import FlextResult as r
from flext_infra.check.services import FlextInfraWorkspaceChecker


class TestFlextInfraWorkspaceChecker:
    """Test suite for FlextInfraWorkspaceChecker."""

    def test_init_creates_instance(self) -> None:
        """Test that checker initializes with default workspace root."""
        checker = FlextInfraWorkspaceChecker()
        assert checker is not None

    def test_init_with_custom_workspace_root(self, tmp_path: Path) -> None:
        """Test that checker accepts custom workspace root."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        assert checker is not None

    def test_execute_returns_failure(self) -> None:
        """Test that execute() returns failure with helpful message."""
        checker = FlextInfraWorkspaceChecker()
        result = checker.execute()

        assert result.is_failure
        assert "Use run()" in result.error or "Use run_projects()" in result.error

    def test_resolve_gates_with_valid_gates(self) -> None:
        """Test that resolve_gates normalizes valid gate names."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["lint", "type"])

        assert result.is_success
        assert "lint" in result.value
        assert "pyrefly" in result.value  # type maps to pyrefly

    def test_resolve_gates_deduplicates(self) -> None:
        """Test that resolve_gates removes duplicate gate names."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["lint", "lint", "format"])

        assert result.is_success
        assert result.value.count("lint") == 1

    def test_resolve_gates_with_invalid_gate(self) -> None:
        """Test that resolve_gates fails on invalid gate name."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["invalid_gate"])

        assert result.is_failure

    def test_run_projects_with_missing_projects(self, tmp_path: Path) -> None:
        """Test that run_projects handles missing project directories gracefully."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["nonexistent"],
            ["lint"],
            reports_dir=tmp_path / "reports",
        )

        assert result.is_success
        # Should skip missing projects
        assert len(result.value) == 0

    def test_run_projects_creates_reports_dir(self, tmp_path: Path) -> None:
        """Test that run_projects creates reports directory if missing."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        result = checker.run_projects(
            [],
            ["lint"],
            reports_dir=reports_dir,
        )

        assert result.is_success
        assert reports_dir.exists()

    def test_lint_returns_gate_result(self, tmp_path: Path) -> None:
        """Test that lint() returns a GateResult."""
        checker = FlextInfraWorkspaceChecker()
        result = checker.lint(tmp_path)

        assert isinstance(result, r)
        assert result.is_success

    def test_format_returns_gate_result(self, tmp_path: Path) -> None:
        """Test that format() returns a GateResult."""
        checker = FlextInfraWorkspaceChecker()
        result = checker.format(tmp_path)

        assert isinstance(result, r)
        assert result.is_success

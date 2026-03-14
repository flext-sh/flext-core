"""Tests for FlextInfraReportingService — report dir/path and ensure operations.

Tests cover report path generation and directory management.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesReporting
from flext_tests import tm


class TestFlextInfraReportingServiceCore:
    """Test suite for FlextInfraReportingService core operations."""

    @pytest.fixture
    def service(self) -> FlextInfraUtilitiesReporting:
        """Create a reporting service instance."""
        return FlextInfraUtilitiesReporting()

    def test_get_report_dir_project_scope(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting project-level report directory."""
        result = service.get_report_dir(tmp_path, "project", "check")
        assert isinstance(result, Path)
        assert result.name == "check"
        assert ".reports" in str(result)
        assert "workspace" not in str(result)

    def test_get_report_dir_workspace_scope(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting workspace-level report directory."""
        result = service.get_report_dir(tmp_path, "workspace", "validate")
        assert isinstance(result, Path)
        assert result.name == "validate"
        assert ".reports" in str(result)
        assert "workspace" in str(result)

    def test_get_report_dir_with_string_root(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting report directory with string root path."""
        result = service.get_report_dir(str(tmp_path), "project", "test")
        assert isinstance(result, Path)
        assert result.name == "test"

    def test_get_report_path_project_scope(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting project-level report file path."""
        result = service.get_report_path(tmp_path, "project", "check", "report.json")
        assert isinstance(result, Path)
        assert result.name == "report.json"
        assert ".reports" in str(result)
        assert "check" in str(result)

    def test_get_report_path_workspace_scope(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting workspace-level report file path."""
        result = service.get_report_path(
            tmp_path,
            "workspace",
            "validate",
            "summary.log",
        )
        assert isinstance(result, Path)
        assert result.name == "summary.log"
        assert ".reports" in str(result)
        assert "workspace" in str(result)
        assert "validate" in str(result)

    def test_get_report_path_with_string_root(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test getting report file path with string root."""
        result = service.get_report_path(
            str(tmp_path),
            "project",
            "test",
            "results.xml",
        )
        assert isinstance(result, Path)
        assert result.name == "results.xml"

    def test_ensure_report_dir_creates_directory(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test ensuring report directory creates it if missing."""
        result = service.ensure_report_dir(tmp_path, "project", "check")
        tm.ok(result)
        report_dir = result.value
        assert report_dir.exists()
        assert report_dir.is_dir()

    def test_ensure_report_dir_idempotent(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test ensuring report directory is idempotent."""
        result1 = service.ensure_report_dir(tmp_path, "project", "check")
        result2 = service.ensure_report_dir(tmp_path, "project", "check")
        tm.ok(result1)
        tm.ok(result2)
        assert result1.value == result2.value

    def test_ensure_report_dir_workspace_scope(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test ensuring workspace-level report directory."""
        result = service.ensure_report_dir(tmp_path, "workspace", "validate")
        tm.ok(result)
        report_dir = result.value
        assert report_dir.exists()
        assert "workspace" in str(report_dir)

    def test_ensure_report_dir_permission_error(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test handling permission errors when creating directory."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(292)
        try:
            result = service.ensure_report_dir(readonly_dir, "project", "check")
            tm.fail(result)
        finally:
            readonly_dir.chmod(493)

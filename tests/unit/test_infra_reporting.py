"""Tests for FlextInfraReportingService.

Tests cover report path generation and directory management.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra import FlextInfraReportingService


class TestFlextInfraReportingService:
    """Test suite for FlextInfraReportingService."""

    @pytest.fixture
    def service(self) -> FlextInfraReportingService:
        """Create a reporting service instance."""
        return FlextInfraReportingService()

    def test_get_report_dir_project_scope(
        self,
        service: FlextInfraReportingService,
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
        service: FlextInfraReportingService,
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
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test getting report directory with string root path."""
        result = service.get_report_dir(str(tmp_path), "project", "test")

        assert isinstance(result, Path)
        assert result.name == "test"

    def test_get_report_path_project_scope(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test getting project-level report file path."""
        result = service.get_report_path(
            tmp_path,
            "project",
            "check",
            "report.json",
        )

        assert isinstance(result, Path)
        assert result.name == "report.json"
        assert ".reports" in str(result)
        assert "check" in str(result)

    def test_get_report_path_workspace_scope(
        self,
        service: FlextInfraReportingService,
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
        service: FlextInfraReportingService,
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
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test ensuring report directory creates it if missing."""
        result = service.ensure_report_dir(tmp_path, "project", "check")

        assert result.is_success
        report_dir = result.value
        assert report_dir.exists()
        assert report_dir.is_dir()

    def test_ensure_report_dir_idempotent(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test ensuring report directory is idempotent."""
        result1 = service.ensure_report_dir(tmp_path, "project", "check")
        result2 = service.ensure_report_dir(tmp_path, "project", "check")

        assert result1.is_success
        assert result2.is_success
        assert result1.value == result2.value

    def test_ensure_report_dir_workspace_scope(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test ensuring workspace-level report directory."""
        result = service.ensure_report_dir(tmp_path, "workspace", "validate")

        assert result.is_success
        report_dir = result.value
        assert report_dir.exists()
        assert "workspace" in str(report_dir)

    def test_ensure_report_dir_permission_error(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test handling permission errors when creating directory."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            result = service.ensure_report_dir(
                readonly_dir,
                "project",
                "check",
            )
            assert result.is_failure
        finally:
            readonly_dir.chmod(0o755)

    def test_get_report_dir_returns_path(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test that get_report_dir returns Path type."""
        result = service.get_report_dir(tmp_path, "project", "check")

        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_get_report_path_returns_path(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test that get_report_path returns Path type."""
        result = service.get_report_path(
            tmp_path,
            "project",
            "check",
            "report.json",
        )

        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_execute_returns_empty_path(
        self,
        service: FlextInfraReportingService,
    ) -> None:
        """Test execute method returns empty Path."""
        result = service.execute()

        assert result.is_success
        assert result.value == Path()

    def test_create_latest_symlink_success(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test creating a latest symlink."""
        report_dir = tmp_path / ".reports" / "tests"
        report_dir.mkdir(parents=True)
        run_id = "run-2025-01-01"

        result = service.create_latest_symlink(report_dir, run_id)

        assert result.is_success
        link = result.value
        assert link.name == "latest"
        assert link.is_symlink()
        assert link.resolve().name == run_id

    def test_create_latest_symlink_update_existing(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test updating an existing latest symlink."""
        report_dir = tmp_path / ".reports" / "tests"
        report_dir.mkdir(parents=True)
        run_id_1 = "run-2025-01-01"
        run_id_2 = "run-2025-01-02"

        # Create first symlink
        result1 = service.create_latest_symlink(report_dir, run_id_1)
        assert result1.is_success

        # Update to second symlink
        result2 = service.create_latest_symlink(report_dir, run_id_2)
        assert result2.is_success
        link = result2.value
        assert link.is_symlink()
        assert link.resolve().name == run_id_2

    def test_create_latest_symlink_oserror(
        self,
        service: FlextInfraReportingService,
        tmp_path: Path,
    ) -> None:
        """Test handling OSError when creating symlink."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            result = service.create_latest_symlink(readonly_dir, "run-id")
            assert result.is_failure
            assert "symlink" in result.error.lower()
        finally:
            readonly_dir.chmod(0o755)

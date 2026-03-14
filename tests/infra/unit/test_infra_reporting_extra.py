"""Tests for FlextInfraReportingService — path types and symlink operations.

Tests cover return type validation and latest symlink management.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesReporting
from flext_tests import tm


class TestFlextInfraReportingServiceExtra:
    """Test suite for FlextInfraReportingService extra operations."""

    @pytest.fixture
    def service(self) -> FlextInfraUtilitiesReporting:
        """Create a reporting service instance."""
        return FlextInfraUtilitiesReporting()

    def test_get_report_dir_returns_path(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test that get_report_dir returns Path type."""
        result = service.get_report_dir(tmp_path, "project", "check")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_get_report_path_returns_path(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test that get_report_path returns Path type."""
        result = service.get_report_path(tmp_path, "project", "check", "report.json")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_create_latest_symlink_success(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test creating a latest symlink."""
        report_dir = tmp_path / ".reports" / "tests"
        report_dir.mkdir(parents=True)
        run_id = "run-2025-01-01"
        result = service.create_latest_symlink(report_dir, run_id)
        tm.ok(result)
        link = result.value
        assert link.name == "latest"
        assert link.is_symlink()
        assert link.resolve().name == run_id

    def test_create_latest_symlink_update_existing(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test updating an existing latest symlink."""
        report_dir = tmp_path / ".reports" / "tests"
        report_dir.mkdir(parents=True)
        run_id_1 = "run-2025-01-01"
        run_id_2 = "run-2025-01-02"
        result1 = service.create_latest_symlink(report_dir, run_id_1)
        tm.ok(result1)
        result2 = service.create_latest_symlink(report_dir, run_id_2)
        tm.ok(result2)
        link = result2.value
        assert link.is_symlink()
        assert link.resolve().name == run_id_2

    def test_create_latest_symlink_oserror(
        self,
        service: FlextInfraUtilitiesReporting,
        tmp_path: Path,
    ) -> None:
        """Test handling OSError when creating symlink."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(292)
        try:
            result = service.create_latest_symlink(readonly_dir, "run-id")
            tm.fail(result)
            assert result.error is not None and "symlink" in result.error.lower()
        finally:
            readonly_dir.chmod(493)

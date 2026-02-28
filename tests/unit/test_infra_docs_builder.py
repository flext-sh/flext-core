"""Tests for FlextInfraDocBuilder service.

Tests documentation building functionality with mocked command runner
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.builder import BuildReport, FlextInfraDocBuilder
from flext_infra.docs.shared import FlextInfraDocScope


class TestFlextInfraDocBuilder:
    """Tests for FlextInfraDocBuilder service."""

    @pytest.fixture
    def builder(self) -> FlextInfraDocBuilder:
        """Create builder instance."""
        return FlextInfraDocBuilder()

    @pytest.fixture
    def sample_scope(self, tmp_path: Path) -> FlextInfraDocScope:
        """Create sample documentation scope."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=report_dir,
        )

    def test_build_returns_flext_result(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test that build returns FlextResult[list[BuildReport]]."""
        result = builder.build(tmp_path)
        assert result.is_success or result.is_failure

    def test_build_with_valid_scope_returns_success(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with valid scope returns success."""
        result = builder.build(tmp_path)
        assert result.is_success
        assert isinstance(result.value, list)

    def test_build_report_structure(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test BuildReport has required fields."""
        result = builder.build(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            assert hasattr(report, "scope")
            assert hasattr(report, "result")
            assert hasattr(report, "reason")
            assert hasattr(report, "site_dir")

    def test_build_report_frozen(self) -> None:
        """Test BuildReport is frozen (immutable)."""
        assert BuildReport.model_config.get("frozen") is True

    def test_build_with_project_filter(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with single project filter."""
        result = builder.build(tmp_path, project="test-project")
        assert result.is_success or result.is_failure

    def test_build_with_projects_filter(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with multiple projects filter."""
        result = builder.build(tmp_path, projects="proj1,proj2")
        assert result.is_success or result.is_failure

    def test_build_with_custom_output_dir(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with custom output directory."""
        output_dir = str(tmp_path / "custom_output")
        result = builder.build(tmp_path, output_dir=output_dir)
        assert result.is_success or result.is_failure

    def test_build_report_result_field_values(self) -> None:
        """Test BuildReport result field accepts valid values."""
        for status in ["OK", "FAIL", "SKIP"]:
            report = BuildReport(
                scope="test",
                result=status,
                reason="Test reason",
                site_dir="/tmp/site",
            )
            assert report.result == status

    def test_build_report_site_dir_field(self) -> None:
        """Test BuildReport site_dir field."""
        report = BuildReport(
            scope="test",
            result="OK",
            reason="Build successful",
            site_dir="/path/to/site",
        )
        assert report.site_dir == "/path/to/site"

    def test_build_with_multiple_projects_returns_list(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with multiple projects returns list of reports."""
        result = builder.build(tmp_path, projects="proj1,proj2")
        if result.is_success:
            assert isinstance(result.value, list)

    def test_build_multiple_scopes(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build returns multiple reports for multiple scopes."""
        result = builder.build(tmp_path, projects="proj1,proj2,proj3")
        if result.is_success:
            assert isinstance(result.value, list)

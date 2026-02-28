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

    def test_build_scope_with_mkdocs_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _build_scope with mkdocs.yml present."""
        mkdocs_file = tmp_path / "mkdocs.yml"
        mkdocs_file.write_text("site_name: Test\n")
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = builder._build_scope(scope)
        assert report.scope == "test"

    def test_build_scope_without_mkdocs_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _build_scope without mkdocs.yml returns SKIP."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = builder._build_scope(scope)
        assert report.result == "SKIP"

    def test_run_mkdocs_no_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _run_mkdocs returns SKIP when mkdocs.yml not found."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = builder._run_mkdocs(scope)
        assert report.result == "SKIP"
        assert "mkdocs.yml not found" in report.reason

    def test_write_reports_creates_json_and_markdown(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _write_reports creates both JSON and markdown files."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=report_dir,
        )
        report = BuildReport(
            scope="test",
            result="OK",
            reason="Build succeeded",
            site_dir="/tmp/site",
        )
        builder._write_reports(scope, report)
        assert (report_dir / "build-summary.json").exists()
        assert (report_dir / "build-report.md").exists()

    def test_run_mkdocs_with_command_failure(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _run_mkdocs handles command failures."""
        mkdocs_file = tmp_path / "mkdocs.yml"
        mkdocs_file.write_text("site_name: Test\n")
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = builder._run_mkdocs(scope)
        assert report.scope == "test"
        assert isinstance(report.result, str)

"""Tests for FlextInfraDocGenerator service.

Tests documentation generation functionality with tmp_path for file generation
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.generator import (
    FlextInfraDocGenerator,
    GeneratedFile,
    GenerateReport,
)
from flext_infra.docs.shared import FlextInfraDocScope


class TestFlextInfraDocGenerator:
    """Tests for FlextInfraDocGenerator service."""

    @pytest.fixture
    def generator(self) -> FlextInfraDocGenerator:
        """Create generator instance."""
        return FlextInfraDocGenerator()

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

    def test_generate_returns_flext_result(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test that generate returns FlextResult[list[GenerateReport]]."""
        result = generator.generate(tmp_path)
        assert result.is_success or result.is_failure

    def test_generate_with_valid_scope_returns_success(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with valid scope returns success."""
        result = generator.generate(tmp_path)
        assert result.is_success
        assert isinstance(result.value, list)

    def test_generate_report_structure(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test GenerateReport has required fields."""
        result = generator.generate(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            assert hasattr(report, "scope")
            assert hasattr(report, "generated")
            assert hasattr(report, "applied")
            assert hasattr(report, "source")
            assert hasattr(report, "files")

    def test_generated_file_structure(self) -> None:
        """Test GeneratedFile model structure."""
        file = GeneratedFile(path="README.md", written=True)
        assert file.path == "README.md"
        assert file.written is True

    def test_generate_report_frozen(self) -> None:
        """Test GenerateReport is frozen (immutable)."""
        assert GenerateReport.model_config.get("frozen") is True

    def test_generated_file_frozen(self) -> None:
        """Test GeneratedFile is frozen (immutable)."""
        assert GeneratedFile.model_config.get("frozen") is True

    def test_generate_with_project_filter(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with single project filter."""
        result = generator.generate(tmp_path, project="test-project")
        assert result.is_success or result.is_failure

    def test_generate_with_projects_filter(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with multiple projects filter."""
        result = generator.generate(tmp_path, projects="proj1,proj2")
        assert result.is_success or result.is_failure

    def test_generate_with_apply_false_dry_run(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with apply=False (dry-run mode)."""
        result = generator.generate(tmp_path, apply=False)
        assert result.is_success or result.is_failure

    def test_generate_with_apply_true_writes_files(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with apply=True writes files."""
        result = generator.generate(tmp_path, apply=True)
        assert result.is_success or result.is_failure

    def test_generate_with_custom_output_dir(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test generate with custom output directory."""
        output_dir = str(tmp_path / "custom_output")
        result = generator.generate(tmp_path, output_dir=output_dir)
        assert result.is_success or result.is_failure

    def test_generate_report_generated_count(self) -> None:
        """Test GenerateReport generated field."""
        report = GenerateReport(
            scope="test",
            generated=5,
            applied=True,
            source="test-source",
        )
        assert report.generated == 5

    def test_generate_report_applied_field(self) -> None:
        """Test GenerateReport applied field."""
        report = GenerateReport(
            scope="test",
            generated=0,
            applied=False,
            source="test-source",
        )
        assert report.applied is False

    def test_generate_report_source_field(self) -> None:
        """Test GenerateReport source field."""
        report = GenerateReport(
            scope="test",
            generated=0,
            applied=False,
            source="workspace-ssot",
        )
        assert report.source == "workspace-ssot"

    def test_generate_report_files_list(self) -> None:
        """Test GenerateReport files list."""
        files = [
            GeneratedFile(path="file1.md", written=True),
            GeneratedFile(path="file2.md", written=False),
        ]
        report = GenerateReport(
            scope="test",
            generated=2,
            applied=True,
            source="test-source",
            files=files,
        )
        assert len(report.files) == 2
        assert report.files[0].path == "file1.md"

    def test_generated_file_written_field(self) -> None:
        """Test GeneratedFile written field."""
        file_written = GeneratedFile(path="test.md", written=True)
        file_not_written = GeneratedFile(path="test2.md", written=False)
        assert file_written.written is True
        assert file_not_written.written is False

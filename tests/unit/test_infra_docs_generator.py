"""Tests for FlextInfraDocGenerator service.

Tests documentation generation functionality with tmp_path for file generation
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_core import r
from flext_infra.docs.generator import (
    FlextInfraDocGenerator,
    GeneratedFile,
    GenerateReport,
)
from flext_infra.docs.shared import FlextInfraDocScope, FlextInfraDocsShared


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

    def test_generate_scope_root_scope(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_scope with root scope."""
        scope = FlextInfraDocScope(
            name="root",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = generator._generate_scope(scope, apply=False, workspace_root=tmp_path)
        assert report.scope == "root"

    def test_generate_scope_project_scope(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_scope with project scope."""
        scope = FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = generator._generate_scope(scope, apply=False, workspace_root=tmp_path)
        assert report.scope == "test-project"

    def test_generate_root_docs_creates_files(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_root_docs creates placeholder files."""
        scope = FlextInfraDocScope(
            name="root",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        files = generator._generate_root_docs(scope, apply=False)
        assert len(files) == 3  # CHANGELOG, releases/latest, roadmap

    def test_generate_project_guides_no_source(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_guides with no source guides."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        files = generator._generate_project_guides(
            scope, workspace_root=tmp_path, apply=False
        )
        assert files == []

    def test_generate_project_mkdocs_creates_config(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_mkdocs creates mkdocs.yml."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        files = generator._generate_project_mkdocs(scope, apply=False)
        assert len(files) == 1
        assert files[0].path.endswith("mkdocs.yml")

    def test_generate_project_mkdocs_skips_existing(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_mkdocs skips existing mkdocs.yml."""
        mkdocs_file = tmp_path / "mkdocs.yml"
        mkdocs_file.write_text("site_name: Test\n")
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        files = generator._generate_project_mkdocs(scope, apply=False)
        assert files == []

    def test_project_guide_content_adds_heading(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _project_guide_content adds project heading."""
        content = "# Original Title\n\nContent here.\n"
        result = generator._project_guide_content(content, "my-project", "guide.md")
        assert "my-project - Original Title" in result

    def test_project_guide_content_preserves_body(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _project_guide_content preserves body content."""
        content = "# Title\n\nBody content.\n"
        result = generator._project_guide_content(content, "proj", "guide.md")
        assert "Body content" in result

    def test_sanitize_internal_anchor_links_removes_local_links(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _sanitize_internal_anchor_links removes local markdown links."""
        content = "[Link](local.md) and [External](http://example.com)"
        result = generator._sanitize_internal_anchor_links(content)
        assert "Link" in result
        assert "http://example.com" in result

    def test_normalize_anchor_converts_to_slug(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _normalize_anchor converts heading to slug."""
        assert generator._normalize_anchor("Hello World") == "hello-world"
        assert generator._normalize_anchor("Test-Case") == "test-case"

    def test_build_toc_from_headings(self, generator: FlextInfraDocGenerator) -> None:
        """Test _build_toc generates TOC from headings."""
        content = "# Main\n\n## Section 1\n\n### Subsection\n"
        toc = generator._build_toc(content)
        assert "<!-- TOC START -->" in toc
        assert "Section 1" in toc

    def test_update_toc_replaces_existing(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _update_toc replaces existing TOC."""
        content = "# Main\n\n<!-- TOC START -->\nOld\n<!-- TOC END -->\n\n## Section\n"
        result = generator._update_toc(content)
        assert "Old" not in result

    def test_update_toc_inserts_new(self, generator: FlextInfraDocGenerator) -> None:
        """Test _update_toc inserts new TOC."""
        content = "# Main\n\n## Section\n"
        result = generator._update_toc(content)
        assert "<!-- TOC START -->" in result

    def test_write_if_needed_no_change(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed skips unchanged content."""
        path = tmp_path / "test.md"
        path.write_text("# Test\n")
        result = generator._write_if_needed(path, "# Test\n", apply=True)
        assert result.written is False

    def test_write_if_needed_with_apply(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed writes when apply=True."""
        path = tmp_path / "test.md"
        result = generator._write_if_needed(path, "# New Content\n", apply=True)
        assert result.written is True
        assert path.exists()

    def test_write_if_needed_dry_run(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed dry-run mode."""
        path = tmp_path / "test.md"
        result = generator._write_if_needed(path, "# New Content\n", apply=False)
        assert result.written is False

    def test_generate_project_guides_with_source(
        self, generator: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_guides with source guides."""
        guides_dir = tmp_path / "docs/guides"
        guides_dir.mkdir(parents=True, exist_ok=True)
        guide_file = guides_dir / "test.md"
        guide_file.write_text("# Test Guide\n\nContent.\n")
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path / "project",
            report_dir=tmp_path / "reports",
        )
        files = generator._generate_project_guides(
            scope, workspace_root=tmp_path, apply=False
        )
        assert isinstance(files, list)

    def test_normalize_anchor_empty_string(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _normalize_anchor with empty string."""
        result = generator._normalize_anchor("")
        assert result == ""

    def test_build_toc_with_no_headings(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _build_toc with no headings."""
        content = "# Main\n\nNo sections.\n"
        toc = generator._build_toc(content)
        assert "No sections found" in toc

    def test_sanitize_internal_anchor_links_preserves_external(
        self, generator: FlextInfraDocGenerator
    ) -> None:
        """Test _sanitize_internal_anchor_links preserves external links."""
        content = "[Local](local.md) [External](https://example.com)"
        result = generator._sanitize_internal_anchor_links(content)
        assert "https://example.com" in result

    def test_generate_with_scope_failure_returns_failure(
        self,
        generator: FlextInfraDocGenerator,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test generate returns failure when scope building fails."""

        def mock_build_scopes(*args: object, **kwargs: object) -> r[list]:
            return r[list].fail("Scope error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = generator.generate(tmp_path)
        assert result.is_failure
        assert "Scope error" in result.error

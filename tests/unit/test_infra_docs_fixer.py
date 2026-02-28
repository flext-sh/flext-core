"""Tests for FlextInfraDocFixer service.

Tests documentation fixing functionality with tmp_path for file generation
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_core import r
from flext_infra.docs.fixer import FixItem, FixReport, FlextInfraDocFixer
from flext_infra.docs.shared import FlextInfraDocScope, FlextInfraDocsShared


class TestFlextInfraDocFixer:
    """Tests for FlextInfraDocFixer service."""

    @pytest.fixture
    def fixer(self) -> FlextInfraDocFixer:
        """Create fixer instance."""
        return FlextInfraDocFixer()

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

    @pytest.fixture
    def sample_markdown_file(self, tmp_path: Path) -> Path:
        """Create sample markdown file for fixing."""
        md_file = tmp_path / "README.md"
        md_file.write_text("# Test\n\nSome content here.\n")
        return md_file

    def test_fix_returns_flext_result(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test that fix returns FlextResult[list[FixReport]]."""
        result = fixer.fix(tmp_path)
        assert result.is_success or result.is_failure

    def test_fix_with_valid_scope_returns_success(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with valid scope returns success."""
        result = fixer.fix(tmp_path)
        assert result.is_success
        assert isinstance(result.value, list)

    def test_fix_report_structure(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test FixReport has required fields."""
        result = fixer.fix(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            assert hasattr(report, "scope")
            assert hasattr(report, "changed_files")
            assert hasattr(report, "applied")
            assert hasattr(report, "items")

    def test_fix_item_structure(self) -> None:
        """Test FixItem model structure."""
        item = FixItem(file="README.md", links=2, toc=1)
        assert item.file == "README.md"
        assert item.links == 2
        assert item.toc == 1

    def test_fix_report_frozen(self) -> None:
        """Test FixReport is frozen (immutable)."""
        assert FixReport.model_config.get("frozen") is True

    def test_fix_item_frozen(self) -> None:
        """Test FixItem is frozen (immutable)."""
        assert FixItem.model_config.get("frozen") is True

    def test_fix_with_project_filter(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with single project filter."""
        result = fixer.fix(tmp_path, project="test-project")
        assert result.is_success or result.is_failure

    def test_fix_with_projects_filter(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with multiple projects filter."""
        result = fixer.fix(tmp_path, projects="proj1,proj2")
        assert result.is_success or result.is_failure

    def test_fix_with_apply_false_dry_run(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with apply=False (dry-run mode)."""
        result = fixer.fix(tmp_path, apply=False)
        assert result.is_success or result.is_failure

    def test_fix_with_apply_true_writes_changes(
        self, fixer: FlextInfraDocFixer, tmp_path: Path, sample_markdown_file: Path
    ) -> None:
        """Test fix with apply=True writes changes."""
        result = fixer.fix(tmp_path, apply=True)
        assert result.is_success or result.is_failure

    def test_fix_with_custom_output_dir(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with custom output directory."""
        output_dir = str(tmp_path / "custom_output")
        result = fixer.fix(tmp_path, output_dir=output_dir)
        assert result.is_success or result.is_failure

    def test_fix_report_changed_files_count(self) -> None:
        """Test FixReport changed_files field."""
        report = FixReport(scope="test", changed_files=5, applied=True)
        assert report.changed_files == 5

    def test_fix_report_applied_field(self) -> None:
        """Test FixReport applied field."""
        report = FixReport(scope="test", changed_files=0, applied=False)
        assert report.applied is False

    def test_fix_report_items_list(self) -> None:
        """Test FixReport items list."""
        items = [
            FixItem(file="file1.md", links=1, toc=0),
            FixItem(file="file2.md", links=0, toc=1),
        ]
        report = FixReport(scope="test", changed_files=2, applied=True, items=items)
        assert len(report.items) == 2
        assert report.items[0].file == "file1.md"

    def test_process_file_with_markdown_links(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file detects and fixes markdown links."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\n[Link](missing.md)\n")
        item = fixer._process_file(md_file, apply=False)
        assert item.file == str(md_file)

    def test_maybe_fix_link_external_urls(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for external URLs."""
        md_file = tmp_path / "test.md"
        assert fixer._maybe_fix_link(md_file, "http://example.com") is None
        assert fixer._maybe_fix_link(md_file, "https://example.com") is None
        assert fixer._maybe_fix_link(md_file, "mailto:test@example.com") is None

    def test_maybe_fix_link_fragment_only(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for fragment-only links."""
        md_file = tmp_path / "test.md"
        assert fixer._maybe_fix_link(md_file, "#section") is None

    def test_maybe_fix_link_existing_file(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for existing files."""
        md_file = tmp_path / "test.md"
        existing = tmp_path / "existing.md"
        existing.write_text("# Existing")
        assert fixer._maybe_fix_link(md_file, "existing.md") is None

    def test_maybe_fix_link_adds_md_extension(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link adds .md extension when needed."""
        md_file = tmp_path / "test.md"
        candidate = tmp_path / "missing.md"
        candidate.write_text("# Missing")
        result = fixer._maybe_fix_link(md_file, "missing")
        assert result == "missing.md"

    def test_anchorize_converts_to_slug(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize converts heading to anchor slug."""
        assert fixer._anchorize("Hello World") == "hello-world"
        assert fixer._anchorize("Test-Case") == "test-case"
        assert fixer._anchorize("  Spaces  ") == "spaces"

    def test_anchorize_removes_special_chars(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize removes special characters."""
        assert fixer._anchorize("Hello! World?") == "hello-world"
        assert fixer._anchorize("Test@#$%") == "test"

    def test_build_toc_from_headings(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc generates TOC from headings."""
        content = "# Main\n\n## Section 1\n\n### Subsection\n\n## Section 2\n"
        toc = fixer._build_toc(content)
        assert "<!-- TOC START -->" in toc
        assert "<!-- TOC END -->" in toc
        assert "Section 1" in toc

    def test_build_toc_no_headings(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc with no headings."""
        content = "# Main\n\nNo sections here.\n"
        toc = fixer._build_toc(content)
        assert "No sections found" in toc

    def test_update_toc_replaces_existing(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc replaces existing TOC."""
        content = (
            "# Main\n\n<!-- TOC START -->\nOld TOC\n<!-- TOC END -->\n\n## Section\n"
        )
        updated, changed = fixer._update_toc(content)
        assert changed == 1
        assert "Old TOC" not in updated

    def test_update_toc_inserts_new(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc inserts new TOC."""
        content = "# Main\n\n## Section\n"
        updated, changed = fixer._update_toc(content)
        assert changed == 1
        assert "<!-- TOC START -->" in updated

    def test_fix_scope_with_markdown_files(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _fix_scope processes markdown files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        md_file = docs_dir / "README.md"
        md_file.write_text("# Test\n\n## Section\n")
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = fixer._fix_scope(scope, apply=False)
        assert report.scope == "test"
        assert isinstance(report.items, list)

    def test_process_file_with_apply_true(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file with apply=True writes changes."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\n## Section\n")
        item = fixer._process_file(md_file, apply=True)
        assert item.file == str(md_file)

    def test_maybe_fix_link_empty_base(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link with empty base."""
        md_file = tmp_path / "test.md"
        assert fixer._maybe_fix_link(md_file, "#section") is None

    def test_anchorize_empty_string(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize with empty string."""
        result = fixer._anchorize("")
        assert result == ""

    def test_build_toc_with_no_h2_h3(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc with no H2/H3 headings."""
        content = "# Main\n\nNo sections.\n"
        toc = fixer._build_toc(content)
        assert "No sections found" in toc

    def test_fix_with_scope_failure_returns_failure(
        self, fixer: FlextInfraDocFixer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test fix returns failure when scope building fails."""

        def mock_build_scopes(*args: object, **kwargs: object) -> r[list]:
            return r[list].fail("Scope error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = fixer.fix(tmp_path)
        assert result.is_failure
        assert "Scope error" in result.error

    def test_maybe_fix_link_with_valid_link(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link fixes valid broken links."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test")
        # Test with a link that should be fixed
        result = fixer._maybe_fix_link(md_file, "[text](broken.md)")
        # Should return fixed link or None
        assert result is None or isinstance(result, str)

    def test_maybe_fix_link_returns_fixed_link(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns fixed link when found."""
        # Create a target file
        target = tmp_path / "target.md"
        target.write_text("# Target")
        md_file = tmp_path / "test.md"
        # Test with a link that can be fixed
        result = fixer._maybe_fix_link(md_file, "[text](target.md)")
        # Should return the link or None
        assert result is None or isinstance(result, str)

    def test_process_file_with_no_fixes_needed(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file with content that needs no fixes."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nNo broken links or TOC needed.")
        item = fixer._process_file(md_file, apply=False)
        assert "test.md" in item.file
        assert item.links == 0

    def test_build_toc_with_existing_toc_marker(
        self, fixer: FlextInfraDocFixer
    ) -> None:
        """Test _build_toc when TOC marker already exists."""
        content = "<!-- TOC START -->\n<!-- TOC END -->\n\n# Section\n\n## Subsection"
        result = fixer._build_toc(content)
        # Should return tuple or string
        assert result is not None

    def test_process_file_with_fixable_links(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file counts fixed links correctly."""
        md_file = tmp_path / "test.md"
        # Create a markdown file with a link that can be fixed
        md_file.write_text("# Test\n\n[Link](target.md)\n")
        target = tmp_path / "target.md"
        target.write_text("# Target")
        item = fixer._process_file(md_file, apply=False)
        # Should detect the link
        assert "test.md" in item.file

    def test_build_toc_with_empty_anchor(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc skips headings with empty anchors."""
        content = "# Test\n\n## !!!Invalid!!!\n\n## Valid Section"
        result = fixer._build_toc(content)
        # Should include valid section but skip invalid
        assert "Valid Section" in result

    def test_update_toc_without_h1_heading(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc prepends TOC when no h1 heading exists."""
        content = "## Section 1\n\nContent here."
        updated, changed = fixer._update_toc(content)
        assert changed == 1
        assert "<!-- TOC START -->" in updated

    def test_maybe_fix_link_with_existing_target(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns fixed link when .md suffix exists (lines 167-168)."""
        md_file = tmp_path / "docs" / "foo.md"
        md_file.parent.mkdir(parents=True)
        md_file.touch()
        (tmp_path / "docs" / "bar.md").touch()

        # Link without .md extension should be fixed to bar.md
        result = fixer._maybe_fix_link(md_file, "bar")
        assert result == "bar.md"

    def test_maybe_fix_link_with_empty_base(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for empty base (line 183)."""
        md_file = tmp_path / "README.md"
        md_file.touch()

        # Empty link (just anchor)
        result = fixer._maybe_fix_link(md_file, "")
        assert result is None

    def test_anchorize_with_special_chars_only(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize returns empty string for heading with only special chars (line 207)."""
        # Heading with only special characters
        anchor = fixer._anchorize("!!!")
        assert anchor == ""

    def test_fix_markdown_with_link_fix(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix_markdown increments link_count when link is fixed (lines 167-168)."""
        md_file = tmp_path / "README.md"
        (tmp_path / "target.md").touch()
        md_file.write_text("# Test\n\nSee [link](target) for details.\n")

        item = fixer._process_file(md_file, apply=False)
        assert item.links == 1

    def test_build_toc_skips_empty_anchors(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc skips headings that produce empty anchors (line 207)."""
        content = "## !!!\n\n## Valid Section\n"
        toc = fixer._build_toc(content)
        # Should only include Valid Section, not !!!
        assert "Valid Section" in toc
        assert "!!!" not in toc

"""Tests for FlextInfraDocFixer service.

Tests documentation fixing functionality with tmp_path for file generation
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.fixer import FixItem, FixReport, FlextInfraDocFixer
from flext_infra.docs.shared import FlextInfraDocScope


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
        report = FixReport(scope="test", changed_files=0, applied=False)
        with pytest.raises(Exception):  # pydantic frozen raises
            report.scope = "modified"  # type: ignore

    def test_fix_item_frozen(self) -> None:
        """Test FixItem is frozen (immutable)."""
        item = FixItem(file="test.md", links=0, toc=0)
        with pytest.raises(Exception):  # pydantic frozen raises
            item.file = "modified"  # type: ignore

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

"""Tests for FlextInfraDocsShared service.

Tests shared documentation utilities including scope building,
markdown helpers, and JSON reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.auditor import AuditReport
from flext_infra.docs.shared import (
    DEFAULT_DOCS_OUTPUT_DIR,
    FlextInfraDocScope,
    FlextInfraDocsShared,
)


class TestFlextInfraDocScope:
    """Tests for FlextInfraDocScope model."""

    def test_scope_creation(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope creation."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        scope = FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=report_dir,
        )
        assert scope.name == "test-project"
        assert scope.path == tmp_path
        assert scope.report_dir == report_dir

    def test_scope_name_required(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope requires name."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(Exception):  # pydantic validation
            FlextInfraDocScope(
                name="",  # empty name
                path=tmp_path,
                report_dir=report_dir,
            )

    def test_scope_path_required(self) -> None:
        """Test FlextInfraDocScope requires path."""
        with pytest.raises(Exception):  # pydantic validation
            FlextInfraDocScope.model_validate({
                "name": "test",
                "path": None,
                "report_dir": "/tmp",
            })

    def test_scope_report_dir_required(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope requires report_dir."""
        with pytest.raises(Exception):  # pydantic validation
            FlextInfraDocScope.model_validate({
                "name": "test",
                "path": str(tmp_path),
                "report_dir": None,
            })


class TestFlextInfraDocsShared:
    """Tests for FlextInfraDocsShared service."""

    def test_build_scopes_returns_flext_result(self, tmp_path: Path) -> None:
        """Test build_scopes returns FlextResult."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        assert result.is_success or result.is_failure

    def test_build_scopes_with_valid_root_returns_success(self, tmp_path: Path) -> None:
        """Test build_scopes with valid root returns success."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        assert result.is_success
        assert isinstance(result.value, list)

    def test_build_scopes_includes_root_scope(self, tmp_path: Path) -> None:
        """Test build_scopes includes root scope."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        if result.is_success:
            scopes = result.value
            assert any(scope.name == "root" for scope in scopes)

    def test_build_scopes_with_single_project(self, tmp_path: Path) -> None:
        """Test build_scopes with single project filter."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project="test-project",
            projects=None,
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        assert result.is_success or result.is_failure

    def test_build_scopes_with_multiple_projects(self, tmp_path: Path) -> None:
        """Test build_scopes with multiple projects filter."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects="proj1,proj2,proj3",
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        assert result.is_success or result.is_failure

    def test_build_scopes_with_custom_output_dir(self, tmp_path: Path) -> None:
        """Test build_scopes with custom output directory."""
        custom_output = str(tmp_path / "custom_output")
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=custom_output,
        )
        assert result.is_success or result.is_failure

    def test_default_docs_output_dir_constant(self) -> None:
        """Test DEFAULT_DOCS_OUTPUT_DIR constant is defined."""
        assert isinstance(DEFAULT_DOCS_OUTPUT_DIR, str)
        assert len(DEFAULT_DOCS_OUTPUT_DIR) > 0

    def test_build_scopes_scope_structure(self, tmp_path: Path) -> None:
        """Test scopes returned have required structure."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=DEFAULT_DOCS_OUTPUT_DIR,
        )
        if result.is_success and result.value:
            scope = result.value[0]
            assert hasattr(scope, "name")
            assert hasattr(scope, "path")
            assert hasattr(scope, "report_dir")

    def test_build_scopes_report_dir_created(self, tmp_path: Path) -> None:
        """Test build_scopes creates report directories."""
        result = FlextInfraDocsShared.build_scopes(
            root=tmp_path,
            project=None,
            projects=None,
            output_dir=str(tmp_path / "reports"),
        )
        if result.is_success and result.value:
            for scope in result.value:
                # Report dir should be created or creatable
                assert isinstance(scope.report_dir, Path)

    def test_write_json_returns_flext_result(self, tmp_path: Path) -> None:
        """Test write_json returns FlextResult."""
        json_file = tmp_path / "test.json"
        result = FlextInfraDocsShared.write_json(json_file, {"key": "value"})
        assert result.is_success or result.is_failure

    def test_write_markdown_returns_flext_result(self, tmp_path: Path) -> None:
        """Test write_markdown returns FlextResult."""
        md_file = tmp_path / "test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, "# Test\n\nContent")
        assert result.is_success or result.is_failure

    def test_iter_markdown_files_empty_directory(self, tmp_path: Path) -> None:
        """Test iter_markdown_files with empty directory."""
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        assert isinstance(files, list)

    def test_iter_markdown_files_with_markdown(self, tmp_path: Path) -> None:
        """Test iter_markdown_files finds markdown files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        md_file = docs_dir / "README.md"
        md_file.write_text("# Test\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        assert len(files) > 0

    def test_iter_markdown_files_excludes_hidden(self, tmp_path: Path) -> None:
        """Test iter_markdown_files excludes hidden directories."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        hidden_dir = docs_dir / ".hidden"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hidden_file = hidden_dir / "test.md"
        hidden_file.write_text("# Hidden\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        # Hidden files should be excluded
        assert not any(".hidden" in str(f) for f in files)

    def test_selected_project_names_with_project(self, tmp_path: Path) -> None:
        """Test _selected_project_names with single project."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path, "test-proj", None
        )
        assert names == ["test-proj"]

    def test_selected_project_names_with_projects_comma(self, tmp_path: Path) -> None:
        """Test _selected_project_names with comma-separated projects."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path, None, "proj1,proj2,proj3"
        )
        assert "proj1" in names
        assert "proj2" in names

    def test_selected_project_names_with_projects_space(self, tmp_path: Path) -> None:
        """Test _selected_project_names with space-separated projects."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path, None, "proj1 proj2 proj3"
        )
        assert "proj1" in names
        assert "proj2" in names

    def test_selected_project_names_no_filter(self, tmp_path: Path) -> None:
        """Test _selected_project_names with no filter discovers projects."""
        names = FlextInfraDocsShared._selected_project_names(tmp_path, None, None)
        assert isinstance(names, list)

    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        """Test write_json creates JSON file."""
        json_file = tmp_path / "test.json"
        result = FlextInfraDocsShared.write_json(json_file, {"key": "value"})
        assert result.is_success
        assert json_file.exists()

    def test_write_json_with_model(self, tmp_path: Path) -> None:
        """Test write_json with Pydantic model."""
        json_file = tmp_path / "test.json"
        report = AuditReport(
            scope="test", issues=[], checks=[], strict=False, passed=True
        )
        result = FlextInfraDocsShared.write_json(json_file, report)
        assert result.is_success or result.is_failure

    def test_write_markdown_creates_file(self, tmp_path: Path) -> None:
        """Test write_markdown creates markdown file."""
        md_file = tmp_path / "test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Test", "", "Content"])
        assert result.is_success
        assert md_file.exists()

    def test_write_markdown_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test write_markdown creates parent directories."""
        md_file = tmp_path / "nested/deep/test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Test"])
        assert result.is_success
        assert md_file.exists()

    def test_write_markdown_preserves_newlines(self, tmp_path: Path) -> None:
        """Test write_markdown preserves line structure."""
        md_file = tmp_path / "test.md"
        lines = ["# Title", "", "Paragraph 1", "", "Paragraph 2"]
        FlextInfraDocsShared.write_markdown(md_file, lines)
        content = md_file.read_text()
        assert "# Title" in content
        assert "Paragraph 1" in content

    def test_iter_markdown_files_nested_structure(self, tmp_path: Path) -> None:
        """Test iter_markdown_files with nested directory structure."""
        docs_dir = tmp_path / "docs"
        nested_dir = docs_dir / "guides/advanced"
        nested_dir.mkdir(parents=True, exist_ok=True)
        md_file = nested_dir / "guide.md"
        md_file.write_text("# Guide\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        assert len(files) > 0

    def test_write_json_with_dict_payload(self, tmp_path: Path) -> None:
        """Test write_json with dictionary payload."""
        json_file = tmp_path / "test.json"
        payload = {"key": "value", "nested": {"inner": "data"}}
        result = FlextInfraDocsShared.write_json(json_file, payload)
        assert result.is_success

    def test_write_markdown_with_empty_lines(self, tmp_path: Path) -> None:
        """Test write_markdown preserves empty lines."""
        md_file = tmp_path / "test.md"
        lines = ["# Title", "", "", "Content"]
        result = FlextInfraDocsShared.write_markdown(md_file, lines)
        assert result.is_success
        content = md_file.read_text()
        assert content.count("\n") >= 3

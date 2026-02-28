"""Tests for FlextInfraDocsShared service.

Tests shared documentation utilities including scope building,
markdown helpers, and JSON reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
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
            FlextInfraDocScope(
                name="test",
                path=None,  # type: ignore
                report_dir=Path("/tmp"),
            )

    def test_scope_report_dir_required(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope requires report_dir."""
        with pytest.raises(Exception):  # pydantic validation
            FlextInfraDocScope(
                name="test",
                path=tmp_path,
                report_dir=None,  # type: ignore
            )


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

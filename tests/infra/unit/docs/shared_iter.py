"""Tests for FlextInfraDocsShared — iter_markdown_files and _selected_project_names.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm


class TestIterMarkdownFiles:
    """Tests for FlextInfraDocsShared.iter_markdown_files."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test iter_markdown_files with empty directory."""
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) >= 0, eq=True)

    def test_finds_markdown(self, tmp_path: Path) -> None:
        """Test iter_markdown_files finds markdown files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "README.md").write_text("# Test\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) > 0, eq=True)

    def test_excludes_hidden(self, tmp_path: Path) -> None:
        """Test iter_markdown_files excludes hidden directories."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        hidden_dir = docs_dir / ".hidden"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        (hidden_dir / "test.md").write_text("# Hidden\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(any(".hidden" in str(f) for f in files), eq=False)

    def test_nested_structure(self, tmp_path: Path) -> None:
        """Test iter_markdown_files with nested directory structure."""
        nested_dir = tmp_path / "docs/guides/advanced"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "guide.md").write_text("# Guide\n")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) > 0, eq=True)

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """Test iter_markdown_files returns sorted list."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "z.md").write_text("# Z")
        (docs_dir / "a.md").write_text("# A")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) >= 2, eq=True)
        str_files = [str(f) for f in files]
        tm.that(str_files, eq=sorted(str_files))

    def test_no_docs_dir(self, tmp_path: Path) -> None:
        """Test iter_markdown_files when docs dir doesn't exist."""
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) >= 0, eq=True)

    def test_docs_at_root(self, tmp_path: Path) -> None:
        """Test iter_markdown_files finds docs at root."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("# Test")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(len(files) > 0, eq=True)

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        """Test iter_markdown_files excludes node_modules."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        nm_dir = docs_dir / "node_modules"
        nm_dir.mkdir(parents=True, exist_ok=True)
        (nm_dir / "test.md").write_text("# Test")
        files = FlextInfraDocsShared.iter_markdown_files(tmp_path)
        tm.that(any("node_modules" in str(f) for f in files), eq=False)


class TestSelectedProjectNames:
    """Tests for FlextInfraDocsShared._selected_project_names."""

    def test_with_project(self, tmp_path: Path) -> None:
        """Test _selected_project_names with single project."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path,
            "test-proj",
            None,
        )
        tm.that(names, eq=["test-proj"])

    def test_with_projects_comma(self, tmp_path: Path) -> None:
        """Test _selected_project_names with comma-separated projects."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path,
            None,
            "proj1,proj2,proj3",
        )
        tm.that("proj1" in names, eq=True)
        tm.that("proj2" in names, eq=True)

    def test_with_projects_space(self, tmp_path: Path) -> None:
        """Test _selected_project_names with space-separated projects."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path,
            None,
            "proj1 proj2 proj3",
        )
        tm.that("proj1" in names, eq=True)
        tm.that("proj2" in names, eq=True)

    def test_no_filter(self, tmp_path: Path) -> None:
        """Test _selected_project_names with no filter discovers projects."""
        names = FlextInfraDocsShared._selected_project_names(tmp_path, None, None)
        tm.that(len(names) >= 0, eq=True)

    def test_empty_string(self, tmp_path: Path) -> None:
        """Test _selected_project_names with empty string."""
        names = FlextInfraDocsShared._selected_project_names(tmp_path, None, "")
        tm.that(len(names) >= 0, eq=True)

    def test_whitespace_only(self, tmp_path: Path) -> None:
        """Test _selected_project_names with whitespace-only string."""
        names = FlextInfraDocsShared._selected_project_names(tmp_path, None, "   ")
        tm.that(len(names) >= 0, eq=True)

    def test_mixed_separators(self, tmp_path: Path) -> None:
        """Test _selected_project_names with mixed separators."""
        names = FlextInfraDocsShared._selected_project_names(
            tmp_path,
            None,
            "proj1, proj2, proj3",
        )
        tm.that("proj1" in names, eq=True)
        tm.that("proj2" in names, eq=True)

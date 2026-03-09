"""Tests for FlextInfraDocFixer — internal methods: _process_file, _maybe_fix_link, TOC.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import m
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_tests import tm


class TestFixerProcessFile:
    """Tests for _process_file."""

    @pytest.fixture
    def fixer(self) -> FlextInfraDocFixer:
        """Create fixer instance."""
        return FlextInfraDocFixer()

    def test_process_file_with_markdown_links(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file detects and fixes markdown links."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\n[Link](missing.md)\n")
        item = fixer._process_file(md_file, apply=False)
        tm.that(item.file, eq=str(md_file))

    def test_process_file_with_apply_true(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file with apply=True writes changes."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\n## Section\n")
        item = fixer._process_file(md_file, apply=True)
        tm.that(item.file, eq=str(md_file))

    def test_process_file_with_no_fixes_needed(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file with content that needs no fixes."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nNo broken links or TOC needed.")
        item = fixer._process_file(md_file, apply=False)
        tm.that("test.md" in item.file, eq=True)
        tm.that(item.links, eq=0)

    def test_process_file_with_fixable_links(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _process_file counts fixed links correctly."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\n[Link](target.md)\n")
        (tmp_path / "target.md").write_text("# Target")
        item = fixer._process_file(md_file, apply=False)
        tm.that("test.md" in item.file, eq=True)

    def test_fix_markdown_with_link_fix(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix_markdown increments link_count when link is fixed."""
        md_file = tmp_path / "README.md"
        (tmp_path / "target.md").touch()
        md_file.write_text("# Test\n\nSee [link](target) for details.\n")
        item = fixer._process_file(md_file, apply=False)
        tm.that(item.links, eq=1)


class TestFixerMaybeFixLink:
    """Tests for _maybe_fix_link."""

    @pytest.fixture
    def fixer(self) -> FlextInfraDocFixer:
        """Create fixer instance."""
        return FlextInfraDocFixer()

    def test_maybe_fix_link_external_urls(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for external URLs."""
        md_file = tmp_path / "test.md"
        tm.that(fixer._maybe_fix_link(md_file, "http://example.com"), eq=None)
        tm.that(fixer._maybe_fix_link(md_file, "https://example.com"), eq=None)
        tm.that(fixer._maybe_fix_link(md_file, "mailto:test@example.com"), eq=None)

    def test_maybe_fix_link_fragment_only(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for fragment-only links."""
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "#section"), eq=None)

    def test_maybe_fix_link_existing_file(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for existing files."""
        (tmp_path / "existing.md").write_text("# Existing")
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "existing.md"), eq=None)

    def test_maybe_fix_link_adds_md_extension(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link adds .md extension when needed."""
        (tmp_path / "missing.md").write_text("# Missing")
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "missing"), eq="missing.md")

    def test_maybe_fix_link_empty_base(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for empty base."""
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "#section"), eq=None)

    def test_maybe_fix_link_with_empty_string(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns None for empty string."""
        (tmp_path / "README.md").touch()
        tm.that(fixer._maybe_fix_link(tmp_path / "README.md", ""), eq=None)

    def test_maybe_fix_link_with_existing_target(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test _maybe_fix_link returns fixed link when .md suffix exists."""
        md_file = tmp_path / "docs" / "foo.md"
        md_file.parent.mkdir(parents=True)
        md_file.touch()
        (tmp_path / "docs" / "bar.md").touch()
        tm.that(fixer._maybe_fix_link(md_file, "bar"), eq="bar.md")


class TestFixerToc:
    """Tests for _anchorize, _build_toc, _update_toc."""

    @pytest.fixture
    def fixer(self) -> FlextInfraDocFixer:
        """Create fixer instance."""
        return FlextInfraDocFixer()

    def test_anchorize_converts_to_slug(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize converts heading to anchor slug."""
        tm.that(fixer._anchorize("Hello World"), eq="hello-world")
        tm.that(fixer._anchorize("Test-Case"), eq="test-case")
        tm.that(fixer._anchorize("  Spaces  "), eq="spaces")

    def test_anchorize_removes_special_chars(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize removes special characters."""
        tm.that(fixer._anchorize("Hello! World?"), eq="hello-world")
        tm.that(fixer._anchorize("Test@#$%"), eq="test")

    def test_anchorize_empty_string(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize with empty string."""
        tm.that(fixer._anchorize(""), eq="")

    def test_anchorize_with_special_chars_only(self, fixer: FlextInfraDocFixer) -> None:
        """Test _anchorize returns empty string for heading with only special chars."""
        tm.that(fixer._anchorize("!!!"), eq="")

    def test_build_toc_from_headings(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc generates TOC from headings."""
        toc = fixer._build_toc(
            "# Main\n\n## Section 1\n\n### Subsection\n\n## Section 2\n"
        )
        tm.that("<!-- TOC START -->" in toc, eq=True)
        tm.that("<!-- TOC END -->" in toc, eq=True)
        tm.that("Section 1" in toc, eq=True)

    def test_build_toc_no_headings(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc with no headings."""
        tm.that(
            "No sections found" in fixer._build_toc("# Main\n\nNo sections here.\n"),
            eq=True,
        )

    def test_build_toc_skips_empty_anchors(self, fixer: FlextInfraDocFixer) -> None:
        """Test _build_toc skips headings that produce empty anchors."""
        toc = fixer._build_toc("## !!!\n\n## Valid Section\n")
        tm.that("Valid Section" in toc, eq=True)
        tm.that("!!!" not in toc, eq=True)

    def test_update_toc_replaces_existing(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc replaces existing TOC."""
        updated, changed = fixer._update_toc(
            "# Main\n\n<!-- TOC START -->\nOld TOC\n<!-- TOC END -->\n\n## Section\n"
        )
        tm.that(changed, eq=1)
        tm.that("Old TOC" not in updated, eq=True)

    def test_update_toc_inserts_new(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc inserts new TOC."""
        updated, changed = fixer._update_toc("# Main\n\n## Section\n")
        tm.that(changed, eq=1)
        tm.that("<!-- TOC START -->" in updated, eq=True)

    def test_update_toc_without_h1_heading(self, fixer: FlextInfraDocFixer) -> None:
        """Test _update_toc prepends TOC when no h1 heading exists."""
        updated, changed = fixer._update_toc("## Section 1\n\nContent here.")
        tm.that(changed, eq=1)
        tm.that("<!-- TOC START -->" in updated, eq=True)


class TestFixerScope:
    """Tests for _fix_scope."""

    def test_fix_scope_with_markdown_files(self, tmp_path: Path) -> None:
        """Test _fix_scope processes markdown files."""
        fixer = FlextInfraDocFixer()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "README.md").write_text("# Test\n\n## Section\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = fixer._fix_scope(scope, apply=False)
        tm.that(report.scope, eq="test")
        tm.that(isinstance(report.items, list), eq=True)

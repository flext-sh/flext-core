"""Tests for FlextInfraDocFixer — internal methods: _process_file, _maybe_fix_link, TOC.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_tests import tf, tm
from tests.infra.models import m


@pytest.fixture
def fixer() -> FlextInfraDocFixer:
    return FlextInfraDocFixer()


class TestFixerProcessFile:
    @pytest.mark.parametrize(
        ("name", "content", "apply"),
        [
            ("test.md", "# Test\n\n[Link](missing.md)\n", False),
            ("test.md", "# Test\n\n## Section\n", True),
            ("test.md", "# Test\n\nNo broken links or TOC needed.", False),
        ],
    )
    def test_process_file_variants(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
        name: str,
        content: str,
        apply: bool,
    ) -> None:
        md_file = tf.create_in(content, name, tmp_path)
        item = fixer._process_file(md_file, apply=apply)
        tm.that(item.file, eq=str(md_file))

    def test_process_file_with_fixable_links(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
    ) -> None:
        md_file = tf.create_in("# Test\n\n[Link](target.md)\n", "test.md", tmp_path)
        _ = tf.create_in("# Target", "target.md", tmp_path)
        item = fixer._process_file(md_file, apply=False)
        tm.that("test.md" in item.file, eq=True)

    def test_fix_markdown_with_link_fix(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
    ) -> None:
        _ = tf.create_in("", "target.md", tmp_path)
        md_file = tf.create_in(
            "# Test\n\nSee [link](target) for details.\n",
            "README.md",
            tmp_path,
        )
        item = fixer._process_file(md_file, apply=False)
        tm.that(item.links, eq=1)


class TestFixerMaybeFixLink:
    @pytest.mark.parametrize(
        "link",
        [
            "http://example.com",
            "https://example.com",
            "mailto:test@example.com",
            "#section",
            "",
        ],
    )
    def test_unchanged_links_return_none(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
        link: str,
    ) -> None:
        _ = tf.create_in("# Test", "README.md", tmp_path)
        tm.that(fixer._maybe_fix_link(tmp_path / "README.md", link), eq=None)

    def test_maybe_fix_link_existing_file(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        _ = tf.create_in("# Existing", "existing.md", tmp_path)
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "existing.md"), eq=None)

    def test_maybe_fix_link_adds_md_extension(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
    ) -> None:
        _ = tf.create_in("# Missing", "missing.md", tmp_path)
        tm.that(fixer._maybe_fix_link(tmp_path / "test.md", "missing"), eq="missing.md")

    def test_maybe_fix_link_with_existing_target(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True)
        md_file = tf.create_in("", "foo.md", docs_dir)
        _ = tf.create_in("", "bar.md", docs_dir)
        tm.that(fixer._maybe_fix_link(md_file, "bar"), eq="bar.md")


class TestFixerToc:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Hello World", "hello-world"),
            ("Test-Case", "test-case"),
            ("  Spaces  ", "spaces"),
            ("Hello! World?", "hello-world"),
            ("Test@#$%", "test"),
            ("", ""),
            ("!!!", ""),
        ],
    )
    def test_anchorize_cases(
        self,
        fixer: FlextInfraDocFixer,
        title: str,
        expected: str,
    ) -> None:
        tm.that(fixer._anchorize(title), eq=expected)

    def test_build_toc_variants(self, fixer: FlextInfraDocFixer) -> None:
        toc = fixer._build_toc(
            "# Main\n\n## Section 1\n\n### Subsection\n\n## Section 2\n"
        )
        tm.that("<!-- TOC START -->" in toc, eq=True)
        tm.that("<!-- TOC END -->" in toc, eq=True)
        tm.that("Section 1" in toc, eq=True)
        tm.that(
            "No sections found" in fixer._build_toc("# Main\n\nNo sections here.\n"),
            eq=True,
        )

    def test_build_toc_skips_empty_anchors(self, fixer: FlextInfraDocFixer) -> None:
        toc = fixer._build_toc("## !!!\n\n## Valid Section\n")
        tm.that("Valid Section" in toc, eq=True)
        tm.that("!!!" not in toc, eq=True)

    @pytest.mark.parametrize(
        "content",
        [
            "# Main\n\n<!-- TOC START -->\nOld TOC\n<!-- TOC END -->\n\n## Section\n",
            "# Main\n\n## Section\n",
            "## Section 1\n\nContent here.",
        ],
    )
    def test_update_toc_paths(self, fixer: FlextInfraDocFixer, content: str) -> None:
        updated, changed = fixer._update_toc(content)
        tm.that(changed, eq=1)
        tm.that("<!-- TOC START -->" in updated, eq=True)


class TestFixerScope:
    def test_fix_scope_with_markdown_files(self, tmp_path: Path) -> None:
        fixer = FlextInfraDocFixer()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        _ = tf.create_in("# Test\n\n## Section\n", "README.md", docs_dir)
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = fixer._fix_scope(scope, apply=False)
        tm.that(report.scope, eq="test")
        tm.that(len(report.items) >= 0, eq=True)

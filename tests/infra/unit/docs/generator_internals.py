"""Tests for FlextInfraDocGenerator — internal methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import m
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_tests import tm


class TestGeneratorScope:
    """Tests for _generate_scope and related methods."""

    @pytest.fixture
    def gen(self) -> FlextInfraDocGenerator:
        """Create generator instance."""
        return FlextInfraDocGenerator()

    def test_generate_scope_root_scope(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_scope with root scope."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="root", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = gen._generate_scope(scope, apply=False, workspace_root=tmp_path)
        tm.that(report.scope, eq="root")

    def test_generate_scope_project_scope(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_scope with project scope."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test-project", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = gen._generate_scope(scope, apply=False, workspace_root=tmp_path)
        tm.that(report.scope, eq="test-project")

    def test_generate_root_docs_creates_files(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_root_docs creates placeholder files."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="root", path=tmp_path, report_dir=tmp_path / "reports"
        )
        files = gen._generate_root_docs(scope, apply=False)
        tm.that(len(files), eq=3)

    def test_generate_project_guides_no_source(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_guides with no source guides."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        files = gen._generate_project_guides(
            scope, workspace_root=tmp_path, apply=False
        )
        tm.that(files, eq=[])

    def test_generate_project_guides_with_source(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_guides with source guides."""
        guides_dir = tmp_path / "docs/guides"
        guides_dir.mkdir(parents=True, exist_ok=True)
        (guides_dir / "test.md").write_text("# Test Guide\n\nContent.\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path / "project", report_dir=tmp_path / "reports"
        )
        files = gen._generate_project_guides(
            scope, workspace_root=tmp_path, apply=False
        )
        tm.that(isinstance(files, list), eq=True)

    def test_generate_project_mkdocs_creates_config(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_mkdocs creates mkdocs.yml."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        files = gen._generate_project_mkdocs(scope, apply=False)
        tm.that(len(files), eq=1)
        tm.that(files[0].path.endswith("mkdocs.yml"), eq=True)

    def test_generate_project_mkdocs_skips_existing(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _generate_project_mkdocs skips existing mkdocs.yml."""
        (tmp_path / "mkdocs.yml").write_text("site_name: Test\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        files = gen._generate_project_mkdocs(scope, apply=False)
        tm.that(files, eq=[])


class TestGeneratorHelpers:
    """Tests for content helpers and write_if_needed."""

    @pytest.fixture
    def gen(self) -> FlextInfraDocGenerator:
        """Create generator instance."""
        return FlextInfraDocGenerator()

    def test_project_guide_content_adds_heading(
        self, gen: FlextInfraDocGenerator
    ) -> None:
        """Test _project_guide_content adds project heading."""
        result = gen._project_guide_content(
            "# Original Title\n\nContent here.\n", "my-project", "guide.md"
        )
        tm.that("my-project - Original Title" in result, eq=True)

    def test_project_guide_content_preserves_body(
        self, gen: FlextInfraDocGenerator
    ) -> None:
        """Test _project_guide_content preserves body content."""
        result = gen._project_guide_content(
            "# Title\n\nBody content.\n", "proj", "guide.md"
        )
        tm.that("Body content" in result, eq=True)

    def test_sanitize_internal_anchor_links_removes_local_links(
        self, gen: FlextInfraDocGenerator
    ) -> None:
        """Test _sanitize_internal_anchor_links removes local markdown links."""
        result = gen._sanitize_internal_anchor_links(
            "[Link](local.md) and [External](http://example.com)"
        )
        tm.that("Link" in result, eq=True)
        tm.that("http://example.com" in result, eq=True)

    def test_sanitize_internal_anchor_links_preserves_external(
        self, gen: FlextInfraDocGenerator
    ) -> None:
        """Test _sanitize_internal_anchor_links preserves external links."""
        result = gen._sanitize_internal_anchor_links(
            "[Local](local.md) [External](https://example.com)"
        )
        tm.that("https://example.com" in result, eq=True)

    def test_normalize_anchor_converts_to_slug(
        self, gen: FlextInfraDocGenerator
    ) -> None:
        """Test _normalize_anchor converts heading to slug."""
        tm.that(gen._normalize_anchor("Hello World"), eq="hello-world")
        tm.that(gen._normalize_anchor("Test-Case"), eq="test-case")

    def test_normalize_anchor_empty_string(self, gen: FlextInfraDocGenerator) -> None:
        """Test _normalize_anchor with empty string."""
        tm.that(gen._normalize_anchor(""), eq="")

    def test_build_toc_from_headings(self, gen: FlextInfraDocGenerator) -> None:
        """Test _build_toc generates TOC from headings."""
        toc = gen._build_toc("# Main\n\n## Section 1\n\n### Subsection\n")
        tm.that("<!-- TOC START -->" in toc, eq=True)
        tm.that("Section 1" in toc, eq=True)

    def test_build_toc_with_no_headings(self, gen: FlextInfraDocGenerator) -> None:
        """Test _build_toc with no headings."""
        tm.that(
            "No sections found" in gen._build_toc("# Main\n\nNo sections.\n"), eq=True
        )

    def test_update_toc_replaces_existing(self, gen: FlextInfraDocGenerator) -> None:
        """Test _update_toc replaces existing TOC."""
        result = gen._update_toc(
            "# Main\n\n<!-- TOC START -->\nOld\n<!-- TOC END -->\n\n## Section\n"
        )
        tm.that("Old" not in result, eq=True)

    def test_update_toc_inserts_new(self, gen: FlextInfraDocGenerator) -> None:
        """Test _update_toc inserts new TOC."""
        result = gen._update_toc("# Main\n\n## Section\n")
        tm.that("<!-- TOC START -->" in result, eq=True)

    def test_write_if_needed_no_change(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed skips unchanged content."""
        path = tmp_path / "test.md"
        path.write_text("# Test\n")
        tm.that(gen._write_if_needed(path, "# Test\n", apply=True).written, eq=False)

    def test_write_if_needed_with_apply(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed writes when apply=True."""
        path = tmp_path / "test.md"
        result = gen._write_if_needed(path, "# New Content\n", apply=True)
        tm.that(result.written, eq=True)
        tm.that(path.exists(), eq=True)

    def test_write_if_needed_dry_run(
        self, gen: FlextInfraDocGenerator, tmp_path: Path
    ) -> None:
        """Test _write_if_needed dry-run mode."""
        path = tmp_path / "test.md"
        tm.that(
            gen._write_if_needed(path, "# New Content\n", apply=False).written, eq=False
        )

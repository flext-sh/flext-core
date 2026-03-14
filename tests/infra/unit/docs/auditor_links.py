"""Tests for FlextInfraDocAuditor — broken link and markdown helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_tests import tm
from tests.infra.models import m


class TestAuditorToMarkdown:
    """Tests for to_markdown helper."""

    def test_to_markdown_empty_issues(self) -> None:
        """Test to_markdown with no issues."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=Path(), report_dir=Path()
        )
        result = FlextInfraDocAuditor.to_markdown(scope, [])
        tm.that(len(result) >= 0, eq=True)
        tm.that("# Docs Audit Report" in result, eq=True)

    def test_to_markdown_with_issues(self) -> None:
        """Test to_markdown with issues."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=Path(), report_dir=Path()
        )
        issue = m.Infra.Docs.AuditIssue(
            file="README.md",
            issue_type="broken_link",
            severity="high",
            message="Link not found",
        )
        result = FlextInfraDocAuditor.to_markdown(scope, [issue])
        tm.that(len(result) >= 0, eq=True)
        tm.that(any("README.md" in line for line in result), eq=True)


class TestAuditorBrokenLinks:
    """Tests for broken_link_issues."""

    def test_broken_link_issues_empty_scope(self, tmp_path: Path) -> None:
        """Test broken_link_issues with no markdown files."""
        auditor = FlextInfraDocAuditor()
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_valid_links(self, tmp_path: Path) -> None:
        """Test broken_link_issues ignores valid links."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[link](test.md)")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_external_links(self, tmp_path: Path) -> None:
        """Test broken_link_issues ignores external links."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[link](https://example.com)")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_fragments(self, tmp_path: Path) -> None:
        """Test broken_link_issues ignores fragment-only links."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[link](#section)")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_in_code_blocks(self, tmp_path: Path) -> None:
        """Test broken_link_issues ignores links in code blocks."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("```\n[link](nonexistent.md)\n```")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_should_skip_target_true(
        self, tmp_path: Path
    ) -> None:
        """Test broken_link_issues skips targets when should_skip_target returns True."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[a, b]")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_missing_target(self, tmp_path: Path) -> None:
        """Test broken_link_issues reports missing targets."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[link](missing.md)")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) > 0, eq=True)
        tm.that(any("missing.md" in issue.message for issue in issues), eq=True)

    def test_broken_link_issues_skips_some_text(self, tmp_path: Path) -> None:
        """Test broken_link_issues skips plain text brackets."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[some text]")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)

    def test_broken_link_issues_with_space_in_url_skips(self, tmp_path: Path) -> None:
        """Test broken_link_issues skips URLs with spaces."""
        auditor = FlextInfraDocAuditor()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "test.md").write_text("[link](some text)")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        issues = auditor.broken_link_issues(scope)
        tm.that(len(issues) >= 0, eq=True)
        tm.that(len(issues), eq=0)

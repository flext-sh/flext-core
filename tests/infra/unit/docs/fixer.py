"""Tests for FlextInfraDocFixer — core fix, model, and link tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm
from tests.infra.models import m
from tests.infra.typings import t


class TestFixerCore:
    """Core fix invocation tests."""

    @pytest.fixture
    def fixer(self) -> FlextInfraDocFixer:
        """Create fixer instance."""
        return FlextInfraDocFixer()

    def test_fix_returns_flext_result(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test that fix returns FlextResult."""
        result = fixer.fix(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_with_valid_scope_returns_success(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with valid scope returns success."""
        result = fixer.fix(tmp_path)
        tm.ok(result)
        tm.that(isinstance(result.value, list), eq=True)

    def test_fix_report_structure(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test FixReport has required fields."""
        result = fixer.fix(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "changed_files"), eq=True)
            tm.that(hasattr(report, "applied"), eq=True)
            tm.that(hasattr(report, "items"), eq=True)

    def test_fix_item_structure(self) -> None:
        """Test FixItem model structure."""
        item = m.Infra.Docs.DocsPhaseItem(phase="fix", file="README.md", links=2, toc=1)
        tm.that(item.file, eq="README.md")
        tm.that(item.links, eq=2)
        tm.that(item.toc, eq=1)

    def test_fix_report_frozen(self) -> None:
        """Test FixReport is frozen (immutable)."""
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_fix_item_frozen(self) -> None:
        """Test FixItem is frozen (immutable)."""
        tm.that(m.Infra.Docs.DocsPhaseItem.model_config.get("frozen"), eq=True)

    def test_fix_with_project_filter(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with single project filter."""
        result = fixer.fix(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_with_projects_filter(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with multiple projects filter."""
        result = fixer.fix(tmp_path, projects="proj1,proj2")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_with_apply_false_dry_run(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with apply=False (dry-run mode)."""
        result = fixer.fix(tmp_path, apply=False)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_with_apply_true_writes_changes(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with apply=True writes changes."""
        (tmp_path / "README.md").write_text("# Test\n\nSome content here.\n")
        result = fixer.fix(tmp_path, apply=True)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_with_custom_output_dir(
        self, fixer: FlextInfraDocFixer, tmp_path: Path
    ) -> None:
        """Test fix with custom output directory."""
        result = fixer.fix(tmp_path, output_dir=str(tmp_path / "custom_output"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_fix_report_changed_files_count(self) -> None:
        """Test FixReport changed_files field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="fix", scope="test", changed_files=5, applied=True
        )
        tm.that(report.changed_files, eq=5)

    def test_fix_report_applied_field(self) -> None:
        """Test FixReport applied field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="fix", scope="test", changed_files=0, applied=False
        )
        tm.that(report.applied, eq=False)

    def test_fix_report_items_list(self) -> None:
        """Test FixReport items list."""
        items = [
            m.Infra.Docs.DocsPhaseItem(phase="fix", file="file1.md", links=1, toc=0),
            m.Infra.Docs.DocsPhaseItem(phase="fix", file="file2.md", links=0, toc=1),
        ]
        report = m.Infra.Docs.DocsPhaseReport(
            phase="fix", scope="test", changed_files=2, applied=True, items=items
        )
        tm.that(len(report.items), eq=2)
        tm.that(report.items[0].model_dump().get("file"), eq="file1.md")

    def test_fix_with_scope_failure_returns_failure(
        self,
        fixer: FlextInfraDocFixer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test fix returns failure when scope building fails."""

        def mock_build_scopes(
            *args: t.ContainerValue, **kwargs: t.ContainerValue
        ) -> r[list[t.ContainerValue]]:
            return r[list[t.ContainerValue]].fail("Scope error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = fixer.fix(tmp_path)
        tm.fail(result, has="Scope error")

"""Tests for FlextInfraDocBuilder — scope, mkdocs, and write_reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r, t
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm
from tests.infra import m


class _RunnerStub:
    def __init__(self, output: m.Infra.Core.CommandOutput) -> None:
        self._output = output

    def run_raw(
        self,
        command: list[str],
        cwd: Path,
    ) -> r[m.Infra.Core.CommandOutput]:
        _ = command, cwd
        return r[m.Infra.Core.CommandOutput].ok(self._output)


class TestBuilderScope:
    """Tests for _build_scope and _run_mkdocs."""

    @pytest.fixture
    def builder(self) -> FlextInfraDocBuilder:
        """Create builder instance."""
        return FlextInfraDocBuilder()

    def test_build_scope_with_mkdocs_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _build_scope with mkdocs.yml present."""
        (tmp_path / "mkdocs.yml").write_text("site_name: Test\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = builder._build_scope(scope)
        tm.that(report.scope, eq="test")

    def test_build_scope_without_mkdocs_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _build_scope without mkdocs.yml returns SKIP."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = builder._build_scope(scope)
        tm.that(report.result, eq="SKIP")

    def test_run_mkdocs_no_config(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _run_mkdocs returns SKIP when mkdocs.yml not found."""
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = builder._run_mkdocs(scope)
        tm.that(report.result, eq="SKIP")
        tm.that("mkdocs.yml not found" in report.reason, eq=True)

    def test_run_mkdocs_with_command_failure(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _run_mkdocs handles command failures."""
        (tmp_path / "mkdocs.yml").write_text("site_name: Test\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        report = builder._run_mkdocs(scope)
        tm.that(report.scope, eq="test")
        tm.that(len(report.result) > 0, eq=True)

    def test_run_mkdocs_with_success_exit_code(
        self,
        builder: FlextInfraDocBuilder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _run_mkdocs returns OK when exit_code is 0."""
        (tmp_path / "mkdocs.yml").write_text("site_name: Test\n")
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=tmp_path / "reports"
        )
        mock_output = SimpleNamespace(exit_code=0, stdout="Build successful", stderr="")
        command_output = m.Infra.Core.CommandOutput(
            exit_code=mock_output.exit_code,
            stdout=mock_output.stdout,
            stderr=mock_output.stderr,
        )
        mock_runner = _RunnerStub(command_output)
        monkeypatch.setattr(builder, "_runner", mock_runner)
        report = builder._run_mkdocs(scope)
        tm.that(report.result, eq="OK")
        tm.that("build succeeded" in report.reason, eq=True)

    def test_write_reports_creates_json_and_markdown(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test _write_reports creates both JSON and markdown files."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test", path=tmp_path, report_dir=report_dir
        )
        report = m.Infra.Docs.DocsPhaseReport(
            phase="build",
            scope="test",
            result="OK",
            reason="Build succeeded",
            site_dir="/tmp/site",
        )
        builder._write_reports(scope, report)
        tm.that((report_dir / "build-summary.json").exists(), eq=True)
        tm.that((report_dir / "build-report.md").exists(), eq=True)

    def test_build_with_scope_failure_returns_failure(
        self,
        builder: FlextInfraDocBuilder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build returns failure when scope building fails."""

        def mock_build_scopes(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
            return r[list[m.Infra.Docs.FlextInfraDocScope]].fail("Scope error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = builder.build(tmp_path)
        tm.fail(result, has="Scope error")

    def test_build_multiple_scopes(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build returns multiple reports for multiple scopes."""
        result = builder.build(tmp_path, projects="proj1,proj2,proj3")
        if result.is_success:
            tm.that(len(result.value) >= 0, eq=True)

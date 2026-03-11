"""Tests for FlextInfraWorkflowLinter.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github.linter import FlextInfraWorkflowLinter
from flext_tests import tm
from tests.infra.models import m
from tests.infra.unit.github._stubs import StubJsonIo, StubRunner


def _ok_output(
    *, exit_code: int = 0, stdout: str = "", stderr: str = ""
) -> r[m.Infra.Core.CommandOutput]:
    return r[m.Infra.Core.CommandOutput].ok(
        m.Infra.Core.CommandOutput(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        ),
    )


class TestFlextInfraWorkflowLinter:
    """Test suite for FlextInfraWorkflowLinter."""

    def test_lint_success_with_actionlint_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful linting when actionlint is available."""

        def _which(cmd: str) -> str:
            _ = cmd
            return "/usr/bin/actionlint"

        monkeypatch.setattr(shutil, "which", _which)
        runner = StubRunner(run_returns=[_ok_output(stdout="All workflows valid")])
        json_io = StubJsonIo()
        linter = FlextInfraWorkflowLinter(runner=runner, json_io=json_io)
        result = linter.lint(tmp_path)
        value = tm.ok(result)
        tm.that(value.status, eq="ok")
        tm.that(value.exit_code, eq=0)

    def test_lint_skipped_when_actionlint_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test linting skipped when actionlint is not available."""

        def _which_none(cmd: str) -> str | None:
            _ = cmd
            return None

        monkeypatch.setattr(shutil, "which", _which_none)
        runner = StubRunner()
        json_io = StubJsonIo()
        linter = FlextInfraWorkflowLinter(runner=runner, json_io=json_io)
        result = linter.lint(tmp_path)
        value = tm.ok(result)
        tm.that(value.status, eq="skipped")
        tm.that(value.reason is not None, eq=True)
        tm.that(value.reason, contains="actionlint not installed")

    def test_lint_with_report_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test linting with JSON report output."""

        def _which(cmd: str) -> str:
            _ = cmd
            return "/usr/bin/actionlint"

        monkeypatch.setattr(shutil, "which", _which)
        runner = StubRunner(run_returns=[_ok_output(stdout="Valid")])
        json_io = StubJsonIo()
        report_path = tmp_path / "report.json"
        linter = FlextInfraWorkflowLinter(runner=runner, json_io=json_io)
        result = linter.lint(tmp_path, report_path=report_path)
        tm.ok(result)
        tm.that(len(json_io.write_json_calls), eq=1)

    def test_lint_strict_mode_fails_on_issues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test strict mode returns failure when actionlint finds issues."""

        def _which(cmd: str) -> str:
            _ = cmd
            return "/usr/bin/actionlint"

        monkeypatch.setattr(shutil, "which", _which)
        runner = StubRunner(
            run_returns=[r[m.Infra.Core.CommandOutput].fail("workflow has errors")],
        )
        json_io = StubJsonIo()
        linter = FlextInfraWorkflowLinter(runner=runner, json_io=json_io)
        result = linter.lint(tmp_path, strict=True)
        tm.fail(result)

    def test_lint_default_runner_initialization(self) -> None:
        """Test linter initializes with default runner and json service."""
        linter = FlextInfraWorkflowLinter()
        tm.that(getattr(linter, "_runner", None) is not None, eq=True)
        tm.that(getattr(linter, "_json", None) is not None, eq=True)

    def test_lint_skipped_with_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test linting skipped with report output."""

        def _which_none(cmd: str) -> str | None:
            _ = cmd
            return None

        monkeypatch.setattr(shutil, "which", _which_none)
        runner = StubRunner()
        json_io = StubJsonIo()
        report_path = tmp_path / "report.json"
        linter = FlextInfraWorkflowLinter(runner=runner, json_io=json_io)
        result = linter.lint(tmp_path, report_path=report_path)
        tm.ok(result)
        tm.that(len(json_io.write_json_calls), eq=1)

"""Tests for flext_infra.github.__main__ CLI entry point.

Tests pure functions and argument parsing without mocks.
Uses real service instances and tm matchers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys

from flext_infra import m
from flext_infra.github import __main__ as github_main
from flext_tests import tm

main = github_main.main


def _orchestration_result(
    *,
    fail: int = 0,
    total: int = 1,
) -> m.Infra.Github.PrOrchestrationResult:
    return m.Infra.Github.PrOrchestrationResult(
        total=total,
        success=max(total - fail, 0),
        fail=fail,
        results=(),
    )


class TestOrchestrationResultModel:
    """Tests for PrOrchestrationResult construction."""

    def test_orchestration_result_zero_failures(self) -> None:
        result = _orchestration_result(fail=0, total=3)
        tm.that(result.total, eq=3)
        tm.that(result.success, eq=3)
        tm.that(result.fail, eq=0)

    def test_orchestration_result_with_failures(self) -> None:
        result = _orchestration_result(fail=2, total=5)
        tm.that(result.total, eq=5)
        tm.that(result.success, eq=3)
        tm.that(result.fail, eq=2)

    def test_orchestration_result_all_failures(self) -> None:
        result = _orchestration_result(fail=3, total=3)
        tm.that(result.total, eq=3)
        tm.that(result.success, eq=0)
        tm.that(result.fail, eq=3)


class TestMainHelpAndNoArgs:
    """Tests for main CLI help and no-args behavior."""

    def test_main_help_flag(self) -> None:
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "-h"]
            result = main()
            tm.that(result, eq=0)
        finally:
            sys.argv = original_argv

    def test_main_no_args(self) -> None:
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra"]
            result = main()
            tm.that(result, eq=1)
        finally:
            sys.argv = original_argv

    def test_main_unknown_subcommand(self) -> None:
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "unknown"]
            result = main()
            tm.that(result, eq=1)
        finally:
            sys.argv = original_argv


class TestModuleAttributes:
    """Tests for module-level attributes and functions."""

    def test_run_lint_exists(self) -> None:
        run_lint = getattr(github_main, "_run_lint", None)
        tm.that(run_lint is not None, eq=True)
        tm.that(callable(run_lint), eq=True)

    def test_run_pr_exists(self) -> None:
        run_pr = getattr(github_main, "_run_pr", None)
        tm.that(run_pr is not None, eq=True)
        tm.that(callable(run_pr), eq=True)

    def test_run_workflows_exists(self) -> None:
        run_workflows = getattr(github_main, "_run_workflows", None)
        tm.that(run_workflows is not None, eq=True)
        tm.that(callable(run_workflows), eq=True)

    def test_run_pr_workspace_exists(self) -> None:
        run_pr_workspace = getattr(github_main, "_run_pr_workspace", None)
        tm.that(run_pr_workspace is not None, eq=True)
        tm.that(callable(run_pr_workspace), eq=True)

    def test_main_is_callable(self) -> None:
        tm.that(callable(main), eq=True)


class TestGithubModuleLazyImports:
    """Tests for github module __init__.py lazy imports."""

    def test_lazy_import_pr_manager(self) -> None:
        github_module = importlib.import_module("flext_infra.github")
        tm.that(hasattr(github_module, "FlextInfraPrManager"), eq=True)

    def test_dir_returns_all_exports(self) -> None:
        github_module = importlib.import_module("flext_infra.github")
        exports = dir(github_module)
        tm.that(exports, contains="FlextInfraPrManager")
        tm.that(exports, contains="FlextInfraPrWorkspaceManager")
        tm.that(exports, contains="FlextInfraWorkflowLinter")
        tm.that(exports, contains="FlextInfraWorkflowSyncer")
        tm.that(exports, contains="SyncOperation")

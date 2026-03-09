"""Workflow linter service for GitHub Actions validation.

Wraps actionlint execution with FlextResult error handling,
replacing scripts/github/lint_workflows.py with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flext_core import r
from flext_infra import FlextInfraCommandRunner, FlextInfraUtilitiesIo, m, p


class FlextInfraWorkflowLinter:
    """Infrastructure service for GitHub Actions workflow linting.

    Delegates to ``actionlint`` for validation and persists JSON reports.
    """

    def __init__(
        self,
        runner: p.Infra.CommandRunner | None = None,
        json_io: FlextInfraUtilitiesIo | None = None,
    ) -> None:
        """Initialize the workflow linter."""
        self._runner: p.Infra.CommandRunner = runner or FlextInfraCommandRunner()
        self._json = json_io or FlextInfraUtilitiesIo()

    def lint(
        self,
        root: Path,
        *,
        report_path: Path | None = None,
        strict: bool = False,
    ) -> r[m.Infra.Github.WorkflowLintResult]:
        """Run actionlint on the repository and return results.

        Args:
            root: Repository root directory.
            report_path: Optional path for JSON report output.
            strict: If True, treat lint failures as errors.

        Returns:
            FlextResult with lint status payload.

        """
        actionlint = shutil.which("actionlint")
        if actionlint is None:
            payload_skipped = m.Infra.Github.WorkflowLintResult(
                status="skipped",
                reason="actionlint not installed",
            )
            if report_path is not None:
                self._json.write_json(report_path, payload_skipped, sort_keys=True)
            return r[m.Infra.Github.WorkflowLintResult].ok(payload_skipped)
        result: r[m.Infra.Core.CommandOutput] = self._runner.run([actionlint], cwd=root)
        if result.is_success:
            output = result.value
            payload = m.Infra.Github.WorkflowLintResult(
                status="ok",
                exit_code=output.exit_code,
                stdout=output.stdout,
                stderr=output.stderr,
            )
        else:
            payload = m.Infra.Github.WorkflowLintResult(
                status="fail",
                exit_code=1,
                detail=result.error or "",
            )
        if report_path is not None:
            self._json.write_json(report_path, payload, sort_keys=True)
        if payload.status == "fail" and strict:
            return r[m.Infra.Github.WorkflowLintResult].fail(
                result.error or "actionlint found issues",
            )
        return r[m.Infra.Github.WorkflowLintResult].ok(payload)


__all__ = ["FlextInfraWorkflowLinter"]

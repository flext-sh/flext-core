"""Workflow linter service for GitHub Actions validation.

Wraps actionlint execution with FlextResult error handling,
replacing scripts/github/lint_workflows.py with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flext_core.result import FlextResult, r

from flext_infra.json_io import JsonService
from flext_infra.subprocess import CommandRunner


class WorkflowLinter:
    """Infrastructure service for GitHub Actions workflow linting.

    Delegates to ``actionlint`` for validation and persists JSON reports.
    """

    def __init__(
        self,
        runner: CommandRunner | None = None,
        json_io: JsonService | None = None,
    ) -> None:
        self._runner = runner or CommandRunner()
        self._json = json_io or JsonService()

    def lint(
        self,
        root: Path,
        *,
        report_path: Path | None = None,
        strict: bool = False,
    ) -> FlextResult[dict[str, object]]:
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
            payload_skipped: dict[str, object] = {
                "status": "skipped",
                "reason": "actionlint not installed",
            }
            if report_path is not None:
                self._json.write(report_path, payload_skipped, sort_keys=True)
            return r[dict[str, object]].ok(payload_skipped)

        result = self._runner.run([actionlint], cwd=root)

        if result.is_success:
            output = result.value
            payload: dict[str, object] = {
                "status": "ok",
                "exit_code": output.exit_code,
                "stdout": output.stdout,
                "stderr": output.stderr,
            }
        else:
            # actionlint returns non-zero on findings
            # Parse from error message since run() returns failure
            payload = {
                "status": "fail",
                "exit_code": 1,
                "detail": result.error or "",
            }

        if report_path is not None:
            self._json.write(report_path, payload, sort_keys=True)

        if payload.get("status") == "fail" and strict:
            return r[dict[str, object]].fail(
                result.error or "actionlint found issues",
            )

        return r[dict[str, object]].ok(payload)


__all__ = ["WorkflowLinter"]

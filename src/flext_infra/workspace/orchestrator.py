"""Multi-project orchestration service.

Executes make verbs across projects with per-project logging and structured
results. Migrated from scripts/workspace_orchestrator.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import override

import structlog
from flext_core.result import r
from flext_core.service import FlextService

from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.subprocess import CommandRunner

logger = structlog.get_logger(__name__)


class OrchestratorService(FlextService[list[m.CommandOutput]]):
    """Infrastructure service for multi-project make orchestration.

    Executes a make verb across a list of projects sequentially, capturing
    per-project output and timing. Supports fail-fast mode to stop on
    first failure.

    """

    def __init__(self) -> None:
        """Initialize the orchestrator service."""
        super().__init__()
        self._runner = CommandRunner()

    @override
    def execute(self) -> r[list[m.CommandOutput]]:
        """Not used; call orchestrate() directly instead."""
        return r[list[m.CommandOutput]].fail("Use orchestrate() method directly")

    def orchestrate(
        self,
        projects: Sequence[str],
        verb: str,
        *,
        fail_fast: bool = False,
        make_args: Sequence[str] = (),
    ) -> r[list[m.CommandOutput]]:
        """Execute make verb across projects with per-project logging.

        Args:
            projects: List of project directory names.
            verb: Make verb to execute (e.g. "check", "test", "help").
            fail_fast: Stop execution on first project failure.
            make_args: Additional arguments to pass to make.

        Returns:
            FlextResult containing list of CommandOutput per project.

        """
        try:
            results: list[m.CommandOutput] = []
            total = len(projects)
            success = 0
            failed = 0
            skipped = 0

            for idx, project in enumerate(projects, start=1):
                if skipped:
                    logger.info(
                        "workspace_project_skipped",
                        index=idx,
                        project=project,
                        verb=verb,
                        elapsed_seconds=0,
                        exit_code=0,
                    )
                    results.append(
                        m.CommandOutput(
                            stdout="",
                            stderr="",
                            exit_code=0,
                            duration=0.0,
                        ),
                    )
                    continue

                output_result = self._run_project(
                    project,
                    verb,
                    idx,
                    make_args=list(make_args),
                )
                if output_result.is_failure:
                    failed += 1
                    results.append(
                        m.CommandOutput(
                            stdout="",
                            stderr=output_result.error or "project execution failed",
                            exit_code=1,
                            duration=0.0,
                        ),
                    )
                    if fail_fast:
                        skipped = total - idx
                    continue

                output = output_result.value
                results.append(output)
                if output.exit_code == 0:
                    success += 1
                else:
                    failed += 1

                if output.exit_code != 0 and fail_fast:
                    skipped = total - idx

            logger.info(
                "workspace_orchestration_summary",
                total=total,
                success=success,
                failed=failed,
                skipped=skipped,
            )
            return r[list[m.CommandOutput]].ok(results)

        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return r[list[m.CommandOutput]].fail(f"Orchestration failed: {exc}")

    def _run_project(
        self,
        project: str,
        verb: str,
        index: int,
        *,
        make_args: list[str],
    ) -> r[m.CommandOutput]:
        """Execute make verb for a single project.

        Args:
            project: Project directory name.
            verb: Make verb to execute.
            index: 1-based project index.
            make_args: Additional make arguments.

        Returns:
            CommandOutput with log path in stdout, exit code, and timing.

        """
        reports_dir_result = self._ensure_report_dir(verb)
        if reports_dir_result.is_failure:
            return r[m.CommandOutput].fail(
                reports_dir_result.error or "failed to create report directory"
            )
        reports_dir = reports_dir_result.value
        log_path = reports_dir / f"{project}.log"
        started = time.monotonic()

        proc_result = self._runner.run_to_file(
            ["make", "-C", project, verb, *make_args],
            log_path,
        )
        return_code = proc_result.value if proc_result.is_success else 1
        stderr = "" if proc_result.is_success else (proc_result.error or "")

        elapsed = time.monotonic() - started
        status = c.Status.OK if return_code == 0 else c.Status.FAIL
        msg = (
            f"{index:02d} [{status}] {project} {verb}"
            f" ({int(elapsed)}s) exit={return_code} log={log_path}"
        )
        logger.info(
            "workspace_project_completed",
            message=msg,
            index=index,
            status=status,
            project=project,
            verb=verb,
            elapsed_seconds=int(elapsed),
            exit_code=return_code,
            log_path=str(log_path),
        )

        return r[m.CommandOutput].ok(
            m.CommandOutput(
                stdout=str(log_path),
                stderr=stderr,
                exit_code=return_code,
                duration=round(elapsed, 2),
            ),
        )

    @staticmethod
    def _ensure_report_dir(verb: str) -> r[Path]:
        """Ensure report directory exists for workspace verb logs.

        Args:
            verb: Make verb used as subdirectory name.

        Returns:
            Path to the report directory.

        """
        reports_dir = Path(".reports") / "workspace" / verb
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            return r[Path].ok(reports_dir)
        except OSError as exc:
            return r[Path].fail(f"failed to create report directory: {exc}")


__all__ = [
    "OrchestratorService",
]

"""Multi-project orchestration service.

Executes make verbs across projects with per-project logging and structured
results. Migrated from scripts/workspace_orchestrator.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import override

from flext_core.loggings import FlextLogger
from flext_core.result import r
from flext_core.service import FlextService

from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.output import output
from flext_infra.reporting import ReportingService
from flext_infra.subprocess import CommandRunner

logger = FlextLogger.create_module_logger(__name__)


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
        self._reporting = ReportingService()

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
        output.header("Workspace Orchestration")
        try:
            results: list[m.CommandOutput] = []
            total = len(projects)
            success = 0
            failed = 0
            skipped = 0
            started_total = time.monotonic()

            for idx, project in enumerate(projects, start=1):
                output.progress(idx, total, project, verb)
                if skipped:
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

                cmd_output = output_result.value
                results.append(cmd_output)
                if cmd_output.exit_code == 0:
                    success += 1
                else:
                    failed += 1

                if cmd_output.exit_code != 0 and fail_fast:
                    skipped = total - idx

            elapsed_total = time.monotonic() - started_total
            output.summary(verb, total, success, failed, skipped, elapsed_total)
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
        log_path = self._reporting.get_report_path(
            Path.cwd(), "workspace", verb, f"{project}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()

        proc_result = self._runner.run_to_file(
            ["make", "-C", project, verb, *make_args],
            log_path,
            env={"NO_COLOR": "1", **os.environ},
        )
        return_code = proc_result.value if proc_result.is_success else 1
        stderr = "" if proc_result.is_success else (proc_result.error or "")

        elapsed = time.monotonic() - started
        status = c.Status.OK if return_code == 0 else c.Status.FAIL
        # Output project completion status
        status_symbol = "✓" if return_code == 0 else "✗"
        output.info(f"  {status_symbol} {project} completed in {int(elapsed)}s (log: {log_path.name})")

        return r[m.CommandOutput].ok(
            m.CommandOutput(
                stdout=str(log_path),
                stderr=stderr,
                exit_code=return_code,
                duration=round(elapsed, 2),
            ),
        )


__all__ = [
    "OrchestratorService",
]

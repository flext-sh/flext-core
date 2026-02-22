"""Multi-project orchestration service.

Executes make verbs across projects with per-project logging and structured
results. Migrated from scripts/workspace_orchestrator.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import override

from flext_core.result import FlextResult as r
from flext_core.service import FlextService

from flext_infra.models import InfraModels

_DEFAULT_ENCODING = "utf-8"
_STATUS_OK = "OK"
_STATUS_FAIL = "FAIL"


class OrchestratorService(FlextService[list[InfraModels.CommandOutput]]):
    """Infrastructure service for multi-project make orchestration.

    Executes a make verb across a list of projects sequentially, capturing
    per-project output and timing. Supports fail-fast mode to stop on
    first failure.

    Example:
        service = OrchestratorService()
        result = service.orchestrate(
            projects=["flext-core", "flext-api"],
            verb="check",
            fail_fast=True,
        )
        if result.is_success:
            for output in result.value:
                print(f"exit={output.exit_code} duration={output.duration}s")

    """

    @override
    def execute(self) -> r[list[InfraModels.CommandOutput]]:
        """Not used; call orchestrate() directly instead."""
        return r[list[InfraModels.CommandOutput]].fail(
            "Use orchestrate() method directly"
        )

    def orchestrate(
        self,
        projects: Sequence[str],
        verb: str,
        *,
        fail_fast: bool = False,
        make_args: Sequence[str] = (),
    ) -> r[list[InfraModels.CommandOutput]]:
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
            results: list[InfraModels.CommandOutput] = []
            total = len(projects)
            success = 0
            failed = 0
            skipped = 0

            for idx, project in enumerate(projects, start=1):
                if skipped:
                    print(f"{idx:02d} [SKIP] {project} {verb} (0s) exit=0")
                    results.append(
                        InfraModels.CommandOutput(
                            stdout="",
                            stderr="",
                            exit_code=0,
                            duration=0.0,
                        )
                    )
                    continue

                output = self._run_project(
                    project,
                    verb,
                    idx,
                    make_args=list(make_args),
                )
                results.append(output)

                if output.exit_code == 0:
                    success += 1
                else:
                    failed += 1

                if output.exit_code != 0 and fail_fast:
                    skipped = total - idx

            print(
                f"summary total={total} success={success} fail={failed} skip={skipped}"
            )
            return r[list[InfraModels.CommandOutput]].ok(results)

        except Exception as exc:
            return r[list[InfraModels.CommandOutput]].fail(
                f"Orchestration failed: {exc}"
            )

    def _run_project(
        self,
        project: str,
        verb: str,
        index: int,
        *,
        make_args: list[str],
    ) -> InfraModels.CommandOutput:
        """Execute make verb for a single project.

        Args:
            project: Project directory name.
            verb: Make verb to execute.
            index: 1-based project index.
            make_args: Additional make arguments.

        Returns:
            CommandOutput with log path in stdout, exit code, and timing.

        """
        reports_dir = self._ensure_report_dir(verb)
        log_path = reports_dir / f"{project}.log"
        started = time.monotonic()

        with log_path.open("w", encoding=_DEFAULT_ENCODING) as log_handle:
            proc = subprocess.run(
                ["make", "-C", project, verb, *make_args],  # noqa: S607
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        elapsed = time.monotonic() - started
        status = _STATUS_OK if proc.returncode == 0 else _STATUS_FAIL
        msg = (
            f"{index:02d} [{status}] {project} {verb}"
            f" ({int(elapsed)}s) exit={proc.returncode} log={log_path}"
        )
        print(msg)

        return InfraModels.CommandOutput(
            stdout=str(log_path),
            stderr="",
            exit_code=proc.returncode,
            duration=round(elapsed, 2),
        )

    @staticmethod
    def _ensure_report_dir(verb: str) -> Path:
        """Ensure report directory exists for workspace verb logs.

        Args:
            verb: Make verb used as subdirectory name.

        Returns:
            Path to the report directory.

        """
        reports_dir = Path(".reports") / "workspace" / verb
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir


__all__ = [
    "OrchestratorService",
]

"""Reporting service for workspace report management.

Wraps reporting operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core.result import FlextResult, r
from flext_core.typings import t

_REPORTS_DIR = ".reports"


class ReportingService:
    """Infrastructure service for report management.

    Structurally satisfies ``InfraProtocols.ReporterProtocol``.
    """

    def reports_root(self, workspace_root: Path) -> FlextResult[Path]:
        """Return the root directory for repository reports.

        Args:
            workspace_root: The root directory of the workspace.

        Returns:
            FlextResult[Path] with the .reports directory path.

        """
        return r[Path].ok(workspace_root / _REPORTS_DIR)

    def ensure_report_dir(
        self,
        workspace_root: Path,
        *parts: str,
    ) -> FlextResult[Path]:
        """Ensure a specific report directory exists.

        Args:
            workspace_root: The root directory of the workspace.
            *parts: Subdirectory parts to join with reports root.

        Returns:
            FlextResult[Path] with the ensured directory path.

        """
        try:
            path = (workspace_root / _REPORTS_DIR).joinpath(*parts)
            path.mkdir(parents=True, exist_ok=True)
            return r[Path].ok(path)
        except OSError as exc:
            return r[Path].fail(f"failed to create report dir: {exc}")

    def report(
        self,
        results: Sequence[FlextResult[Mapping[str, t.ScalarValue]]],
    ) -> FlextResult[Path]:
        """Generate a report from a sequence of results.

        Protocol-compliant method satisfying ``InfraProtocols.ReporterProtocol``.

        Args:
            results: Sequence of FlextResult objects to report on.

        Returns:
            FlextResult[Path] with the report file path.

        """
        # Base implementation - subclasses should override with
        # workspace_root context for actual report generation
        passed = sum(1 for res in results if res.is_success)
        failed = len(results) - passed
        summary = f"Results: {passed} passed, {failed} failed"
        return r[Path].fail(
            f"report generation requires workspace context. {summary}",
        )


__all__ = ["ReportingService"]

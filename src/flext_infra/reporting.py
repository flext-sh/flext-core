"""Reporting service for workspace report management.

Wraps reporting operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core.result import FlextResult, r

_REPORTS_DIR = ".reports"


class ReportingService:
    """Infrastructure service for report management.

    Structurally satisfies ``InfraProtocols.ReporterProtocol``.
    """

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


__all__ = ["ReportingService"]

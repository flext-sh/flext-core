"""Reporting service for standardized .reports/ path management.

Convention::

    .reports/
    ├── {verb}/              # Project-level (check, test, validate, docs, …)
    │   └── {report-files}
    └── workspace/           # Workspace-level
        └── {verb}/
            └── {project}.log

Known verbs: build, check, dependencies, docs, preflight, release, tests,
validate, workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import FlextResult, FlextService, r
from flext_core.constants import KNOWN_VERBS

from flext_infra.constants import c

class FlextInfraReportingService(FlextService[Path]):
    """Infrastructure service for standardized report path management.

    Convention::

        .reports/
        ├── {verb}/              # Project-level reports
        │   └── {report-files}
        └── workspace/           # Workspace-level reports
            └── {verb}/
                └── {project}.log

    Structurally satisfies ``InfraProtocols.ReporterProtocol``.
    Structurally satisfies ``InfraProtocols.ReporterProtocol``.
    """

    def execute(self) -> FlextResult[Path]:
        """Execute reporting (default: empty path).

        Returns:
            FlextResult with empty path by default.

        """
        return r[Path].ok(Path())

    def get_report_dir(self, root: Path | str, scope: str, verb: str) -> Path:
        """Build a standardized report directory path (no I/O).

        Args:
            root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).

        Returns:
            Absolute Path to the report directory.

        """
        root_path = Path(root) if isinstance(root, str) else root
        base = root_path / c.Infra.Reporting.REPORTS_DIR_NAME
        if scope == "workspace":
            return (base / "workspace" / verb).resolve()
        return (base / verb).resolve()

    def get_report_path(
        self, root: Path | str, scope: str, verb: str, filename: str
    ) -> Path:
        """Build a standardized report file path (no I/O).

        Args:
            root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).
            filename: Report filename.

        Returns:
            Absolute Path to the report file.

        """
        return self.get_report_dir(root, scope, verb) / filename

    def ensure_report_dir(
        self, root: Path | str, scope: str, verb: str
    ) -> FlextResult[Path]:
        """Ensure report directory exists, creating it if necessary.

        Args:
            root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).

        Returns:
            FlextResult[Path] with the report directory path.

        """
        try:
            report_dir = self.get_report_dir(root, scope, verb)
            report_dir.mkdir(parents=True, exist_ok=True)
            return r[Path].ok(report_dir)
        except OSError as exc:
            return r[Path].fail(f"failed to create report directory: {exc}")

    def create_latest_symlink(self, report_dir: Path, run_id: str) -> FlextResult[Path]:
        """Create or update a ``latest`` symlink pointing to *run_id*.

        Args:
            report_dir: Base report directory (e.g. ``.reports/tests``).
            run_id: The run-specific subdirectory name.

        Returns:
            FlextResult[Path] with the symlink path.

        """
        link = report_dir / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(run_id)
            return r[Path].ok(link)
        except OSError as exc:
            return r[Path].fail(f"failed to create latest symlink: {exc}")


__all__ = ["KNOWN_VERBS", "FlextInfraReportingService"]

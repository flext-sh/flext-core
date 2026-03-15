"""Reporting utilities for standardized .reports/ path management.

Convention::

    .reports/
    ├── {verb}/              # Project-level (check, test, validate, docs, …)
    │   └── {report-files}
    └── workspace/           # Workspace-level
        └── {verb}/
            └── {project}.log

Known verbs: build, check, dependencies, docs, preflight, release, tests,
validate, workspace.

All methods are static — exposed via u.Infra.get_report_dir() through MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.constants import FlextInfraConstants as c


class FlextInfraUtilitiesReporting:
    """Static reporting utilities for standardized report path management.

    All methods are ``@staticmethod`` — no instantiation required.
    Exposed via ``u.Infra.get_report_dir()`` etc. through MRO.
    """

    @staticmethod
    def get_report_dir(workspace_root: Path | str, scope: str, verb: str) -> Path:
        """Build a standardized report directory path (no I/O).

        Args:
            workspace_root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).

        Returns:
            Absolute Path to the report directory.

        """
        root_path = (
            Path(workspace_root) if isinstance(workspace_root, str) else workspace_root
        )
        base = root_path / c.Infra.Reporting.REPORTS_DIR_NAME
        if scope == c.Infra.ReportKeys.WORKSPACE:
            return (base / c.Infra.ReportKeys.WORKSPACE / verb).resolve()
        return (base / verb).resolve()

    @staticmethod
    def get_report_path(
        workspace_root: Path | str,
        scope: str,
        verb: str,
        filename: str,
    ) -> Path:
        """Build a standardized report file path (no I/O).

        Args:
            workspace_root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).
            filename: Report filename.

        Returns:
            Absolute Path to the report file.

        """
        return (
            FlextInfraUtilitiesReporting.get_report_dir(workspace_root, scope, verb)
            / filename
        )

    @staticmethod
    def ensure_report_dir(workspace_root: Path | str, scope: str, verb: str) -> r[Path]:
        """Ensure report directory exists, creating it if necessary.

        Args:
            workspace_root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).

        Returns:
            r[Path] with the report directory path.

        """
        try:
            report_dir = FlextInfraUtilitiesReporting.get_report_dir(
                workspace_root,
                scope,
                verb,
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            return r[Path].ok(report_dir)
        except OSError as exc:
            return r[Path].fail(f"failed to create report directory: {exc}")

    @staticmethod
    def create_latest_symlink(report_dir: Path, run_id: str) -> r[Path]:
        """Create or update a ``latest`` symlink pointing to *run_id*.

        Args:
            report_dir: Base report directory (e.g. ``.reports/tests``).
            run_id: The run-specific subdirectory name.

        Returns:
            r[Path] with the symlink path.

        """
        link = report_dir / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(run_id)
            return r[Path].ok(link)
        except OSError as exc:
            return r[Path].fail(f"failed to create latest symlink: {exc}")


__all__ = ["FlextInfraUtilitiesReporting"]

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
from typing import Final

from flext_core import FlextResult, r

REPORTS_DIR_NAME: Final[str] = ".reports"
# Top-level reports directory name (always `.reports`).

KNOWN_VERBS: Final[frozenset[str]] = frozenset({
    "build",
    "check",
    "dependencies",
    "docs",
    "preflight",
    "release",
    "tests",
    "validate",
    "workspace",
})
# Standard subdirectory verbs under `.reports/`.

# Backward-compat alias (internal callers used `_REPORTS_DIR`).
_REPORTS_DIR = REPORTS_DIR_NAME


class ReportingService:
    """Infrastructure service for standardized report path management.

    Convention::

        .reports/
        ├── {verb}/              # Project-level reports
        │   └── {report-files}
        └── workspace/           # Workspace-level reports
            └── {verb}/
                └── {project}.log

    Structurally satisfies ``InfraProtocols.ReporterProtocol``.
    """

    # ------------------------------------------------------------------
    # Existing API (kept for backward compatibility)
    # ------------------------------------------------------------------

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
            path = (workspace_root / REPORTS_DIR_NAME).joinpath(*parts)
            path.mkdir(parents=True, exist_ok=True)
            return r[Path].ok(path)
        except OSError as exc:
            return r[Path].fail(f"failed to create report dir: {exc}")

    # ------------------------------------------------------------------
    # New standardized API
    # ------------------------------------------------------------------

    def get_report_dir(
        self,
        root: Path | str,
        scope: str,
        verb: str,
    ) -> Path:
        """Build a standardized report directory path (no I/O).

        Args:
            root: Workspace or project root.
            scope: ``"project"`` or ``"workspace"``.
            verb: Action verb (check, test, validate, docs, …).

        Returns:
            Absolute Path to the report directory.

        """
        root_path = Path(root) if isinstance(root, str) else root
        base = root_path / REPORTS_DIR_NAME
        if scope == "workspace":
            return (base / "workspace" / verb).resolve()
        return (base / verb).resolve()

    def get_report_path(
        self,
        root: Path | str,
        scope: str,
        verb: str,
        filename: str,
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

    def create_latest_symlink(
        self,
        report_dir: Path,
        run_id: str,
    ) -> FlextResult[Path]:
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


__all__ = ["KNOWN_VERBS", "REPORTS_DIR_NAME", "ReportingService"]

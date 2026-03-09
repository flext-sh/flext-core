"""Base.mk sync validation service.

Checks that vendored project base.mk files match the root base.mk,
detecting configuration drift across workspace projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from flext_core import r
from flext_infra import c, m


class FlextInfraBaseMkValidator:
    """Validates base.mk synchronization across workspace projects.

    Compares SHA-256 hashes of vendored base.mk copies against the
    root base.mk to detect configuration drift.
    """

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(self, workspace_root: Path) -> r[m.Infra.Core.ValidationReport]:
        """Validate that all vendored base.mk copies match root base.mk.

        Args:
            workspace_root: Root directory of the workspace.

        Returns:
            r with ValidationReport indicating sync status.

        """
        try:
            source = workspace_root / c.Infra.Files.BASE_MK
            if not source.exists():
                return r[m.Infra.Core.ValidationReport].ok(
                    m.Infra.Core.ValidationReport(
                        passed=False,
                        violations=["missing root base.mk"],
                        summary="missing root base.mk",
                    ),
                )
            source_hash = self._sha256(source)
            mismatched: list[str] = []
            checked = 0
            for pyproject in sorted(
                workspace_root.glob(f"*/{c.Infra.Files.PYPROJECT_FILENAME}"),
            ):
                local_base = pyproject.parent / c.Infra.Files.BASE_MK
                if not local_base.exists():
                    continue
                checked += 1
                if self._sha256(local_base) != source_hash:
                    rel = str(local_base.relative_to(workspace_root))
                    mismatched.append(f"drift: {rel}")
            passed = len(mismatched) == 0
            summary = (
                f"all vendored base.mk copies in sync ({checked} checked)"
                if passed
                else f"{len(mismatched)} base.mk files out of sync"
            )
            return r[m.Infra.Core.ValidationReport].ok(
                m.Infra.Core.ValidationReport(
                    passed=passed,
                    violations=mismatched,
                    summary=summary,
                ),
            )
        except OSError as exc:
            return r[m.Infra.Core.ValidationReport].fail(
                f"base.mk validation failed: {exc}",
            )


__all__ = ["FlextInfraBaseMkValidator"]

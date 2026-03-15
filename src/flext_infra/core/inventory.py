"""Scripts inventory generation service.

Generates script inventory artifacts for workspace governance,
cataloging all scripts and their wiring status.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flext_core import r
from flext_infra import c, m, t, u


class FlextInfraInventoryService:
    """Generates and manages scripts inventory for workspace governance.

    Scans the workspace for Python and Bash scripts and produces
    structured inventory, wiring, and external-candidate reports.
    """

    def generate(
        self,
        workspace_root: Path,
        *,
        output_dir: Path | None = None,
    ) -> r[m.Infra.Core.InventoryReport]:
        """Build and write scripts inventory reports.

        Args:
            workspace_root: Root of the workspace to scan.
            output_dir: Optional directory for reports. Defaults to
                ``workspace_root / ".reports"``.

        Returns:
            r with combined inventory metadata.

        """
        try:
            root = workspace_root.resolve()
            scripts_dir = root / c.Infra.Directories.SCRIPTS
            scripts: list[str] = []
            if scripts_dir.exists():
                scripts = sorted(
                    path.relative_to(root).as_posix()
                    for path in scripts_dir.rglob("*")
                    if path.is_file()
                    and path.suffix in {c.Infra.Extensions.PYTHON, ".sh"}
                )
            now = datetime.now(UTC).isoformat()
            scripts_infra: list[t.Infra.InfraValue] = list(scripts)
            inventory: t.Infra.ContainerDict = {
                "generated_at": now,
                "repo_root": str(root),
                "total_scripts": len(scripts),
                "scripts": scripts_infra,
            }
            wiring: t.Infra.ContainerDict = {
                "generated_at": now,
                "root_makefile": [c.Infra.Files.MAKEFILE_FILENAME],
                "unwired_scripts": [],
            }
            external: t.Infra.ContainerDict = {"generated_at": now, "candidates": []}
            reports_dir = output_dir or root / c.Infra.Reporting.REPORTS_DIR_NAME
            written: list[str] = []
            inventory_path = reports_dir / "scripts-infra--json--scripts-inventory.json"
            wiring_path = reports_dir / "scripts-infra--json--scripts-wiring.json"
            external_path = (
                reports_dir / "scripts-infra--json--external-scripts-candidates.json"
            )
            write_result = u.Infra.write_json(inventory_path, inventory, sort_keys=True)
            if write_result.is_failure:
                return r[m.Infra.Core.InventoryReport].fail(
                    write_result.error or "write failed",
                )
            written.append(str(inventory_path))
            write_result = u.Infra.write_json(wiring_path, wiring, sort_keys=True)
            if write_result.is_failure:
                return r[m.Infra.Core.InventoryReport].fail(
                    write_result.error or "write failed",
                )
            written.append(str(wiring_path))
            write_result = u.Infra.write_json(external_path, external, sort_keys=True)
            if write_result.is_failure:
                return r[m.Infra.Core.InventoryReport].fail(
                    write_result.error or "write failed",
                )
            written.append(str(external_path))
            result = m.Infra.Core.InventoryReport(
                total_scripts=len(scripts),
                reports_written=written,
            )
            return r[m.Infra.Core.InventoryReport].ok(result)
        except (OSError, TypeError, ValueError) as exc:
            return r[m.Infra.Core.InventoryReport].fail(
                f"inventory generation failed: {exc}",
            )


__all__ = ["FlextInfraInventoryService"]

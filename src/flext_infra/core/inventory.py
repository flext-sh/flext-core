"""Scripts inventory generation service.

Generates script inventory artifacts for workspace governance,
cataloging all scripts and their wiring status.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path

from flext_core import FlextResult, r, t

from flext_infra import FlextInfraJsonService, c


class FlextInfraInventoryService:
    """Generates and manages scripts inventory for workspace governance.

    Scans the workspace for Python and Bash scripts and produces
    structured inventory, wiring, and external-candidate reports.
    """

    def __init__(self) -> None:
        """Initialize the inventory service."""
        self._json = FlextInfraJsonService()

    def generate(
        self,
        workspace_root: Path,
        *,
        output_dir: Path | None = None,
    ) -> FlextResult[Mapping[str, t.ConfigMapValue]]:
        """Build and write scripts inventory reports.

        Args:
            workspace_root: Root of the workspace to scan.
            output_dir: Optional directory for reports. Defaults to
                ``workspace_root / ".reports"``.

        Returns:
            FlextResult with combined inventory metadata.

        """
        try:
            root = workspace_root.resolve()
            scripts_dir = root / "scripts"

            scripts: list[str] = []
            if scripts_dir.exists():
                scripts = sorted(
                    path.relative_to(root).as_posix()
                    for path in scripts_dir.rglob("*")
                    if path.is_file() and path.suffix in {".py", ".sh"}
                )

            now = datetime.now(UTC).isoformat()
            inventory: Mapping[str, t.ConfigMapValue] = {
                "generated_at": now,
                "repo_root": str(root),
                "total_scripts": len(scripts),
                "scripts": scripts,
            }
            wiring: Mapping[str, t.ConfigMapValue] = {
                "generated_at": now,
                "root_makefile": [c.Files.MAKEFILE_FILENAME],
                "unwired_scripts": [],
            }
            external: Mapping[str, t.ConfigMapValue] = {
                "generated_at": now,
                "candidates": [],
            }

            reports_dir = output_dir or (root / ".reports")
            outputs: Mapping[Path, Mapping[str, t.ConfigMapValue]] = {
                reports_dir / "scripts-infra--json--scripts-inventory.json": inventory,
                reports_dir / "scripts-infra--json--scripts-wiring.json": wiring,
                reports_dir
                / "scripts-infra--json--external-scripts-candidates.json": external,
            }

            written: list[str] = []
            for path, payload in outputs.items():
                write_result = self._json.write(
                    path,
                    payload,
                    sort_keys=True,
                )
                if write_result.is_failure:
                    return r[Mapping[str, t.ConfigMapValue]].fail(
                        write_result.error or "write failed",
                    )
                written.append(str(path))

            result: MutableMapping[str, t.ConfigMapValue] = {
                "total_scripts": len(scripts),
                "reports_written": written,
            }
            return r[Mapping[str, t.ConfigMapValue]].ok(result)
        except (OSError, TypeError, ValueError) as exc:
            return r[Mapping[str, t.ConfigMapValue]].fail(
                f"inventory generation failed: {exc}",
            )


__all__ = ["FlextInfraInventoryService"]

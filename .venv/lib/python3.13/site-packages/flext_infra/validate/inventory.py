"""Scripts inventory generation service.

Generates script inventory artifacts for workspace governance,
cataloging all scripts and their wiring status.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast, override

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.base import s


class FlextInfraInventoryService(s[bool]):
    """Generates and manages scripts inventory for workspace governance.

    Scans the workspace for Python and Bash scripts and produces
    structured inventory, wiring, and external-candidate reports.
    """

    output_dir: Annotated[
        Path | None, m.Field(description="Output directory for reports")
    ] = None

    def generate(
        self, repository_root: Path, *, output_dir: Path | None = None
    ) -> p.Result[m.Infra.InventoryReport]:
        """Build and write scripts inventory reports.

        Args:
            repository_root: Root of the workspace to scan.
            output_dir: Optional directory for reports. Defaults to
                ``repository_root / ".reports"``.

        Returns:
            r with combined inventory metadata.

        """
        try:
            return self._generate_inventory_report(repository_root, output_dir)
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[m.Infra.InventoryReport].fail_op("inventory generation", exc)

    @staticmethod
    def _script_paths(root: Path) -> t.StrSequence:
        """Return workspace script paths relative to root."""
        scripts_dir = root / c.Infra.DIR_SCRIPTS
        if not scripts_dir.exists():
            return ()
        return tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in u.Infra.iter_matching_files(
                    scripts_dir, includes=[c.Infra.EXT_PYTHON_GLOB, "*.sh"]
                )
            )
        )

    @staticmethod
    def _report_payloads(
        root: Path, scripts: t.StrSequence
    ) -> tuple[t.JsonMapping, t.JsonMapping, t.JsonMapping]:
        """Build inventory, wiring, and external candidate payloads."""
        now = u.now().isoformat()
        inventory_payload = t.json_value_adapter().validate_python({
            "generated_at": now,
            "repo_root": str(root),
            "total_scripts": len(scripts),
            "scripts": list(scripts),
        })
        wiring_payload = t.json_value_adapter().validate_python({
            "generated_at": now,
            "root_makefile": [c.Infra.MAKEFILE_FILENAME],
            "unwired_scripts": [],
        })
        external_payload = t.json_value_adapter().validate_python({
            "generated_at": now,
            "candidates": [],
        })
        return (
            cast("t.JsonMapping", inventory_payload),
            cast("t.JsonMapping", wiring_payload),
            cast("t.JsonMapping", external_payload),
        )

    @staticmethod
    def _write_json_report(path: Path, payload: t.JsonMapping) -> p.Result[str]:
        """Write a single JSON report and return its path."""
        write_result = u.Cli.json_write(
            path, payload, m.Cli.JsonWriteOptions(sort_keys=True)
        )
        if write_result.failure:
            return r[str].fail(write_result.error or "write failed")
        return r[str].ok(str(path))

    def _write_inventory_reports(
        self,
        reports_dir: Path,
        inventory: t.JsonMapping,
        wiring: t.JsonMapping,
        external: t.JsonMapping,
    ) -> p.Result[list[str]]:
        """Write all inventory report payloads."""
        report_specs: tuple[tuple[str, t.JsonMapping], ...] = (
            ("scripts-infra--json--scripts-inventory.json", inventory),
            ("scripts-infra--json--scripts-wiring.json", wiring),
            ("scripts-infra--json--external-scripts-candidates.json", external),
        )
        written: t.MutableSequenceOf[str] = []
        for filename, payload in report_specs:
            write_result = self._write_json_report(reports_dir / filename, payload)
            if write_result.failure:
                return r[list[str]].fail(write_result.error or "write failed")
            written.append(write_result.value)
        return r[list[str]].ok(list(written))

    def _generate_inventory_report(
        self, repository_root: Path, output_dir: Path | None
    ) -> p.Result[m.Infra.InventoryReport]:
        """Generate inventory reports after path resolution."""
        root = repository_root.resolve()
        scripts = self._script_paths(root)
        inventory, wiring, external = self._report_payloads(root, scripts)
        reports_dir = output_dir or root / c.Infra.REPORTS_DIR_NAME
        written_result = self._write_inventory_reports(
            reports_dir, inventory, wiring, external
        )
        if written_result.failure:
            return r[m.Infra.InventoryReport].fail(
                written_result.error or "inventory report write failed"
            )
        return r[m.Infra.InventoryReport].ok(
            m.Infra.InventoryReport(
                total_scripts=len(scripts), reports_written=written_result.value
            )
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the inventory CLI flow."""
        result = self.generate(self.repository_root, output_dir=self.output_dir)
        if result.failure:
            return r[bool].fail(result.error or "inventory generation failed")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraInventoryService"]

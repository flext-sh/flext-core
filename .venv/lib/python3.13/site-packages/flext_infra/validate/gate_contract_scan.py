"""Gate contract script discovery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import u
from flext_infra.validate.gate_contract_errors import GateContractInfraError

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraGateContractScanMixin:
    """Discover tracked workspace scripts."""

    @staticmethod
    def _tracked_scripts(root: Path) -> t.SequenceOf[Path]:
        scripts_root = root / "scripts"
        if not scripts_root.exists() or not scripts_root.is_dir():
            return ()

        result = u.Cli.run_raw(
            [
                "/usr/bin/env",
                "git",
                "ls-files",
                "scripts/*.sh",
                "scripts/*.py",
                "scripts/**/*.sh",
                "scripts/**/*.py",
            ],
            cwd=root,
        )
        if result.failure:
            raise GateContractInfraError(result.error or "git ls-files failed")
        output = result.value
        if output.exit_code != 0:
            stderr = (output.stderr or "").strip()
            raise GateContractInfraError(stderr or "git ls-files failed")

        scripts = (
            Path(line.strip())
            for line in sorted(set(output.stdout.splitlines()))
            if line.strip()
        )
        return tuple(path for path in scripts if path.name != "__init__.py")


__all__: list[str] = ["FlextInfraGateContractScanMixin"]

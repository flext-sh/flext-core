"""PEP 561 ``py.typed`` marker generator.

Scans all package directories under ``src/``, ``tests/``, ``examples/``,
and ``scripts/`` and ensures every directory containing ``.py`` files has
a ``py.typed`` marker.  Removes stale markers from directories without
``.py`` files.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import c, output


class FlextInfraCodegenPyTyped(s[int]):
    """Creates and removes PEP 561 ``py.typed`` markers across workspace packages."""

    _PY_TYPED_FILENAME: str = "py.typed"

    def __init__(self, workspace_root: Path) -> None:
        """Initialize py.typed marker generator with workspace root."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        self._root: Path = workspace_root

    @override
    def execute(self) -> r[int]:
        return r[int].ok(self.run())

    def run(self, *, check_only: bool = False) -> int:
        """Ensure ``py.typed`` markers exist in every package directory.

        Args:
            check_only: If True, only report changes without writing.

        Returns the number of marker files created or removed.

        """
        dirs_to_scan: list[Path] = []
        for pattern in c.Infra.Codegen.ALL_SCAN_PATTERNS:
            base = pattern.split("/*")[0]
            base_dir = self._root / base
            if base_dir.is_dir():
                dirs_to_scan.append(base_dir)
        created = 0
        removed = 0
        for base_dir in dirs_to_scan:
            for dirpath in sorted(base_dir.rglob("*")):
                if not dirpath.is_dir():
                    continue
                if any(
                    part.startswith(".") or part in {"vendor", "node_modules", ".venv"}
                    for part in dirpath.relative_to(self._root).parts
                ):
                    continue
                marker = dirpath / self._PY_TYPED_FILENAME
                has_py = self._dir_has_py_files(dirpath)
                if has_py and not marker.exists():
                    if not check_only:
                        marker.touch()
                    created += 1
                elif not has_py and marker.exists():
                    if not check_only:
                        marker.unlink()
                    removed += 1
        mode = "check" if check_only else "apply"
        output.info(
            f"py.typed {mode}: {created} created, {removed} removed",
        )
        return created + removed

    @staticmethod
    def _dir_has_py_files(dirpath: Path) -> bool:
        return any(
            f.suffix == c.Infra.Extensions.PYTHON and f.name != c.Infra.Files.INIT_PY
            for f in dirpath.iterdir()
            if f.is_file()
        )


__all__ = ["FlextInfraCodegenPyTyped"]

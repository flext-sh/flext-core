"""PEP 561 ``py.typed`` marker generator.

Scans all package directories under ``src/``, ``tests/``, ``examples/``,
and ``scripts/`` and ensures every directory containing ``.py`` files has
a ``py.typed`` marker.  Removes stale markers from directories without
``.py`` files.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, u
from flext_infra.base import s

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraCodegenPyTyped(s[bool]):
    """Creates and removes PEP 561 ``py.typed`` markers across workspace packages."""

    _PY_TYPED_FILENAME: str = c.Infra.PY_TYPED

    @override
    def execute(self) -> p.Result[bool]:
        """Execute ``py.typed`` synchronization from the validated CLI model."""
        self.run(check_only=self.check_only)
        return r[bool].ok(True)

    def run(self, *, check_only: bool = False) -> int:
        """Ensure ``py.typed`` markers exist in every package directory.

        Args:
            check_only: If True, only report changes without writing.

        Returns the number of marker files created or removed.

        """
        dirs_to_scan: t.SequenceOf[Path] = [
            self.repository_root / pattern.split("/*")[0]
            for pattern in c.Infra.ALL_SCAN_PATTERNS
            if (self.repository_root / pattern.split("/*")[0]).is_dir()
        ]
        created = 0
        removed = 0
        for base_dir in dirs_to_scan:
            for dirpath in sorted(base_dir.rglob("*")):
                if not dirpath.is_dir():
                    continue
                if any(
                    part.startswith(".") or part in {"vendor", "node_modules", ".venv"}
                    for part in dirpath.relative_to(self.repository_root).parts
                ):
                    continue
                marker = dirpath / self._PY_TYPED_FILENAME
                has_py = u.Infra.dir_has_py_files(dirpath)
                if has_py and not marker.exists():
                    if not check_only:
                        marker.touch()
                    created += 1
                elif not has_py and marker.exists():
                    if not check_only:
                        marker.unlink()
                    removed += 1
        mode = "check" if check_only else "apply"
        u.Cli.info(f"py.typed {mode}: {created} created, {removed} removed")
        return created + removed


__all__: list[str] = ["FlextInfraCodegenPyTyped"]

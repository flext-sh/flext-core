"""Workspace Python file iteration helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.constants import FlextInfraConstants as c


class FlextInfraUtilitiesIteration:
    @staticmethod
    def _discover_project_roots(
        workspace_root: Path,
        *,
        scan_dirs: frozenset[str] | None = None,
    ) -> list[Path]:
        roots: list[Path] = []
        effective_scan_dirs = scan_dirs or c.Infra.Refactor.MRO_SCAN_DIRECTORIES

        def _looks_like_project(path: Path) -> bool:
            if (
                not path.is_dir()
                or not (path / c.Infra.Files.MAKEFILE_FILENAME).exists()
            ):
                return False
            if (
                not (path / c.Infra.Files.PYPROJECT_FILENAME).exists()
                and not (path / c.Infra.Files.GO_MOD).exists()
            ):
                return False
            return any((path / dir_name).is_dir() for dir_name in effective_scan_dirs)

        if _looks_like_project(workspace_root):
            roots.append(workspace_root)
        roots.extend(
            [
                entry
                for entry in sorted(
                    workspace_root.iterdir(), key=lambda item: item.name
                )
                if entry.is_dir()
                and (not entry.name.startswith("."))
                and _looks_like_project(entry)
            ],
        )
        if (
            len(roots) == 0
            and (workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir()
        ):
            return [workspace_root]
        return roots

    @staticmethod
    def iter_directory_python_files(
        directory: Path,
        *,
        pattern: str | None = None,
        skip_pycache: bool = True,
    ) -> list[Path]:
        """Iterate Python files in a single directory tree.

        Scoped to one directory (project src, subdirectory, etc.) — unlike
        ``iter_python_files`` which discovers across the whole workspace.

        Args:
            directory: Root directory to scan.
            pattern: Glob pattern (defaults to ``c.Infra.Extensions.PYTHON_GLOB``).
            skip_pycache: Exclude ``__pycache__`` paths (default True).

        Returns:
            Sorted list of matching file paths. Empty list if directory
            does not exist.

        """
        if not directory.is_dir():
            return []
        effective_pattern = pattern or c.Infra.Extensions.PYTHON_GLOB
        files = sorted(directory.rglob(effective_pattern))
        if skip_pycache:
            return [f for f in files if "__pycache__" not in f.parts]
        return files

    @staticmethod
    def iter_python_files(
        workspace_root: Path,
        *,
        project_roots: list[Path] | None = None,
        include_tests: bool = True,
        include_examples: bool = True,
        include_scripts: bool = True,
        src_dirs: frozenset[str] | None = None,
    ) -> r[list[Path]]:
        try:
            roots = (
                project_roots
                or FlextInfraUtilitiesIteration._discover_project_roots(
                    workspace_root=workspace_root,
                )
            )
            selected_dirs = src_dirs or frozenset(
                {
                    c.Infra.Paths.DEFAULT_SRC_DIR,
                    c.Infra.Directories.TESTS,
                    c.Infra.Directories.EXAMPLES,
                    c.Infra.Directories.SCRIPTS,
                },
            )
            include_flags = {
                c.Infra.Paths.DEFAULT_SRC_DIR: True,
                c.Infra.Directories.TESTS: include_tests,
                c.Infra.Directories.EXAMPLES: include_examples,
                c.Infra.Directories.SCRIPTS: include_scripts,
            }
            files: list[Path] = []
            for project_root in roots:
                for dir_name, enabled in include_flags.items():
                    if (not enabled) or (dir_name not in selected_dirs):
                        continue
                    directory = project_root / dir_name
                    if directory.is_dir():
                        files.extend(directory.rglob(c.Infra.Extensions.PYTHON_GLOB))
            return r[list[Path]].ok(sorted(set(files)))
        except OSError as exc:
            return r[list[Path]].fail(f"python file iteration failed: {exc}")


__all__ = ["FlextInfraUtilitiesIteration"]

"""Project discovery utilities for workspace scanning.

All methods are static — exposed via u.Infra.discover_projects() through MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

from flext_core import r
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.models import FlextInfraModels as m


class FlextInfraUtilitiesDiscovery:
    """Static discovery utilities for workspace project scanning.

    All methods are ``@staticmethod`` — no instantiation required.
    Exposed via ``u.Infra.discover_projects()`` through MRO.
    """

    @staticmethod
    def _is_git_project(path: Path) -> bool:
        """Check if a directory is a Git repository."""
        return (path / c.Infra.Git.DIR).exists()

    @staticmethod
    def _submodule_names(workspace_root: Path) -> set[str]:
        """Retrieve submodule names from .gitmodules."""
        gitmodules = workspace_root / c.Infra.Files.GITMODULES
        if not gitmodules.exists():
            return set()
        try:
            content = gitmodules.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except OSError:
            return set()
        return set(re.findall(r"^\s*path\s*=\s*(.+?)\s*$", content, re.MULTILINE))

    @staticmethod
    def discover_projects(
        workspace_root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        try:
            projects: list[m.Infra.Workspace.ProjectInfo] = []
            submodules = FlextInfraUtilitiesDiscovery._submodule_names(workspace_root)
            for entry in sorted(workspace_root.iterdir(), key=lambda v: v.name):
                if (
                    not entry.is_dir()
                    or entry.name == "cmd"
                    or entry.name.startswith(".")
                ):
                    continue
                if not FlextInfraUtilitiesDiscovery._is_git_project(entry):
                    continue
                if not (entry / c.Infra.Files.MAKEFILE_FILENAME).exists():
                    continue
                has_pyproject = (entry / c.Infra.Files.PYPROJECT_FILENAME).exists()
                has_gomod = (entry / c.Infra.Files.GO_MOD).exists()
                if not has_pyproject and (not has_gomod):
                    continue
                stack = c.Infra.Toml.PYTHON if has_pyproject else c.Infra.Gates.GO
                kind = "submodule" if entry.name in submodules else "external"
                projects.append(
                    m.Infra.Workspace.ProjectInfo(
                        path=entry,
                        name=entry.name,
                        stack=f"{stack}/{kind}",
                        has_tests=(entry / c.Infra.Directories.TESTS).is_dir(),
                        has_src=(entry / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir(),
                    ),
                )
            return r[list[m.Infra.Workspace.ProjectInfo]].ok(projects)
        except OSError as exc:
            return r[list[m.Infra.Workspace.ProjectInfo]].fail(
                f"project discovery failed: {exc}",
            )

    @staticmethod
    def discover_project_roots(
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
                or FlextInfraUtilitiesDiscovery.discover_project_roots(
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

    @staticmethod
    def find_all_pyproject_files(
        workspace_root: Path,
        *,
        skip_dirs: frozenset[str] | None = None,
        project_paths: list[Path] | None = None,
    ) -> r[list[Path]]:
        try:
            if project_paths:
                selected: list[Path] = []
                for proj_path in project_paths:
                    target = (
                        proj_path
                        if proj_path.name == c.Infra.Files.PYPROJECT_FILENAME
                        else proj_path / c.Infra.Files.PYPROJECT_FILENAME
                    )
                    if target.exists() and target.is_file():
                        selected.append(target)
                return r[list[Path]].ok(sorted(set(selected)))
            effective_skip = (
                skip_dirs
                if skip_dirs is not None
                else c.Infra.Excluded.PYPROJECT_SKIP_DIRS
            )
            result = [
                found
                for found in sorted(
                    workspace_root.rglob(c.Infra.Files.PYPROJECT_FILENAME),
                )
                if not any(skip in found.parts for skip in effective_skip)
            ]
            return r[list[Path]].ok(result)
        except OSError as exc:
            return r[list[Path]].fail(f"pyproject file scan failed: {exc}")


__all__ = ["FlextInfraUtilitiesDiscovery"]

"""Project discovery service for workspace scanning.

Wraps project discovery logic with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

from flext_core.result import FlextResult, r

from flext_infra.constants import ic
from flext_infra.models import im


class DiscoveryService:
    """Infrastructure service for discovering workspace projects.

    Structurally satisfies ``InfraProtocols.DiscoveryProtocol``.
    """

    def discover_projects(
        self,
        workspace_root: Path,
    ) -> FlextResult[list[im.ProjectInfo]]:
        """Discover all subprojects in the workspace.

        Scans the workspace root for directories that are Git repositories
        with Makefile and pyproject.toml/go.mod markers.

        Args:
            workspace_root: The root directory of the workspace.

        Returns:
            FlextResult with list of discovered ProjectInfo models.

        """
        try:
            projects: list[im.ProjectInfo] = []
            submodules = self._submodule_names(workspace_root)

            for entry in sorted(
                workspace_root.iterdir(),
                key=lambda v: v.name,
            ):
                if (
                    not entry.is_dir()
                    or entry.name == "cmd"
                    or entry.name.startswith(".")
                ):
                    continue
                if not self._is_git_project(entry):
                    continue
                if not (entry / ic.Files.MAKEFILE_FILENAME).exists():
                    continue

                has_pyproject = (entry / ic.Files.PYPROJECT_FILENAME).exists()
                has_gomod = (entry / "go.mod").exists()
                if not has_pyproject and not has_gomod:
                    continue

                stack = "python" if has_pyproject else "go"
                kind = "submodule" if entry.name in submodules else "external"

                projects.append(
                    im.ProjectInfo(
                        path=entry,
                        name=entry.name,
                        stack=f"{stack}/{kind}",
                        has_tests=(entry / "tests").is_dir(),
                        has_src=(entry / ic.Paths.DEFAULT_SRC_DIR).is_dir(),
                    ),
                )

            return r[list[im.ProjectInfo]].ok(projects)
        except Exception as exc:
            return r[list[im.ProjectInfo]].fail(
                f"project discovery failed: {exc}",
            )

    def discover(
        self,
        root: object,
    ) -> FlextResult[list[im.ProjectInfo]]:
        """Protocol-compliant discover method.

        Satisfies ``InfraProtocols.DiscoveryProtocol.discover``.
        """
        if not isinstance(root, Path):
            root = Path(str(root))
        return self.discover_projects(root)

    def find_all_pyproject_files(
        self,
        workspace_root: Path,
        *,
        skip_dirs: frozenset[str] | None = None,
        project_paths: list[Path] | None = None,
    ) -> FlextResult[list[Path]]:
        """Find every pyproject.toml under workspace_root recursively.

        Args:
            workspace_root: Root of the workspace tree.
            skip_dirs: Directory names to exclude from traversal.
            project_paths: If given, only return files for these project dirs.

        Returns:
            FlextResult with sorted list of pyproject.toml paths.

        """
        try:
            if project_paths:
                selected: list[Path] = []
                for p in project_paths:
                    target = (
                        p
                        if p.name == ic.Files.PYPROJECT_FILENAME
                        else p / ic.Files.PYPROJECT_FILENAME
                    )
                    if target.exists() and target.is_file():
                        selected.append(target)
                return r[list[Path]].ok(sorted(set(selected)))

            effective_skip = (
                skip_dirs if skip_dirs is not None else ic.Excluded.PYPROJECT_SKIP_DIRS
            )
            result = [
                p
                for p in sorted(
                    workspace_root.rglob(ic.Files.PYPROJECT_FILENAME),
                )
                if not any(skip in p.parts for skip in effective_skip)
            ]
            return r[list[Path]].ok(result)
        except Exception as exc:
            return r[list[Path]].fail(
                f"pyproject file scan failed: {exc}",
            )

    @staticmethod
    def _is_git_project(path: Path) -> bool:
        """Check if a directory is a Git repository."""
        return (path / ".git").exists()

    @staticmethod
    def _submodule_names(workspace_root: Path) -> set[str]:
        """Retrieve submodule names from .gitmodules."""
        gitmodules = workspace_root / ".gitmodules"
        if not gitmodules.exists():
            return set()
        try:
            content = gitmodules.read_text(encoding=ic.Encoding.DEFAULT)
        except OSError:
            return set()
        return set(
            re.findall(
                r"^\s*path\s*=\s*(.+?)\s*$",
                content,
                re.MULTILINE,
            ),
        )


__all__ = ["DiscoveryService"]

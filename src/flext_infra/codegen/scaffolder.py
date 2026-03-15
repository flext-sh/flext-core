"""Module scaffolder for base module generation.

Generates missing base modules (constants, typings, protocols, models, utilities)
in both src/ and tests/ directories for workspace projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from pydantic import BaseModel

from flext_core import r, s, t
from flext_infra import (
    FlextInfraNamespaceValidator,
    c,
    m,
    u,
)

__all__ = ["FlextInfraCodegenScaffolder"]


class FlextInfraCodegenScaffolder(s):
    """Generates missing base modules in src/ and tests/ directories."""

    def __init__(self, workspace_root: Path) -> None:
        """Initialize scaffolder with workspace root."""
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
        self._workspace_root = workspace_root

    _find_package_dir = staticmethod(u.Infra.find_package_dir)

    @override
    def execute(
        self,
    ) -> r[t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]]:
        return r[
            t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]
        ].fail("Use run() directly")

    def run(self) -> list[m.Infra.Codegen.ScaffoldResult]:
        """Scaffold missing base modules for all projects in workspace.

        Returns:
            List of ScaffoldResult models, one per project.

        """
        projects_result = u.Infra.discover_projects(self._workspace_root)
        if not projects_result.is_success:
            return []
        results: list[m.Infra.Codegen.ScaffoldResult] = []
        discovered: list[m.Infra.Workspace.ProjectInfo] = projects_result.unwrap()
        projects = discovered
        for project in projects:
            if project.name in c.Infra.Codegen.EXCLUDED_PROJECTS:
                continue
            if project.stack.startswith(c.Infra.Gates.GO):
                continue
            result = self.scaffold_project(project.path)
            results.append(result)
        return results

    def scaffold_project(self, project_path: Path) -> m.Infra.Codegen.ScaffoldResult:
        """Scaffold missing base modules for a single project.

        Args:
            project_path: Path to the project root directory.

        Returns:
            ScaffoldResult with lists of created and skipped files.

        """
        prefix = FlextInfraNamespaceValidator.derive_prefix(project_path)
        if not prefix:
            return m.Infra.Codegen.ScaffoldResult(
                project=project_path.name,
                files_created=[],
                files_skipped=[],
            )
        files_created: list[str] = []
        files_skipped: list[str] = []
        pkg_dir = self._find_package_dir(project_path)
        if pkg_dir is not None:
            self._scaffold_dir(
                target_dir=pkg_dir,
                prefix=prefix,
                modules=c.Infra.Codegen.SRC_MODULES,
                test_prefix="",
                files_created=files_created,
                files_skipped=files_skipped,
            )
        tests_dir = project_path / c.Infra.Directories.TESTS
        if tests_dir.is_dir():
            self._scaffold_dir(
                target_dir=tests_dir,
                prefix=prefix,
                modules=c.Infra.Codegen.TESTS_MODULES,
                test_prefix="Tests",
                files_created=files_created,
                files_skipped=files_skipped,
            )
        return m.Infra.Codegen.ScaffoldResult(
            project=project_path.name,
            files_created=files_created,
            files_skipped=files_skipped,
        )

    def _scaffold_dir(
        self,
        *,
        target_dir: Path,
        prefix: str,
        modules: tuple[tuple[str, str, str, str], ...],
        test_prefix: str,
        files_created: list[str],
        files_skipped: list[str],
    ) -> None:
        """Generate missing modules in a directory."""
        for filename, suffix, base_class, doc_suffix in modules:
            filepath = target_dir / filename
            if filepath.exists():
                files_skipped.append(str(filepath))
                continue
            class_name = f"{test_prefix}{prefix}{suffix}"
            docstring = f"{doc_suffix} for {prefix.lower()}."
            content = u.Infra.generate_module_skeleton(
                class_name=class_name,
                base_class=base_class,
                docstring=docstring,
            )
            u.write_file(filepath, content, encoding=c.Infra.Encoding.DEFAULT)
            u.Infra.run_ruff_fix(filepath)
            files_created.append(str(filepath))


__all__ = ["FlextInfraCodegenScaffolder"]

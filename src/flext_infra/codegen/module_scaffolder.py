"""Module scaffolder for base module generation.

Generates missing base modules (constants, typings, protocols, models, utilities)
in both src/ and tests/ directories for workspace projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from flext_core import FlextService, r

from flext_infra.codegen.ast_utils import FlextInfraAstUtils
from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.models import FlextInfraModels

__all__ = ["FlextInfraModuleScaffolder"]

_EXCLUDED_PROJECTS: Final[frozenset[str]] = frozenset({"flexcore"})

# Base module definitions: (filename, class_suffix, base_class, docstring_suffix)
_SRC_MODULES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("constants.py", "Constants", "FlextConstants", "Constants"),
    ("typings.py", "Types", "FlextTypes", "Type aliases"),
    ("protocols.py", "Protocols", "FlextProtocols", "Protocol definitions"),
    ("models.py", "Models", "FlextModels", "Domain models"),
    ("utilities.py", "Utilities", "FlextUtilities", "Utility functions"),
)

_TESTS_MODULES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("constants.py", "Constants", "FlextTestsConstants", "Test constants"),
    ("typings.py", "Types", "FlextTestsTypes", "Test type aliases"),
    ("protocols.py", "Protocols", "FlextTestsProtocols", "Test protocols"),
    ("models.py", "Models", "FlextTestsModels", "Test models"),
    ("utilities.py", "Utilities", "FlextTestsUtilities", "Test utilities"),
)


class FlextInfraModuleScaffolder(FlextService[list[FlextInfraModels.ScaffoldResult]]):
    """Generates missing base modules in src/ and tests/ directories."""

    def __init__(self, workspace_root: Path) -> None:  # noqa: D107
        super().__init__()
        self._workspace_root = workspace_root

    @override
    def execute(self) -> r[list[FlextInfraModels.ScaffoldResult]]:
        """Execute scaffolding across all workspace projects."""
        return r[list[FlextInfraModels.ScaffoldResult]].ok(self.run())

    def run(self) -> list[FlextInfraModels.ScaffoldResult]:
        """Scaffold missing base modules for all projects in workspace.

        Returns:
            List of ScaffoldResult models, one per project.

        """
        discovery = FlextInfraDiscoveryService()
        projects_result = discovery.discover_projects(self._workspace_root)
        if not projects_result.is_success:
            return []

        results: list[FlextInfraModels.ScaffoldResult] = []
        for project in projects_result.unwrap():
            if project.name in _EXCLUDED_PROJECTS:
                continue
            if project.stack.startswith("go"):
                continue
            result = self.scaffold_project(project.path)
            results.append(result)
        return results

    def scaffold_project(self, project_path: Path) -> FlextInfraModels.ScaffoldResult:
        """Scaffold missing base modules for a single project.

        Args:
            project_path: Path to the project root directory.

        Returns:
            ScaffoldResult with lists of created and skipped files.

        """
        prefix = FlextInfraNamespaceValidator.derive_prefix(project_path)
        if not prefix:
            return FlextInfraModels.ScaffoldResult(
                project=project_path.name,
                files_created=[],
                files_skipped=[],
            )

        files_created: list[str] = []
        files_skipped: list[str] = []

        # Scaffold src/ modules
        pkg_dir = self._find_package_dir(project_path)
        if pkg_dir is not None:
            self._scaffold_dir(
                target_dir=pkg_dir,
                prefix=prefix,
                modules=_SRC_MODULES,
                test_prefix="",
                files_created=files_created,
                files_skipped=files_skipped,
            )

        # Scaffold tests/ modules
        tests_dir = project_path / "tests"
        if tests_dir.is_dir():
            self._scaffold_dir(
                target_dir=tests_dir,
                prefix=prefix,
                modules=_TESTS_MODULES,
                test_prefix="Tests",
                files_created=files_created,
                files_skipped=files_skipped,
            )

        return FlextInfraModels.ScaffoldResult(
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
            content = FlextInfraAstUtils.generate_module_skeleton(
                class_name=class_name,
                base_class=base_class,
                docstring=docstring,
            )

            filepath.write_text(content, encoding="utf-8")
            FlextInfraAstUtils.run_ruff_fix(filepath)
            files_created.append(str(filepath))

    @staticmethod
    def _find_package_dir(project_root: Path) -> Path | None:
        """Find the first Python package under src/."""
        src_dir = project_root / "src"
        if not src_dir.is_dir():
            return None
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                return child
        return None

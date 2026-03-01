"""Auto-fix engine for namespace violations.

AST-based auto-fixer that moves standalone Final constants to constants.py
and standalone TypeVar/TypeAlias definitions to typings.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, override

from flext_core import FlextService, r

from flext_infra.codegen.ast_utils import FlextInfraAstUtils
from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.models import FlextInfraModels

__all__ = ["FlextInfraAutoFixer"]

_EXCLUDED_PROJECTS: Final[frozenset[str]] = frozenset({"flexcore"})

_TYPEVAR_CALLABLES: Final[frozenset[str]] = frozenset({
    "TypeVar",
    "ParamSpec",
    "TypeVarTuple",
})


class FlextInfraAutoFixer(FlextService[list[FlextInfraModels.AutoFixResult]]):
    """AST-based auto-fixer for namespace violations (Rules 1-2)."""

    def __init__(self, workspace_root: Path) -> None:  # noqa: D107
        super().__init__()
        self._workspace_root = workspace_root

    @override
    def execute(self) -> r[list[FlextInfraModels.AutoFixResult]]:
        """Execute auto-fix across all workspace projects."""
        return r[list[FlextInfraModels.AutoFixResult]].ok(self.run())

    def run(self) -> list[FlextInfraModels.AutoFixResult]:
        """Run auto-fix on all projects in workspace.

        Returns:
            List of AutoFixResult models, one per project.

        """
        discovery = FlextInfraDiscoveryService()
        projects_result = discovery.discover_projects(self._workspace_root)
        if not projects_result.is_success:
            return []

        results: list[FlextInfraModels.AutoFixResult] = []
        for project in projects_result.unwrap():
            if project.name in _EXCLUDED_PROJECTS:
                continue
            if project.stack.startswith("go"):
                continue
            result = self.fix_project(project.path)
            results.append(result)
        return results

    def fix_project(self, project_path: Path) -> FlextInfraModels.AutoFixResult:
        """Auto-fix namespace violations in a single project.

        Args:
            project_path: Path to the project root directory.

        Returns:
            AutoFixResult with lists of fixed and skipped violations.

        """
        prefix = FlextInfraNamespaceValidator._derive_prefix(project_path)
        if not prefix:
            return FlextInfraModels.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )

        pkg_dir = self._find_package_dir(project_path)
        if pkg_dir is None:
            return FlextInfraModels.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )

        violations_fixed: list[FlextInfraModels.CensusViolation] = []
        violations_skipped: list[FlextInfraModels.CensusViolation] = []
        files_modified: set[str] = set()

        # Scan all non-exempt files in src/
        src_dir = project_path / "src"
        if not src_dir.is_dir():
            return FlextInfraModels.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )

        for py_file in sorted(src_dir.rglob("*.py")):
            if py_file.name in {"__init__.py", "conftest.py", "__main__.py"}:
                continue
            if py_file.name.startswith(("test_", "_")):
                continue
            if py_file.name in {"constants.py", "typings.py"}:
                continue

            tree = FlextInfraAstUtils.parse_module(py_file)
            if tree is None:
                continue

            self._fix_rule1(
                source_file=py_file,
                tree=tree,
                pkg_dir=pkg_dir,
                prefix=prefix,
                violations_fixed=violations_fixed,
                violations_skipped=violations_skipped,
                files_modified=files_modified,
            )
            self._fix_rule2(
                source_file=py_file,
                tree=tree,
                pkg_dir=pkg_dir,
                prefix=prefix,
                violations_fixed=violations_fixed,
                violations_skipped=violations_skipped,
                files_modified=files_modified,
            )

        return FlextInfraModels.AutoFixResult(
            project=project_path.name,
            violations_fixed=violations_fixed,
            violations_skipped=violations_skipped,
            files_modified=sorted(files_modified),
        )

    def _fix_rule1(
        self,
        *,
        source_file: Path,
        tree: ast.Module,
        pkg_dir: Path,
        prefix: str,
        violations_fixed: list[FlextInfraModels.CensusViolation],
        violations_skipped: list[FlextInfraModels.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 1 — move loose Final constants to constants.py."""
        finals = FlextInfraAstUtils.find_standalone_finals(tree)
        if not finals:
            return

        for node in finals:
            target_name = ""
            if isinstance(node.target, ast.Name):
                target_name = node.target.id

            violation = FlextInfraModels.CensusViolation(
                module=str(source_file),
                rule="NS-001",
                line=node.lineno,
                message=f"Loose Final constant '{target_name}' belongs in constants.py",
                fixable=True,
            )

            if FlextInfraAstUtils.is_used_in_context(node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Final constant '{target_name}' used in-context — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            violations_fixed.append(violation)
            files_modified.add(str(source_file))
            files_modified.add(str(pkg_dir / "constants.py"))

    def _fix_rule2(
        self,
        *,
        source_file: Path,
        tree: ast.Module,
        pkg_dir: Path,
        prefix: str,
        violations_fixed: list[FlextInfraModels.CensusViolation],
        violations_skipped: list[FlextInfraModels.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 2 — move loose TypeVars/TypeAliases to typings.py."""
        typevars = FlextInfraAstUtils.find_standalone_typevars(tree)
        typealiases = FlextInfraAstUtils.find_standalone_typealiases(tree)

        for node in typevars:
            target_name = ""
            if isinstance(node, ast.Assign) and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    target_name = target.id

            violation = FlextInfraModels.CensusViolation(
                module=str(source_file),
                rule="NS-002",
                line=node.lineno,
                message=f"TypeVar '{target_name}' belongs in typings.py",
                fixable=True,
            )

            if FlextInfraAstUtils.is_used_in_context(node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeVar '{target_name}' used in-context — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            violations_fixed.append(violation)
            files_modified.add(str(source_file))
            files_modified.add(str(pkg_dir / "typings.py"))

        for node in typealiases:
            target_name = ""
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id

            violation = FlextInfraModels.CensusViolation(
                module=str(source_file),
                rule="NS-002",
                line=node.lineno,
                message=f"TypeAlias '{target_name}' belongs in typings.py",
                fixable=True,
            )

            if FlextInfraAstUtils.is_used_in_context(node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeAlias '{target_name}' used in-context — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            violations_fixed.append(violation)
            files_modified.add(str(source_file))
            files_modified.add(str(pkg_dir / "typings.py"))

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

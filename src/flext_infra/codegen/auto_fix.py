"""Auto-fix engine for namespace violations.

AST-based auto-fixer that moves standalone Final constants to constants.py
and standalone TypeVar/TypeAlias definitions to typings.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
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
        prefix = FlextInfraNamespaceValidator.derive_prefix(project_path)
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
                violations_fixed=violations_fixed,
                violations_skipped=violations_skipped,
                files_modified=files_modified,
            )
            self._fix_rule2(
                source_file=py_file,
                tree=tree,
                pkg_dir=pkg_dir,
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
        violations_fixed: list[FlextInfraModels.CensusViolation],
        violations_skipped: list[FlextInfraModels.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 1 — move loose Final constants to constants.py."""
        finals = FlextInfraAstUtils.find_standalone_finals(tree)
        if not finals:
            return

        target_path = pkg_dir / "constants.py"
        if not target_path.exists():
            return

        # Parse target file once
        target_tree = FlextInfraAstUtils.parse_module(target_path)
        if target_tree is None:
            return

        nodes_to_move: list[ast.AnnAssign] = []
        for node in finals:
            target_name = ""
            if isinstance(node.target, ast.Name):
                target_name = node.target.id

            # Skip private names (underscore-prefixed)
            if target_name.startswith("_"):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Final constant '{target_name}' is private — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

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

            # Check if name is referenced anywhere else in the file (not just classes)
            if self._is_name_referenced_in_file(target_name, node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Final constant '{target_name}' referenced locally — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            nodes_to_move.append(node)

        if not nodes_to_move:
            return

        pkg_name = pkg_dir.name
        actually_moved: list[ast.AnnAssign] = []
        for node in nodes_to_move:
            target_name = ""
            if isinstance(node.target, ast.Name):
                target_name = node.target.id

            # Check if already in target (idempotency)
            if self._name_exists_in_module(target_name, target_tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Final constant '{target_name}' already in constants.py — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            # Copy required imports from source to target, then verify all deps resolvable
            self._copy_required_imports(node, tree, target_tree)
            if not self._all_deps_resolvable(node, target_tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Final constant '{target_name}' has unresolvable deps — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue
            # Remove from source tree
            if node in tree.body:
                tree.body.remove(node)

            # Add to target tree (after last import)
            insert_idx = self._find_insert_position(target_tree)
            target_tree.body.insert(insert_idx, node)

            # Add import in source file
            self._add_import_to_tree(
                tree=tree,
                pkg_name=pkg_name,
                module_name="constants",
                name=target_name,
            )

            violation = FlextInfraModels.CensusViolation(
                module=str(source_file),
                rule="NS-001",
                line=node.lineno,
                message=f"Loose Final constant '{target_name}' moved to constants.py",
                fixable=True,
            )
            violations_fixed.append(violation)
            files_modified.add(str(source_file))
            files_modified.add(str(target_path))
            actually_moved.append(node)

        if actually_moved:
            # Write both files
            self._write_tree(source_file, tree)
            self._write_tree(target_path, target_tree)

    def _fix_rule2(
        self,
        *,
        source_file: Path,
        tree: ast.Module,
        pkg_dir: Path,
        violations_fixed: list[FlextInfraModels.CensusViolation],
        violations_skipped: list[FlextInfraModels.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 2 — move loose TypeVars/TypeAliases to typings.py."""
        typevars = FlextInfraAstUtils.find_standalone_typevars(tree)
        typealiases = FlextInfraAstUtils.find_standalone_typealiases(tree)

        target_path = pkg_dir / "typings.py"
        if not target_path.exists():
            return

        # Parse target file once
        target_tree = FlextInfraAstUtils.parse_module(target_path)
        if target_tree is None:
            return

        nodes_to_move: list[ast.stmt] = []

        for node in typevars:
            target_name = ""
            if isinstance(node, ast.Assign) and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    target_name = target.id

            # Skip private names
            if target_name.startswith("_"):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeVar '{target_name}' is private — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

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

            # Check if name is referenced anywhere else in the file
            if self._is_name_referenced_in_file(target_name, node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeVar '{target_name}' referenced locally — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            nodes_to_move.append(node)

        for node in typealiases:
            target_name = ""
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id

            # Skip private names
            if target_name.startswith("_"):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeAlias '{target_name}' is private — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

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

            # Check if name is referenced anywhere else in the file
            if self._is_name_referenced_in_file(target_name, node, tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"TypeAlias '{target_name}' referenced locally — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            nodes_to_move.append(node)

        if not nodes_to_move:
            return

        pkg_name = pkg_dir.name
        actually_moved: list[ast.stmt] = []

        for node in nodes_to_move:
            target_name = self._get_node_name(node)
            if not target_name:
                continue

            # Check if already in target (idempotency)
            if self._name_exists_in_module(target_name, target_tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"'{target_name}' already in typings.py — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue

            # Copy required imports from source to target, then verify all deps resolvable
            self._copy_required_imports(node, tree, target_tree)
            if not self._all_deps_resolvable(node, target_tree):
                violation = FlextInfraModels.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=node.lineno,
                    message=f"'{target_name}' has unresolvable deps — skipped",
                    fixable=False,
                )
                violations_skipped.append(violation)
                continue
            # Remove from source tree
            if node in tree.body:
                tree.body.remove(node)

            # Add to target tree
            insert_idx = self._find_insert_position(target_tree)
            target_tree.body.insert(insert_idx, node)

            # Add import in source file only if name is used elsewhere
            if self._is_name_referenced_in_file(target_name, node, tree):
                self._add_import_to_tree(
                    tree=tree,
                    pkg_name=pkg_name,
                    module_name="typings",
                    name=target_name,
                )

            rule = "NS-002"
            kind = "TypeVar" if isinstance(node, ast.Assign) else "TypeAlias"
            violation = FlextInfraModels.CensusViolation(
                module=str(source_file),
                rule=rule,
                line=node.lineno,
                message=f"{kind} '{target_name}' moved to typings.py",
                fixable=True,
            )
            violations_fixed.append(violation)
            files_modified.add(str(source_file))
            files_modified.add(str(target_path))
            actually_moved.append(node)

        if actually_moved:
            # Write both files
            self._write_tree(source_file, tree)
            self._write_tree(target_path, target_tree)

    @staticmethod
    def _get_node_name(node: ast.stmt) -> str:
        """Extract the name from an assignment node."""
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return ""

    @staticmethod
    def _is_name_referenced_in_file(
        name: str, definition_node: ast.stmt, tree: ast.Module
    ) -> bool:
        """Check if a name is referenced anywhere in the file besides its definition."""
        for stmt in tree.body:
            if stmt is definition_node:
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and node.id == name:
                    return True
        return False

    @staticmethod
    def _name_exists_in_module(name: str, tree: ast.Module) -> bool:
        """Check if a name is already defined in the module."""
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == name
            ):
                return True
        return False

    @classmethod
    def _copy_required_imports(
        cls,
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> None:
        """Copy imports needed by node from source_tree to target_tree."""
        names_used = cls._get_top_level_names_in_node(node)
        node_name = cls._get_node_name(node)
        names_used = frozenset(n for n in names_used if n != node_name)

        if not names_used:
            return

        # Build map of name -> import stmt from source
        source_imports: dict[str, ast.stmt] = {}
        for stmt in source_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    top_name = imported_name.split(".")[0]
                    source_imports[top_name] = stmt
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        source_imports[imported_name] = stmt

        # Get names already available in target
        target_available: set[str] = set()
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    target_available.add(imported_name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        target_available.add(imported_name)

        # Copy missing imports from source to target
        seen_modules: set[str] = set()
        imports_to_add: list[ast.stmt] = []
        for name in sorted(names_used):
            if name in target_available:
                continue
            if name not in source_imports:
                continue
            import_stmt = source_imports[name]
            import_key = ast.unparse(import_stmt)
            if import_key in seen_modules:
                continue
            seen_modules.add(import_key)
            imports_to_add.append(import_stmt)

        if not imports_to_add:
            return

        # Insert after last import in target
        last_import_idx = 0
        for i, stmt in enumerate(target_tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            ):
                last_import_idx = i + 1

        for i, imp in enumerate(imports_to_add):
            target_tree.body.insert(last_import_idx + i, imp)

    @staticmethod
    def _get_top_level_names_in_node(node: ast.stmt) -> frozenset[str]:
        """Collect all Name references used in a node."""
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return frozenset(names)

    @classmethod
    def _all_deps_resolvable(cls, node: ast.stmt, target_tree: ast.Module) -> bool:
        """Check if all names used in node are available in the target module.

        Called AFTER _copy_required_imports to verify the copy succeeded.
        A name is available if it's imported or defined in the target module.
        """
        names_used = cls._get_top_level_names_in_node(node)
        node_name = cls._get_node_name(node)
        names_used = frozenset(n for n in names_used if n != node_name)

        if not names_used:
            return True

        available: set[str] = set(dir(_builtins_module))
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    available.add(imported_name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        available.add(imported_name)
            else:
                name = cls._get_node_name(stmt)
                if name:
                    available.add(name)

        return all(n in available for n in names_used)

    @staticmethod
    def _find_insert_position(tree: ast.Module) -> int:
        """Find the position after the last import statement in the module."""
        last_import_idx = 0
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                last_import_idx = i + 1
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Module docstring
                last_import_idx = i + 1
        return last_import_idx

    @staticmethod
    def _add_import_to_tree(
        tree: ast.Module,
        pkg_name: str,
        module_name: str,
        name: str,
    ) -> None:
        """Add a from-import to the tree if not already present."""
        full_module = f"{pkg_name}.{module_name}"

        # Check if import already exists
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == full_module:
                # Check if name already imported
                for alias in stmt.names:
                    if alias.name == name:
                        return
                # Add name to existing import
                stmt.names.append(ast.alias(name=name))
                return

        # Create new import statement
        new_import = ast.ImportFrom(
            module=full_module,
            names=[ast.alias(name=name)],
            level=0,
        )
        ast.fix_missing_locations(new_import)

        # Insert after last import
        last_import_idx = 0
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                last_import_idx = i + 1
        tree.body.insert(last_import_idx, new_import)

    @staticmethod
    def _write_tree(path: Path, tree: ast.Module) -> None:
        """Write an AST module back to disk and run ruff fix."""
        ast.fix_missing_locations(tree)
        source = ast.unparse(tree)
        path.write_text(source, encoding="utf-8")
        FlextInfraAstUtils.run_ruff_fix(path)

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

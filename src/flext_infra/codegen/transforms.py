"""AST utility library for safe Python code transformations.

Provides stateless helper functions for parsing, analyzing, and generating
Python code using the ast module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path


class FlextInfraCodegenTransforms:
    """Utility helpers for AST-based code transformations."""

    @staticmethod
    def _resolve_base_class_import(base_class: str) -> str:
        """Resolve the import statement for a base class name.

        Maps base class names to their canonical import paths.
        FlextTests* classes come from flext_tests, others from flext_core.
        """
        if base_class.startswith("FlextTests"):
            return f"from flext_tests import {base_class}"
        return f"from flext_core import {base_class}"

    @staticmethod
    def add_import_to_tree(
        tree: ast.Module, pkg_name: str, module_name: str, name: str
    ) -> None:
        """Add a from-import to the tree when it is missing."""
        full_module = f"{pkg_name}.{module_name}"
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == full_module:
                for alias in stmt.names:
                    if alias.name == name:
                        return
                stmt.names.append(ast.alias(name=name))
                return
        new_import = ast.ImportFrom(
            module=full_module,
            names=[ast.alias(name=name)],
            level=0,
        )
        _ = ast.fix_missing_locations(new_import)
        insert_idx = FlextInfraCodegenTransforms.find_insert_position(tree)
        tree.body.insert(insert_idx, new_import)

    @staticmethod
    def extract_public_classes(tree: ast.Module, prefix: str) -> list[str]:
        """Extract class names that match the provided public prefix."""
        return [
            stmt.name
            for stmt in tree.body
            if isinstance(stmt, ast.ClassDef) and stmt.name.startswith(prefix)
        ]

    @staticmethod
    def find_insert_position(tree: ast.Module) -> int:
        """Find insertion point after module docstring/imports."""
        last_import_idx = 0
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            ):
                last_import_idx = i + 1
        return last_import_idx

    @staticmethod
    def find_standalone_finals(tree: ast.Module) -> list[ast.AnnAssign]:
        """Find module-level Final-annotated assignments.

        Returns all top-level ``X: Final = ...`` and ``X: Final[T] = ...``
        statements.  The caller decides whether to move them based on
        additional guards (private prefix, circular-import risk, etc.).
        """
        matches: list[ast.AnnAssign] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            annotation = stmt.annotation
            if isinstance(annotation, ast.Name) and annotation.id == "Final":
                matches.append(stmt)
                continue
            if (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id == "Final"
            ):
                matches.append(stmt)
        return matches

    @staticmethod
    def find_standalone_typealiases(tree: ast.Module) -> list[ast.stmt]:
        """Find module-level TypeAlias declarations (old-style and PEP 695).

        Detects both ``X: TypeAlias = ...`` (ast.AnnAssign) and
        ``type X = ...`` (ast.TypeAlias, PEP 695 / Python 3.12+).
        """
        matches: list[ast.stmt] = []
        for stmt in tree.body:
            # Old-style: X: TypeAlias = ...
            if isinstance(stmt, ast.AnnAssign):
                annotation = stmt.annotation
                if isinstance(annotation, ast.Name) and annotation.id == "TypeAlias":
                    matches.append(stmt)
                continue
            # PEP 695: type X = ...
            if isinstance(stmt, ast.TypeAlias):
                matches.append(stmt)
        return matches

    @staticmethod
    def find_standalone_typevars(tree: ast.Module) -> list[ast.Assign]:
        """Find module-level TypeVar/ParamSpec/TypeVarTuple assignments.

        Detects ``X = TypeVar(...)``, ``P = ParamSpec(...)``, and
        ``Ts = TypeVarTuple(...)`` at module level.
        """
        typevar_names = {"TypeVar", "ParamSpec", "TypeVarTuple"}
        matches: list[ast.Assign] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            func = stmt.value.func
            if (isinstance(func, ast.Name) and func.id in typevar_names) or (
                isinstance(func, ast.Attribute) and func.attr in typevar_names
            ):
                matches.append(stmt)
        return matches

    @staticmethod
    def generate_module_skeleton(
        class_name: str,
        base_class: str,
        docstring: str,
    ) -> str:
        """Generate a minimal base module file with correct imports."""
        import_line = FlextInfraCodegenTransforms._resolve_base_class_import(base_class)
        return f'"""Module skeleton for {class_name}.\n\n{docstring}\n\nCopyright (c) 2025 FLEXT Team. All rights reserved.\nSPDX-License-Identifier: MIT\n"""\n\nfrom __future__ import annotations\n\n{import_line}\n\n\nclass {class_name}({base_class}):\n    """{docstring}"""\n'

    @staticmethod
    def get_node_name(node: ast.stmt) -> str:
        """Extract assignment target name from a statement.

        Handles ast.Assign, ast.AnnAssign, and ast.TypeAlias (PEP 695).
        """
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        if isinstance(node, ast.TypeAlias):
            return node.name.id
        return ""

    @staticmethod
    def get_top_level_names_in_node(node: ast.stmt) -> frozenset[str]:
        """Collect all Name references used in a node."""
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return frozenset(names)

    @staticmethod
    def get_type_param_names(node: ast.stmt) -> frozenset[str]:
        """Extract locally-scoped type parameter names from a PEP 695 type alias.

        ``type X[T, *Ts, **P] = ...`` defines T, Ts, P as local type params.
        These names are part of the node itself and must NOT be treated as
        external dependencies when checking resolvability.
        """
        if not isinstance(node, ast.TypeAlias):
            return frozenset()
        names: set[str] = set()
        for tp in node.type_params:
            if isinstance(tp, (ast.TypeVar, ast.TypeVarTuple, ast.ParamSpec)):
                names.add(tp.name)
        return frozenset(names)

    @staticmethod
    def is_used_in_context(node: ast.stmt, tree: ast.Module) -> bool:
        """Check if a definition's name is referenced elsewhere in the module.

        Checks all statements (class headers, function signatures, annotations,
        body code). Handles ast.Assign, ast.AnnAssign, and ast.TypeAlias
        (PEP 695). A name that appears anywhere beyond its own definition is
        considered "in context".
        """
        name: str | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.TypeAlias):
            name = node.name.id
        if name is None:
            return False
        for stmt in tree.body:
            if stmt is node:
                continue
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name) and child.id == name:
                    return True
        return False

    @staticmethod
    def name_exists_in_module(name: str, tree: ast.Module) -> bool:
        """Check if a top-level name is already defined in a module.

        Handles ast.Assign, ast.AnnAssign, and ast.TypeAlias (PEP 695).
        """
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and (stmt.target.id == name)
            ):
                return True
            if isinstance(stmt, ast.TypeAlias) and stmt.name.id == name:
                return True
        return False

    @staticmethod
    def unparse_and_format(tree: ast.Module, path: Path) -> str:
        """Normalize locations and return unparsed source code."""
        del path
        fixed = ast.fix_missing_locations(tree)
        return ast.unparse(fixed)

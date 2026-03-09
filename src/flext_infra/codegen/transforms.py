"""AST utility library for safe Python code transformations.

Provides stateless helper functions for parsing, analyzing, and generating
Python code using the ast module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from flext_infra import FlextInfraUtilitiesSubprocess, c


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
        """Find Final assignments not used by classes in same file."""
        matches: list[ast.AnnAssign] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            annotation = stmt.annotation
            if isinstance(annotation, ast.Name) and annotation.id == "Final":
                if not FlextInfraCodegenTransforms.is_used_in_context(stmt, tree):
                    matches.append(stmt)
                continue
            if (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and (annotation.value.id == "Final")
                and (not FlextInfraCodegenTransforms.is_used_in_context(stmt, tree))
            ):
                matches.append(stmt)
        return matches

    @staticmethod
    def find_standalone_typealiases(tree: ast.Module) -> list[ast.stmt]:
        """Find TypeAlias declarations not used by classes in same file."""
        matches: list[ast.stmt] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            annotation = stmt.annotation
            if (
                isinstance(annotation, ast.Name)
                and annotation.id == "TypeAlias"
                and (not FlextInfraCodegenTransforms.is_used_in_context(stmt, tree))
            ):
                matches.append(stmt)
        return matches

    @staticmethod
    def find_standalone_typevars(tree: ast.Module) -> list[ast.Assign]:
        """Find TypeVar assignments not used by classes in same file."""
        matches: list[ast.Assign] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            func = stmt.value.func
            is_typevar = False
            if (isinstance(func, ast.Name) and func.id == "TypeVar") or (
                isinstance(func, ast.Attribute) and func.attr == "TypeVar"
            ):
                is_typevar = True
            if is_typevar and (
                not FlextInfraCodegenTransforms.is_used_in_context(stmt, tree)
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
        """Extract assignment target name from a statement."""
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
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
    def is_used_in_context(node: ast.stmt, tree: ast.Module) -> bool:
        """Check if TypeVar/TypeAlias name is used in class header context."""
        name: str | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name is None:
            return False
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.ClassDef):
                bases_module = ast.fix_missing_locations(
                    ast.Module(
                        body=[ast.Expr(value=base) for base in stmt.bases],
                        type_ignores=[],
                    ),
                )
                for base_node in ast.walk(bases_module):
                    if isinstance(base_node, ast.Name) and base_node.id == name:
                        return True
                keywords_module = ast.fix_missing_locations(
                    ast.Module(
                        body=[ast.Expr(value=kw.value) for kw in stmt.keywords],
                        type_ignores=[],
                    ),
                )
                for kw_node in ast.walk(keywords_module):
                    if isinstance(kw_node, ast.Name) and kw_node.id == name:
                        return True
        return False

    @staticmethod
    def name_exists_in_module(name: str, tree: ast.Module) -> bool:
        """Check if a top-level name is already defined in a module."""
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
        return False

    @staticmethod
    def parse_module(path: Path) -> ast.Module | None:
        """Parse a Python module from disk and return None on syntax error."""
        try:
            source = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            return ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            return None

    @staticmethod
    def run_ruff_fix(path: Path) -> None:
        """Run ruff check --fix and ruff format on a file."""
        runner = FlextInfraUtilitiesSubprocess()
        runner.run_checked([
            c.Infra.Cli.RUFF,
            c.Infra.Cli.RuffCmd.CHECK,
            "--fix",
            str(path),
        ])
        runner.run_checked([c.Infra.Cli.RUFF, c.Infra.Cli.RuffCmd.FORMAT, str(path)])

    @staticmethod
    def unparse_and_format(tree: ast.Module, path: Path) -> str:
        """Normalize locations and return unparsed source code."""
        del path
        fixed = ast.fix_missing_locations(tree)
        return ast.unparse(fixed)

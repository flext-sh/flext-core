"""AST utility library for safe Python code transformations.

Provides stateless helper functions for parsing, analyzing, and generating
Python code using the ast module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
from collections.abc import Sequence
from pathlib import Path

from flext_infra import c
from flext_infra._utilities.parsing import FlextInfraUtilitiesParsing


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
    def all_deps_resolvable(node: ast.stmt, target_tree: ast.Module) -> bool:
        """Check if all names used in node are available in the target module.

        Called AFTER copy_required_imports to verify the copy succeeded.
        A name is available if it's imported or defined in the target module.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
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
                name = FlextInfraCodegenTransforms.get_node_name(stmt)
                if name:
                    available.add(name)
        return all(n in available for n in names_used)

    @staticmethod
    def _all_deps_resolvable(node: ast.stmt, target_tree: ast.Module) -> bool:
        return FlextInfraCodegenTransforms.all_deps_resolvable(node, target_tree)

    @staticmethod
    def copy_required_imports(
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> None:
        """Copy imports needed by node from source_tree to target_tree.

        Mutates target_tree for analysis accumulation only - the tree is
        never written to disk via ast.unparse.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
        if not names_used:
            return
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
        last_import_idx = 0
        for i, stmt in enumerate(target_tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            ):
                last_import_idx = i + 1
        for i, imp in enumerate(imports_to_add):
            target_tree.body.insert(last_import_idx + i, imp)

    @staticmethod
    def _copy_required_imports(
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> None:
        FlextInfraCodegenTransforms.copy_required_imports(
            node,
            source_tree,
            target_tree,
        )

    @staticmethod
    def needs_first_party_import(
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> bool:
        """Check if moving node to target would require a first-party import.

        First-party imports (from flext_*) into typings.py create circular
        import chains because typings.py is imported by result.py, runtime.py,
        and other foundational modules. If the node depends on names that come
        from first-party imports and those names are NOT already available in
        the target module, moving the node is unsafe.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
        if not names_used:
            return False
        target_available: set[str] = set(dir(_builtins_module))
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                target_available.update(
                    (alias.asname or alias.name).split(".")[0] for alias in stmt.names
                )
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        target_available.add(imported)
            else:
                found = FlextInfraCodegenTransforms.get_node_name(stmt)
                if found:
                    target_available.add(found)
        missing = names_used - target_available
        if not missing:
            return False
        for stmt in source_tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module:
                if stmt.module.startswith("flext"):
                    for alias in stmt.names:
                        imported = alias.asname or alias.name
                        if imported in missing:
                            return True
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    top = (alias.asname or alias.name).split(".")[0]
                    if top.startswith("flext") and top in missing:
                        return True
        return False

    @staticmethod
    def _needs_first_party_import(
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> bool:
        return FlextInfraCodegenTransforms.needs_first_party_import(
            node,
            source_tree,
            target_tree,
        )

    @staticmethod
    def collect_import_texts_for_nodes(
        nodes: Sequence[ast.stmt],
        source_lines: list[str],
        source_tree: ast.Module,
        target_text: str,
    ) -> list[str]:
        """Collect import text lines from source needed by moved nodes.

        Returns import statement strings that should be added to the target
        file. Skips imports already present in the target text.
        """
        all_names: set[str] = set()
        for node in nodes:
            names = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
            node_name = FlextInfraCodegenTransforms.get_node_name(node)
            type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
            all_names.update(
                n for n in names if n != node_name and n not in type_params
            )
        if not all_names:
            return []
        import_texts: list[str] = []
        seen: set[str] = set()
        for stmt in source_tree.body:
            if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            provided: set[str] = set()
            if isinstance(stmt, ast.Import):
                provided.update(
                    (alias.asname or alias.name).split(".")[0] for alias in stmt.names
                )
            else:
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        provided.add(imported)
            if not (provided & all_names):
                continue
            start = stmt.lineno
            end = stmt.end_lineno or start
            text = "\n".join(source_lines[start - 1 : end]).strip()
            if text not in seen and text not in target_text:
                seen.add(text)
                import_texts.append(text)
        return import_texts

    @staticmethod
    def _collect_import_texts_for_nodes(
        nodes: Sequence[ast.stmt],
        source_lines: list[str],
        source_tree: ast.Module,
        target_text: str,
    ) -> list[str]:
        return FlextInfraCodegenTransforms.collect_import_texts_for_nodes(
            nodes,
            source_lines,
            source_tree,
            target_text,
        )

    @staticmethod
    def prune_stale_all_assignment(*, path: Path) -> bool:
        """Remove stale entries from __all__ that no longer exist in the module."""
        try:
            source = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except (OSError, UnicodeDecodeError):
            return False
        tree = FlextInfraUtilitiesParsing.parse_ast_from_source(source)
        if tree is None:
            return False
        assignment: ast.Assign | None = None
        exports: list[str] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id != "__all__":
                continue
            if not isinstance(stmt.value, (ast.List, ast.Tuple)):
                continue
            names: list[str] = []
            is_literal_list = True
            for element in stmt.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.append(element.value)
                    continue
                is_literal_list = False
                break
            if not is_literal_list:
                continue
            assignment = stmt
            exports = names
            break
        if assignment is None or len(exports) == 0:
            return False
        available: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.FunctionDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.AsyncFunctionDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    available.add(imported.split(".")[0])
                continue
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        available.add(imported)
                continue
            found_name = FlextInfraCodegenTransforms.get_node_name(stmt)
            if found_name:
                available.add(found_name)
        filtered = [
            name for name in exports if name in available or name.startswith("__")
        ]
        if filtered == exports:
            return False
        block = "__all__ = [\n" + "\n".join(f'    "{name}",' for name in filtered)
        if len(filtered) == 0:
            block = "__all__ = []"
        else:
            block += "\n]"
        lines = source.splitlines()
        if assignment.lineno <= 0 or assignment.end_lineno is None:
            return False
        start = assignment.lineno - 1
        end = assignment.end_lineno
        updated_lines = [*lines[:start], block, *lines[end:]]
        updated = "\n".join(updated_lines)
        if source.endswith("\n"):
            updated += "\n"
        if updated == source:
            return False
        path.write_text(updated, encoding=c.Infra.Encoding.DEFAULT)
        return True

    @staticmethod
    def _prune_stale_all_assignment(*, path: Path) -> bool:
        return FlextInfraCodegenTransforms.prune_stale_all_assignment(path=path)

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

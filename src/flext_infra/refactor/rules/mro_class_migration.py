"""Rule that migrates module-level constants into constants facade classes."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import override

import libcst as cst

from flext_infra import c, m, u
from flext_infra.refactor.mro_migrator import FlextInfraRefactorMROMigrationTransformer
from flext_infra.refactor.mro_resolver import FlextInfraRefactorMROResolver
from flext_infra.refactor.rule import FlextInfraRefactorRule


class FlextInfraRefactorMROClassMigrationRule(FlextInfraRefactorRule):
    """Apply MRO constants-class migration to a single module."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        if _file_path is None:
            return (tree, [])
        if _file_path.name != c.Infra.Refactor.CONSTANTS_FILE_GLOB:
            return (tree, [])
        source = tree.code
        # given source is in-memory CST output, parse from string is required
        module_ast = u.Infra.parse_ast_from_source(source)
        if module_ast is None:
            return (tree, [])
        candidates: list[m.Infra.Refactor.MROSymbolCandidate] = []
        for stmt in module_ast.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            if not self._is_constant_candidate(stmt.target.id):
                continue
            if not self._is_final_annotation(stmt.annotation):
                continue
            candidates.append(
                m.Infra.Refactor.MROSymbolCandidate(
                    symbol=stmt.target.id,
                    line=stmt.lineno,
                ),
            )
        if len(candidates) == 0:
            return (tree, [])
        constants_class = self._first_constants_class_name(module_ast)
        scan_result = m.Infra.Refactor.MROScanReport(
            file=str(_file_path),
            module="",
            constants_class=constants_class,
            candidates=tuple(candidates),
        )
        updated_source, migration, _ = (
            FlextInfraRefactorMROMigrationTransformer.migrate_file(
                scan_result=scan_result,
            )
        )
        if len(migration.moved_symbols) == 0 or updated_source == source:
            return (tree, [])
        updated_module = u.Infra.parse_cst_from_source(updated_source)
        if updated_module is None:
            return (tree, [])
        syms = ", ".join(migration.moved_symbols)
        return (updated_module, [f"migrated constants into facade class: {syms}"])

    @staticmethod
    def _first_constants_class_name(tree: ast.Module) -> str:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name.endswith(
                c.Infra.Refactor.CONSTANTS_CLASS_SUFFIX,
            ):
                return stmt.name
        return ""

    @staticmethod
    def _is_constant_candidate(symbol: str) -> bool:
        return FlextInfraRefactorMROResolver.CONSTANT_PATTERN.match(symbol) is not None

    @staticmethod
    def _is_final_annotation(annotation: ast.expr) -> bool:
        final_name = c.Infra.Refactor.FINAL_ANNOTATION_NAME
        if isinstance(annotation, ast.Name):
            return annotation.id == final_name
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == final_name
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == final_name
            if isinstance(base, ast.Attribute):
                return base.attr == final_name
        return False


__all__ = ["FlextInfraRefactorMROClassMigrationRule"]

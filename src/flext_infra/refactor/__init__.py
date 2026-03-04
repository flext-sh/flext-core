"""Refactor engine baseado em libcst e regras declarativas.

Uso:
    # Refatorar projeto inteiro
    python -m flext_infra refactor --project ../projeto --dry-run

    # Refatorar arquivo específico
    python -m flext_infra refactor --file src/module.py --dry-run

    # Rodar regras específicas
    python -m flext_infra refactor --project ../projeto --rules legacy,import --dry-run

    # Listar regras disponíveis
    python -m flext_infra refactor --list-rules
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import QualifiedNameProvider, QualifiedNameSource
import yaml

from flext_core import r
from flext_infra.output import output


class MethodCategory(Enum):
    """Categorias de métodos para ordenação."""

    MAGIC = auto()
    PROPERTY = auto()
    STATIC = auto()
    CLASS = auto()
    PUBLIC = auto()
    PROTECTED = auto()
    PRIVATE = auto()


@dataclass
class MethodInfo:
    """Informações sobre um método para ordenação."""

    name: str
    category: MethodCategory
    node: cst.FunctionDef
    decorators: list[str] = field(default_factory=list)


@dataclass
class RefactorResult:
    """Resultado da refatoração de um arquivo."""

    file_path: Path
    success: bool
    modified: bool
    error: str | None = None
    changes: list[str] = field(default_factory=list)


class RefactorRule:
    """Base para regras de refatoração."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rule_id = config.get("id", "unknown")
        self.name = config.get("name", self.rule_id)
        self.description = config.get("description", "")
        self.enabled = config.get("enabled", True)
        self.severity = config.get("severity", "warning")

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        """Aplica a regra na AST.

        Retorna:
            Tuple de (árvore modificada, lista de mudanças)
        """
        return tree, []

    def matches_filter(self, filter_pattern: str) -> bool:
        """Verifica se a regra corresponde ao filtro."""
        pattern_lower = filter_pattern.lower()
        return (
            pattern_lower in self.rule_id.lower()
            or pattern_lower in self.name.lower()
            or pattern_lower in self.description.lower()
        )


class LegacyRemovalRule(RefactorRule):
    """Remove código legado: aliases, deprecated."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []

        if "alias" in self.rule_id:
            tree, alias_changes = self._remove_aliases(tree)
            changes.extend(alias_changes)

        if "deprecated" in self.rule_id:
            tree, deprecated_changes = self._remove_deprecated(tree)
            changes.extend(deprecated_changes)

        if "wrapper" in self.rule_id:
            tree, wrapper_changes = self._remove_wrappers(tree)
            changes.extend(wrapper_changes)

        if "bypass" in self.rule_id:
            tree, bypass_changes = self._remove_import_bypasses(tree)
            changes.extend(bypass_changes)

        return tree, changes

    def _remove_wrappers(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []

        new_body: list[cst.BaseStatement] = []
        for stmt in tree.body:
            if not isinstance(stmt, cst.FunctionDef):
                new_body.append(stmt)
                continue

            target_name = self._get_passthrough_target(stmt)
            if target_name is None:
                new_body.append(stmt)
                continue

            alias_assign = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(stmt.name.value))],
                        value=cst.Name(target_name),
                    )
                ]
            )
            new_body.append(alias_assign)
            changes.append(f"Inlined wrapper: {stmt.name.value} -> {target_name}")

        return tree.with_changes(body=new_body), changes

    def _get_passthrough_target(self, func: cst.FunctionDef) -> str | None:
        if not isinstance(func.body, cst.IndentedBlock):
            return None
        if len(func.body.body) != 1:
            return None

        stmt = func.body.body[0]
        if not isinstance(stmt, cst.SimpleStatementLine):
            return None
        if len(stmt.body) != 1:
            return None

        small_stmt = stmt.body[0]
        if not isinstance(small_stmt, cst.Return):
            return None
        if not isinstance(small_stmt.value, cst.Call):
            return None
        if not isinstance(small_stmt.value.func, cst.Name):
            return None

        call_args = small_stmt.value.args
        param_names = [
            param.name.value
            for param in func.params.params
            if isinstance(param.name, cst.Name)
        ]
        if len(call_args) != len(param_names):
            return None

        for idx, arg in enumerate(call_args):
            if arg.keyword is not None:
                return None
            if arg.star not in {"", None}:
                return None
            if not isinstance(arg.value, cst.Name):
                return None
            if arg.value.value != param_names[idx]:
                return None

        return small_stmt.value.func.value

    def _remove_import_bypasses(
        self,
        tree: cst.Module,
    ) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []

        class ImportBypassRemover(cst.CSTTransformer):
            def leave_Try(self, original_node, updated_node):
                if len(updated_node.body.body) != 1:
                    return updated_node
                if len(updated_node.handlers) != 1:
                    return updated_node

                body_stmt = updated_node.body.body[0]
                handler = updated_node.handlers[0]
                if not isinstance(handler, cst.ExceptHandler):
                    return updated_node
                if not isinstance(handler.body, cst.IndentedBlock):
                    return updated_node
                if len(handler.body.body) != 1:
                    return updated_node

                fallback_stmt = handler.body.body[0]
                if not (
                    isinstance(body_stmt, cst.SimpleStatementLine)
                    and isinstance(fallback_stmt, cst.SimpleStatementLine)
                ):
                    return updated_node
                if len(body_stmt.body) != 1 or len(fallback_stmt.body) != 1:
                    return updated_node

                primary_import = body_stmt.body[0]
                fallback_import = fallback_stmt.body[0]
                if not (
                    isinstance(primary_import, cst.ImportFrom)
                    and isinstance(fallback_import, cst.ImportFrom)
                ):
                    return updated_node

                handler_type = handler.type
                if not isinstance(handler_type, cst.Name):
                    return updated_node
                if handler_type.value != "ImportError":
                    return updated_node

                changes.append("Removed import bypass fallback")
                return body_stmt

        return tree.visit(ImportBypassRemover()), changes

    def _remove_aliases(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        """Remove aliases de compatibilidade no nível do módulo."""
        changes: list[str] = []
        allow_aliases = set(self.config.get("allow_aliases", []))
        allow_target_suffixes = tuple(self.config.get("allow_target_suffixes", []))

        class AliasRemover(cst.CSTTransformer):
            def __init__(self) -> None:
                self._scope_depth = 0

            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                self._scope_depth += 1

            def leave_ClassDef(
                self, original_node: cst.ClassDef, updated_node: cst.ClassDef
            ) -> cst.ClassDef:
                self._scope_depth -= 1
                return updated_node

            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                self._scope_depth += 1

            def leave_FunctionDef(
                self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
            ) -> cst.FunctionDef:
                self._scope_depth -= 1
                return updated_node

            def visit_AsyncFunctionDef(self, node: cst.FunctionDef) -> None:
                self._scope_depth += 1

            def leave_AsyncFunctionDef(
                self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
            ) -> cst.FunctionDef:
                self._scope_depth -= 1
                return updated_node

            def leave_Assign(self, original_node, updated_node):
                if self._scope_depth > 0:
                    return updated_node
                # Verificar se é simples: Name = Name (alias)
                if (
                    len(original_node.targets) == 1
                    and isinstance(original_node.targets[0].target, cst.Name)
                    and isinstance(original_node.value, cst.Name)
                ):
                    target = original_node.targets[0].target.value
                    value = original_node.value.value
                    if target in allow_aliases:
                        return updated_node
                    if allow_target_suffixes and value.endswith(allow_target_suffixes):
                        return updated_node
                    if target != value and target not in {"__version__", "__all__"}:
                        changes.append(f"Removed alias: {target} = {value}")
                        return cst.RemovalSentinel.REMOVE
                return updated_node

        return tree.visit(AliasRemover()), changes

    def _remove_deprecated(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        """Remove classes marcadas como deprecated."""
        changes = []

        class DeprecatedRemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                class_name = original_node.name.value

                # Verificar se nome contém Deprecated
                if "deprecated" in class_name.lower():
                    changes.append(f"Removed deprecated class: {class_name}")
                    return cst.RemovalSentinel.REMOVE

                # Verificar se tem warnings.warn com DeprecationWarning
                for stmt in original_node.body.body:
                    if (
                        isinstance(stmt, cst.FunctionDef)
                        and stmt.name.value == "__init__"
                    ):
                        for sub_stmt in stmt.body.body:
                            if isinstance(sub_stmt, cst.SimpleStatementLine):
                                for line in sub_stmt.body:
                                    if isinstance(line, cst.Expr):
                                        if isinstance(line.value, cst.Call):
                                            func = line.value.func
                                            if isinstance(func, cst.Attribute):
                                                if func.attr.value == "warn":
                                                    changes.append(
                                                        f"Removed deprecated class: {class_name}"
                                                    )
                                                    return cst.RemovalSentinel.REMOVE

                return updated_node

        return tree.visit(DeprecatedRemover()), changes


class ImportModernizerRule(RefactorRule):
    """Moderniza imports para usar runtime aliases explícitos."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        if "lazy-import" in self.rule_id:
            return self._fix_lazy_imports(tree)

        forbidden = self.config.get("forbidden_imports")
        if forbidden is None:
            forbidden = [self.config]

        if not forbidden:
            return tree, []

        imports_to_remove: list[str] = []
        symbols_to_replace: dict[str, str] = {}

        for rule in forbidden:
            module = rule.get("module", "")
            mapping = rule.get("symbol_mapping", {})

            imports_to_remove.append(module)

            for symbol, alias_path in mapping.items():
                symbols_to_replace[symbol] = alias_path
        changes = []

        class ImportModernizer(cst.CSTTransformer):
            METADATA_DEPENDENCIES = (QualifiedNameProvider,)

            def __init__(self):
                self.modified_imports = False
                self.aliases_needed: set[str] = set()
                self.aliases_present: set[str] = set()
                self.active_symbol_replacements: dict[str, str] = {}

            def leave_ImportFrom(self, original_node, updated_node):
                module_name = self._get_module_name(original_node.module)

                if module_name == "flext_core":
                    imported_aliases = self._extract_import_aliases(original_node.names)
                    for imported_alias in imported_aliases:
                        if not isinstance(imported_alias.name, cst.Name):
                            continue
                        imported_name = imported_alias.name.value
                        bound_name = imported_name
                        if imported_alias.asname is not None and isinstance(
                            imported_alias.asname.name, cst.Name
                        ):
                            bound_name = imported_alias.asname.name.value
                        if bound_name in {
                            "c",
                            "m",
                            "r",
                            "t",
                            "u",
                            "p",
                            "d",
                            "e",
                            "h",
                            "s",
                            "x",
                        }:
                            self.aliases_present.add(bound_name)

                for mod in imports_to_remove:
                    if module_name == mod:
                        imported_aliases = self._extract_import_aliases(
                            original_node.names
                        )
                        if not imported_aliases:
                            return updated_node

                        mapped_aliases: list[cst.ImportAlias] = []
                        unmapped_aliases: list[cst.ImportAlias] = []
                        for imported_alias in imported_aliases:
                            if not isinstance(imported_alias.name, cst.Name):
                                unmapped_aliases.append(imported_alias)
                                continue
                            imported_symbol = imported_alias.name.value
                            if imported_symbol not in symbols_to_replace:
                                unmapped_aliases.append(imported_alias)
                                continue
                            mapped_aliases.append(imported_alias)
                            local_symbol = imported_symbol
                            if imported_alias.asname is not None and isinstance(
                                imported_alias.asname.name, cst.Name
                            ):
                                local_symbol = imported_alias.asname.name.value
                            alias_path = symbols_to_replace[imported_symbol]
                            self.active_symbol_replacements[local_symbol] = alias_path
                            self.aliases_needed.add(alias_path.split(".")[0])

                        if not mapped_aliases:
                            return updated_node

                        self.modified_imports = True
                        changes.append(f"Removed import: from {module_name}")
                        if unmapped_aliases:
                            return updated_node.with_changes(
                                names=tuple(unmapped_aliases)
                            )
                        return cst.RemovalSentinel.REMOVE

                return updated_node

            def leave_Name(self, original_node, updated_node):
                if original_node.value in self.active_symbol_replacements:
                    qualified_names = self.get_metadata(
                        QualifiedNameProvider,
                        original_node,
                        default=set(),
                    )
                    if not qualified_names:
                        return updated_node
                    if any(
                        qualified_name.source != QualifiedNameSource.IMPORT
                        for qualified_name in qualified_names
                    ):
                        return updated_node

                    alias_path = self.active_symbol_replacements[original_node.value]
                    parts = alias_path.split(".")

                    # Construir atributo: c.System.PLATFORM
                    result = cst.Name(parts[0])
                    for part in parts[1:]:
                        result = cst.Attribute(value=result, attr=cst.Name(part))

                    changes.append(f"Replaced: {original_node.value} -> {alias_path}")
                    return result
                return updated_node

            def _extract_import_aliases(self, names: Any) -> list[cst.ImportAlias]:
                imported_aliases: list[cst.ImportAlias] = []
                if isinstance(names, cst.ImportStar):
                    return imported_aliases

                aliases: tuple[cst.ImportAlias, ...] = tuple(names)
                for alias in aliases:
                    imported_aliases.append(alias)

                return imported_aliases

            def _get_module_name(self, module) -> str:
                if isinstance(module, cst.Name):
                    return module.value
                elif isinstance(module, cst.Attribute):
                    parts = []
                    current = module
                    while isinstance(current, cst.Attribute):
                        parts.append(current.attr.value)
                        current = current.value
                    if isinstance(current, cst.Name):
                        parts.append(current.value)
                    return ".".join(reversed(parts))
                return ""

            def leave_Module(self, original_node, updated_node):
                missing_aliases = sorted(self.aliases_needed - self.aliases_present)
                if self.modified_imports and missing_aliases:
                    alias_imports = [
                        cst.ImportAlias(name=cst.Name(alias))
                        for alias in missing_aliases
                    ]
                    new_import = cst.SimpleStatementLine(
                        body=[
                            cst.ImportFrom(
                                module=cst.Name("flext_core"), names=alias_imports
                            )
                        ]
                    )

                    # Inserir após docstring e __future__
                    body = list(updated_node.body)
                    insert_idx = 0

                    # Pular docstring
                    if (
                        body
                        and isinstance(body[0], cst.SimpleStatementLine)
                        and len(body[0].body) == 1
                        and isinstance(body[0].body[0], cst.Expr)
                        and isinstance(body[0].body[0].value, cst.SimpleString)
                    ):
                        insert_idx = 1

                    # Pular __future__ imports
                    while insert_idx < len(body) and isinstance(
                        body[insert_idx], cst.SimpleStatementLine
                    ):
                        stmt = body[insert_idx]
                        if (
                            len(stmt.body) == 1
                            and isinstance(stmt.body[0], cst.ImportFrom)
                            and isinstance(stmt.body[0].module, cst.Name)
                            and stmt.body[0].module.value == "__future__"
                        ):
                            insert_idx += 1
                        else:
                            break

                    changes.append(
                        f"Added: from flext_core import {', '.join(missing_aliases)}"
                    )
                    new_body = body[:insert_idx] + [new_import] + body[insert_idx:]
                    return updated_node.with_changes(body=new_body)
                return updated_node

        modernizer = ImportModernizer()
        wrapper = cst.MetadataWrapper(tree)
        return wrapper.visit(modernizer), changes

    def _fix_lazy_imports(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []

        class LazyImportFixer(cst.CSTTransformer):
            def __init__(self) -> None:
                self.hoisted_imports: list[cst.SimpleStatementLine] = []

            def leave_FunctionDef(self, original_node, updated_node):
                if not isinstance(updated_node.body, cst.IndentedBlock):
                    return updated_node

                new_function_body: list[cst.BaseStatement] = []
                for stmt in updated_node.body.body:
                    if (
                        isinstance(stmt, cst.SimpleStatementLine)
                        and len(stmt.body) == 1
                        and isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))
                    ):
                        self.hoisted_imports.append(stmt)
                        changes.append(
                            f"Hoisted lazy import in function {original_node.name.value}"
                        )
                        continue
                    new_function_body.append(stmt)

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_function_body)
                )

            def leave_Module(self, original_node, updated_node):
                if not self.hoisted_imports:
                    return updated_node

                existing_import_codes: set[str] = set()
                for stmt in updated_node.body:
                    if not isinstance(stmt, cst.SimpleStatementLine):
                        continue
                    if len(stmt.body) != 1:
                        continue
                    if isinstance(stmt.body[0], (cst.Import, cst.ImportFrom)):
                        existing_import_codes.add(cst.Module(body=[stmt]).code)

                unique_hoisted: list[cst.SimpleStatementLine] = []
                for stmt in self.hoisted_imports:
                    stmt_code = cst.Module(body=[stmt]).code
                    if stmt_code in existing_import_codes:
                        continue
                    existing_import_codes.add(stmt_code)
                    unique_hoisted.append(stmt)

                if not unique_hoisted:
                    return updated_node

                body = list(updated_node.body)
                insert_idx = 0
                if (
                    body
                    and isinstance(body[0], cst.SimpleStatementLine)
                    and len(body[0].body) == 1
                    and isinstance(body[0].body[0], cst.Expr)
                    and isinstance(body[0].body[0].value, cst.SimpleString)
                ):
                    insert_idx = 1

                while insert_idx < len(body) and isinstance(
                    body[insert_idx], cst.SimpleStatementLine
                ):
                    stmt = body[insert_idx]
                    if (
                        len(stmt.body) == 1
                        and isinstance(stmt.body[0], cst.ImportFrom)
                        and isinstance(stmt.body[0].module, cst.Name)
                        and stmt.body[0].module.value == "__future__"
                    ):
                        insert_idx += 1
                        continue
                    break

                new_body = body[:insert_idx] + unique_hoisted + body[insert_idx:]
                return updated_node.with_changes(body=new_body)

        fixer = LazyImportFixer()
        return tree.visit(fixer), changes


class EnsureFutureAnnotationsRule(RefactorRule):
    """Garante que 'from __future__ import annotations' existe no arquivo."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []
        body = list(tree.body)
        insert_idx = 0
        has_docstring = False

        if (
            body
            and isinstance(body[0], cst.SimpleStatementLine)
            and len(body[0].body) == 1
            and isinstance(body[0].body[0], cst.Expr)
            and isinstance(body[0].body[0].value, cst.SimpleString)
        ):
            has_docstring = True
            insert_idx = 1

        existing_annotations_stmt: cst.SimpleStatementLine | None = None
        non_annotation_future_stmts: list[cst.BaseStatement] = []
        body_without_future: list[cst.BaseStatement] = []

        for stmt in body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                body_without_future.append(stmt)
                continue

            if (
                len(stmt.body) == 1
                and isinstance(stmt.body[0], cst.ImportFrom)
                and isinstance(stmt.body[0].module, cst.Name)
                and stmt.body[0].module.value == "__future__"
            ):
                import_from = stmt.body[0]
                aliases: tuple[cst.ImportAlias, ...] = tuple(import_from.names)
                contains_annotations = any(
                    isinstance(alias.name, cst.Name)
                    and alias.name.value == "annotations"
                    for alias in aliases
                )
                if contains_annotations:
                    existing_annotations_stmt = stmt
                else:
                    non_annotation_future_stmts.append(stmt)
                continue

            body_without_future.append(stmt)

        needs_leading_blank_line = has_docstring
        if existing_annotations_stmt is not None:
            annotations_stmt = existing_annotations_stmt
        else:
            annotations_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name("__future__"),
                        names=[cst.ImportAlias(name=cst.Name("annotations"))],
                    )
                ]
            )
            changes.append("Ensured: from __future__ import annotations")

        if needs_leading_blank_line:
            annotations_stmt = annotations_stmt.with_changes(
                leading_lines=[cst.EmptyLine()]
            )

        future_block = [annotations_stmt, *non_annotation_future_stmts]
        new_body = (
            body_without_future[:insert_idx]
            + future_block
            + body_without_future[insert_idx:]
        )

        if (
            new_body != body
            and "Ensured: from __future__ import annotations" not in changes
        ):
            changes.append("Moved: from __future__ import annotations")

        return tree.with_changes(body=new_body), changes


class ClassReconstructorRule(RefactorRule):
    """Reconstrói classes: ordena métodos, organiza estrutura."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        order_config = self.config.get("method_order") or self.config.get("order", [])

        if not order_config:
            return tree, []

        changes = []

        class ClassReconstructor(cst.CSTTransformer):
            def __init__(self):
                self.order_config = order_config

            def leave_ClassDef(self, original_node, updated_node):
                # Separar métodos de outros membros
                methods: list[MethodInfo] = []
                other_members = []

                for stmt in updated_node.body.body:
                    if isinstance(stmt, cst.FunctionDef):
                        info = self._analyze_method(stmt)
                        methods.append(info)
                    else:
                        other_members.append(stmt)

                if not methods:
                    return updated_node

                # Ordenar métodos
                sorted_methods = self._sort_methods(methods)

                original_method_names = [method.name for method in methods]
                sorted_method_names = [method.name for method in sorted_methods]
                if original_method_names == sorted_method_names:
                    return updated_node

                new_body = other_members + [m.node for m in sorted_methods]

                changes.append(
                    f"Reordered {len(methods)} methods in class {original_node.name.value}"
                )

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )

            def _analyze_method(self, node: cst.FunctionDef) -> MethodInfo:
                """Analisa um método e determina sua categoria."""
                name = node.name.value
                decorators = []

                for dec in node.decorators:
                    if isinstance(dec.decorator, cst.Name):
                        decorators.append(dec.decorator.value)
                    elif isinstance(dec.decorator, cst.Attribute):
                        decorators.append(dec.decorator.attr.value)

                # Determinar categoria
                category = self._categorize(name, decorators)

                return MethodInfo(
                    name=name, category=category, node=node, decorators=decorators
                )

            def _categorize(self, name: str, decorators: list[str]) -> MethodCategory:
                """Categoriza um método."""
                # Verificar decorators
                if any(
                    d in decorators
                    for d in ["property", "cached_property", "computed_field"]
                ):
                    return MethodCategory.PROPERTY
                if "staticmethod" in decorators:
                    return MethodCategory.STATIC
                if "classmethod" in decorators:
                    return MethodCategory.CLASS

                # Verificar nome
                if name.startswith("__") and name.endswith("__"):
                    return MethodCategory.MAGIC
                elif name.startswith("__"):
                    return MethodCategory.PRIVATE
                elif name.startswith("_"):
                    return MethodCategory.PROTECTED
                else:
                    return MethodCategory.PUBLIC

            def _sort_methods(self, methods: list[MethodInfo]) -> list[MethodInfo]:
                def matches_rule(method: MethodInfo, rule: dict[str, Any]) -> bool:
                    decorators = set(method.decorators)
                    exclude_decorators = set(rule.get("exclude_decorators", []))
                    if exclude_decorators and decorators.intersection(
                        exclude_decorators
                    ):
                        return False

                    visibility = rule.get("visibility")
                    if visibility == "public" and method.name.startswith("_"):
                        return False
                    if visibility == "protected" and not (
                        method.name.startswith("_") and not method.name.startswith("__")
                    ):
                        return False
                    if visibility == "private" and not (
                        method.name.startswith("__") and not method.name.endswith("__")
                    ):
                        return False

                    rule_decorators = rule.get("decorators", [])
                    if rule_decorators and not decorators.intersection(rule_decorators):
                        return False

                    patterns = rule.get("patterns", [])
                    if patterns:
                        matched = False
                        for pattern in patterns:
                            if isinstance(pattern, str):
                                if re.match(pattern, method.name):
                                    matched = True
                                continue

                            regex = pattern.get("regex")
                            if regex and re.match(regex, method.name):
                                matched = True

                            pattern_decorators = pattern.get("decorators", [])
                            if pattern_decorators and decorators.intersection(
                                pattern_decorators
                            ):
                                matched = True
                        if not matched:
                            return False

                    return True

                def sort_key(method: MethodInfo) -> tuple[int, int, str]:
                    for idx, rule in enumerate(self.order_config):
                        if rule.get("category") == "class_attributes":
                            continue
                        if matches_rule(method, rule):
                            explicit_order = rule.get("order", [])
                            if explicit_order:
                                if method.name in explicit_order:
                                    return (
                                        idx,
                                        explicit_order.index(method.name),
                                        method.name,
                                    )
                                if "*" in explicit_order:
                                    return (
                                        idx,
                                        explicit_order.index("*") + 1,
                                        method.name,
                                    )
                            return (idx, 0, method.name)
                    return (len(self.order_config), 0, method.name)

                return sorted(methods, key=sort_key)

        return tree.visit(ClassReconstructor()), changes


class MRORedundancyChecker(RefactorRule):
    """Detecta e corrige redeclarações via MRO."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        changes = []

        class MRORemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                # Verificar classes aninhadas que herdam de algo do pai
                if not isinstance(updated_node.body, cst.IndentedBlock):
                    return updated_node

                new_body = []
                for stmt in updated_node.body.body:
                    if isinstance(stmt, cst.ClassDef) and stmt.bases:
                        # Verificar se herda de classe externa
                        for base in stmt.bases:
                            if isinstance(base.value, cst.Attribute):
                                # Ex: class Platform(FlextConstants.Platform)
                                changes.append(
                                    f"Fixed MRO redeclaration: {stmt.name.value}"
                                )
                                stmt = stmt.with_changes(bases=(), lpar=(), rpar=())
                                break
                    new_body.append(stmt)

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )

        return tree.visit(MRORemover()), changes


class FlextRefactorEngine:
    """Engine de refatoração que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[RefactorRule] = []
        self.rule_filters: list[str] = []

    def _default_config_path(self) -> Path:
        """Retorna caminho padrão do config dentro de flext_infra."""
        return Path(__file__).parent / "config.yml"

    def set_rule_filters(self, filters: list[str]) -> None:
        """Define filtros para regras (apenas regras que correspondem serão executadas)."""
        self.rule_filters = [f.lower() for f in filters]

    def load_config(self) -> r[dict[str, Any]]:
        """Carrega configuração do YAML."""
        try:
            content = self.config_path.read_text()
            loaded = yaml.safe_load(content)
            self.config = loaded if loaded is not None else {}
            output.info(f"Loaded config from {self.config_path}")
            return r[dict[str, Any]].ok(self.config)
        except Exception as e:
            return r[dict[str, Any]].fail(f"Failed to load config: {e}")

    def load_rules(self) -> r[list[RefactorRule]]:
        """Carrega regras de arquivos YAML."""
        try:
            rules_dir = self.config_path.parent / "rules"
            loaded_rules: list[RefactorRule] = []

            for rule_file in sorted(rules_dir.glob("*.yml")):
                output.info(f"Loading rules from {rule_file.name}")
                rule_config = yaml.safe_load(rule_file.read_text())
                if rule_config is None:
                    continue
                rules = rule_config.get("rules", [])

                for rule_def in rules:
                    if not rule_def.get("enabled", True):
                        continue

                    rule_id = rule_def.get("id", "unknown")

                    # Instanciar regra apropriada
                    rule: RefactorRule | None = None
                    if "ensure-future" in rule_id or "future-annotations" in rule_id:
                        rule = EnsureFutureAnnotationsRule(rule_def)
                    elif any(
                        x in rule_id
                        for x in ["legacy", "alias", "deprecated", "wrapper", "bypass"]
                    ):
                        rule = LegacyRemovalRule(rule_def)
                    elif any(x in rule_id for x in ["import", "modernize"]):
                        rule = ImportModernizerRule(rule_def)
                    elif any(x in rule_id for x in ["class", "reorder", "method"]):
                        rule = ClassReconstructorRule(rule_def)
                    elif "mro" in rule_id:
                        rule = MRORedundancyChecker(rule_def)

                    if rule:
                        # Aplicar filtros se existirem
                        if self.rule_filters:
                            if any(rule.matches_filter(f) for f in self.rule_filters):
                                loaded_rules.append(rule)
                        else:
                            loaded_rules.append(rule)

            self.rules = loaded_rules
            output.info(f"Loaded {len(self.rules)} rules")
            if self.rule_filters:
                output.info(f"Active filters: {', '.join(self.rule_filters)}")
            return r[list[RefactorRule]].ok(loaded_rules)
        except Exception as e:
            return r[list[RefactorRule]].fail(f"Failed to load rules: {e}")

    def list_rules(self) -> list[dict[str, Any]]:
        """Lista todas as regras disponíveis."""
        rules_info = []
        for rule in self.rules:
            rules_info.append({
                "id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity,
            })
        return rules_info

    def refactor_file(self, file_path: Path, dry_run: bool = False) -> RefactorResult:
        """Refatora um único arquivo."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)

            all_changes = []

            # Aplicar todas as regras
            for rule in self.rules:
                if rule.enabled:
                    tree, changes = rule.apply(tree, file_path)
                    all_changes.extend(changes)

            result_code = tree.code if hasattr(tree, "code") else str(tree)
            modified = result_code != source

            if not dry_run and modified:
                file_path.write_text(result_code, encoding="utf-8")

            return RefactorResult(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=all_changes,
            )
        except Exception as e:
            return RefactorResult(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(e),
                changes=[],
            )

    def refactor_files(
        self, file_paths: list[Path], dry_run: bool = False
    ) -> list[RefactorResult]:
        """Refatora múltiplos arquivos."""
        results = []

        for file_path in file_paths:
            result = self.refactor_file(file_path, dry_run=dry_run)
            results.append(result)

            # Log do resultado
            if result.success:
                if result.modified:
                    output.info(
                        f"{'[DRY-RUN] ' if dry_run else ''}Modified: {file_path.name}"
                    )
                    for change in result.changes:
                        output.info(f"  - {change}")
                else:
                    output.info(f"Unchanged: {file_path.name}")
            else:
                output.error(f"Failed: {file_path.name} - {result.error}")

        return results

    def refactor_project(
        self, project_path: Path, dry_run: bool = False, pattern: str = "*.py"
    ) -> list[RefactorResult]:
        """Refatora projeto inteiro."""
        src_dir = project_path / "src"
        if not src_dir.exists():
            output.error(f"No src/ directory in {project_path}")
            return []

        # Coletar arquivos
        files = []
        for py_file in src_dir.rglob(pattern):
            # Ignorar arquivos especiais
            if py_file.name in {"__init__.py", "conftest.py"}:
                continue
            files.append(py_file)

        output.info(f"Found {len(files)} files to process")

        # Processar arquivos
        return self.refactor_files(files, dry_run=dry_run)

    def initialize(self) -> r[bool]:
        """Inicializa engine carregando config e regras."""
        result = self.load_config()
        if not result.is_success:
            return r[bool].fail(f"Config error: {result.error}")

        result = self.load_rules()
        if not result.is_success:
            return r[bool].fail(f"Rules error: {result.error}")

        return r[bool].ok(True)


def print_rules_table(rules: list[dict[str, Any]]) -> None:
    """Imprime tabela de regras formatada."""
    output.header("Available Rules")

    if not rules:
        output.info("No rules loaded.")
        return

    # Calcular larguras
    id_width = max(len(r["id"]) for r in rules) + 2
    name_width = max(len(r["name"]) for r in rules) + 2

    # Cabeçalho
    header = f"{'ID':<{id_width}} {'Name':<{name_width}} {'Severity':<10} {'Status'}"
    output.info(header)
    output.info("-" * len(header))

    # Linhas
    for rule in rules:
        status = "✓" if rule["enabled"] else "✗"
        line = f"{rule['id']:<{id_width}} {rule['name']:<{name_width}} {rule['severity']:<10} {status}"
        output.info(line)
        if rule["description"]:
            output.info(f"  └─ {rule['description']}")


def print_diff(original: str, refactored: str, file_path: Path) -> None:
    """Imprime diff entre código original e refatorado."""
    output.header(f"Diff for {file_path.name}")

    original_lines = original.splitlines(keepends=True)
    refactored_lines = refactored.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        refactored_lines,
        fromfile=f"{file_path.name} (original)",
        tofile=f"{file_path.name} (refactored)",
        lineterm="",
    )

    diff_text = "".join(diff)
    if diff_text:
        print(diff_text)
    else:
        output.info("No changes")


def print_summary(results: list[RefactorResult], dry_run: bool) -> None:
    """Imprime resumo dos resultados."""
    modified = sum(1 for r in results if r.modified)
    failed = sum(1 for r in results if not r.success)
    unchanged = sum(1 for r in results if r.success and not r.modified)

    output.header("Summary")
    output.info(f"Total files: {len(results)}")
    output.info(f"Modified: {modified}")
    output.info(f"Unchanged: {unchanged}")
    output.info(f"Failed: {failed}")

    if dry_run:
        output.info("\n[DRY-RUN] No changes applied")
    elif failed == 0:
        output.info("\n✓ All changes applied successfully")
    else:
        output.info(f"\n⚠ {failed} files failed")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Flext Refactor Engine - Declarative code transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available rules
  python -m flext_infra refactor --list-rules
  
  # Refactor entire project (dry-run)
  python -m flext_infra refactor --project ../flext-quality --dry-run
  
  # Refactor specific file
  python -m flext_infra refactor --file src/module.py
  
  # Run only specific rules
  python -m flext_infra refactor --project ../flext-quality --rules legacy,import
  
  # Refactor all test files
  python -m flext_infra refactor --project ../flext-quality --pattern "test_*.py"
        """,
    )

    # Modo de operação
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--project",
        "-p",
        type=Path,
        help="Path do projeto a refatorar (processa src/*.py)",
    )
    mode_group.add_argument(
        "--file", "-f", type=Path, help="Path do arquivo específico a refatorar"
    )
    mode_group.add_argument(
        "--files", nargs="+", type=Path, help="Paths dos arquivos a refatorar"
    )
    mode_group.add_argument(
        "--list-rules",
        "-l",
        action="store_true",
        help="Listar regras disponíveis e sair",
    )

    # Opções de filtro
    parser.add_argument(
        "--rules",
        "-r",
        type=str,
        help="Regras específicas a executar (comma-separated, ex: legacy,import,mro)",
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Padrão de arquivos a processar (default: *.py)",
    )

    # Opções de execução
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Mostrar o que seria feito sem aplicar",
    )
    parser.add_argument(
        "--show-diff",
        "-d",
        action="store_true",
        help="Mostrar diff detalhado das mudanças",
    )
    parser.add_argument(
        "--config", "-c", type=Path, help="Path do arquivo de configuração YAML"
    )

    args = parser.parse_args()

    # Inicializar engine
    engine = FlextRefactorEngine(config_path=args.config)

    # Carregar config e regras
    result = engine.load_config()
    if not result.is_success:
        output.error(f"Config error: {result.error}")
        sys.exit(1)

    result = engine.load_rules()
    if not result.is_success:
        output.error(f"Rules error: {result.error}")
        sys.exit(1)

    # Modo: listar regras
    if args.list_rules:
        rules = engine.list_rules()
        print_rules_table(rules)
        sys.exit(0)

    # Aplicar filtros de regras
    if args.rules:
        rule_filters = [f.strip() for f in args.rules.split(",")]
        engine.set_rule_filters(rule_filters)

        # Recarregar regras com filtros
        engine.rules = []  # Limpar regras
        result = engine.load_rules()  # Recarregar com filtros
        if not result.is_success:
            output.error(f"Rules error: {result.error}")
            sys.exit(1)

    # Executar refatoração
    output.header(f"Refactoring")
    output.info(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    output.info(f"Rules: {len(engine.rules)} active")
    if args.rules:
        output.info(f"Filter: {args.rules}")

    results: list[RefactorResult] = []

    if args.project:
        output.info(f"Project: {args.project}")
        output.info(f"Pattern: {args.pattern}")
        results = engine.refactor_project(
            args.project, dry_run=args.dry_run, pattern=args.pattern
        )
    elif args.file:
        output.info(f"File: {args.file}")
        if not args.file.exists():
            output.error(f"File not found: {args.file}")
            sys.exit(1)

        # Ler código original para diff
        original_code = args.file.read_text(encoding="utf-8")
        result_single = engine.refactor_file(args.file, dry_run=args.dry_run)
        results = [result_single]

        # Mostrar diff se solicitado e arquivo foi modificado
        if args.show_diff and result_single.modified:
            # Ler código refatorado (do resultado ou do arquivo se não for dry-run)
            if args.dry_run:
                # Para dry-run, precisamos aplicar as regras novamente para obter o código
                import libcst as cst

                tree = cst.parse_module(original_code)
                for rule in engine.rules:
                    if rule.enabled:
                        tree, _ = rule.apply(tree, args.file)
                refactored_code = tree.code if hasattr(tree, "code") else str(tree)
            else:
                refactored_code = args.file.read_text(encoding="utf-8")

            print_diff(original_code, refactored_code, args.file)
    elif args.files:
        output.info(f"Files: {len(args.files)}")
        existing_files = [f for f in args.files if f.exists()]
        missing = [f for f in args.files if not f.exists()]
        for f in missing:
            output.error(f"File not found: {f}")
        results = engine.refactor_files(existing_files, dry_run=args.dry_run)

    # Imprimir resumo
    print_summary(results, args.dry_run)

    # Exit code
    failed = sum(1 for r in results if not r.success)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

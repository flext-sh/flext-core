"""Import modernizer transformer for runtime alias migration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import override

import libcst as cst
from libcst.metadata import QualifiedNameProvider, QualifiedNameSource

from flext_infra import c


class FlextInfraRefactorImportModernizer(cst.CSTTransformer):
    """Rewrite forbidden imports and replace symbols with runtime alias paths."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(
        self,
        imports_to_remove: list[str],
        symbols_to_replace: Mapping[str, str],
        runtime_aliases: set[str],
        blocked_aliases: set[str],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize import rewrite configuration and result tracking."""
        self._imports_to_remove = imports_to_remove
        self._symbols_to_replace = symbols_to_replace
        self._runtime_aliases = runtime_aliases
        self._blocked_aliases = blocked_aliases
        self._on_change = on_change
        self.modified_imports = False
        self.aliases_needed: set[str] = set()
        self.aliases_present: set[str] = set()
        self.active_symbol_replacements: dict[str, str] = {}
        self.changes: list[str] = []

    @override
    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        """Replace forbidden imports and capture symbol replacement map."""
        module_name = self._module_name_from_expr(original_node.module)
        if module_name == c.Infra.Packages.CORE_UNDERSCORE:
            imported_aliases = self._extract_import_aliases(original_node.names)
            for imported_alias in imported_aliases:
                if not isinstance(imported_alias.name, cst.Name):
                    continue
                imported_name = imported_alias.name.value
                bound_name = imported_name
                if imported_alias.asname is not None and isinstance(
                    imported_alias.asname.name,
                    cst.Name,
                ):
                    bound_name = imported_alias.asname.name.value
                if bound_name in self._runtime_aliases:
                    self.aliases_present.add(bound_name)
        for mod in self._imports_to_remove:
            if module_name != mod:
                continue
            imported_aliases = self._extract_import_aliases(original_node.names)
            if not imported_aliases:
                return updated_node
            mapped_aliases: list[cst.ImportAlias] = []
            unmapped_aliases: list[cst.ImportAlias] = []
            for imported_alias in imported_aliases:
                if not isinstance(imported_alias.name, cst.Name):
                    unmapped_aliases.append(imported_alias)
                    continue
                imported_symbol = imported_alias.name.value
                if imported_symbol not in self._symbols_to_replace:
                    unmapped_aliases.append(imported_alias)
                    continue
                mapped_aliases.append(imported_alias)
                local_symbol = imported_symbol
                if imported_alias.asname is not None and isinstance(
                    imported_alias.asname.name,
                    cst.Name,
                ):
                    local_symbol = imported_alias.asname.name.value
                alias_path = self._symbols_to_replace[imported_symbol]
                alias_root = alias_path.split(".")[0]
                if alias_root in self._blocked_aliases:
                    unmapped_aliases.append(imported_alias)
                    continue
                self.active_symbol_replacements[local_symbol] = alias_path
                self.aliases_needed.add(alias_root)
            if not mapped_aliases:
                return updated_node
            self.modified_imports = True
            self._record_change(f"Removed import: from {module_name}")
            if unmapped_aliases:
                return updated_node.with_changes(names=tuple(unmapped_aliases))
            return cst.RemovalSentinel.REMOVE
        return updated_node

    @override
    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        """Inject missing runtime aliases import at module header."""
        del original_node
        missing_aliases = sorted(self.aliases_needed - self.aliases_present)
        if not (self.modified_imports and missing_aliases):
            return updated_node
        alias_imports = [
            cst.ImportAlias(name=cst.Name(alias_name)) for alias_name in missing_aliases
        ]
        new_import = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Name(c.Infra.Packages.CORE_UNDERSCORE),
                    names=alias_imports,
                ),
            ],
        )
        body = list(updated_node.body)
        insert_idx = 0
        if (
            body
            and isinstance(body[0], cst.SimpleStatementLine)
            and (len(body[0].body) == 1)
            and isinstance(body[0].body[0], cst.Expr)
            and isinstance(body[0].body[0].value, cst.SimpleString)
        ):
            insert_idx = 1
        while insert_idx < len(body):
            stmt = body[insert_idx]
            if not isinstance(stmt, cst.SimpleStatementLine):
                break
            if len(stmt.body) != 1:
                break
            only_stmt = stmt.body[0]
            if (
                isinstance(only_stmt, cst.ImportFrom)
                and isinstance(only_stmt.module, cst.Name)
                and (only_stmt.module.value == "__future__")
            ):
                insert_idx += 1
                continue
            break
        self._record_change(
            f"Added: from flext_core import {', '.join(missing_aliases)}",
        )
        new_body = body[:insert_idx] + [new_import] + body[insert_idx:]
        return updated_node.with_changes(body=new_body)

    @override
    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.BaseExpression:
        """Replace imported symbol usages with configured runtime alias paths."""
        if original_node.value not in self.active_symbol_replacements:
            return updated_node
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
        result: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            result = cst.Attribute(value=result, attr=cst.Name(part))
        self._record_change(f"Replaced: {original_node.value} -> {alias_path}")
        return result

    def _extract_import_aliases(
        self,
        names: Sequence[cst.ImportAlias] | cst.ImportStar,
    ) -> list[cst.ImportAlias]:
        if isinstance(names, cst.ImportStar):
            return []
        return list(names)

    def _module_name_from_expr(self, module: cst.BaseExpression | None) -> str:
        if module is None:
            return ""
        if isinstance(module, cst.Name):
            return module.value
        if isinstance(module, cst.Attribute):
            parts: list[str] = []
            current: cst.BaseExpression | cst.BaseAssignTargetExpression = module
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
            return ".".join(reversed(parts))
        return ""

    def _record_change(self, message: str) -> None:
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


__all__ = ["FlextInfraRefactorImportModernizer"]

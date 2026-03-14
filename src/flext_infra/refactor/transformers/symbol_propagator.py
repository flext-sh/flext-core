"""Workspace-wide symbol propagation transformer for refactor API renames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import override

import libcst as cst
from libcst.metadata import QualifiedNameProvider, QualifiedNameSource


class FlextInfraRefactorSymbolPropagator(cst.CSTTransformer):
    """Propagate import/symbol renames safely using CST + import metadata."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(
        self,
        *,
        target_modules: set[str],
        module_renames: Mapping[str, str],
        import_symbol_renames: Mapping[str, str],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize symbol propagation configuration and change collector."""
        self._target_modules = target_modules
        self._module_renames = module_renames
        self._import_symbol_renames = import_symbol_renames
        self._on_change = on_change
        self._local_name_renames: dict[str, str] = {}
        self.changes: list[str] = []

    @override
    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.ImportFrom:
        module_name = self._module_name_from_expr(original_node.module)
        next_node = updated_node
        if module_name in self._module_renames:
            next_module = self._module_expr_from_dotted(
                self._module_renames[module_name],
            )
            next_node = next_node.with_changes(module=next_module)
            self._record_change(
                f"Renamed import module: {module_name} -> {self._module_renames[module_name]}",
            )
            module_name = self._module_renames[module_name]
        if module_name not in self._target_modules or isinstance(
            next_node.names,
            cst.ImportStar,
        ):
            return next_node
        next_aliases: list[cst.ImportAlias] = []
        changed = False
        for alias in list(next_node.names):
            if not isinstance(alias.name, cst.Name):
                next_aliases.append(alias)
                continue
            imported_name = alias.name.value
            renamed_symbol = self._import_symbol_renames.get(imported_name)
            if renamed_symbol is None:
                next_aliases.append(alias)
                continue
            changed = True
            next_alias = alias.with_changes(name=cst.Name(renamed_symbol))
            next_aliases.append(next_alias)
            local_name = imported_name
            if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                local_name = alias.asname.name.value
            if alias.asname is None:
                self._local_name_renames[imported_name] = renamed_symbol
            self._record_change(
                f"Renamed imported symbol: {imported_name} -> {renamed_symbol} (local={local_name})",
            )
        if not changed:
            return next_node
        return next_node.with_changes(names=tuple(next_aliases))

    @override
    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.BaseExpression:
        rename_to = self._local_name_renames.get(original_node.value)
        if rename_to is None:
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
        if updated_node.value == rename_to:
            return updated_node
        self._record_change(
            f"Propagated local symbol rename: {updated_node.value} -> {rename_to}",
        )
        return cst.Name(rename_to)

    def _module_expr_from_dotted(self, dotted_name: str) -> cst.BaseExpression:
        parts = [part for part in dotted_name.split(".") if part]
        if not parts:
            return cst.Name("")
        expr: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            expr = cst.Attribute(value=expr, attr=cst.Name(part))
        return expr

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


__all__ = ["FlextInfraRefactorSymbolPropagator"]

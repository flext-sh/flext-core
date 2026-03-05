from __future__ import annotations

from typing import override

import libcst as cst
from libcst.metadata import ParentNodeProvider


class NestedClassPropagationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, class_renames: dict[str, str]) -> None:
        self._class_renames = class_renames
        self._name_renames: dict[str, str] = dict(class_renames)

    @override
    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.ImportFrom:
        _ = original_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node

        updated_aliases: list[cst.ImportAlias] = []
        changed = False

        for alias in updated_node.names:
            if not isinstance(alias.name, cst.Name):
                updated_aliases.append(alias)
                continue

            rename_to = self._class_renames.get(alias.name.value)
            if rename_to is None:
                updated_aliases.append(alias)
                continue

            rename_parts = self._split_dotted(rename_to)
            import_name = rename_parts[0]
            next_alias = alias.with_changes(name=cst.Name(import_name))
            updated_aliases.append(next_alias)
            changed = True

            if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                local_name = alias.asname.name.value
                if len(rename_parts) > 1:
                    local_rename = ".".join((local_name, *rename_parts[1:]))
                    self._name_renames[local_name] = local_rename
                else:
                    self._name_renames[local_name] = local_name

        if not changed:
            return updated_node
        return updated_node.with_changes(names=tuple(updated_aliases))

    @override
    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.BaseExpression:
        rename_to = self._name_renames.get(original_node.value)
        if rename_to is None:
            return updated_node
        if self._should_skip_name(original_node):
            return updated_node
        return self._expression_from_dotted(rename_to)

    @override
    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        rename_to = self._class_renames.get(original_node.attr.value)
        if rename_to is None:
            return updated_node

        rename_parts = self._split_dotted(rename_to)
        if not rename_parts:
            return updated_node

        if self._rightmost_name(updated_node.value) == rename_parts[0]:
            suffix_parts = rename_parts[1:]
            if not suffix_parts:
                return updated_node.with_changes(attr=cst.Name(rename_parts[0]))
            return self._attribute_from_base(updated_node.value, suffix_parts)

        return self._attribute_from_base(updated_node.value, rename_parts)

    def _should_skip_name(self, node: cst.Name) -> bool:
        parent = self.get_metadata(ParentNodeProvider, node, default=None)
        if parent is None:
            return False

        if isinstance(parent, cst.Attribute) and parent.attr is node:
            return True
        if isinstance(parent, cst.ImportAlias) and (
            parent.name is node
            or (parent.asname is not None and parent.asname.name is node)
        ):
            return True
        if isinstance(parent, cst.ClassDef) and parent.name is node:
            return True
        if isinstance(parent, cst.FunctionDef) and parent.name is node:
            return True
        if isinstance(parent, cst.Param) and parent.name is node:
            return True
        if isinstance(parent, cst.AsName) and parent.name is node:
            return True
        return False

    def _expression_from_dotted(self, dotted_name: str) -> cst.BaseExpression:
        parts = self._split_dotted(dotted_name)
        expr: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            expr = cst.Attribute(value=expr, attr=cst.Name(part))
        return expr

    def _attribute_from_base(
        self,
        base: cst.BaseExpression,
        dotted_parts: list[str],
    ) -> cst.BaseExpression:
        expr: cst.BaseExpression = base
        for part in dotted_parts:
            expr = cst.Attribute(value=expr, attr=cst.Name(part))
        return expr

    def _rightmost_name(
        self,
        expr: cst.BaseExpression,
    ) -> str | None:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        return None

    def _split_dotted(self, dotted_name: str) -> list[str]:
        return [part for part in dotted_name.split(".") if part]


__all__ = ["NestedClassPropagationTransformer"]

"""CST transformer for propagating nested class references after nesting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

import libcst as cst
from libcst.metadata import ParentNodeProvider

from flext_infra import m, t
from flext_infra.refactor.transformers.policy import (
    FlextInfraRefactorTransformerPolicyUtilities,
)


class NestedClassPropagationTransformer(cst.CSTTransformer):
    """Propagate import and name references after classes are nested into namespaces."""

    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(
        self,
        class_renames: dict[str, str],
        policy_context: t.Infra.PolicyContext | None = None,
        class_families: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize with class rename mappings and optional policy context."""
        self._class_renames = class_renames
        self._name_renames: dict[str, str] = dict(class_renames)
        self._policy_context = policy_context
        self._class_families = class_families or {}

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
            if not self._should_propagate(alias.name.value, "propagate_imports"):
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
        if not self._should_propagate(original_node.value, "propagate_name_references"):
            return updated_node
        if self._should_skip_name(original_node):
            return updated_node
        if self._blocked_by_prefix(original_node.value):
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
        if not self._should_propagate(
            original_node.attr.value,
            "propagate_attribute_references",
        ):
            return updated_node
        if self._blocked_by_prefix(original_node.attr.value):
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
        try:
            parent = self.get_metadata(ParentNodeProvider, node)
        except KeyError:
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
        return bool(isinstance(parent, cst.AsName) and parent.name is node)

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

    def _rightmost_name(self, expr: cst.BaseExpression) -> str | None:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        return None

    def _split_dotted(self, dotted_name: str) -> list[str]:
        return [part for part in dotted_name.split(".") if part]

    def _should_propagate(self, symbol_name: str, policy_key: str) -> bool:
        policy = self._policy_for_symbol(symbol_name)
        if policy is None:
            return True
        if policy_key == "propagate_imports":
            return policy.propagate_imports
        if policy_key == "propagate_name_references":
            return policy.propagate_name_references
        if policy_key == "propagate_attribute_references":
            return policy.propagate_attribute_references
        return False

    def _blocked_by_prefix(self, symbol_name: str) -> bool:
        policy = self._policy_for_symbol(symbol_name)
        if policy is None:
            return False
        blocked_prefixes = tuple(policy.blocked_reference_prefixes)
        return any(symbol_name.startswith(prefix) for prefix in blocked_prefixes)

    def _policy_for_symbol(
        self,
        symbol_name: str,
    ) -> m.Infra.Refactor.ClassNestingPolicy | None:
        return FlextInfraRefactorTransformerPolicyUtilities.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._class_families,
            symbol_name=symbol_name,
        )


__all__ = ["NestedClassPropagationTransformer"]

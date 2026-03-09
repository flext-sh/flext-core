"""Class reconstructor transformer for method ordering rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import c, m, t


class FlextInfraRefactorClassReconstructor(cst.CSTTransformer):
    """Reorder class methods based on declarative ordering configuration."""

    def __init__(
        self,
        order_config: list[t.Infra.RuleConfig],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize with rule order config and optional change callback."""
        try:
            self._order_config = TypeAdapter(
                list[m.Infra.Refactor.RuleConfigs.MethodOrderRule],
            ).validate_python(order_config)
        except ValidationError:
            self._order_config = []
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """Sort methods in every contiguous method block of a class body."""
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        body = list(updated_node.body.body)
        if not body:
            return updated_node
        new_body: list[cst.BaseStatement] = list(body)
        block_start = 0
        changed_blocks = 0
        reordered_methods_total = 0
        while block_start < len(body):
            if not isinstance(body[block_start], cst.FunctionDef):
                block_start += 1
                continue
            block_end = block_start
            while block_end < len(body) and isinstance(
                body[block_end],
                cst.FunctionDef,
            ):
                block_end += 1
            method_indices = list(range(block_start, block_end))
            methods: list[m.Infra.Refactor.MethodInfo] = []
            for idx in method_indices:
                block_item = body[idx]
                if isinstance(block_item, cst.FunctionDef):
                    methods.append(self._analyze_method(block_item))
            sorted_methods = self._sort_methods(methods)
            original_method_names = [method.name for method in methods]
            sorted_method_names = [method.name for method in sorted_methods]
            if original_method_names != sorted_method_names:
                changed_blocks += 1
                reordered_methods_total += len(methods)
                for idx, method in zip(method_indices, sorted_methods, strict=False):
                    new_body[idx] = method.node
            block_start = block_end
        if changed_blocks == 0:
            return updated_node
        self._record_change(
            f"Reordered {reordered_methods_total} methods in class {original_node.name.value} across {changed_blocks} contiguous block(s)",
        )
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body),
        )

    def _analyze_method(self, node: cst.FunctionDef) -> m.Infra.Refactor.MethodInfo:
        name = node.name.value
        decorators: list[str] = []
        for dec in node.decorators:
            if isinstance(dec.decorator, cst.Name):
                decorators.append(dec.decorator.value)
            elif isinstance(dec.decorator, cst.Attribute):
                decorators.append(dec.decorator.attr.value)
        category = self._categorize(name, decorators)
        return m.Infra.Refactor.MethodInfo(
            name=name,
            category=category,
            node=node,
            decorators=decorators,
        )

    def _categorize(self, name: str, decorators: list[str]) -> str:
        if any(
            decorator_name in decorators
            for decorator_name in ["property", "cached_property", "computed_field"]
        ):
            return c.Infra.Refactor.MethodCategory.PROPERTY
        if "staticmethod" in decorators:
            return c.Infra.Refactor.MethodCategory.STATIC
        if "classmethod" in decorators:
            return c.Infra.Refactor.MethodCategory.CLASS
        if name.startswith("__") and name.endswith("__"):
            return c.Infra.Refactor.MethodCategory.MAGIC
        if name.startswith("__"):
            return c.Infra.Refactor.MethodCategory.PRIVATE
        if name.startswith("_"):
            return c.Infra.Refactor.MethodCategory.PROTECTED
        return c.Infra.Refactor.MethodCategory.PUBLIC

    def _record_change(self, message: str) -> None:
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)

    def _sort_methods(
        self,
        methods: list[m.Infra.Refactor.MethodInfo],
    ) -> list[m.Infra.Refactor.MethodInfo]:

        def matches_rule(
            method: m.Infra.Refactor.MethodInfo,
            rule_config: m.Infra.Refactor.RuleConfigs.MethodOrderRule,
        ) -> bool:
            decorators = set(method.decorators)
            exclude_decorators = set(rule_config.exclude_decorators)
            if exclude_decorators and decorators.intersection(exclude_decorators):
                return False
            visibility = rule_config.visibility
            if visibility == "public" and method.name.startswith("_"):
                return False
            if visibility == "protected" and (
                not (method.name.startswith("_") and (not method.name.startswith("__")))
            ):
                return False
            if visibility == "private" and (
                not (method.name.startswith("__") and (not method.name.endswith("__")))
            ):
                return False
            rule_decorators = rule_config.decorators
            if rule_decorators and (not decorators.intersection(rule_decorators)):
                return False
            patterns = rule_config.patterns
            if patterns:
                matched = False
                for pattern in patterns:
                    if re.match(pattern, method.name):
                        matched = True
                if not matched:
                    return False
            return True

        def sort_key(method: m.Infra.Refactor.MethodInfo) -> tuple[int, int, str]:
            for idx, rule_config in enumerate(self._order_config):
                if rule_config.category == "class_attributes":
                    continue
                if not matches_rule(method, rule_config):
                    continue
                explicit_order = rule_config.order
                if explicit_order:
                    if method.name in explicit_order:
                        return (idx, explicit_order.index(method.name), method.name)
                    if "*" in explicit_order:
                        return (idx, explicit_order.index("*") + 1, method.name)
                return (idx, 0, method.name)
            return (len(self._order_config), 0, method.name)

        return sorted(methods, key=sort_key)


__all__ = ["FlextInfraRefactorClassReconstructor"]

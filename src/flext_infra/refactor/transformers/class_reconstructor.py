"""Class reconstructor transformer for method ordering rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast, override

import libcst as cst

from flext_infra import c
from flext_infra.refactor.method_info import FlextInfraRefactorMethodInfo


class FlextInfraRefactorClassReconstructor(cst.CSTTransformer):
    """Reorder class methods based on declarative ordering configuration."""

    def __init__(
        self,
        order_config: list[dict[str, Any]],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize with rule order config and optional change callback."""
        self._order_config = order_config
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """Sort methods in contiguous method blocks of a class body."""
        methods: list[FlextInfraRefactorMethodInfo] = []
        method_indices: list[int] = []

        # Cast the body to list[Any] to bypass Pyrefly's incorrect inference of Union type
        body = cast("list[Any]", list(updated_node.body.body))

        for idx, stmt in enumerate(body):
            if isinstance(stmt, cst.FunctionDef):
                methods.append(self._analyze_method(stmt))
                method_indices.append(idx)

        if not methods:
            return updated_node

        first_method_idx = method_indices[0]
        last_method_idx = method_indices[-1]
        if any(
            not isinstance(body[idx], cst.FunctionDef)
            for idx in range(first_method_idx, last_method_idx + 1)
        ):
            return updated_node

        sorted_methods = self._sort_methods(methods)
        original_method_names = [method.name for method in methods]
        sorted_method_names = [method.name for method in sorted_methods]
        if original_method_names == sorted_method_names:
            return updated_node

        new_body = list(body)
        for idx, method in zip(method_indices, sorted_methods, strict=False):
            new_body[idx] = method.node

        self._record_change(
            f"Reordered {len(methods)} methods in class {original_node.name.value}"
        )
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )

    def _analyze_method(self, node: cst.FunctionDef) -> FlextInfraRefactorMethodInfo:
        name = node.name.value
        decorators: list[str] = []

        for dec in node.decorators:
            if isinstance(dec.decorator, cst.Name):
                decorators.append(dec.decorator.value)
            elif isinstance(dec.decorator, cst.Attribute):
                decorators.append(dec.decorator.attr.value)

        category = self._categorize(name, decorators)
        return FlextInfraRefactorMethodInfo(
            name=name,
            category=category,
            node=node,
            decorators=decorators,
        )

    def _categorize(
        self,
        name: str,
        decorators: list[str],
    ) -> c.Infra.Refactor.MethodCategory:
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

    def _sort_methods(
        self,
        methods: list[FlextInfraRefactorMethodInfo],
    ) -> list[FlextInfraRefactorMethodInfo]:
        def matches_rule(
            method: FlextInfraRefactorMethodInfo,
            rule_config: dict[str, Any],
        ) -> bool:
            decorators = set(method.decorators)
            exclude_decorators = set(rule_config.get("exclude_decorators", []))
            if exclude_decorators and decorators.intersection(exclude_decorators):
                return False

            visibility = rule_config.get("visibility")
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

            rule_decorators = rule_config.get("decorators", [])
            if rule_decorators and not decorators.intersection(rule_decorators):
                return False

            patterns = rule_config.get("patterns", [])
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

        def sort_key(method: FlextInfraRefactorMethodInfo) -> tuple[int, int, str]:
            for idx, rule_config in enumerate(self._order_config):
                if rule_config.get("category") == "class_attributes":
                    continue
                if not matches_rule(method, rule_config):
                    continue
                explicit_order = rule_config.get("order", [])
                if explicit_order:
                    if method.name in explicit_order:
                        return (idx, explicit_order.index(method.name), method.name)
                    if "*" in explicit_order:
                        return (idx, explicit_order.index("*") + 1, method.name)
                return (idx, 0, method.name)
            return (len(self._order_config), 0, method.name)

        return sorted(methods, key=sort_key)

    def _record_change(self, message: str) -> None:
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


__all__ = ["FlextInfraRefactorClassReconstructor"]

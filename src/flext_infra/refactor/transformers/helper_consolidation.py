from __future__ import annotations

from collections import defaultdict
from typing import override

import libcst as cst


class HelperConsolidationTransformer(cst.CSTTransformer):
    def __init__(self, helper_mappings: dict[str, str]) -> None:
        self._helper_mappings = helper_mappings
        self._scope_depth = 0
        self._existing_namespaces: set[str] = set()
        self._collected_helpers: dict[str, list[cst.FunctionDef]] = defaultdict(list)

    @override
    def visit_Module(self, node: cst.Module) -> bool:
        self._existing_namespaces = {
            statement.name.value
            for statement in node.body
            if isinstance(statement, cst.ClassDef)
        }
        return True

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        _ = node
        self._scope_depth += 1
        return True

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        _ = node
        self._scope_depth += 1
        return True

    @override
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        _ = original_node
        if not isinstance(updated_node.func, cst.Name):
            return updated_node

        helper_name = updated_node.func.value
        target_namespace = self._helper_mappings.get(helper_name)
        if target_namespace is None:
            return updated_node

        return updated_node.with_changes(
            func=cst.Attribute(
                value=cst.Name(target_namespace),
                attr=cst.Name(helper_name),
            )
        )

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        _ = original_node
        self._scope_depth -= 1
        return updated_node

    @override
    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef | cst.RemovalSentinel:
        is_top_level_function = self._scope_depth == 1
        self._scope_depth -= 1

        if not is_top_level_function:
            return updated_node

        target_namespace = self._helper_mappings.get(original_node.name.value)
        if target_namespace is None:
            return updated_node

        helper_method = self._ensure_staticmethod(updated_node)
        self._collected_helpers[target_namespace].append(helper_method)
        return cst.RemoveFromParent()

    @override
    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        _ = original_node
        if not self._collected_helpers:
            return updated_node

        module_body = list(updated_node.body)
        for namespace, helper_methods in self._collected_helpers.items():
            namespace_index = None
            if namespace in self._existing_namespaces:
                namespace_index = next(
                    (
                        index
                        for index, statement in enumerate(module_body)
                        if isinstance(statement, cst.ClassDef)
                        and statement.name.value == namespace
                    ),
                    None,
                )

            if namespace_index is None:
                module_body.append(
                    cst.ClassDef(
                        name=cst.Name(namespace),
                        body=cst.IndentedBlock(body=tuple(helper_methods)),
                    )
                )
                self._existing_namespaces.add(namespace)
                continue

            namespace_class = module_body[namespace_index]
            if not isinstance(namespace_class, cst.ClassDef):
                continue

            existing_method_names = {
                statement.name.value
                for statement in namespace_class.body.body
                if isinstance(statement, cst.FunctionDef)
            }
            methods_to_insert = [
                method
                for method in helper_methods
                if method.name.value not in existing_method_names
            ]
            if not methods_to_insert:
                continue

            namespace_body = [
                statement
                for statement in namespace_class.body.body
                if not (
                    isinstance(statement, cst.SimpleStatementLine)
                    and self._is_pass_only(statement)
                )
            ]
            namespace_body.extend(methods_to_insert)

            module_body[namespace_index] = namespace_class.with_changes(
                body=namespace_class.body.with_changes(body=tuple(namespace_body))
            )

        return updated_node.with_changes(body=tuple(module_body))

    def _ensure_staticmethod(self, function_node: cst.FunctionDef) -> cst.FunctionDef:
        if self._has_staticmethod(function_node):
            return function_node

        decorators = [cst.Decorator(decorator=cst.Name("staticmethod"))]
        decorators.extend(function_node.decorators)
        return function_node.with_changes(decorators=tuple(decorators))

    def _has_staticmethod(self, function_node: cst.FunctionDef) -> bool:
        for decorator in function_node.decorators:
            decorated = decorator.decorator
            if isinstance(decorated, cst.Name) and decorated.value == "staticmethod":
                return True
            if (
                isinstance(decorated, cst.Attribute)
                and decorated.attr.value == "staticmethod"
            ):
                return True
        return False

    def _is_pass_only(self, statement: cst.SimpleStatementLine) -> bool:
        return len(statement.body) == 1 and isinstance(statement.body[0], cst.Pass)


__all__ = ["HelperConsolidationTransformer"]

from __future__ import annotations

from collections import defaultdict
from typing import override

import libcst as cst


class ClassNestingTransformer(cst.CSTTransformer):
    def __init__(self, mappings: dict[str, str]) -> None:
        self._mappings = mappings
        self._class_depth = 0
        self._existing_namespaces: set[str] = set()
        self._collected_nested: dict[str, list[cst.ClassDef]] = defaultdict(list)

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
        self._class_depth += 1
        return True

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef | cst.RemovalSentinel:
        is_top_level_class = self._class_depth == 1
        self._class_depth -= 1

        if not is_top_level_class:
            return updated_node

        class_name = original_node.name.value
        target_namespace = self._mappings.get(class_name)
        if target_namespace is None:
            return updated_node

        self._collected_nested[target_namespace].append(updated_node)
        return cst.RemoveFromParent()

    @override
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        _ = original_node
        if not self._collected_nested:
            return updated_node

        module_body = list(updated_node.body)
        for namespace, nested_classes in self._collected_nested.items():
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
                        body=cst.IndentedBlock(body=tuple(nested_classes)),
                    )
                )
                self._existing_namespaces.add(namespace)
                continue

            namespace_class = module_body[namespace_index]
            if not isinstance(namespace_class, cst.ClassDef):
                continue

            existing_nested_names = {
                statement.name.value
                for statement in namespace_class.body.body
                if isinstance(statement, cst.ClassDef)
            }
            classes_to_insert = [
                class_def
                for class_def in nested_classes
                if class_def.name.value not in existing_nested_names
            ]
            if not classes_to_insert:
                continue

            namespace_body = [
                statement
                for statement in namespace_class.body.body
                if not (
                    isinstance(statement, cst.SimpleStatementLine)
                    and self._is_pass_only(statement)
                )
            ]
            namespace_body.extend(classes_to_insert)
            module_body[namespace_index] = namespace_class.with_changes(
                body=namespace_class.body.with_changes(body=tuple(namespace_body))
            )

        return updated_node.with_changes(body=tuple(module_body))

    def _is_pass_only(self, statement: cst.SimpleStatementLine) -> bool:
        return len(statement.body) == 1 and isinstance(statement.body[0], cst.Pass)


__all__ = ["ClassNestingTransformer"]

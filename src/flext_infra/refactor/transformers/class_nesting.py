"""CST transformer for nesting top-level classes into namespace classes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import override

import libcst as cst

from flext_infra import m, t
from flext_infra.refactor._utilities import FlextInfraRefactorUtilities


class FlextInfraRefactorClassNestingTransformer(cst.CSTTransformer):
    """Transform top-level classes into nested classes under namespace parents."""

    def __init__(
        self,
        mappings: dict[str, str],
        policy_context: t.Infra.PolicyContext,
        class_families: t.Infra.ClassFamilyMap,
    ) -> None:
        self._mappings = mappings
        self._policy_context = policy_context
        self._class_families = class_families
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
        if not self._is_nesting_allowed(class_name, target_namespace):
            return updated_node

        self._collected_nested[target_namespace].append(updated_node)
        return cst.RemoveFromParent()

    @override
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        _ = original_node
        if len(self._collected_nested) == 0:
            return updated_node

        module_body = list(updated_node.body)
        for namespace, nested_classes in self._collected_nested.items():
            if not self._can_operate_for_namespace(
                nested_classes=nested_classes,
                target_namespace=namespace,
                operation_key="allow_namespace_creation",
            ):
                continue

            namespace_index = self._namespace_index(
                module_body=module_body, namespace=namespace
            )

            if namespace_index is not None and not self._can_operate_for_namespace(
                nested_classes=nested_classes,
                target_namespace=namespace,
                operation_key="allow_existing_namespace_merge",
            ):
                continue

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
            if len(classes_to_insert) == 0:
                continue

            namespace_body = [
                statement
                for statement in namespace_class.body.body
                if not self._is_pass_only_statement(statement)
            ]
            namespace_body.extend(classes_to_insert)
            module_body[namespace_index] = namespace_class.with_changes(
                body=namespace_class.body.with_changes(body=tuple(namespace_body))
            )

        return updated_node.with_changes(body=tuple(module_body))

    def _namespace_index(
        self,
        *,
        module_body: Sequence[cst.CSTNode],
        namespace: str,
    ) -> int | None:
        if namespace not in self._existing_namespaces:
            return None
        for index, statement in enumerate(module_body):
            if (
                isinstance(statement, cst.ClassDef)
                and statement.name.value == namespace
            ):
                return index
        return None

    def _is_pass_only_statement(self, statement: cst.CSTNode) -> bool:
        return (
            isinstance(statement, cst.SimpleStatementLine)
            and len(statement.body) == 1
            and isinstance(statement.body[0], cst.Pass)
        )

    def _policy_for_symbol(
        self,
        *,
        symbol_name: str,
    ) -> m.Infra.Refactor.ClassNestingPolicy:
        return FlextInfraRefactorUtilities.policy_for_symbol(
            symbol_name=symbol_name,
            policy_context=self._policy_context,
            class_families=self._class_families,
        )

    def _policy_flag(
        self,
        *,
        policy: m.Infra.Refactor.ClassNestingPolicy,
        key: str,
    ) -> bool:
        return FlextInfraRefactorUtilities.policy_bool(policy=policy, key=key)

    def _target_allowed(
        self,
        *,
        policy: m.Infra.Refactor.ClassNestingPolicy,
        target_namespace: str,
    ) -> bool:
        return FlextInfraRefactorUtilities.target_allowed(
            policy=policy,
            target_namespace=target_namespace,
        )

    def _is_nesting_allowed(self, class_name: str, target_namespace: str) -> bool:
        policy = self._policy_for_symbol(symbol_name=class_name)
        if not self._policy_flag(policy=policy, key="enable_class_nesting"):
            return False
        return self._target_allowed(policy=policy, target_namespace=target_namespace)

    def _can_operate_for_namespace(
        self,
        *,
        nested_classes: list[cst.ClassDef],
        target_namespace: str,
        operation_key: str,
    ) -> bool:
        for class_def in nested_classes:
            policy = self._policy_for_symbol(symbol_name=class_def.name.value)
            if not self._policy_flag(policy=policy, key=operation_key):
                return False
            if not self._target_allowed(
                policy=policy, target_namespace=target_namespace
            ):
                return False
        return True


__all__ = ["FlextInfraRefactorClassNestingTransformer"]

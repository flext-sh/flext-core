"""CST transformer for nesting top-level classes into namespace classes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import override

import libcst as cst

from flext_infra import m, t
from flext_infra.refactor.transformers.policy import (
    FlextInfraRefactorTransformerPolicyUtilities,
)


class FlextInfraRefactorClassNestingTransformer(cst.CSTTransformer):
    """Transform top-level classes into nested classes under namespace parents."""

    def __init__(
        self,
        mappings: dict[str, str],
        policy_context: t.Infra.PolicyContext,
        class_families: Mapping[str, str],
    ) -> None:
        """Store mappings and policy context used during class nesting."""
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
        policy = self._policy_for_symbol(class_name)
        if policy is not None:
            if not policy.enable_class_nesting:
                return updated_node
            if not FlextInfraRefactorTransformerPolicyUtilities.target_allowed(
                policy=policy,
                target_namespace=target_namespace,
            ):
                return updated_node
        self._collected_nested[target_namespace].append(updated_node)
        return cst.RemoveFromParent()

    @override
    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        _ = original_node
        if not self._collected_nested:
            return updated_node
        module_body = list(updated_node.body)
        for namespace, nested_classes in self._collected_nested.items():
            if not self._namespace_operation_allowed(
                nested_classes=nested_classes,
                target_namespace=namespace,
                operation="creation",
            ):
                continue
            namespace_index = self._namespace_index(module_body, namespace)
            if namespace_index is not None and (
                not self._namespace_operation_allowed(
                    nested_classes=nested_classes,
                    target_namespace=namespace,
                    operation="merge",
                )
            ):
                continue
            if namespace_index is None:
                module_body.append(
                    cst.ClassDef(
                        name=cst.Name(namespace),
                        body=cst.IndentedBlock(body=tuple(nested_classes)),
                    ),
                )
                self._existing_namespaces.add(namespace)
                continue
            namespace_class = module_body[namespace_index]
            if not isinstance(namespace_class, cst.ClassDef):
                continue
            classes_to_insert = self._deduplicated_nested(
                namespace_class=namespace_class,
                nested_classes=nested_classes,
            )
            if not classes_to_insert:
                continue
            module_body[namespace_index] = namespace_class.with_changes(
                body=namespace_class.body.with_changes(
                    body=tuple(
                        self._merged_namespace_body(namespace_class, classes_to_insert),
                    ),
                ),
            )
        return updated_node.with_changes(body=tuple(module_body))

    def _policy_for_symbol(
        self,
        symbol_name: str,
    ) -> m.Infra.Refactor.ClassNestingPolicy | None:
        return FlextInfraRefactorTransformerPolicyUtilities.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._class_families,
            symbol_name=symbol_name,
        )

    def _namespace_operation_allowed(
        self,
        *,
        nested_classes: list[cst.ClassDef],
        target_namespace: str,
        operation: str,
    ) -> bool:
        for class_def in nested_classes:
            policy = self._policy_for_symbol(class_def.name.value)
            if policy is None:
                continue
            if operation == "creation" and (not policy.allow_namespace_creation):
                return False
            if operation == "merge" and (not policy.allow_existing_namespace_merge):
                return False
            if not FlextInfraRefactorTransformerPolicyUtilities.target_allowed(
                policy=policy,
                target_namespace=target_namespace,
            ):
                return False
        return True

    def _namespace_index(
        self,
        module_body: Sequence[cst.CSTNode],
        namespace: str,
    ) -> int | None:
        if namespace not in self._existing_namespaces:
            return None
        return next(
            (
                index
                for index, statement in enumerate(module_body)
                if isinstance(statement, cst.ClassDef)
                and statement.name.value == namespace
            ),
            None,
        )

    def _deduplicated_nested(
        self,
        *,
        namespace_class: cst.ClassDef,
        nested_classes: list[cst.ClassDef],
    ) -> list[cst.ClassDef]:
        existing_nested_names = {
            statement.name.value
            for statement in namespace_class.body.body
            if isinstance(statement, cst.ClassDef)
        }
        return [
            class_def
            for class_def in nested_classes
            if class_def.name.value not in existing_nested_names
        ]

    def _merged_namespace_body(
        self,
        namespace_class: cst.ClassDef,
        classes_to_insert: list[cst.ClassDef],
    ) -> list[cst.BaseStatement]:
        namespace_body: list[cst.BaseStatement] = []
        for statement in namespace_class.body.body:
            if isinstance(statement, cst.BaseSmallStatement):
                continue
            typed_statement: cst.BaseStatement = statement
            if self._is_pass_line(typed_statement):
                continue
            namespace_body.append(typed_statement)
        namespace_body.extend(classes_to_insert)
        return namespace_body

    def _is_pass_line(self, statement: cst.BaseStatement) -> bool:
        if not isinstance(statement, cst.SimpleStatementLine):
            return False
        return len(statement.body) == 1 and isinstance(statement.body[0], cst.Pass)


__all__ = ["FlextInfraRefactorClassNestingTransformer"]

"""CST transformer for nesting top-level classes into namespace classes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict, override

import libcst as cst


class FamilyPolicy(TypedDict):
    enable_class_nesting: bool
    allow_namespace_creation: bool
    allow_existing_namespace_merge: bool
    allowed_targets: list[str] | tuple[str, ...]
    forbidden_targets: list[str] | tuple[str, ...]


type PolicyContext = Mapping[str, FamilyPolicy]


class FlextInfraRefactorClassNestingTransformer(cst.CSTTransformer):
    """Transform top-level classes into nested classes under namespace parents."""

    def __init__(
        self,
        mappings: dict[str, str],
        policy_context: PolicyContext,
        class_families: Mapping[str, str],
    ) -> None:
        """Initialize with class-to-namespace mappings and strict policy context."""
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
        if not self._collected_nested:
            return updated_node

        module_body = list(updated_node.body)
        for namespace, nested_classes in self._collected_nested.items():
            if not self._can_operate_for_namespace(
                nested_classes,
                namespace,
                "allow_namespace_creation",
            ):
                continue

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

            if namespace_index is not None and not self._can_operate_for_namespace(
                nested_classes,
                namespace,
                "allow_existing_namespace_merge",
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

    def _is_nesting_allowed(self, class_name: str, target_namespace: str) -> bool:
        policy = self._policy_for_symbol(class_name)
        if not self._bool_from_policy(policy, "enable_class_nesting"):
            return False
        return self._target_allowed(policy, target_namespace)

    def _can_operate_for_namespace(
        self,
        nested_classes: list[cst.ClassDef],
        target_namespace: str,
        operation_key: str,
    ) -> bool:
        for class_def in nested_classes:
            policy = self._policy_for_symbol(class_def.name.value)
            if not self._bool_from_policy(policy, operation_key):
                return False
            if not self._target_allowed(policy, target_namespace):
                return False
        return True

    def _policy_for_symbol(self, symbol_name: str) -> FamilyPolicy:
        family = self._class_families.get(symbol_name)
        if family is None:
            msg = f"missing class family mapping for symbol: {symbol_name}"
            raise ValueError(msg)
        policy = self._policy_context.get(family)
        if policy is None:
            msg = f"missing policy for family: {family}"
            raise ValueError(msg)
        return policy

    def _bool_from_policy(
        self,
        policy: FamilyPolicy,
        key: str,
    ) -> bool:
        raw = policy.get(key)
        if isinstance(raw, bool):
            return raw
        msg = f"policy key {key!r} must be bool"
        raise ValueError(msg)

    def _target_allowed(self, policy: FamilyPolicy, target_namespace: str) -> bool:
        allowed = self._string_collection(policy.get("allowed_targets"))
        if allowed and target_namespace not in allowed:
            return False

        forbidden = self._string_collection(policy.get("forbidden_targets"))
        return target_namespace not in forbidden

    def _string_collection(
        self,
        value: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """Extract a tuple of strings from a policy value."""
        return tuple(value)


__all__ = ["FlextInfraRefactorClassNestingTransformer"]

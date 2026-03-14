"""CST transformer for consolidating helper functions into namespace classes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import override

import libcst as cst

from flext_infra import m, t
from flext_infra.refactor.transformers.policy import (
    FlextInfraRefactorTransformerPolicyUtilities,
)


class HelperConsolidationTransformer(cst.CSTTransformer):
    """Move top-level helper functions into target namespace classes."""

    def __init__(
        self,
        helper_mappings: dict[str, str],
        policy_context: t.Infra.PolicyContext | None = None,
        helper_families: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize with helper-to-namespace mappings and optional policy context."""
        self._helper_mappings = helper_mappings
        self._policy_context = policy_context
        self._helper_families = helper_families or {}
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
        if not self._is_call_rewrite_allowed(helper_name, target_namespace):
            return updated_node
        return updated_node.with_changes(
            func=cst.Attribute(
                value=cst.Name(target_namespace),
                attr=cst.Name(helper_name),
            ),
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
        if not self._is_helper_move_allowed(original_node.name.value, target_namespace):
            return updated_node
        if not self._signature_allowed(original_node):
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
                    ),
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
                body=namespace_class.body.with_changes(body=tuple(namespace_body)),
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

    def _is_call_rewrite_allowed(self, helper_name: str, target_namespace: str) -> bool:
        policy = self._policy_for_helper(helper_name)
        if policy is None:
            return True
        if not policy.allow_helper_call_rewrite:
            return False
        return FlextInfraRefactorTransformerPolicyUtilities.target_allowed(
            policy=policy,
            target_namespace=target_namespace,
        )

    def _is_helper_move_allowed(self, helper_name: str, target_namespace: str) -> bool:
        policy = self._policy_for_helper(helper_name)
        if policy is None:
            return True
        if not policy.enable_helper_consolidation:
            return False
        return FlextInfraRefactorTransformerPolicyUtilities.target_allowed(
            policy=policy,
            target_namespace=target_namespace,
        )

    def _signature_allowed(self, function_node: cst.FunctionDef) -> bool:
        policy = self._policy_for_helper(function_node.name.value)
        if policy is None:
            return True
        if not policy.require_signature_validation:
            return True
        required = tuple(policy.required_parameters)
        forbidden = tuple(policy.forbidden_parameters)
        allow_vararg = policy.allow_vararg
        allow_kwarg = policy.allow_kwarg
        allow_positional_only = policy.allow_positional_only_params
        allow_keyword_only = policy.allow_keyword_only_params
        seen_parameters: set[str] = set()
        parameters = function_node.params
        if isinstance(parameters.star_arg, cst.Param | cst.ParamStar) and (
            not allow_vararg
        ):
            return False
        if parameters.star_kwarg is not None and (not allow_kwarg):
            return False
        if parameters.posonly_params and (not allow_positional_only):
            return False
        if parameters.kwonly_params and (not allow_keyword_only):
            return False
        seen_parameters.update(param.name.value for param in parameters.posonly_params)
        seen_parameters.update(param.name.value for param in parameters.params)
        seen_parameters.update(param.name.value for param in parameters.kwonly_params)
        for required_parameter in required:
            if required_parameter not in seen_parameters:
                return False
        for forbidden_parameter in forbidden:
            if forbidden_parameter in seen_parameters:
                return False
        return True

    def _policy_for_helper(
        self,
        helper_name: str,
    ) -> m.Infra.Refactor.ClassNestingPolicy | None:
        return FlextInfraRefactorTransformerPolicyUtilities.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._helper_families,
            symbol_name=helper_name,
        )


__all__ = ["HelperConsolidationTransformer"]
